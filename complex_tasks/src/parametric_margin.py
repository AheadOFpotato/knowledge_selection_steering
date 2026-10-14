#!/usr/bin/env python3
"""Parametric confidence margin m_i = log P(org) - log P(sub) on clean prompts."""
from __future__ import annotations

import math
from typing import Any

import torch

from chat_prompt import format_prompt_for_inference
from stress_test_common import clean_prompt


def _answer_token_variants(answer: str, tokenizer) -> list[list[int]]:
    """Token-id sequences for an answer string (with/without leading space)."""
    a = str(answer).strip()
    if not a:
        return []
    cands = [a, f" {a}"]
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for c in cands:
        ids = tokenizer.encode(c, add_special_tokens=False)
        key = tuple(ids)
        if ids and key not in seen:
            seen.add(key)
            out.append(ids)
    return out


@torch.inference_mode()
def sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    answer_token_ids: list[int],
    *,
    model_key: str = "",
    use_chat_template: bool = True,
) -> float:
    if not answer_token_ids:
        return float("-inf")
    formatted = format_prompt_for_inference(
        tokenizer, prompt, model_key, use_chat_template=use_chat_template
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    full_ids = prompt_ids + answer_token_ids
    input_ids = torch.tensor([full_ids], device=model.device)
    out = model(input_ids=input_ids)
    logits = out.logits[0]
    start = len(prompt_ids)
    total = 0.0
    for j, tid in enumerate(answer_token_ids):
        pos = start + j - 1
        if pos < 0:
            continue
        lp = torch.log_softmax(logits[pos], dim=-1)[tid].item()
        total += lp
    return float(total)


def best_answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answers: list[str],
    *,
    model_key: str = "",
    use_chat_template: bool = True,
) -> float:
    best = float("-inf")
    for ans in answers:
        for ids in _answer_token_variants(ans, tokenizer):
            lp = sequence_logprob(
                model,
                tokenizer,
                prompt,
                ids,
                model_key=model_key,
                use_chat_template=use_chat_template,
            )
            best = max(best, lp)
    return best


@torch.inference_mode()
def parametric_margin_batch(
    model,
    tokenizer,
    questions: list[str],
    org_answers_list: list[list[str]],
    sub_answers_list: list[list[str]],
    *,
    model_key: str = "",
    use_chat_template: bool = True,
) -> list[float]:
    margins: list[float] = []
    for q, org, sub in zip(questions, org_answers_list, sub_answers_list):
        prompt = clean_prompt(q)
        lp_org = best_answer_logprob(
            model, tokenizer, prompt, org, model_key=model_key, use_chat_template=use_chat_template
        )
        lp_sub = best_answer_logprob(
            model, tokenizer, prompt, sub, model_key=model_key, use_chat_template=use_chat_template
        )
        m = lp_org - lp_sub
        if math.isnan(m):
            m = float("-inf")
        margins.append(float(m))
    return margins


def margin_summary(margins: list[float]) -> dict[str, Any]:
    xs = [m for m in margins if math.isfinite(m)]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return {"n": len(xs), "mean": mean, "std": math.sqrt(var)}
