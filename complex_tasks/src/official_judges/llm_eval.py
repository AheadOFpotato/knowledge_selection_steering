#!/usr/bin/env python3
"""LLM-as-judge for datasets that have **no official eval script**.

Uses ``llm_settings.py`` (Paratera / Qwen-Long by default).

Covered:
  - NQ-Swap: semantic answer equivalence (after local substring miss)
  - Mem2ActBench: tool-call equivalence (name + args), format-flexible
  - generic_qa: fallback open-QA yes/no judge
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def _chat(messages: list[dict[str, str]], *, max_tokens: int = 32) -> str:
    from .llm_client import ChatLLM

    return ChatLLM().chat(messages, max_tokens=max_tokens)


def _parse_yes_no(raw: str) -> bool:
    t = (raw or "").strip().lower()
    # prefer first line / first token
    first = re.split(r"[\s,.;:!?]", t, maxsplit=1)[0]
    if first in {"yes", "y", "true", "1", "correct"}:
        return True
    if first in {"no", "n", "false", "0", "incorrect", "wrong"}:
        return False
    if "yes" in t and "no" not in t.split("yes", 1)[0][-10:]:
        return True
    return False


def _parse_json_blob(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# NQ-Swap / open factual QA
# ---------------------------------------------------------------------------
_NQSWAP_PROMPT = """You are evaluating a question-answering model on an entity-swap / factual QA task (NQ-Swap style).

Question (optional): {question}
Gold answer(s): {gold}
Model prediction: {prediction}

Decide if the prediction correctly conveys the gold answer.
- Answer YES if the prediction contains the gold answer, or an equivalent paraphrase / alias / abbreviation.
- Answer NO if the prediction gives a different entity/fact, refuses, or is unrelated.
- Minor wording differences are OK; wrong entity is NOT OK.

Reply with a single word: YES or NO."""


def llm_judge_nqswap(
    *,
    prediction: str,
    gold_answers: Any,
    question: str = "",
    llm: Any = None,
) -> bool:
    if isinstance(gold_answers, list):
        gold = " | ".join(str(x) for x in gold_answers if str(x).strip())
    else:
        gold = str(gold_answers or "").strip()
    if not gold:
        return False
    prompt = _NQSWAP_PROMPT.format(
        question=question or "(not provided)",
        gold=gold,
        prediction=str(prediction or "").strip() or "(empty)",
    )
    if llm is None:
        raw = _chat([{"role": "user", "content": prompt}], max_tokens=8)
    else:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=8)
    return _parse_yes_no(raw)


_GUIDEBENCH_MC_PROMPT = """你是 GuideBench 选择题判分器。
任务类别: {category}
标准答案选项字母: {gold}
模型输出:
{prediction}

判断模型输出是否在选择与标准答案相同的选项（允许模型先写分析再给出字母；也允许用选项原文表达同一选择）。
只回答 YES 或 NO。"""


def llm_judge_guidebench_mc(
    *,
    prediction: str,
    gold: str,
    category: str = "",
    llm: Any = None,
) -> bool:
    prompt = _GUIDEBENCH_MC_PROMPT.format(
        category=category or "chat",
        gold=str(gold).strip().upper(),
        prediction=str(prediction or "").strip() or "(empty)",
    )
    if llm is None:
        raw = _chat([{"role": "user", "content": prompt}], max_tokens=8)
    else:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=8)
    return _parse_yes_no(raw)


# ---------------------------------------------------------------------------
# Mem2ActBench tool call
# ---------------------------------------------------------------------------
_MEM2ACT_PROMPT = """You are evaluating a tool-calling agent (Mem2ActBench style).

User query (optional): {query}
Gold tool call (JSON): {gold}
Model prediction: {prediction}

Decide if the prediction is the SAME tool call as the gold.
- Tool/API name must match (ignore trivial formatting like leading '/').
- Arguments must be semantically the same (types/string forms may differ, e.g. true vs True, 50 vs "50").
- Extra unrelated text around a correct JSON/API call is OK.
- Wrong tool name or materially wrong argument values → NO.

Reply with JSON only: {{"match": true/false, "reason": "short"}}"""


def llm_judge_mem2act(
    *,
    prediction: str,
    gold_tool_call: dict[str, Any],
    query: str = "",
    llm: Any = None,
) -> bool:
    gold = {
        "name": gold_tool_call.get("name"),
        "arguments": gold_tool_call.get("arguments") or {},
    }
    prompt = _MEM2ACT_PROMPT.format(
        query=query or "(not provided)",
        gold=json.dumps(gold, ensure_ascii=False),
        prediction=str(prediction or "").strip() or "(empty)",
    )
    if llm is None:
        raw = _chat([{"role": "user", "content": prompt}], max_tokens=128)
    else:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=128)
    obj = _parse_json_blob(raw)
    if "match" in obj:
        return bool(obj["match"])
    return _parse_yes_no(raw)


# ---------------------------------------------------------------------------
# Generic open QA (fallback for any dataset without eval)
# ---------------------------------------------------------------------------
_GENERIC_QA_PROMPT = """You are a careful grader for a QA benchmark.

Question: {question}
Gold answer: {gold}
Model prediction: {prediction}

Is the prediction correct relative to the gold answer?
Reply with a single word: YES or NO."""


def llm_judge_generic_qa(
    *,
    prediction: str,
    gold: Any,
    question: str = "",
    llm: Any = None,
) -> bool:
    if isinstance(gold, list):
        g = " | ".join(str(x) for x in gold)
    else:
        g = str(gold)
    prompt = _GENERIC_QA_PROMPT.format(
        question=question or "(not provided)",
        gold=g,
        prediction=str(prediction or "").strip() or "(empty)",
    )
    if llm is None:
        raw = _chat([{"role": "user", "content": prompt}], max_tokens=8)
    else:
        raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=8)
    return _parse_yes_no(raw)


def llm_judge_dispatch(
    dataset: str,
    *,
    prediction: str,
    gold: Any,
    question: str = "",
    llm: Any = None,
) -> bool:
    """Route LLM judge for no-official-eval datasets."""
    ds = dataset.strip().lower()
    if ds in {"nq-swap", "nqswap"}:
        return llm_judge_nqswap(
            prediction=prediction, gold_answers=gold, question=question, llm=llm
        )
    if ds in {"mem2actbench", "mem2act"}:
        if isinstance(gold, dict):
            tc = gold
        else:
            try:
                tc = json.loads(str(gold))
            except Exception:
                tc = {"name": str(gold), "arguments": {}}
        return llm_judge_mem2act(
            prediction=prediction, gold_tool_call=tc, query=question, llm=llm
        )
    return llm_judge_generic_qa(
        prediction=prediction, gold=gold, question=question, llm=llm
    )
