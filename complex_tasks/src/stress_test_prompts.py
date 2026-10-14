#!/usr/bin/env python3
"""Prompt builders for context-strength controlled stress tests (NQ-Swap)."""
from __future__ import annotations

import re
from typing import Any

from stress_test_common import clean_prompt, strip_nq_tags

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_NEUTRAL_FILLER_SENTENCES = [
    "The production team announced several updates during the regular press cycle.",
    "Industry observers noted steady audience engagement across multiple regions.",
    "Official records mention routine scheduling and standard broadcast practices.",
    "Media coverage highlighted general background details without changing core facts.",
    "Archival notes describe ordinary release patterns for comparable programs.",
]


def split_sentences(text: str) -> list[str]:
    t = strip_nq_tags(text)
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts if parts else ([t] if t else [])


def _norm_token(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def extract_conflict_sentence(sub_context: str, sub_answers: list[str]) -> str:
    """Pick the sentence in sub_context that states the conflicting (context) answer."""
    sents = split_sentences(sub_context)
    if not sents:
        return ""
    norms = [_norm_token(a) for a in sub_answers if str(a).strip()]
    for sent in sents:
        ns = _norm_token(sent)
        if any(n and n in ns for n in norms):
            return sent.strip()
    return sents[-1].strip()


def paragraph_context(sub_context: str) -> str:
    return strip_nq_tags(sub_context)


def atomic_context(sub_context: str, sub_answers: list[str]) -> str:
    sent = extract_conflict_sentence(sub_context, sub_answers)
    return sent.strip()


def repeat_sentence(sentence: str, times: int) -> str:
    n = max(1, int(times))
    s = str(sentence).strip()
    return "\n".join([s] * n)


def build_neutral_filler(*, target_token_count: int, tokenizer) -> str:
    """Build neutral filler with approximately the same token count as repetition block."""
    target = max(1, int(target_token_count))
    chunks: list[str] = []
    total = 0
    i = 0
    while total < target:
        sent = _NEUTRAL_FILLER_SENTENCES[i % len(_NEUTRAL_FILLER_SENTENCES)]
        chunks.append(sent)
        total = len(tokenizer.encode("\n".join(chunks), add_special_tokens=False))
        i += 1
        if i > 200:
            break
    text = "\n".join(chunks)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target:
        text = tokenizer.decode(ids[:target], skip_special_tokens=True)
    return text.strip()


def conflict_qa_prompt(question: str, context: str) -> str:
    q = str(question).strip()
    c = str(context).strip()
    if not c:
        return clean_prompt(q)
    return f"{c}\n\nQuestion: {q}\nAnswer:"


def context_position_prompt(
    question: str,
    *,
    paragraph: str,
    atomic: str,
    position: str,
) -> str:
    """Place context at beginning / middle / immediately before question."""
    q = str(question).strip()
    para = str(paragraph).strip()
    atom = str(atomic).strip() or para
    pos = str(position).strip().lower()

    if pos in ("begin", "beginning", "start"):
        return conflict_qa_prompt(q, para)

    if pos in ("before_question", "before", "end", "immediate"):
        # Minimal cue right before the QA block (strong recency bias test).
        return conflict_qa_prompt(q, atom)

    if pos in ("middle", "mid"):
        sents = split_sentences(para)
        if len(sents) <= 1:
            return conflict_qa_prompt(q, para)
        mid = len(sents) // 2
        before = " ".join(sents[:mid]).strip()
        after = " ".join(sents[mid:]).strip()
        return f"{before}\n\nQuestion: {q}\n\n{after}\nAnswer:"

    raise ValueError(f"unknown context position: {position!r}")


def build_stress_variants(
    row: dict[str, Any],
    *,
    tokenizer=None,
    repetition_counts: tuple[int, ...] = (1, 2, 4),
    include_positions: bool = True,
) -> dict[str, str]:
    """Return named conflict prompts for one NQ-Swap row."""
    q = str(row.get("question", "")).strip()
    sub_ctx = str(row.get("sub_context", ""))
    sub_answers = row.get("sub_answer", [])
    if isinstance(sub_answers, str):
        sub_answers = [sub_answers]

    atom = atomic_context(sub_ctx, sub_answers)
    para = paragraph_context(sub_ctx)

    out: dict[str, str] = {
        "detail_atomic": conflict_qa_prompt(q, atom),
        "detail_paragraph": conflict_qa_prompt(q, para),
    }

    for r in repetition_counts:
        rep_block = repeat_sentence(atom, r)
        out[f"repetition_r{r}"] = conflict_qa_prompt(q, rep_block)
        if tokenizer is not None:
            tok_n = len(tokenizer.encode(rep_block, add_special_tokens=False))
            filler = build_neutral_filler(target_token_count=tok_n, tokenizer=tokenizer)
            out[f"filler_r{r}"] = conflict_qa_prompt(q, filler)

    if include_positions:
        for pos in ("beginning", "middle", "before_question"):
            out[f"position_{pos}"] = context_position_prompt(
                q, paragraph=para, atomic=atom, position=pos
            )

    return out


STRESS_VARIANT_GROUPS: dict[str, list[str]] = {
    "context_detail": ["detail_atomic", "detail_paragraph"],
    "answer_repetition": ["repetition_r1", "repetition_r2", "repetition_r4"],
    "length_control": ["filler_r1", "filler_r2", "filler_r4"],
    "context_position": ["position_beginning", "position_middle", "position_before_question"],
}
