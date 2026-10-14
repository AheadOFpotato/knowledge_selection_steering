"""Single-target correctness judges.

Every evaluation asks one question only: does the prediction satisfy the
dataset's official target? Source-choice comparisons are not part of this API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from . import native


@dataclass
class JudgeResult:
    label: str  # correct | wrong
    official_score: float
    meta: dict[str, Any] = field(default_factory=dict)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def match_gold(dataset: str, prediction: str, gold: Any, *, meta: Optional[dict[str, Any]] = None, llm: Any = None) -> bool:
    """Return whether ``prediction`` satisfies one official dataset target."""
    meta = meta or {}
    ds = dataset.strip().lower()
    if ds in {"nq-swap", "nqswap"}:
        return native.nqswap_match(prediction, gold, question=str(meta.get("question") or meta.get("query") or ""), llm=llm)
    if ds == "rgb":
        return native.rgb_match(prediction, gold)
    if ds == "guidebench":
        return native.guidebench_match(prediction, gold, category=str(meta.get("category", "")), llm=llm)
    if ds == "contractnli":
        return native.legalbench_yes_no_match(prediction, str(_first(gold)))
    if ds == "legalbench":
        return native.legalbench_label_match(prediction, str(_first(gold)))
    if ds == "crag":
        label, _ = native.crag_match(query=str(meta.get("query") or meta.get("question") or ""), ground_truth=str(_first(gold) or ""), prediction=prediction, llm=llm)
        return label == "correct"
    if ds == "longmemeval":
        answer = str(_first(gold) or "")
        if meta.get("eval_function"):
            return native.longmemeval_v2_match(prediction=prediction, answer=answer, eval_function=str(meta["eval_function"]), question=str(meta.get("question") or meta.get("query") or ""), llm=llm)
        return native.longmemeval_v1_match(question=str(meta.get("question") or ""), answer=answer, prediction=prediction, question_type=str(meta.get("question_type") or "multi-session"), llm=llm)
    if ds in {"memoryagentbench", "memory-agent-bench"}:
        if str(meta.get("question_type") or "") != "fact_consolidation":
            return False
        return native.memoryagentbench_fact_match(prediction, gold)
    if ds == "api-bank":
        return native.apibank_match(prediction, str(_first(gold) or ""))
    if ds in {"mem2actbench", "mem2act"}:
        target = _first(gold)
        if not isinstance(target, dict):
            try:
                target = json.loads(str(target))
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
        return isinstance(target, dict) and native.mem2act_match(prediction, target)
    if ds == "followbench":
        return native.followbench_match(prediction, source=str(meta.get("source") or ""), target=str(meta.get("target") if meta.get("target") is not None else _first(gold)), level=int(meta.get("level") or 0), example_id=str(meta.get("example_id") or ""), category=str(meta.get("category") or meta.get("question_type") or ""))
    if ds == "iheval":
        return native.iheval_match(prediction, _first(gold), task=meta.get("iheval_task") or meta.get("category") or meta.get("question_type"), prompt=str(meta.get("iheval_prompt") or meta.get("question") or ""))
    if ds == "ifeval":
        from .ifeval_scorer import ifeval_strict_score
        return ifeval_strict_score(
            prediction,
            meta.get("ifeval_target") or _first(gold),
            prompt=str(meta.get("ifeval_prompt") or meta.get("question") or ""),
        ) >= 1.0 - 1e-8
    if ds == "freshqa":
        from .freshqa_scorer import freshqa_strict_score
        return freshqa_strict_score(
            prediction,
            meta.get("freshqa_answers") or gold,
            question=str(meta.get("freshqa_question") or meta.get("question") or ""),
        ) >= 1.0 - 1e-8
    if ds == "locomo":
        return native.locomo_f1_match(prediction, str(_first(gold) or ""), category=meta.get("locomo_category") or meta.get("question_type") or meta.get("category"), option_a=meta.get("locomo_cat5_a"), option_b=meta.get("locomo_cat5_b"))
    if ds in {"locomo-plus", "locomo_plus"}:
        return native.locomo_plus_match(prediction, str(meta.get("constraint") or _first(gold) or ""), trigger=str(meta.get("question") or ""), llm=llm)
    if ds == "toolsandbox":
        score = meta.get("toolsandbox_similarity")
        return score is not None and float(score) >= 1.0 - 1e-8
    if ds == "bfcl":
        from .bfcl_scorer import bfcl_official_score
        return bfcl_official_score(prediction, meta) >= 1.0 - 1e-8
    raise KeyError(f"No single-target judge registered for dataset={dataset!r}")


def judge_prediction(dataset: str, prediction: str, gold: Any, *, meta: Optional[dict[str, Any]] = None, llm: Any = None) -> JudgeResult:
    """Score a prediction against the official target for one sample."""
    hit = match_gold(dataset, prediction, gold, meta=meta, llm=llm)
    return JudgeResult("correct" if hit else "wrong", 1.0 if hit else 0.0, {"dataset": dataset})
