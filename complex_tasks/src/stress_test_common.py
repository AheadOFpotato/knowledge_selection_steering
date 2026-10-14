#!/usr/bin/env python3
"""Shared helpers for strict-conflict verification and stress-test experiments."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baseline_eval_metrics import as_answer_list, score_prediction  # noqa: E402
from dump_activations_from_testjsonl import (  # noqa: E402
    _answer_matches,
    _generate_batch,
    _get_model_and_tokenizer,
    _gold_list_from_row_field,
    _load_rows_nq_swap,
    _nq_swap_question_only_prompt,
    _truncate_for_judge,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NQSWAP = ROOT / "data" / "NQ-Swap" / "dev_dedup_p_only.jsonl"

_P_TAG = re.compile(r"</?P>", re.IGNORECASE)


def strip_nq_tags(text: str) -> str:
    return _P_TAG.sub(" ", str(text or "")).strip()


def load_nqswap_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    return _load_rows_nq_swap(path.resolve())


def clean_prompt(question: str) -> str:
    return _nq_swap_question_only_prompt(str(question).strip())


def gold_from_row(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    org = _gold_list_from_row_field(row.get("org_answer"))
    sub = _gold_list_from_row_field(row.get("sub_answer"))
    return org, sub


def is_parametric_correct(output: str, org_answers: list[str]) -> bool:
    return _answer_matches(_truncate_for_judge(output), org_answers)


def score_conflict(output: str, org_answers: list[str], sub_answers: list[str]) -> dict[str, Any]:
    return score_prediction(_truncate_for_judge(output), sub_answers, org_answers)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tier_label(value: float, *, low_q: float, high_q: float) -> str:
    if value <= low_q:
        return "low"
    if value >= high_q:
        return "high"
    return "medium"


def compute_tertile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    xs = sorted(values)
    n = len(xs)
    lo = xs[max(0, int(round(0.33 * (n - 1))))]
    hi = xs[max(0, int(round(0.67 * (n - 1))))]
    return lo, hi


__all__ = [
    "ROOT",
    "DEFAULT_NQSWAP",
    "as_answer_list",
    "clean_prompt",
    "compute_tertile_thresholds",
    "gold_from_row",
    "is_parametric_correct",
    "load_nqswap_rows",
    "read_jsonl",
    "score_conflict",
    "strip_nq_tags",
    "tier_label",
    "write_jsonl",
    "_generate_batch",
    "_get_model_and_tokenizer",
]
