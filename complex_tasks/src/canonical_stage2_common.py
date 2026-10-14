#!/usr/bin/env python3
"""Shared Stage2 eval contract (aligned with run_resid_projection_ablation Stage2)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from baseline_eval_metrics import (
    aggregate_knowledge_rates,
    as_answer_list,
    score_prediction,
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stage2_prompt_for_record(rec: dict[str, Any]) -> str:
    return str(
        rec.get("stage2_prompt_input") or rec.get("conflict_prompt_input") or ""
    ).strip()


def gold_lists_for_record(rec: dict[str, Any]) -> tuple[list[str], list[str]]:
    sub = as_answer_list(rec.get("sub_answer"))
    if not sub:
        sub = as_answer_list(rec.get("distracted_tokens"))
    org = as_answer_list(rec.get("org_answer"))
    return sub, org


def score_record(rec: dict[str, Any], prediction: str) -> dict[str, Any]:
    sub, org = gold_lists_for_record(rec)
    sc = score_prediction(prediction, sub, org)
    row = dict(rec)
    row["prediction_raw"] = prediction
    row["prediction"] = prediction
    row.update(sc)
    return row


def infer_manifest(
    records: list[dict[str, Any]],
    model,
    tokenizer,
    generate_batch_fn: Callable,
    *,
    batch_size: int = 32,
    progress_prefix: str = "canonical_infer",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in range(0, len(records), batch_size):
        chunk = records[st : st + batch_size]
        prompts = [stage2_prompt_for_record(r) for r in chunk]
        preds = generate_batch_fn(model, tokenizer, prompts)
        for rec, pred in zip(chunk, preds):
            out.append(score_record(rec, pred))
        done = min(st + batch_size, len(records))
        print(f"[{progress_prefix}] {done}/{len(records)}", flush=True)
    return out


def merge_predictions(
    records: list[dict[str, Any]],
    predictions_by_line: dict[int, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        li = int(rec.get("line_idx", len(out)))
        pred = predictions_by_line.get(li)
        if pred is None:
            continue
        out.append(score_record(rec, str(pred)))
    return out


def load_predictions_map(path: Path) -> dict[int, str]:
    by_line: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            li = row.get("line_idx", row.get("input_index", i))
            pred = row.get("prediction_raw", row.get("prediction", row.get("Output", "")))
            if isinstance(pred, list):
                pred = pred[0] if pred else ""
            by_line[int(li)] = str(pred)
    return by_line


def write_predictions_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for rec in records:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_metrics(
    records: list[dict[str, Any]],
    *,
    method: str = "",
    mode: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = [
        {
            "knowledge_choice": r["knowledge_choice"],
            "correct_external": r["correct_external"],
            "correct_internal": r["correct_internal"],
        }
        for r in records
    ]
    m = aggregate_knowledge_rates(scores)
    m["eval_contract"] = {
        "prompt_field": "stage2_prompt_input",
        "metric": "first_sentence_bidirectional_contain",
        "method": method,
        "mode": mode,
        **(extra or {}),
    }
    return m


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
