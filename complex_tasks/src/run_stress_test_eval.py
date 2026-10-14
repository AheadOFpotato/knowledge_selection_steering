#!/usr/bin/env python3
"""Run context-strength stress tests on strict-conflict (or full) NQ-Swap subset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stress_test_common import (  # noqa: E402
    gold_from_row,
    read_jsonl,
    score_conflict,
    write_jsonl,
    _generate_batch,
    _get_model_and_tokenizer,
)
from stress_test_prompts import STRESS_VARIANT_GROUPS, build_stress_variants  # noqa: E402


def _variant_list(include_positions: bool, variants: str) -> list[str]:
    if variants.strip().lower() == "all":
        names: list[str] = []
        for group in STRESS_VARIANT_GROUPS.values():
            names.extend(group)
        if not include_positions:
            names = [n for n in names if not n.startswith("position_")]
        return names
    return [v.strip() for v in variants.split(",") if v.strip()]


def run_eval(
    rows: list[dict[str, Any]],
    *,
    model,
    tokenizer,
    model_key: str,
    use_chat_template: bool,
    batch_size: int,
    include_positions: bool,
    variant_names: list[str],
) -> list[dict[str, Any]]:
    # Flatten (row, variant) jobs
    jobs: list[tuple[int, str, str, dict[str, Any]]] = []
    for row in rows:
        li = int(row.get("line_idx", 0))
        variants = build_stress_variants(
            row,
            tokenizer=tokenizer,
            include_positions=include_positions,
        )
        for vname in variant_names:
            if vname not in variants:
                continue
            jobs.append((li, vname, variants[vname], row))

    out_records: list[dict[str, Any]] = []
    for st in range(0, len(jobs), batch_size):
        chunk = jobs[st : st + batch_size]
        prompts = [p for _, _, p, _ in chunk]
        preds = _generate_batch(
            model,
            tokenizer,
            prompts,
            model_key=model_key,
            use_chat_template=use_chat_template,
        )
        for (li, vname, prompt, row), pred in zip(chunk, preds):
            org, sub = gold_from_row(row)
            sc = score_conflict(pred, org, sub)
            out_records.append(
                {
                    "line_idx": li,
                    "question": row.get("question", ""),
                    "variant": vname,
                    "variant_group": _group_for_variant(vname),
                    "prompt": prompt,
                    "prediction_raw": pred,
                    "org_answer": org,
                    "sub_answer": sub,
                    "parametric_margin": row.get("parametric_margin"),
                    "paraphrase_consistency": row.get("paraphrase_consistency"),
                    "confidence_tier": row.get("confidence_tier"),
                    "consistency_tier": row.get("consistency_tier"),
                    **sc,
                }
            )
        print(f"[stress_eval] {min(st + batch_size, len(jobs))}/{len(jobs)}", flush=True)
    return out_records


def _group_for_variant(name: str) -> str:
    for group, names in STRESS_VARIANT_GROUPS.items():
        if name in names:
            return group
    if name.startswith("repetition_"):
        return "answer_repetition"
    if name.startswith("filler_"):
        return "length_control"
    if name.startswith("position_"):
        return "context_position"
    if name.startswith("detail_"):
        return "context_detail"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate context-strength stress test variants.")
    ap.add_argument("--input", type=Path, required=True, help="strict subset or NQ-Swap jsonl")
    ap.add_argument("--output", type=Path, required=True, help="per-sample per-variant predictions jsonl")
    ap.add_argument("--metrics", type=Path, default=None, help="aggregated metrics json")
    ap.add_argument("--model", default="llama3.1-8b-it")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names or 'all'",
    )
    ap.add_argument("--no-position", action="store_true", help="Skip context-position variants")
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.input.resolve())
    if args.limit > 0:
        rows = rows[: args.limit]

    variant_names = _variant_list(not args.no_position, args.variants)
    model, tokenizer = _get_model_and_tokenizer(str(args.model))

    records = run_eval(
        rows,
        model=model,
        tokenizer=tokenizer,
        model_key=str(args.model),
        use_chat_template=not args.no_chat_template,
        batch_size=args.batch_size,
        include_positions=not args.no_position,
        variant_names=variant_names,
    )
    write_jsonl(args.output.resolve(), records)

    metrics = aggregate_stress_metrics(records)
    metrics_path = args.metrics or args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records -> {args.output}", flush=True)
    print(f"metrics -> {metrics_path}", flush=True)


def aggregate_stress_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate internal/external rates by variant and difficulty bucket."""
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_variant.setdefault(str(r["variant"]), []).append(r)

    def _rates(rs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        if n == 0:
            return {"n": 0, "external_rate": 0.0, "internal_rate": 0.0, "other_rate": 0.0}
        ext = sum(int(x.get("correct_external", 0)) for x in rs)
        inn = sum(int(x.get("correct_internal", 0)) for x in rs)
        other = n - ext - inn
        return {
            "n": n,
            "external_rate": ext / n,
            "internal_rate": inn / n,
            "other_rate": other / n,
        }

    out: dict[str, Any] = {
        "overall_by_variant": {v: _rates(rs) for v, rs in sorted(by_variant.items())},
        "by_variant_group": {},
        "by_confidence_tier": {},
        "by_consistency_tier": {},
        "cross_confidence_x_variant": {},
        "cross_consistency_x_variant": {},
    }

    # Group-level
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(str(r.get("variant_group", "other")), []).append(r)
    out["by_variant_group"] = {g: _rates(rs) for g, rs in sorted(by_group.items())}

    for bucket_field, bucket_key in (
        ("confidence_tier", "by_confidence_tier"),
        ("consistency_tier", "by_consistency_tier"),
    ):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            b = str(r.get(bucket_field) or "unknown")
            buckets.setdefault(b, []).append(r)
        out[bucket_key] = {b: _rates(rs) for b, rs in sorted(buckets.items())}

    for bucket_field, cross_key in (
        ("confidence_tier", "cross_confidence_x_variant"),
        ("consistency_tier", "cross_consistency_x_variant"),
    ):
        cross: dict[str, dict[str, Any]] = {}
        for r in records:
            b = str(r.get(bucket_field) or "unknown")
            v = str(r["variant"])
            cross.setdefault(b, {}).setdefault(v, []).append(r)
        out[cross_key] = {
            b: {v: _rates(rs) for v, rs in sorted(variant_map.items())}
            for b, variant_map in sorted(cross.items())
        }

    return out


if __name__ == "__main__":
    main()
