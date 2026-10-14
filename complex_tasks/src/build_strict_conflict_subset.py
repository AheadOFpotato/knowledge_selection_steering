#!/usr/bin/env python3
"""Build strict-conflict subset via paraphrase consistency + parametric margin.

Filter criterion (default):
  1. Original clean prompt answers org_answer (stage-1 pass)
  2. Paraphrase consistency >= threshold:
       (1/K) * sum_k 1[f(q^(k)) = a_gold] >= 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from parametric_margin import parametric_margin_batch  # noqa: E402
from stress_test_common import (  # noqa: E402
    DEFAULT_NQSWAP,
    clean_prompt,
    compute_tertile_thresholds,
    gold_from_row,
    is_parametric_correct,
    load_nqswap_rows,
    read_jsonl,
    strip_nq_tags,
    tier_label,
    write_jsonl,
    _generate_batch,
    _get_model_and_tokenizer,
)


def _load_rows_with_paraphrases(
    input_path: Path,
    paraphrase_path: Path | None,
) -> list[dict[str, Any]]:
    if paraphrase_path and paraphrase_path.is_file():
        rows = read_jsonl(paraphrase_path)
        return rows

    rows: list[dict[str, Any]] = []
    for line_idx, row in load_nqswap_rows(input_path):
        rec = dict(row)
        rec["line_idx"] = line_idx
        rec.setdefault("paraphrases", [])
        rows.append(rec)
    return rows


def evaluate_paraphrase_consistency(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    model_key: str,
    use_chat_template: bool,
    batch_size: int,
) -> None:
    """Mutates rows in place with paraphrase eval fields."""
    for row in rows:
        org, _ = gold_from_row(row)
        q0 = str(row.get("question", "")).strip()
        paras = [str(p).strip() for p in row.get("paraphrases", []) if str(p).strip()]
        all_qs = [q0] + paras
        prompts = [clean_prompt(q) for q in all_qs]

        outputs: list[str] = []
        for st in range(0, len(prompts), batch_size):
            chunk = prompts[st : st + batch_size]
            outputs.extend(
                _generate_batch(
                    model,
                    tokenizer,
                    chunk,
                    model_key=model_key,
                    use_chat_template=use_chat_template,
                )
            )

        correct_flags = [is_parametric_correct(o, org) for o in outputs]
        row["stage1_correct_original"] = bool(correct_flags[0])
        row["stage1_output_original"] = outputs[0]
        row["paraphrase_outputs"] = outputs[1:]
        row["paraphrase_correct_flags"] = correct_flags[1:]
        k = len(correct_flags[1:])
        row["paraphrase_correct_count"] = sum(correct_flags[1:])
        row["paraphrase_consistency"] = (sum(correct_flags[1:]) / k) if k else 0.0
        row["paraphrase_total"] = k


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strict-conflict subset for NQ-Swap.")
    ap.add_argument("--input", type=Path, default=DEFAULT_NQSWAP)
    ap.add_argument(
        "--paraphrases",
        type=Path,
        default=None,
        help="Optional jsonl with pre-generated paraphrases (must include line_idx).",
    )
    ap.add_argument("--output", type=Path, required=True, help="strict subset jsonl")
    ap.add_argument("--summary", type=Path, default=None, help="filter summary json")
    ap.add_argument("--model", default="llama3.1-8b-it")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--consistency-threshold", type=float, default=0.8)
    ap.add_argument("--min-paraphrases", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-margin", action="store_true")
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template wrapping.",
    )
    args = ap.parse_args()

    rows = _load_rows_with_paraphrases(args.input.resolve(), args.paraphrases)
    if args.limit > 0:
        rows = rows[: args.limit]

    model, tokenizer = _get_model_and_tokenizer(str(args.model))
    use_chat = not args.no_chat_template

    evaluate_paraphrase_consistency(
        model,
        tokenizer,
        rows,
        model_key=str(args.model),
        use_chat_template=use_chat,
        batch_size=args.batch_size,
    )

    if not args.skip_margin:
        questions = [str(r.get("question", "")).strip() for r in rows]
        org_list = [gold_from_row(r)[0] for r in rows]
        sub_list = [gold_from_row(r)[1] for r in rows]
        margins = parametric_margin_batch(
            model,
            tokenizer,
            questions,
            org_list,
            sub_list,
            model_key=str(args.model),
            use_chat_template=use_chat,
        )
        for r, m in zip(rows, margins):
            r["parametric_margin"] = m
    else:
        for r in rows:
            r.setdefault("parametric_margin", None)

    # Filter strict subset
    strict: list[dict[str, Any]] = []
    for r in rows:
        if not r.get("stage1_correct_original"):
            continue
        if int(r.get("paraphrase_total", 0)) < args.min_paraphrases:
            continue
        if float(r.get("paraphrase_consistency", 0.0)) < args.consistency_threshold:
            continue
        rec = dict(r)
        rec["org_context_clean"] = strip_nq_tags(str(r.get("org_context", "")))
        rec["sub_context_clean"] = strip_nq_tags(str(r.get("sub_context", "")))
        strict.append(rec)

    # Difficulty tiers (computed on strict subset)
    margins_strict = [float(r["parametric_margin"]) for r in strict if r.get("parametric_margin") is not None]
    cons_strict = [float(r["paraphrase_consistency"]) for r in strict]
    m_lo, m_hi = compute_tertile_thresholds(margins_strict)
    c_lo, c_hi = compute_tertile_thresholds(cons_strict)
    for r in strict:
        m = r.get("parametric_margin")
        r["confidence_tier"] = (
            tier_label(float(m), low_q=m_lo, high_q=m_hi) if m is not None else "unknown"
        )
        r["consistency_tier"] = tier_label(float(r["paraphrase_consistency"]), low_q=c_lo, high_q=c_hi)

    write_jsonl(args.output.resolve(), strict)

    summary = {
        "input": str(args.input.resolve()),
        "paraphrases": str(args.paraphrases) if args.paraphrases else None,
        "output": str(args.output.resolve()),
        "model": args.model,
        "consistency_threshold": args.consistency_threshold,
        "min_paraphrases": args.min_paraphrases,
        "total_rows": len(rows),
        "stage1_correct": sum(1 for r in rows if r.get("stage1_correct_original")),
        "strict_subset_size": len(strict),
        "strict_fraction_of_stage1_correct": (
            len(strict) / max(1, sum(1 for r in rows if r.get("stage1_correct_original")))
        ),
        "margin_tertiles": {"low_q": m_lo, "high_q": m_hi},
        "consistency_tertiles": {"low_q": c_lo, "high_q": c_hi},
        "confidence_tier_counts": {},
        "consistency_tier_counts": {},
    }
    for r in strict:
        summary["confidence_tier_counts"][r["confidence_tier"]] = (
            summary["confidence_tier_counts"].get(r["confidence_tier"], 0) + 1
        )
        summary["consistency_tier_counts"][r["consistency_tier"]] = (
            summary["consistency_tier_counts"].get(r["consistency_tier"], 0) + 1
        )

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"strict subset: {len(strict)}/{len(rows)} "
        f"(stage1_correct={summary['stage1_correct']}) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
