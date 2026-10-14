#!/usr/bin/env python3
"""Summarize stress-test predictions into rebuttal-ready tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_stress_test_eval import aggregate_stress_metrics  # noqa: E402
from stress_test_common import read_jsonl  # noqa: E402


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(metrics: dict[str, Any]) -> str:
    parts: list[str] = ["# Stress Test Summary\n"]

    # A. Context detail
    detail_rows: list[list[str]] = []
    for v in ("detail_atomic", "detail_paragraph"):
        m = metrics["overall_by_variant"].get(v, {})
        detail_rows.append([v, str(m.get("n", 0)), _pct(m.get("internal_rate", 0)), _pct(m.get("external_rate", 0))])
    parts.append("## A. Contextual detail\n")
    parts.append(markdown_table(["Variant", "N", "Acc_par (internal)", "Acc_ctx (external)"], detail_rows))
    parts.append("")

    # B. Answer repetition vs filler
    rep_rows: list[list[str]] = []
    for r in (1, 2, 4):
        rep = metrics["overall_by_variant"].get(f"repetition_r{r}", {})
        fil = metrics["overall_by_variant"].get(f"filler_r{r}", {})
        rep_rows.append(
            [
                str(r),
                _pct(rep.get("internal_rate", 0)),
                _pct(rep.get("external_rate", 0)),
                _pct(fil.get("internal_rate", 0)),
                _pct(fil.get("external_rate", 0)),
            ]
        )
    parts.append("## B. Answer repetition vs length-matched neutral filler\n")
    parts.append(
        markdown_table(
            ["r", "Rep internal", "Rep external", "Filler internal", "Filler external"],
            rep_rows,
        )
    )
    parts.append("")

    # C. Context position
    pos_rows: list[list[str]] = []
    for v in ("position_beginning", "position_middle", "position_before_question"):
        m = metrics["overall_by_variant"].get(v, {})
        pos_rows.append([v.replace("position_", ""), str(m.get("n", 0)), _pct(m.get("internal_rate", 0)), _pct(m.get("external_rate", 0))])
    parts.append("## C. Context position\n")
    parts.append(markdown_table(["Position", "N", "Acc_par", "Acc_ctx"], pos_rows))
    parts.append("")

    # D. Question difficulty — confidence tier
    conf_rows: list[list[str]] = []
    for tier in ("low", "medium", "high", "unknown"):
        m = metrics.get("by_confidence_tier", {}).get(tier)
        if not m:
            continue
        conf_rows.append([tier, str(m.get("n", 0)), _pct(m.get("internal_rate", 0)), _pct(m.get("external_rate", 0))])
    parts.append("## D. By clean confidence (parametric margin tertiles)\n")
    parts.append(markdown_table(["Tier", "N", "Acc_par", "Acc_ctx"], conf_rows))
    parts.append("")

    # E. Paraphrase consistency tier
    cons_rows: list[list[str]] = []
    for tier in ("low", "medium", "high", "unknown"):
        m = metrics.get("by_consistency_tier", {}).get(tier)
        if not m:
            continue
        cons_rows.append([tier, str(m.get("n", 0)), _pct(m.get("internal_rate", 0)), _pct(m.get("external_rate", 0))])
    parts.append("## E. By paraphrase consistency tertiles (strict subset)\n")
    parts.append(markdown_table(["Tier", "N", "Acc_par", "Acc_ctx"], cons_rows))
    parts.append("")

    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=None, help="stress test predictions jsonl")
    ap.add_argument("--metrics", type=Path, default=None, help="precomputed metrics json")
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if args.metrics and args.metrics.is_file():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    elif args.predictions and args.predictions.is_file():
        records = read_jsonl(args.predictions.resolve())
        metrics = aggregate_stress_metrics(records)
    else:
        raise SystemExit("Provide --metrics or --predictions")

    report = build_report(metrics)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")

    if args.output_json:
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"report -> {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
