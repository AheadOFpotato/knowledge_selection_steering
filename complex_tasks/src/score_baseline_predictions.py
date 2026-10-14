#!/usr/bin/env python3
"""Score canonical predictions jsonl against manifest gold (unified metrics)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from canonical_stage2_common import (  # noqa: E402
    compute_metrics,
    load_manifest,
    load_predictions_map,
    merge_predictions,
    write_metrics,
    write_predictions_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-metrics", type=Path, required=True)
    ap.add_argument("--output-predictions", type=Path, default=None)
    ap.add_argument("--method", default="")
    ap.add_argument("--mode", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = load_manifest(args.manifest.resolve())
    if args.limit > 0:
        records = records[: args.limit]
    pred_map = load_predictions_map(args.predictions.resolve())
    scored = merge_predictions(records, pred_map)
    if not scored:
        raise SystemExit(f"no overlapping predictions: {args.predictions}")

    metrics = compute_metrics(scored, method=args.method, mode=args.mode)
    write_metrics(args.output_metrics.resolve(), metrics)
    if args.output_predictions:
        write_predictions_jsonl(args.output_predictions.resolve(), scored)
    print(
        f"scored n={len(scored)} ext={metrics['external_rate']:.4f} "
        f"int={metrics['internal_rate']:.4f} -> {args.output_metrics}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
