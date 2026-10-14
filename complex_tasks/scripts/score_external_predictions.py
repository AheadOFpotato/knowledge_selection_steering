#!/usr/bin/env python3
"""Score externally generated predictions with the taxonomy official judges.

This is deliberately generation-method agnostic.  A baseline writes one JSONL
row per input with ``line_idx`` and ``prediction_raw`` (or ``prediction``), and
this script joins it to an ``internal_aware_data__context_all.json`` manifest.
It keeps the same per-item official scoring used by the residual evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from run_resid_projection_ablation import _context_item_score, _score_label  # noqa: E402


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"expected taxonomy JSON manifest with samples: {path}")
    return payload


def _load_predictions(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row.get("line_idx", row.get("input_index", i)))
        value = row.get("prediction_raw", row.get("prediction", row.get("Output", "")))
        if isinstance(value, list):
            value = value[0] if value else ""
        out[idx] = str(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--coefficient", type=float, default=None)
    ap.add_argument("--judge", choices=("official", "legacy"), default="official")
    args = ap.parse_args()

    manifest = _load_manifest(args.manifest)
    preds = _load_predictions(args.predictions)
    dataset = str(manifest.get("dataset") or "")
    category = str(manifest.get("category") or "")
    rows, score_sum = [], 0.0
    for sample in manifest["samples"]:
        idx = int(sample.get("line_idx", -1))
        if idx not in preds:
            continue
        pred = preds[idx]
        score = _context_item_score(
            pred,
            [str(x) for x in sample.get("gold_answers", [])],
            dataset_name=dataset,
            category_tag=category,
            sample=sample,
            judge_mode=args.judge,
            is_kre=False,
        )
        score_sum += score
        rows.append({
            "line_idx": idx,
            "prediction_raw": pred,
            "official_score": score,
            "label": _score_label(score),
        })
    result = {
        "method": args.method,
        "coefficient": args.coefficient,
        "dataset": dataset,
        "category": category,
        "judge": args.judge,
        "num_manifest": len(manifest["samples"]),
        "num_predicted": len(rows),
        "score_sum": score_sum,
        "pass_rate": score_sum / max(1, len(rows)),
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.output} n={len(rows)} score={result['pass_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
