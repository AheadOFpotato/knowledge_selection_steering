#!/usr/bin/env python3
"""Export native baseline outputs → canonical predictions jsonl (line_idx + prediction_raw)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from canonical_stage2_common import load_manifest, write_predictions_jsonl  # noqa: E402


def _pred_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "line_idx": int(r["line_idx"]),
            "prediction_raw": str(r["prediction_raw"]),
        }
        for r in records
        if "line_idx" in r and r.get("prediction_raw") is not None
    ]


def export_ircan(pred_path: Path, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(preds, list):
        raise ValueError(f"expected list in {pred_path}")
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(preds):
        if not isinstance(rec, dict):
            continue
        text = str(rec.get("full_prediction", rec.get("prediction", "")))
        li = manifest[i]["line_idx"] if i < len(manifest) else i
        out.append({"line_idx": int(li), "prediction_raw": text})
    return out


def export_juice(result_path: Path, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {result_path}")
    samples = [x for x in data if isinstance(x, dict) and "Output" in x]
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(samples):
        if rec.get("line_idx") is not None:
            li = int(rec["line_idx"])
        elif i < len(manifest):
            li = int(manifest[i]["line_idx"])
        else:
            li = i
        out.append({"line_idx": li, "prediction_raw": str(rec.get("Output", ""))})
    return out


def export_ck(preds_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(preds_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    else:
        rows = raw.get("predictions", [])
    out: list[dict[str, Any]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        li = rec.get("line_idx")
        pred = rec.get("prediction_raw", rec.get("prediction"))
        if li is None or pred is None:
            continue
        out.append({"line_idx": int(li), "prediction_raw": str(pred)})
    return out


def export_adacad(pred_path: Path, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_idx: dict[int, str] = {}
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("placeholder"):
                continue
            row = json.loads(line)
            proc = row.get("assigned_process", 0)
            if proc not in (0, "0"):
                continue
            idx = int(row["input_index"])
            pred = row.get("string", [""])[0]
            by_idx[idx] = str(pred)
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(manifest):
        li = int(rec["line_idx"])
        pred = by_idx.get(i)
        if pred is None:
            continue
        out.append({"line_idx": li, "prediction_raw": pred})
    return out


def export_spare(
    results_path: Path,
    nqswap_jsonl: Path,
    *,
    steer_direction: str,
) -> list[dict[str, Any]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    idx_to_line: dict[int, int] = {}
    with nqswap_jsonl.open(encoding="utf-8") as f:
        for row_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row.get("idx", row_i))
            li = row.get("line_idx", row_i)
            idx_to_line[idx] = int(li)

    ids = [int(x) for x in results.get("ids", [])]
    if steer_direction == "internal":
        preds = results.get("use_parameter_predictions", [])
    else:
        preds = results.get("use_context_predictions", [])
    if not preds and steer_direction == "both":
        preds = results.get("use_context_predictions", [])

    out: list[dict[str, Any]] = []
    for item_idx, pred in zip(ids, preds):
        li = idx_to_line.get(int(item_idx), int(item_idx))
        out.append({"line_idx": li, "prediction_raw": str(pred)})
    return out


def export_spare_detail_log(log_path: Path, nqswap_jsonl: Path, steer_direction: str) -> list[dict[str, Any]]:
    idx_to_line: dict[int, int] = {}
    with nqswap_jsonl.open(encoding="utf-8") as f:
        for row_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row.get("idx", row_i))
            idx_to_line[idx] = int(row.get("line_idx", row_i))

    key = "use_parameter" if steer_direction == "internal" else "use_context"
    pattern = re.compile(
        rf"\[item (\d+)\]\[{key}\].*?\noutput: (.+?)\n",
        re.DOTALL,
    )
    out: list[dict[str, Any]] = []
    text = log_path.read_text(encoding="utf-8")
    for m in pattern.finditer(text):
        item_idx = int(m.group(1))
        pred = m.group(2).strip().strip("'\"")
        li = idx_to_line.get(item_idx, item_idx)
        out.append({"line_idx": li, "prediction_raw": pred})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--source",
        required=True,
        choices=("ircan", "juice", "ck", "adacad", "spare", "spare_log", "jsonl"),
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--nqswap-jsonl", type=Path, default=None)
    ap.add_argument("--steer-direction", default="external", choices=("external", "internal", "both"))
    args = ap.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    src = args.source

    if src == "jsonl":
        rows = []
        with args.input.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        exported = _pred_rows(rows) if rows and "prediction_raw" in rows[0] else rows
    elif src == "ircan":
        exported = export_ircan(args.input, manifest)
    elif src == "juice":
        exported = export_juice(args.input, manifest)
    elif src == "ck":
        exported = export_ck(args.input)
    elif src == "adacad":
        exported = export_adacad(args.input, manifest)
    elif src == "spare":
        if not args.nqswap_jsonl:
            raise SystemExit("--nqswap-jsonl required for spare")
        exported = export_spare(args.input, args.nqswap_jsonl, steer_direction=args.steer_direction)
    elif src == "spare_log":
        if not args.nqswap_jsonl:
            raise SystemExit("--nqswap-jsonl required for spare_log")
        exported = export_spare_detail_log(args.input, args.nqswap_jsonl, args.steer_direction)
    else:
        raise SystemExit(f"unknown source {src}")

    if not exported:
        raise SystemExit(f"no predictions exported from {args.input}")
    write_predictions_jsonl(args.output.resolve(), exported)
    print(f"exported {len(exported)} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
