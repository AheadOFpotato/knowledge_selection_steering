#!/usr/bin/env python3
"""Build matched A/B/C/D NQ-Swap directions and report geometry.

Conditions per question:
  A: question only; B: org_context (correct); C: sub_context (conflicting);
  D: another example's org_context (structure-matched distractor).

The output direction NPZs are directly consumable by
``run_resid_projection_ablation.py``.  No labels or model generations are
needed: all four conditions are measured on the same examples.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dump_activations_from_testjsonl import (
    _collect_resid_last_token_batch,
    _get_model_and_tokenizer,
    _load_rows_nq_swap,
    _nq_swap_prompt,
    _nq_swap_question_only_prompt,
)
from project_paths import ROOT


def _parse_layers(value: str) -> list[int]:
    # Kept for metadata validation; hidden states are collected for all layers.
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _collect(model: Any, tokenizer: Any, prompts: list[str], batch_size: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        print(f"[activations] batch {start}:{end}/{len(prompts)}", flush=True)
        rows.extend(_collect_resid_last_token_batch(model, tokenizer, prompts[start:end]))
    if not rows:
        raise ValueError("no NQ-Swap rows selected")
    return np.stack(rows, axis=0).astype(np.float32)  # [N, layer, hidden]


def _direction_npz(out_dir: Path, name: str, left: np.ndarray, right: np.ndarray, metadata: dict[str, Any]) -> Path:
    path = out_dir / f"direction_{name}.npz"
    # Object arrays are intentionally avoided: the existing evaluator indexes [row, layer, hidden].
    np.savez_compressed(path, resid_act_clean=left, resid_act_conflict=right,
                        direction_name=np.array(name), num_samples=np.array(left.shape[0]))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama3-8b-it")
    ap.add_argument("--gpu", default="1")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/home/lyb/conflict_steer/data/NQ-Swap/dev_dedup_by_question_only_p_no_table_li.jsonl"),
    )
    ap.add_argument("--out-root", type=Path, default=ROOT / "output/four_condition_control")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all rows")
    ap.add_argument("--layers", default="10", help="Layers recorded in metadata")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rows_indexed = _load_rows_nq_swap(args.input)
    rows = [row for _, row in rows_indexed]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if len(rows) < 2:
        raise SystemExit("need at least two NQ-Swap rows so D can use a different context")

    prompts: dict[str, list[str]] = {k: [] for k in ("A", "B", "C", "D")}
    for i, row in enumerate(rows):
        q = str(row.get("question", "")).strip()
        prompts["A"].append(_nq_swap_question_only_prompt(q))
        prompts["B"].append(_nq_swap_prompt(q, str(row.get("org_context", ""))))
        prompts["C"].append(_nq_swap_prompt(q, str(row.get("sub_context", ""))))
        # Circularly shifted context preserves context length/style without matching the question.
        other = rows[(i + 1) % len(rows)]
        prompts["D"].append(_nq_swap_prompt(q, str(other.get("org_context", ""))))

    print(f"[setup] model={args.model} gpu={args.gpu} rows={len(rows)} batch_size={args.batch_size}", flush=True)
    print("[setup] loading model and tokenizer...", flush=True)
    model, tokenizer = _get_model_and_tokenizer(args.model)
    print("[setup] model loaded; extracting A/B/C/D residual activations", flush=True)
    activations: dict[str, np.ndarray] = {}
    for key in ("A", "B", "C", "D"):
        print(f"[condition {key}] {len(prompts[key])} prompts", flush=True)
        activations[key] = _collect(model, tokenizer, prompts[key], args.batch_size)
        print(f"[condition {key}] complete", flush=True)
    out_dir = (args.out_root / args.model).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "activations_ABCD.npz", **activations)

    pairs = {"A-C": ("A", "C"), "B-C": ("B", "C"), "A-B": ("A", "B"), "A-D": ("A", "D")}
    stats: dict[str, Any] = {
        "model": args.model,
        "input": str(args.input.resolve()),
        "num_samples": len(rows),
        "layers": _parse_layers(args.layers),
        "conditions": {
            "A": "question only", "B": "correct context", "C": "conflicting context",
            "D": "unrelated context from a different NQ-Swap example",
        },
        "directions": {},
    }
    direction_deltas: dict[str, np.ndarray] = {}
    for name, (left_key, right_key) in pairs.items():
        left, right = activations[left_key], activations[right_key]
        mean_left, mean_right = left.mean(axis=0), right.mean(axis=0)
        delta = mean_left - mean_right
        direction_deltas[name] = delta
        norms = np.linalg.norm(delta, axis=1)
        denom = np.linalg.norm(mean_left, axis=1) * np.linalg.norm(mean_right, axis=1)
        cosine = np.divide((mean_left * mean_right).sum(axis=1), denom,
                           out=np.zeros_like(norms), where=denom > 0)
        path = _direction_npz(out_dir, name, left, right, stats)
        print(f"[direction {name}] wrote {path}", flush=True)
        stats["directions"][name] = {
            "left": left_key, "right": right_key, "npz": str(path),
            "delta_norm_by_layer": norms.tolist(), "mean_delta_norm": float(norms.mean()),
            "cosine_left_right_by_layer": cosine.tolist(),
        }
    direction_cosines: dict[str, list[float]] = {}
    names = list(direction_deltas)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left_delta, right_delta = direction_deltas[left_name], direction_deltas[right_name]
            denom = np.linalg.norm(left_delta, axis=1) * np.linalg.norm(right_delta, axis=1)
            value = np.divide((left_delta * right_delta).sum(axis=1), denom,
                              out=np.zeros_like(denom), where=denom > 0)
            direction_cosines[f"{left_name}__vs__{right_name}"] = value.tolist()
    stats["direction_cosine_by_layer"] = direction_cosines
    (out_dir / "geometry.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] geometry written to {out_dir / 'geometry.json'}", flush=True)
    print(json.dumps({"out_dir": str(out_dir), "directions": stats["directions"]}, indent=2))


if __name__ == "__main__":
    main()
