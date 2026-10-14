#!/usr/bin/env python3
"""Evaluate residual steering on fixed WebArena observations.

This is next-action replay, not an official end-to-end WebArena score. The
environment does not transition after a newly generated action.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from run_resid_projection_ablation import (
    _build_projection_dirs,
    _iter_alpha_grid,
    _layer_hook_with_vector,
    generate_batch,
    get_model_and_tokenizer,
)


DEFAULT_DATA = Path(os.environ.get("WEBAGENT_DATA", str(Path(__file__).resolve().parents[2] / "data" / "policy_web" / "webarena_offline.jsonl")))
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = Path(os.environ.get("DIR_NPZ", str(_ROOT / "output" / "dump_taxonomy" / "llama3.1-8b-it" / "NQ-Swap" / "dev_dedup_p_only" / "resid_hidden_state__unfiltered_all_pairs.npz")))
DEFAULT_OUT = Path(os.environ.get("STEER_OUT_ROOT", str(_ROOT / "output" / "steer_taxonomy"))) / "WebArenaOffline"


SYSTEM = """You are a web navigation agent. Given a task, prior actions, and the current accessibility tree, choose exactly one next action.
Return only one WebArena-style function call such as click('30'), fill('8', 'query'), scroll(0, 600), send_msg_to_user('answer'), or report_infeasible('reason').
Use element IDs exactly as shown in the accessibility tree. Do not include reasoning or markdown."""


def prompt(row: dict[str, Any]) -> str:
    history = row.get("action_history") or []
    history_text = "\n".join(f"- {item}" for item in history) if history else "(none)"
    return (
        f"{SYSTEM}\n\nTask: {row['task']}\nCurrent URL: {row.get('url', '')}\n\n"
        f"Prior actions:\n{history_text}\n\nAccessibility tree:\n{row['observation']}\n\nNext action:"
    )


def parse_action(text: str) -> dict[str, Any] | None:
    candidate = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", candidate, re.S)
    if match:
        candidate = match.group(0)
    value: Any
    try:
        node = ast.parse(candidate, mode="eval").body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return {
                "function": node.func.id.strip().lower(),
                "args": [ast.literal_eval(arg) for arg in node.args],
                "kwargs": {
                    kw.arg: ast.literal_eval(kw.value)
                    for kw in node.keywords if kw.arg is not None
                },
            }
    except Exception:
        # Tolerate the one released human label that contains SQL-style
        # doubled quotes inside send_msg_to_user(...).
        call_match = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", candidate, re.S)
        if call_match:
            payload = call_match.group(2).strip()
            if len(payload) >= 2 and payload[0] == payload[-1] == "'":
                payload = payload[1:-1]
            return {
                "function": call_match.group(1).strip().lower(),
                "args": [payload.replace("''", "'")],
                "kwargs": {},
                "parse_repaired": True,
            }
    try:
        value = json.loads(candidate)
    except Exception:
        try:
            value = ast.literal_eval(candidate)
        except Exception:
            return None
    if not isinstance(value, dict):
        return None
    function = value.get("function") or value.get("name") or value.get("action")
    kwargs = value.get("kwargs") or value.get("arguments") or value.get("params") or {}
    args = value.get("args") or []
    if not isinstance(function, str) or not isinstance(kwargs, dict) or not isinstance(args, list):
        return None
    return {"function": function.strip().lower(), "args": args, "kwargs": kwargs}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def score(pred: dict[str, Any] | None, ref: dict[str, Any]) -> dict[str, bool]:
    if pred is None:
        return {"valid": False, "function": False, "target": False, "arguments": False, "exact": False}
    pf, rf = norm(pred["function"]), norm(ref.get("function", ""))
    pa, ra = list(pred.get("args") or []), list(ref.get("args") or [])
    pk, rk = pred.get("kwargs") or {}, ref.get("kwargs") or {}
    target_keys = ("element_id", "bid", "node_id", "id")
    element_functions = {"click", "fill", "type", "hover", "select_option"}
    ref_target = ra[0] if ra and rf in element_functions else next((rk[k] for k in target_keys if k in rk), None)
    pred_target = pa[0] if pa and pf in element_functions else next((pk[k] for k in target_keys if k in pk), None)
    target_ok = ref_target is None or norm(pred_target) == norm(ref_target)
    ref_tail = ra[1:] if ref_target is not None and ra else ra
    pred_tail = pa[1:] if pred_target is not None and pa else pa
    positional_ok = len(ref_tail) == len(pred_tail) and all(norm(a) == norm(b) for a, b in zip(pred_tail, ref_tail))
    non_target = {k: v for k, v in rk.items() if k not in target_keys}
    keyword_ok = all(k in pk and norm(pk[k]) == norm(v) for k, v in non_target.items())
    args_ok = positional_ok and keyword_ok
    exact = pf == rf and target_ok and args_ok and len(pk) == len(rk)
    return {"valid": True, "function": pf == rf, "target": target_ok, "arguments": args_ok, "exact": exact}


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="llama3.1-8b-it")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--layers", default="10")
    ap.add_argument("--alpha-start", type=float, default=0)
    ap.add_argument("--alpha-end", type=float, default=2)
    ap.add_argument("--alpha-step", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument(
        "--direction-mode",
        choices=("true", "shuffled", "random_split", "gaussian"),
        default="true",
    )
    ap.add_argument("--direction-control-seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Missing offline dataset: {args.data}")
    if not args.npz.is_file():
        raise SystemExit(f"Missing NQ-Swap direction: {args.npz}")
    rows = load_rows(args.data, args.max_samples)
    alphas = _iter_alpha_grid(args.alpha_start, args.alpha_end, args.alpha_step)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model, tokenizer = get_model_and_tokenizer(args.model)
    layers = [int(x) for x in str(args.layers).split(",") if x.strip()]
    with np.load(args.npz, allow_pickle=True) as archive:
        n_direction_rows = len(archive["resid_act_clean"])
    vectors, _scales, direction_meta = _build_projection_dirs(
        args.npz, layers, list(range(n_direction_rows)), steer_coeff_mode="alpha",
        direction_mode=args.direction_mode,
        direction_control_seed=args.direction_control_seed,
        steer_toward="context",
    )
    model_layers = model.model.layers
    current_alpha = {"value": 0.0}
    handles = []
    for layer_idx, vector in vectors.items():
        base = _layer_hook_with_vector(vector, 1.0)
        def hook(module, inputs, output, *, _base=base):
            alpha = current_alpha["value"]
            if alpha == 0:
                return output
            shifted = _base(module, inputs, output)
            original_h = output[0] if isinstance(output, tuple) else output
            shifted_h = shifted[0] if isinstance(shifted, tuple) else shifted
            mixed = original_h + alpha * (shifted_h - original_h)
            return (mixed,) + output[1:] if isinstance(output, tuple) else mixed
        handles.append(model_layers[layer_idx].register_forward_hook(hook))

    args.out_root.mkdir(parents=True, exist_ok=True)
    try:
        for alpha in alphas:
            out_path = args.out_root / args.model / f"L{args.layers.replace(',', '_')}_alpha{alpha:g}.json"
            if args.skip_existing and out_path.is_file():
                print(f"[webarena alpha={alpha:g}] skip {out_path}", flush=True)
                continue
            current_alpha["value"] = float(alpha)
            details: list[dict[str, Any]] = []
            counts = {k: 0 for k in ("valid", "function", "target", "arguments", "exact")}
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                outputs = generate_batch(
                    model, tokenizer, [prompt(row) for row in batch], model_key=args.model,
                    max_length=args.max_length, max_new_tokens=args.max_new_tokens,
                )
                for row, raw in zip(batch, outputs):
                    parsed = parse_action(raw)
                    metrics = score(parsed, row["reference_action"])
                    for key, passed in metrics.items():
                        counts[key] += int(passed)
                    details.append({
                        "sample_id": row["sample_id"], "task": row["task"], "url": row.get("url", ""),
                        "reference_action": row["reference_action"], "raw_prediction": raw,
                        "parsed_prediction": parsed, "scores": metrics,
                    })
                done = min(start + len(batch), len(rows))
                print(f"[webarena alpha={alpha:g}] {done}/{len(rows)} exact={counts['exact']}", flush=True)
            total = len(rows)
            payload = {
                "benchmark": "WebArena offline human-expert fixed-observation next-action replay",
                "official_end_to_end_score": False,
                "alpha": alpha, "layers": layers, "samples": total,
                "direction_mode": args.direction_mode,
                "direction_control_seed": args.direction_control_seed,
                "metrics": {f"{key}_accuracy": counts[key] / total if total else 0.0 for key in counts},
                "counts": counts, "direction": direction_meta, "results": details,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[webarena alpha={alpha:g}] wrote {out_path}", flush=True)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
