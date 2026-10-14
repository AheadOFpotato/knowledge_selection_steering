#!/usr/bin/env python3
"""Residual steering ablation on LooGLE (portable bundle, all paths relative to bundle root)."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT / "src"))

from steer_core import (  # noqa: E402
    BUNDLE_ROOT as ROOT,
    _alphas_from_cli,
    _layer_hook_with_vector,
    _load_resid_vectors,
    _safe_name,
    generate_batch,
    get_model_and_tokenizer,
)

DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_STEER_OUT_DIR = ROOT / "output" / "new_steer"
DEFAULT_MAX_PROMPT_LEN = 8192
PROGRESS_EVERY = 10

TASK_TO_FILE = {
    "shortdep_qa": "shortdep_qa.jsonl",
    "shortdep_cloze": "shortdep_cloze.jsonl",
    "longdep_qa": "longdep_qa.jsonl",
    "longdep_summarization": "summarization.jsonl",
    "summarization": "summarization.jsonl",
}

TASK_TO_PROMPT_KEY = {
    "shortdep_qa": "shortdep_qa",
    "shortdep_cloze": "shortdep_cloze",
    "longdep_qa": "longdep_qa",
    "longdep_summarization": "longdep_summarization",
    "summarization": "longdep_summarization",
}

TASK_TO_MAX_NEW = {
    "shortdep_qa": 300,
    "shortdep_cloze": 50,
    "longdep_qa": 500,
    "longdep_summarization": 500,
    "summarization": 500,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _is_mcq(question: str) -> bool:
    q = str(question or "")
    return bool(re.search(r"\n\s*1[\.\)]\s", q))


def _cloze_reference(question: str, answer: Any) -> dict[str, str]:
    masks = sorted(set(re.findall(r"<mask-\d+>", str(question))))
    if isinstance(answer, dict):
        return {str(k): str(v) for k, v in answer.items()}
    vals = list(answer) if isinstance(answer, list) else [answer]
    return {m: str(v) for m, v in zip(masks, vals)}


def _parse_cloze_pred(text: str) -> dict[str, str] | None:
    s = str(text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except json.JSONDecodeError:
        pass
    try:
        obj = eval(s, {"__builtins__": {}})  # noqa: S307
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        return None
    return None


def _extract_mcq_choice(pred: str) -> str | None:
    s = str(pred or "")
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*([1-4])\b", s, re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"\b([1-4])\b", s)
    return nums[-1] if nums else None


def _rouge_l_f1(ref: str, hyp: str) -> float:
    try:
        from rouge import Rouge

        if not str(hyp or "").strip():
            return 0.0
        score = Rouge().get_scores(hyps=[str(hyp)], refs=[str(ref)])[0]["rouge-l"]["f"]
        return float(score)
    except Exception:
        return 0.0


def judge_prediction(task: str, item: dict[str, Any], pred: str) -> dict[str, Any]:
    ref = item.get("answer")
    question = str(item.get("question") or "")
    task_key = TASK_TO_PROMPT_KEY.get(task, task)

    if task_key == "shortdep_cloze":
        reference = _cloze_reference(question, ref)
        hypothesis = _parse_cloze_pred(pred)
        if not reference or hypothesis is None:
            return {"judge": False, "metric": "cloze", "exact": 0.0, "partial": 0.0}
        exact = sum(1 for k, v in reference.items() if hypothesis.get(k) == v)
        partial = sum(
            1
            for k, v in reference.items()
            if k in hypothesis and len(set(_norm(v).split()) & set(_norm(hypothesis[k]).split())) > 0
        )
        n = max(1, len(reference))
        exact_rate = exact / n
        partial_rate = partial / n
        return {
            "judge": exact_rate == 1.0,
            "metric": "cloze",
            "exact": exact_rate,
            "partial": partial_rate,
        }

    if task_key == "longdep_summarization":
        rl = _rouge_l_f1(str(ref), pred)
        return {"judge": rl >= 0.15, "metric": "rouge-l", "rouge_l_f1": rl}

    ref_s = str(ref).strip()
    if task_key == "longdep_qa" and _is_mcq(question):
        pred_choice = _extract_mcq_choice(pred)
        ok = pred_choice is not None and pred_choice == ref_s
        return {"judge": ok, "metric": "mcq", "pred_choice": pred_choice, "ref_choice": ref_s}

    rl = _rouge_l_f1(ref_s, pred)
    ok = rl >= 0.5 or _norm(ref_s) in _norm(pred)
    return {"judge": ok, "metric": "rouge-l", "rouge_l_f1": rl}


def load_prompt_template(task: str, prompt_json: Path | None = None) -> str:
    p = prompt_json or (ROOT / "config" / "task2prompt.json")
    templates = json.loads(p.read_text(encoding="utf-8"))
    key = TASK_TO_PROMPT_KEY[task]
    if key not in templates:
        raise KeyError(f"prompt key {key!r} not in {p}")
    return templates[key]


def resolve_data_json(data_root: Path, task: str, data_json: Path | None) -> Path:
    if data_json is not None:
        p = data_json if data_json.is_absolute() else (ROOT / data_json)
        p = p.resolve()
        if not p.is_file():
            raise SystemExit(f"data json not found: {p}")
        return p
    fname = TASK_TO_FILE.get(task)
    if not fname:
        raise SystemExit(f"unknown task {task!r}")
    p = (data_root / fname).resolve()
    if not p.is_file():
        raise SystemExit(f"data file not found: {p}")
    return p


def resolve_npz(npz_arg: Path) -> Path:
    p = npz_arg if npz_arg.is_absolute() else (ROOT / npz_arg)
    p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"NPZ not found: {p}")
    return p


def load_loogle_rows(path: Path, max_samples: int = 0, seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(rows)), max_samples))
        rows = [rows[i] for i in idxs]
    return rows


def build_prompt(template: str, item: dict[str, Any]) -> str:
    context = str(item.get("context") or item.get("input") or "").strip()
    question = str(item.get("question") or "").strip()
    if "{Q}" in template:
        return template.format(input=context, Q=question)
    return template.format(input=context)


def truncate_prompt_middle(tokenizer, prompt: str, max_len: int) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) <= max_len:
        return prompt
    half = max_len // 2
    input_ids = input_ids[:half] + input_ids[-half:]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def summarize_rows(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("judge"))
    out: dict[str, Any] = {
        "overall_acc": round(100.0 * correct / max(1, total), 2),
        "total": total,
        "correct": correct,
    }
    metric = rows[0].get("metric") if rows else None
    if metric == "rouge-l":
        vals = [float(r.get("rouge_l_f1") or 0.0) for r in rows]
        out["mean_rouge_l_f1"] = round(sum(vals) / max(1, len(vals)), 4)
    elif metric == "cloze":
        out["mean_exact"] = round(
            sum(float(r.get("exact") or 0.0) for r in rows) / max(1, total), 4
        )
        out["mean_partial"] = round(
            sum(float(r.get("partial") or 0.0) for r in rows) / max(1, total), 4
        )
    if task == "longdep_qa":
        buckets = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in rows:
            tag = "mcq" if _is_mcq(r.get("question", "")) else "freeform"
            buckets[tag]["n"] += 1
            buckets[tag]["correct"] += int(bool(r.get("judge")))
        for tag, stat in buckets.items():
            out[f"{tag}_acc"] = round(100.0 * stat["correct"] / max(1, stat["n"]), 2)
            out[f"{tag}_n"] = stat["n"]
    return out


def _out_path(
    out_root: Path,
    model: str,
    task: str,
    direction_tag: str,
    layers: str,
    alpha: float,
) -> Path:
    cat = _safe_name(direction_tag or "default")
    ls = str(layers).replace(",", "_").replace(" ", "")
    out_dir = out_root / _safe_name(model) / "LooGLE" / _safe_name(task) / f"__dir_{cat}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"L{ls}_alpha{float(alpha):g}_loogle_result.json"


@torch.inference_mode()
def run_alpha(
    model,
    tokenizer,
    prompts: list[str],
    items: list[dict[str, Any]],
    task: str,
    projection_dirs: dict[int, Any] | None,
    alpha: float,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    hooks = []
    if projection_dirs and float(alpha) != 0.0:
        layers_mod = model.model.layers  # type: ignore[attr-defined]
        for li, vec in projection_dirs.items():
            if 0 <= li < len(layers_mod):
                hooks.append(
                    layers_mod[li].register_forward_hook(
                        _layer_hook_with_vector(vec, float(alpha))
                    )
                )
    rows: list[dict[str, Any]] = []
    try:
        done = 0
        for st in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[st : st + batch_size]
            chunk_items = items[st : st + batch_size]
            outputs = generate_batch(
                model,
                tokenizer,
                chunk_prompts,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            for item, raw in zip(chunk_items, outputs):
                metrics = judge_prediction(task, item, raw)
                rows.append(
                    {
                        "id": item.get("id"),
                        "doc_id": item.get("doc_id"),
                        "title": item.get("title"),
                        "task": item.get("task") or task,
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        "response": raw,
                        **metrics,
                    }
                )
            done += len(chunk_prompts)
            if done % PROGRESS_EVERY == 0 or done == len(prompts):
                print(f"[loogle] progress: {done}/{len(prompts)}", flush=True)
    finally:
        for h in hooks:
            h.remove()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="LooGLE steering ablation (portable bundle).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASK_TO_FILE.keys()))
    ap.add_argument("--gpu", type=str, required=True)
    ap.add_argument("--layers", type=str, required=True)
    ap.add_argument("--alpha-start", type=float, default=0.0)
    ap.add_argument("--alpha-end", type=float, default=0.0)
    ap.add_argument("--alpha-step", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_STEER_OUT_DIR)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--data-json", type=Path, default=None)
    ap.add_argument("--prompt-json", type=Path, default=None)
    ap.add_argument("--max-prompt-len", type=int, default=DEFAULT_MAX_PROMPT_LEN)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--direction-category-tag", type=str, default="")
    args = ap.parse_args()

    npz = resolve_npz(args.npz)
    data_root = args.data_root if args.data_root.is_absolute() else (ROOT / args.data_root)
    out_root = args.out_root if args.out_root.is_absolute() else (ROOT / args.out_root)

    data_json = resolve_data_json(data_root.resolve(), args.task, args.data_json)
    template = load_prompt_template(args.task, args.prompt_json)
    items = load_loogle_rows(data_json, int(args.max_samples), int(args.sample_seed))
    if not items:
        raise SystemExit(f"no rows in {data_json}")
    print(f"[loogle] bundle={ROOT} task={args.task} rows={len(items)}", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    dirs = _load_resid_vectors(npz, layers)
    used_layers = [li for li in layers if li in dirs]
    if not used_layers:
        raise SystemExit("no valid layers from NPZ")
    proj = {li: (dirs[li][0] - dirs[li][1]).astype("float32") for li in used_layers}

    model, tokenizer = get_model_and_tokenizer(args.model)
    max_prompt_len = int(args.max_prompt_len)
    max_length = int(args.max_length) if args.max_length is not None else max_prompt_len
    max_new_tokens = int(args.max_new_tokens or TASK_TO_MAX_NEW.get(args.task, 300))
    prompts = [
        truncate_prompt_middle(tokenizer, build_prompt(template, item), max_prompt_len)
        for item in items
    ]

    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    alphas = _alphas_from_cli(args.alpha_start, args.alpha_end, args.alpha_step)
    dir_tag = str(args.direction_category_tag or npz.stem).strip()

    for ai, alpha in enumerate(alphas):
        out_path = _out_path(out_root, str(args.model), args.task, dir_tag, str(args.layers), float(alpha))
        if out_path.is_file():
            print(f"[alpha {ai + 1}/{len(alphas)}] skip {out_path.name}", flush=True)
            continue
        print(f"[alpha {ai + 1}/{len(alphas)}] alpha={alpha:g}", flush=True)
        rows = run_alpha(
            model,
            tokenizer,
            prompts,
            items,
            args.task,
            proj if float(alpha) != 0.0 else None,
            float(alpha),
            int(args.batch_size),
            max_length,
            max_new_tokens,
        )
        summary = summarize_rows(args.task, rows)
        payload = {
            "benchmark": "LooGLE",
            "bundle_root": ".",
            "task": args.task,
            "eval_method": "automatic_heuristic",
            "direction_npz": str(npz.relative_to(ROOT)) if npz.is_relative_to(ROOT) else str(npz),
            "data_json": str(data_json.relative_to(ROOT)) if data_json.is_relative_to(ROOT) else str(data_json),
            "model": args.model,
            "alpha": float(alpha),
            "summary": summary,
            "predictions": rows,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[result] {out_path.relative_to(ROOT)} overall={summary['overall_acc']}%", flush=True)


if __name__ == "__main__":
    main()
