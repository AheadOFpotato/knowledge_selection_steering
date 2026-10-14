#!/usr/bin/env python3
"""Residual-stream steering with conflict-correct/conflict-wrong direction.

This script mirrors src/run_resid_projection_ablation.py but changes how the
steering direction is computed:

- Direction vectors are computed from *conflict(stage2)* residual stream vectors
  restricted to samples whose manifest stage2_label matches --direction-label
  (correct or wrong).

Specialized to:
  <DATA_DIR>/conflict_steer/llama3-8b-it/NQ-Swap/dev_dedup_by_question_only_p_no_table_li/
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEER_OUT_DIR = ROOT / "output" / "new_steer"

MODEL_PATHS: dict[str, str] = {
    "llama3-8b-it": "<DATA_DIR>/model/Llama3-8b-it",
    "gemma2-9b-it": "<DATA_DIR>/model/Gemma2-9b-it",
    "qwen3-8b": "<DATA_DIR>/model/Qwen3-8B",
    "mistral-7b-v0.1": "<DATA_DIR>/model/Mistral-7B-v0.1",
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 8192
PROGRESS_EVERY = 10


def _abspath(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _safe_name(s: str) -> str:
    t = str(s).strip().replace("\\", "/")
    if "/" in t:
        t = Path(t).stem
    return t.replace(" ", "_").replace("/", "_")


def _truncate_for_judge(s: str) -> str:
    if not s:
        return s
    m = re.search(r"[。！？.!?]", s)
    return s[: m.end()].rstrip() if m else s


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _normalized_answers(gold_answers: list[str]) -> list[str]:
    return [t for t in (_norm(a) for a in gold_answers) if len(t) >= 2]


def _answer_matches(generated: str, gold_answers: list[str]) -> bool:
    g = _norm(generated)
    if not g:
        return False
    return any(t in g for t in _normalized_answers(gold_answers))


def _distractor_in_text(generated: str, distracted_token: str) -> bool:
    g = _norm(generated)
    if not g:
        return False
    t = _norm(str(distracted_token or "").strip())
    if not t:
        return False
    return t in g


def _first_option_letter(generated: str) -> str:
    g = str(generated or "")
    m = re.search(r"[A-Za-z]", g)
    return m.group(0).upper() if m else ""


def _answer_matches_letters(generated: str, letters: list[str]) -> bool:
    first = _first_option_letter(generated)
    if not first:
        return False
    allowed = {str(l).strip().upper() for l in letters if str(l).strip()}
    return first in allowed


def _stage2_label_from_strings(raw_text: str, gold_answers: list[str], distracted_tokens: list[str]) -> str:
    if _answer_matches(raw_text, gold_answers):
        return "correct"
    if any(_distractor_in_text(raw_text, d) for d in distracted_tokens if str(d).strip()):
        return "wrong"
    return "skipped"


def _stage2_label_from_letters(raw_text: str, gold_letters: list[str], wrong_letters: list[str]) -> str:
    if _answer_matches_letters(raw_text, gold_letters):
        return "correct"
    if _answer_matches_letters(raw_text, wrong_letters):
        return "wrong"
    return "skipped"


def mean_vector(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    arrs = []
    for v in vectors:
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        arrs.append(a)
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float64)


def _load_conflict_stage2_vectors(npz_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    try:
        if "resid_act_stage2" in z.files:
            key = "resid_act_stage2"
        elif "resid_act_conflict" in z.files:
            key = "resid_act_conflict"
        else:
            raise KeyError(f"missing resid_act_stage2/resid_act_conflict, keys={z.files}")
        if "line_idx" not in z.files:
            raise KeyError(f"missing line_idx, keys={z.files}")
        return key, z[key], z["line_idx"].astype(np.int64)
    finally:
        z.close()


def _conflict_subset_mean_dirs(
    npz_path: Path, manifest_json: Path, layers: list[int], direction_label: str
) -> dict[int, np.ndarray]:
    vec_key, raw_vecs, line_idx = _load_conflict_stage2_vectors(npz_path)
    want = str(direction_label or "").strip().lower()
    if want not in {"correct", "wrong"}:
        raise SystemExit("--direction-label must be 'correct' or 'wrong'")
    obj = json.loads(manifest_json.read_text(encoding="utf-8"))
    samples = obj.get("samples", [])
    if not isinstance(samples, list):
        raise SystemExit("manifest missing samples list")

    label_by_line: dict[int, str] = {}
    for s in samples:
        if isinstance(s, dict):
            try:
                li = int(s.get("line_idx", -1))
            except Exception:
                continue
            label_by_line[li] = str(s.get("stage2_label", "")).strip().lower()

    per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    for i in range(len(line_idx)):
        lii = int(line_idx[i])
        if label_by_line.get(lii, "") != want:
            continue
        arr = np.asarray(raw_vecs[i], dtype=np.float64)
        if arr.ndim != 2:
            continue
        for l in layers:
            if 0 <= l < arr.shape[0]:
                per_layer[l].append(arr[l])

    out: dict[int, np.ndarray] = {}
    for l in layers:
        mv = mean_vector(per_layer.get(l, []))
        if mv is not None and np.isfinite(mv).all():
            out[l] = mv.astype(np.float32)

    if not out:
        raise SystemExit(
            f"no valid conflict-{want} direction vectors computed (npz_key={vec_key}). "
            f"Check manifest stage2_label=='{want}' and NPZ vectors."
        )
    return out


def _load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


def get_model_and_tokenizer(model_name: str):
    model_path = MODEL_PATHS.get(model_name, model_name)
    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "left"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device_map = "cuda" if (visible and "," not in visible) else "auto"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "seq_length") and not hasattr(config, "max_length"):
        config.max_length = int(getattr(config, "seq_length"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_batch(model: torch.nn.Module, tokenizer, prompts: list[str]) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TOKENIZER_MAX_LEN,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    w = input_ids.shape[1]
    return [tokenizer.decode(out[i, w:], skip_special_tokens=True).strip() for i in range(input_ids.shape[0])]


def _layer_hook_with_vector(v_mean: np.ndarray, alpha: float):
    d = torch.tensor(v_mean, dtype=torch.float32)
    a = float(alpha)

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return out
        dv = d.to(device=hs.device, dtype=hs.dtype)
        if not torch.isfinite(dv).all():
            dv = torch.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
        if float(torch.linalg.vector_norm(dv).item()) == 0.0:
            return out
        mixed = hs + a * dv.view(1, 1, -1)
        return (mixed,) + out[1:] if isinstance(out, tuple) else mixed

    return hook


@dataclass
class ModeResult:
    stage2_correct: int
    stage2_wrong: int
    stage2_skipped: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        denom = max(1, self.total)
        return {
            "total_samples": self.total,
            "stage2_runs": self.total,
            "stage2_correct": self.stage2_correct,
            "stage2_wrong": self.stage2_wrong,
            "stage2_skipped": self.stage2_skipped,
            "stage2_correct_rate": self.stage2_correct / denom,
            "stage2_wrong_rate": self.stage2_wrong / denom,
        }


def _iter_alpha_grid(a0: float, a1: float, step: float) -> list[float]:
    if step == 0:
        raise SystemExit(
            "alpha-step must be non-zero when alpha-start differs from alpha-end"
        )
    out: list[float] = []
    x = float(a0)
    n = 0
    if step > 0:
        if a0 > a1:
            raise SystemExit(
                "alpha-start must be <= alpha-end when alpha-step is positive"
            )
        while x <= a1 + 1e-9 and n < 1_000_000:
            out.append(round(x, 12))
            x += step
            n += 1
    else:
        if a0 < a1:
            raise SystemExit(
                "alpha-start must be >= alpha-end when alpha-step is negative"
            )
        while x >= a1 - 1e-9 and n < 1_000_000:
            out.append(round(x, 12))
            x += step
            n += 1
    if not out:
        raise SystemExit("empty alpha grid (check alpha-start/end/step)")
    return out


def _alphas_from_cli(a0: float, a1: float, step: float) -> list[float]:
    if abs(float(a0) - float(a1)) < 1e-12:
        return [float(a0)]
    return _iter_alpha_grid(float(a0), float(a1), float(step))


def _steer_out_path(
    out_root: Path,
    manifest_obj: dict[str, Any],
    model_name: str,
    layers_tag: str,
    alpha: float,
    npz_category_tag: str,
    direction_label: str,
) -> Path:
    mod = _safe_name(str(manifest_obj.get("model") or model_name))
    ds = _safe_name(str(manifest_obj.get("dataset") or "unknown"))
    cat = _safe_name(str(manifest_obj.get("category") or npz_category_tag or ""))
    group = _safe_name(str(manifest_obj.get("group") or ""))
    dlab = _safe_name(str(direction_label or ""))
    # Keep same naming spirit as run_resid_projection_ablation.py
    # <out_root>/<model>/<dataset>/<category__npz_tag>__<group>/L<layers>_alpha<alpha>_steer_result.json
    cat_tag = cat or _safe_name(str(npz_category_tag or ""))
    if cat_tag and npz_category_tag and _safe_name(npz_category_tag) not in cat_tag:
        cat_tag = f"{cat_tag}__{_safe_name(npz_category_tag)}"
    group_dir = f"{cat_tag}__{group}" if group else f"{cat_tag}"
    if dlab:
        group_dir = f"{group_dir}__dir_{dlab}"
    out_dir = out_root / mod / ds / group_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    a = f"{alpha:g}".replace(".", "p")
    return out_dir / f"L{_safe_name(layers_tag)}_alpha{a}_steer_result.json"


def _run_stage2_with_steer(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    dataset_name: str,
    projection_dirs: dict[int, np.ndarray],
    alpha: float,
    batch_size: int,
    mode_name: str,
    no_skip_in_stage2: bool,
) -> tuple[ModeResult, list[dict[str, Any]]]:
    layers = model.model.layers  # type: ignore[attr-defined]
    hooks = []
    for li, d in projection_dirs.items():
        if 0 <= li < len(layers):
            hooks.append(layers[li].register_forward_hook(_layer_hook_with_vector(d, alpha)))

    per_sample: list[dict[str, Any]] = []
    is_kre = str(dataset_name).strip().lower() == "kre"
    for s in samples:
        stage2_prompt = s.get("stage2_prompt_input")
        if stage2_prompt is None:
            stage2_prompt = s.get("conflict_prompt_input")
        per_sample.append(
            {
                "line_idx": int(s.get("line_idx", -1)),
                "manifest_stage2_label": str(s.get("stage2_label", "")),
                "conflict_prompt_input": stage2_prompt,
                "gold_answers": list(s.get("gold_answers", [])),
                "distracted_tokens": list(s.get("distracted_tokens", []))
                if isinstance(s.get("distracted_tokens"), list)
                else [s.get("distracted_token", "")],
                "stage2_output_raw": None,
                "stage2_output_judged": None,
                "stage2_label": "not_run",
            }
        )

    try:
        s2_correct = s2_wrong = s2_skipped = 0
        done = 0
        for st in range(0, len(samples), batch_size):
            chunk = samples[st : st + batch_size]
            prompts = [
                str(
                    x.get("stage2_prompt_input")
                    if x.get("stage2_prompt_input") is not None
                    else x.get("conflict_prompt_input", "")
                )
                for x in chunk
            ]
            outs = generate_batch(model, tokenizer, prompts)
            for j, row in enumerate(chunk):
                raw = outs[j]
                judged = _truncate_for_judge(raw)
                gold = [str(v) for v in row.get("gold_answers", [])]
                dlist = row.get("distracted_tokens")
                if not isinstance(dlist, list):
                    dlist = [str(row.get("distracted_token", ""))]
                dlist = [str(x) for x in dlist]

                idx = st + j
                per_sample[idx]["stage2_output_raw"] = raw
                per_sample[idx]["stage2_output_judged"] = judged

                if is_kre:
                    # For KRE, gold answers are option letters; distracted_tokens are wrong letters.
                    if _answer_matches_letters(judged, gold):
                        lab = "correct"
                    elif no_skip_in_stage2:
                        lab = "wrong"
                    else:
                        lab = _stage2_label_from_letters(judged, gold, dlist)
                else:
                    if _answer_matches(judged, gold):
                        lab = "correct"
                    elif no_skip_in_stage2:
                        lab = "wrong"
                    else:
                        lab = _stage2_label_from_strings(judged, gold, dlist)

                per_sample[idx]["stage2_label"] = lab
                if lab == "correct":
                    s2_correct += 1
                elif lab == "wrong":
                    s2_wrong += 1
                else:
                    s2_skipped += 1

            done += len(chunk)
            if done % PROGRESS_EVERY == 0 or done == len(samples):
                print(f"[{mode_name}] stage2 progress: {done}/{len(samples)}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    return (
        ModeResult(
            stage2_correct=s2_correct,
            stage2_wrong=s2_wrong,
            stage2_skipped=s2_skipped,
            total=len(samples),
        ),
        per_sample,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run steer ablation, direction = mean(conflict vectors filtered by stage2_label (correct/wrong)."
    )
    ap.add_argument("--gpu", type=str, required=True, help="CUDA_VISIBLE_DEVICES (e.g. 0).")
    ap.add_argument("--layers", type=str, required=True, help="Comma-separated layer indices, e.g. 10 or 27,28,29.")
    ap.add_argument("--alpha-start", type=float, default=0.0)
    ap.add_argument("--alpha-end", type=float, default=1.5)
    ap.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="Step between alphas. Negative allowed when alpha-start >= alpha-end.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", type=str, default="llama3-8b-it")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_STEER_OUT_DIR)
    ap.add_argument("--npz-category-tag", type=str, default="dev_dedup_by_question_only_p_no_table_li")
    ap.add_argument(
        "--direction-label",
        type=str,
        default="correct",
        choices=["correct", "wrong"],
        help="Which subset in conflict(stage2) to use for direction mean.",
    )

    base_dir = Path("<DATA_DIR>/conflict_steer/llama3-8b-it/NQ-Swap/dev_dedup_by_question_only_p_no_table_li")
    ap.add_argument(
        "--npz",
        type=Path,
        default=base_dir / "resid_hidden_state__stage1_wrong_use_correct_context.npz",
    )
    ap.add_argument(
        "--test-json",
        type=Path,
        default=base_dir / "internal_aware_data__stage1_wrong_use_correct_context.json",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    out_root = _abspath(Path(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)

    npz_path = _abspath(Path(args.npz))
    test_json = _abspath(Path(args.test_json))
    if not npz_path.is_file():
        raise SystemExit(f"npz not found: {npz_path}")
    if not test_json.is_file():
        raise SystemExit(f"manifest json not found: {test_json}")

    obj = json.loads(test_json.read_text(encoding="utf-8"))
    samples_all = list(obj.get("samples", []))
    if not samples_all:
        raise SystemExit("test manifest has no samples")

    dataset_name = str(obj.get("dataset") or "NQ-Swap")
    group_name = str(obj.get("group") or "")
    no_skip_in_stage2 = group_name == "stage1_wrong_use_correct_context"

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    alphas = _alphas_from_cli(args.alpha_start, args.alpha_end, args.alpha_step)

    dirs = _conflict_subset_mean_dirs(npz_path, test_json, layers, str(args.direction_label))
    used_layers = sorted(dirs.keys())

    model, tokenizer = get_model_and_tokenizer(str(args.model))

    out_paths: list[str] = []
    for ai, alpha in enumerate(alphas):
        out_path = _steer_out_path(
            out_root=out_root,
            manifest_obj=obj,
            model_name=str(args.model),
            layers_tag=str(args.layers),
            alpha=float(alpha),
            npz_category_tag=str(args.npz_category_tag or ""),
            direction_label=str(args.direction_label),
        )
        print(f"[alpha {ai + 1}/{len(alphas)}] alpha={alpha:g} -> {out_path.name}", flush=True)

        r, ps = _run_stage2_with_steer(
            model=model,
            tokenizer=tokenizer,
            samples=samples_all,
            dataset_name=dataset_name,
            projection_dirs={li: dirs[li] for li in used_layers},
            alpha=float(alpha),
            batch_size=int(args.batch_size),
            mode_name="project_to_conflict_correct_mean_direction",
            no_skip_in_stage2=no_skip_in_stage2,
        )

        results: dict[str, Any] = {
            "after_steer_stage2_on_all_samples": {
                "stage2_runs_with_steer": r.total,
                "stage2_correct_after_steer": r.stage2_correct,
                "stage2_wrong_after_steer": r.stage2_wrong,
                "stage2_skipped_after_steer": r.stage2_skipped,
                "rate_correct": r.stage2_correct / max(1, r.total),
                "details": r.to_dict(),
            }
        }
        per_sample_out: dict[str, Any] = {"project_to_conflict_correct_mean_direction": ps}

        out = {
            "direction_label": str(args.direction_label),
            "direction_note": "Direction is mean(conflict/stage2 resid vectors) over subset where manifest stage2_label matches direction_label.",
            "steer_formula": "h + alpha * v_conflict_subset_mean",
            "direction_npz": str(npz_path),
            "test_json": str(test_json),
            "out_json": str(out_path),
            "gpu": str(args.gpu),
            "batch_size": int(args.batch_size),
            "model": str(args.model),
            "alpha": float(alpha),
            "alpha_grid": {
                "alpha_start": float(args.alpha_start),
                "alpha_end": float(args.alpha_end),
                "alpha_step": float(args.alpha_step),
                "alphas": [float(x) for x in alphas],
                "index_in_grid": ai,
            },
            "requested_resid_layers": layers,
            "used_resid_layers_with_valid_dirs": used_layers,
            "results": results,
            "per_sample_outputs_and_judgements": per_sample_out,
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out_paths.append(str(out_path))

    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()

