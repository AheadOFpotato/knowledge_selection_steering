#!/usr/bin/env python3
"""Residual-stream steering (self-contained).

CLI: ``--model``, ``--npz``, ``--test-json``, ``--gpu``, ``--layers``,
``--alpha-start`` / ``--alpha-end`` / ``--alpha-step``, ``--batch-size``.

``--npz``: resid activations used to compute steer directions (``resid_act_clean`` /
``resid_act_conflict``). Steer on conflict: per hooked layer ``h' = h + alpha * v_clean`` (``v_clean`` = clean residual mean from NPZ).

**Evaluation:** For each alpha in the grid, run **full Stage1 then Stage2** (steer on conflict).
Stage2 runs for **all** samples whose rerun Stage1 passes (not restricted to manifest ``stage2_label==wrong``).
Output: ``output/steer_result/<model>/<dataset>/<category>/L<layer>_alpha<value>_steer_result.json``.
Labels match ``dump_activations_from_testjsonl.py``.
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
NQ_OUT_ROOT = ROOT / "output" / "nq_swap"
DEFAULT_STEER_OUT_DIR = NQ_OUT_ROOT / "steer_result"
DEFAULT_CATEGORY = "who"

MODEL_PATHS: dict[str, str] = {
    "llama3-8b-it": "<DATA_DIR>/model/Llama3-8b-it",
    "gemma2-9b-it": "<DATA_DIR>/model/Gemma2-9b-it",
    "qwen3-8b": "<DATA_DIR>/model/Qwen3-8B",
    "mistral-7b-v0.1": "<DATA_DIR>/model/Mistral-7B-v0.1",
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 8192
PROGRESS_EVERY = 10


def _safe_name(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def _steer_out_path(
    manifest: dict[str, Any],
    args_model: str,
    test_json: Path,
    layers_arg: str,
    alpha: float,
    npz_category_tag: str = "",
) -> Path:
    """``.../L<layer>_alpha<numeric>_steer_result.json`` (alpha formatted with ``:g``, e.g. ``0.05``, ``1.5``)."""
    mod = _safe_name(str(manifest.get("model") or args_model or "unknown"))
    ds = "NQ-Swap"
    cat = _safe_name(str(manifest.get("category") or ""))
    if not cat:
        p = test_json.resolve()
        cat = _safe_name(p.parent.name) if p.parent != p else _safe_name(p.stem)
    tag = _safe_name(str(npz_category_tag))
    if tag:
        cat = f"{cat}__npz_{tag}"
    ls = str(layers_arg).replace(",", "_").replace(" ", "")
    a_str = f"{float(alpha):g}"
    out_dir = DEFAULT_STEER_OUT_DIR / mod / ds / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"L{ls}_alpha{a_str}_steer_result.json"


def _abspath(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


# ----- Labeling (same rules as dump_activations_from_testjsonl.py) -----


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _truncate_for_judge(s: str) -> str:
    if not s:
        return s
    m = re.search(r"[。！？.!?]", s)
    return s[: m.end()].rstrip() if m else s


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


def _stage2_label_from_strings(
    raw_text: str, gold_answers: list[str], distracted_tokens: list[str]
) -> str:
    if _answer_matches(raw_text, gold_answers):
        return "correct"
    if any(_distractor_in_text(raw_text, d) for d in distracted_tokens if str(d).strip()):
        return "wrong"
    return "skipped"


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


def _stage2_label_from_letters(raw_text: str, gold_letters: list[str], wrong_letters: list[str]) -> str:
    if _answer_matches_letters(raw_text, gold_letters):
        return "correct"
    if _answer_matches_letters(raw_text, wrong_letters):
        return "wrong"
    return "skipped"


def _load_tokenizer(model_path: str):
    model_path = str(model_path)
    errors: list[str] = []
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as e:
        errors.append(f"AutoTokenizer (default): {e!r}")
        if "object has no attribute 'keys'" in str(e):
            try:
                return AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, extra_special_tokens={}
                )
            except Exception as e2:
                errors.append(f"AutoTokenizer (extra_special_tokens={{}}): {e2!r}")
        else:
            raise
    except Exception as e:
        errors.append(f"AutoTokenizer (default): {e!r}")
    try:
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        errors.append(f"PreTrainedTokenizerFast.from_pretrained: {e!r}")
    tj = Path(model_path) / "tokenizer.json"
    if tj.is_file():
        try:
            return PreTrainedTokenizerFast(tokenizer_file=str(tj))
        except Exception as e:
            errors.append(f"PreTrainedTokenizerFast(tokenizer_file=...): {e!r}")
    try:
        return AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    except Exception as e:
        errors.append(f"AutoTokenizer (use_fast=False): {e!r}")
    raise RuntimeError(
        f"Could not load tokenizer from {model_path}.\n" + "\n".join(errors)
    ) from None


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


def _load_resid_vectors(activations_npz: Path, layers: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean of last-token resid for clean vs conflict (same NPZ layout as analyze_angle)."""
    z = np.load(activations_npz, allow_pickle=True)
    rc = z["resid_act_clean"]
    rw = z["resid_act_conflict"]
    z.close()
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for li in layers:
        vc, vw = [], []
        for i in range(len(rc)):
            a = np.asarray(rc[i], dtype=np.float64)
            if a.ndim == 2 and li < a.shape[0]:
                vc.append(a[li])
        for i in range(len(rw)):
            a = np.asarray(rw[i], dtype=np.float64)
            if a.ndim == 2 and li < a.shape[0]:
                vw.append(a[li])
        dc, dw = mean_vector(vc), mean_vector(vw)
        if dc is not None and dw is not None and dc.shape == dw.shape:
            out[li] = (dc.astype(np.float32), dw.astype(np.float32))
    return out


@dataclass
class ModeResult:
    stage1_pass: int
    stage2_correct: int
    stage2_wrong: int
    stage2_skipped: int
    total: int
    stage2_runs: int

    def to_dict(self) -> dict[str, Any]:
        denom_s2 = max(1, self.stage2_runs)
        return {
            "total_samples": self.total,
            "stage1_pass": self.stage1_pass,
            "stage1_pass_rate": self.stage1_pass / max(1, self.total),
            "stage2_runs": self.stage2_runs,
            "stage2_correct": self.stage2_correct,
            "stage2_wrong": self.stage2_wrong,
            "stage2_skipped": self.stage2_skipped,
            "stage2_correct_rate_among_stage2_runs": self.stage2_correct / denom_s2,
            "stage2_wrong_rate_among_stage2_runs": self.stage2_wrong / denom_s2,
        }


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


def _manifest_stage2_is_wrong(s: dict[str, Any]) -> bool:
    return str(s.get("stage2_label", "")).strip().lower() == "wrong"


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


def _run_one_mode(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    dataset_name: str,
    projection_dirs: dict[int, np.ndarray],
    projection_alpha: float,
    batch_size: int,
    mode_name: str,
) -> tuple[ModeResult, list[dict[str, Any]]]:
    layers = model.model.layers  # type: ignore[attr-defined]
    hooks = []
    for li, d in projection_dirs.items():
        if 0 <= li < len(layers):
            hooks.append(layers[li].register_forward_hook(_layer_hook_with_vector(d, projection_alpha)))

    per_sample: list[dict[str, Any]] = []
    is_kre = str(dataset_name).strip().lower() == "kre"
    for s in samples:
        per_sample.append(
            {
                "line_idx": int(s.get("line_idx", -1)),
                "manifest_stage2_label": str(s.get("stage2_label", "")),
                "clean_prompt_input": s.get("clean_prompt_input"),
                "conflict_prompt_input": s.get("conflict_prompt_input"),
                "gold_answers": list(s.get("gold_answers", [])),
                "distracted_token": s.get("distracted_token"),
                "stage1_output_raw": None,
                "stage1_output_judged": None,
                "stage1_pass": False,
                "stage2_output_raw": None,
                "stage2_output_judged": None,
                "stage2_label": "not_run",
                "stage2_evaluated": False,
                "stage2_skip_reason": None,
            }
        )

    try:
        s1_flags = [False] * len(samples)
        s1_done = 0
        for st in range(0, len(samples), batch_size):
            chunk = samples[st : st + batch_size]
            outs = generate_batch(model, tokenizer, [str(x["clean_prompt_input"]) for x in chunk])
            for j, row in enumerate(chunk):
                idx = st + j
                raw = outs[j]
                t = _truncate_for_judge(raw)
                gold = [str(v) for v in row.get("gold_answers", [])]
                ok = _answer_matches_letters(t, gold) if is_kre else _answer_matches(t, gold)
                s1_flags[idx] = ok
                per_sample[idx]["stage1_output_raw"] = raw
                per_sample[idx]["stage1_output_judged"] = t
                per_sample[idx]["stage1_pass"] = bool(ok)
            s1_done += len(chunk)
            if s1_done % PROGRESS_EVERY == 0 or s1_done == len(samples):
                print(f"[{mode_name}] stage1 progress: {s1_done}/{len(samples)}", flush=True)

        for i in range(len(samples)):
            if not s1_flags[i]:
                per_sample[i]["stage2_label"] = "not_run"
                per_sample[i]["stage2_skip_reason"] = "stage1_fail"

        s2_correct = s2_wrong = s2_skipped = 0
        pass_idxs = [i for i, ok in enumerate(s1_flags) if ok]
        s2_done = 0
        for st in range(0, len(pass_idxs), batch_size):
            idxs = pass_idxs[st : st + batch_size]
            chunk = [samples[i] for i in idxs]
            outs = generate_batch(model, tokenizer, [str(x["conflict_prompt_input"]) for x in chunk])
            for j, row in enumerate(chunk):
                raw = outs[j]
                t = _truncate_for_judge(raw)
                gold = [str(v) for v in row.get("gold_answers", [])]
                dlist = row.get("distracted_tokens")
                if not isinstance(dlist, list):
                    dlist = [str(row.get("distracted_token", ""))]
                dlist = [str(x) for x in dlist]
                src_idx = idxs[j]
                per_sample[src_idx]["stage2_output_raw"] = raw
                per_sample[src_idx]["stage2_output_judged"] = t
                if is_kre:
                    lab = _stage2_label_from_letters(raw, gold, dlist)
                else:
                    lab = _stage2_label_from_strings(raw, gold, dlist)
                per_sample[src_idx]["stage2_label"] = lab
                per_sample[src_idx]["stage2_evaluated"] = True
                per_sample[src_idx]["stage2_skip_reason"] = None
                if lab == "correct":
                    s2_correct += 1
                elif lab == "wrong":
                    s2_wrong += 1
                else:
                    s2_skipped += 1
            s2_done += len(chunk)
            if s2_done % PROGRESS_EVERY == 0 or s2_done == len(pass_idxs):
                print(f"[{mode_name}] stage2 progress: {s2_done}/{len(pass_idxs)}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    s2_runs = len(pass_idxs)
    return (
        ModeResult(
            sum(1 for x in s1_flags if x),
            s2_correct,
            s2_wrong,
            s2_skipped,
            len(samples),
            s2_runs,
        ),
        per_sample,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Residual steer: alpha grid; each alpha runs full Stage1 then Stage2 on all manifest rows."
    )
    ap.add_argument("--model", default="llama3-8b-it", help="Model key in MODEL_PATHS or HF path.")
    ap.add_argument("--category", type=str, default=DEFAULT_CATEGORY, help="NQ-Swap split/category tag (e.g. where, who, how).")
    ap.add_argument("--npz", type=Path, default=None, help="NPZ with resid_act_clean / resid_act_conflict (directions).")
    ap.add_argument("--test-json", type=Path, default=None, help="internal_aware_data.json (test samples).")
    ap.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES (e.g. 0).")
    ap.add_argument("--layers", type=str, default="10", help="Comma-separated layer indices, e.g. 10 or 27,28,29.")
    ap.add_argument("--alpha-start", type=float, default=0.0, help="First alpha (inclusive).")
    ap.add_argument("--alpha-end", type=float, default=1, help="Last alpha (inclusive when stepping from start).")
    ap.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Step between alphas (ignored if start==end). May be negative if alpha-start >= alpha-end.",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--npz-category-tag",
        type=str,
        default="",
        help="Optional tag appended to output category (e.g. World Capital) to avoid overwrite across NPZ sources.",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    if args.npz is None:
        direction_npz = (
            NQ_OUT_ROOT / "resid_hidden_state" / str(args.model) / str(args.category) / "resid_hidden_state.npz"
        ).resolve()
    else:
        direction_npz = _abspath(args.npz)
    if args.test_json is None:
        test_manifest = (
            NQ_OUT_ROOT / "internal_aware_data" / str(args.model) / str(args.category) / "internal_aware_data.json"
        ).resolve()
    else:
        test_manifest = _abspath(args.test_json)

    if not direction_npz.is_file():
        raise SystemExit(f"direction activations not found: {direction_npz}")
    if not test_manifest.is_file():
        raise SystemExit(f"test manifest not found: {test_manifest}")

    obj = json.loads(test_manifest.read_text(encoding="utf-8"))
    samples_all = list(obj.get("samples", []))
    dataset_name = str(obj.get("dataset") or "")
    if not samples_all:
        raise SystemExit("test manifest has no samples")

    alphas = _alphas_from_cli(args.alpha_start, args.alpha_end, args.alpha_step)

    n_manifest_s2_wrong = sum(1 for s in samples_all if _manifest_stage2_is_wrong(s))

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    dirs = _load_resid_vectors(direction_npz, layers)
    used_layers = [li for li in layers if li in dirs]
    if not used_layers:
        raise SystemExit("no valid layers from direction activations NPZ")

    model, tokenizer = get_model_and_tokenizer(args.model)
    n_total_manifest = len(samples_all)
    proj = {li: dirs[li][0] for li in used_layers}

    out_paths: list[str] = []
    for ai, alpha in enumerate(alphas):
        out_path = _steer_out_path(
            obj,
            str(args.model),
            test_manifest,
            str(args.layers),
            float(alpha),
            str(args.npz_category_tag),
        )
        print(
            f"[alpha {ai + 1}/{len(alphas)}] alpha={alpha:g} -> {out_path.name}",
            flush=True,
        )
        r, ps = _run_one_mode(
            model,
            tokenizer,
            samples_all,
            dataset_name,
            proj,
            float(alpha),
            int(args.batch_size),
            "project_to_correct_mean_direction",
        )
        s1p = r.stage1_pass
        flip_correct = r.stage2_correct
        flip_wrong = r.stage2_wrong
        flip_skip = r.stage2_skipped
        s2_runs = r.stage2_runs

        results: dict[str, Any] = {
            "manifest": {
                "total_samples_in_json": n_total_manifest,
                "num_manifest_stage2_wrong": n_manifest_s2_wrong,
            },
            "stage1_all_samples": {
                "rerun_stage1_pass": s1p,
                "rerun_stage1_pass_rate": s1p / max(1, n_total_manifest),
            },
            "after_steer_stage2_on_stage1_pass": {
                "stage2_runs_with_steer": s2_runs,
                "stage2_correct_after_steer": flip_correct,
                "stage2_wrong_after_steer": flip_wrong,
                "stage2_skipped_after_steer": flip_skip,
                "rate_correct_among_stage2_runs": flip_correct / max(1, s2_runs),
                "details": r.to_dict(),
            },
        }
        per_sample_out: dict[str, Any] = {"project_to_correct_mean_direction": ps}

        out = {
            "direction_npz": str(direction_npz),
            "test_json": str(test_manifest),
            "out_json": str(out_path),
            "gpu": str(args.gpu),
            "batch_size": int(args.batch_size),
            "labeling_note": "For this alpha: full Stage1 on all manifest rows, then Stage2 (steer) on every row "
            "that passes rerun Stage1. Residual steer: h' = h + alpha * v_clean (layer mean from NPZ). "
            "Judgement rules match dump_activations_from_testjsonl.py.",
            "steer_formula": "h + alpha * v_clean_mean",
            "model": args.model,
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
        out_paths.append(str(out_path.relative_to(ROOT)))

    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()
