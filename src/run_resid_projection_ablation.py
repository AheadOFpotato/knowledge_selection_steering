#!/usr/bin/env python3
"""Residual-stream steering (self-contained).

CLI: ``--model``, ``--npz``, ``--test-json``, ``--gpu``, ``--layers``,
``--alpha-start`` / ``--alpha-end`` / ``--alpha-step``, ``--batch-size``.

``--npz``: resid activations (``resid_act_clean`` and ``resid_act_conflict`` / ``resid_act_stage2``).
Per layer, ``v = mean(resid_act_clean) - mean(resid_act_conflict)``; steer with ``h' = h + alpha * v``
(alpha sign selects direction: positive → ``h + |alpha|*v``, negative → ``h - |alpha|*v``).
Output paths include ``__dir_<direction_category>`` / ``__test_<test_category>`` when set (via CLI).

**eval-mode=legacy:** For each alpha, rerun Stage1+Stage2 on ``--test-json`` (same as before).

**eval-mode=train_val_test:** ``--npz`` = train dump (direction only); sweep alphas on ``--val-json``;
pick best alpha by val stage2 correct rate; evaluate once on ``--test-json``.

Labels match ``dump_activations_from_testjsonl.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from project_paths import ROOT, STEER_OUT_ROOT, display_path, resolve_model_path

DEFAULT_STEER_OUT_DIR = STEER_OUT_ROOT

MODEL_PATHS: dict[str, str] = {
    "llama3-8b-it": "Llama3-8b-it",
    "gemma2-9b-it": "Gemma2-9b-it",
    "qwen3-8b": "Qwen3-8B",
    "mistral-7b-v0.1": "Mistral-7B-v0.1",
    "yi-6b": "Yi-6B",
    "yi-6b-chat": "Yi-6B-Chat",
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 8192
PROGRESS_EVERY = 10


def _safe_name(s: str) -> str:
    t = str(s).strip().replace("\\", "/")
    # If user/category carries a path, collapse to stem to avoid escaping out_root.
    if "/" in t:
        t = Path(t).stem
    t = t.replace(" ", "_").replace("/", "_")
    return t


def _output_category_slug(manifest: dict[str, Any], override: str = "") -> str:
    """Folder slug for ``<out-root>/<model>/<dataset>/<category>/`` (not manifest path strings)."""
    o = str(override or "").strip()
    if o:
        if "/" in o or o.endswith((".jsonl", ".json")):
            return _safe_name(Path(o).expanduser().stem)
        return _safe_name(o)
    raw = str(manifest.get("category") or "")
    if "/" in raw or raw.endswith((".jsonl", ".json")):
        return _safe_name(Path(raw).stem)
    return _safe_name(raw)


def _steer_out_path(
    out_root: Path,
    manifest: dict[str, Any],
    args_model: str,
    layers_arg: str,
    alpha: float,
    output_category: str = "",
    direction_category_tag: str = "",
    test_category_tag: str = "",
    legacy_npz_category_tag: str = "",
) -> Path:
    """``.../L<layer>_alpha<numeric>_steer_result.json`` under --out-root."""
    mod = _safe_name(str(manifest.get("model") or args_model or "unknown"))
    ds = _safe_name(str(manifest.get("dataset") or "unknown"))
    cat = _output_category_slug(manifest, output_category)
    dir_tag = str(direction_category_tag or legacy_npz_category_tag or "").strip()
    if dir_tag:
        cat = f"{cat}__dir_{_safe_name(dir_tag)}"
    tst_tag = str(test_category_tag).strip()
    if tst_tag:
        cat = f"{cat}__test_{_safe_name(tst_tag)}"
    ls = str(layers_arg).replace(",", "_").replace(" ", "")
    a_str = f"{float(alpha):g}"
    out_dir = out_root / mod / ds / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"L{ls}_alpha{a_str}_steer_result.json"


def _abspath(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _display_path(p: Path) -> str:
    return display_path(p, ROOT)


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


def _disable_torch_compile_for_hooks() -> None:
    """Forward hooks break under torch.compile / TorchDynamo (common on Gemma-2)."""
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo

        torch._dynamo.config.disable = True
    except Exception:
        pass


def get_model_and_tokenizer(model_name: str):
    _disable_torch_compile_for_hooks()
    model_path = resolve_model_path(model_name)
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
    if "resid_act_conflict" in z.files:
        rw = z["resid_act_conflict"]
    elif "resid_act_stage2" in z.files:
        rw = z["resid_act_stage2"]
    else:
        z.close()
        raise KeyError(
            "Neither 'resid_act_conflict' nor 'resid_act_stage2' found in NPZ. "
            f"Available keys: {z.files}"
        )
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


def _default_stage1_pass_from_manifest(sample: dict[str, Any], group_name: str) -> bool:
    """Gate for rows not rerun in Stage1 (use dump-time cohort / stage1_is_correct)."""
    if "stage1_is_correct" in sample:
        correct = bool(sample["stage1_is_correct"])
        if str(group_name).strip() == "stage1_wrong_use_correct_context":
            return not correct
        return correct
    return True


@dataclass
class ModeResult:
    stage1_pass: int
    stage2_correct: int
    stage2_wrong: int
    stage2_skipped: int
    total: int
    stage2_runs: int
    stage2_early_stopped: bool = False
    stage1_eval_n: int = 0

    def to_dict(self) -> dict[str, Any]:
        denom_s2 = max(1, self.stage2_runs)
        denom_s1 = max(1, self.stage1_eval_n or self.total)
        return {
            "total_samples": self.total,
            "stage1_eval_n": self.stage1_eval_n or self.total,
            "stage1_pass": self.stage1_pass,
            "stage1_pass_rate": self.stage1_pass / denom_s1,
            "stage2_runs": self.stage2_runs,
            "stage2_correct": self.stage2_correct,
            "stage2_wrong": self.stage2_wrong,
            "stage2_skipped": self.stage2_skipped,
            "stage2_early_stopped": self.stage2_early_stopped,
            "stage2_correct_rate_among_stage2_runs": self.stage2_correct / denom_s2,
            "stage2_wrong_rate_among_stage2_runs": self.stage2_wrong / denom_s2,
        }


@torch.inference_mode()
def generate_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length if max_length is not None else TOKENIZER_MAX_LEN,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    w = input_ids.shape[1]
    return [tokenizer.decode(out[i, w:], skip_special_tokens=True).strip() for i in range(input_ids.shape[0])]


def _layer_hook_with_vector(v_mean: np.ndarray, alpha: float):
    """Apply ``h' = h + alpha * v`` on the layer hidden state."""
    v = np.asarray(v_mean, dtype=np.float32).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0 or not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0.0:
        return lambda _m, _inp, out: out

    a = float(alpha)
    _dv_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return out
        if int(v.shape[0]) != int(hs.shape[-1]):
            return out
        key = (hs.device, hs.dtype)
        dv = _dv_cache.get(key)
        if dv is None:
            dv = torch.as_tensor(v, device=hs.device, dtype=hs.dtype)
            _dv_cache[key] = dv
        mixed = hs + (a * dv).view(1, 1, -1)
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


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    samples = list(obj.get("samples", []))
    if not samples:
        raise SystemExit(f"manifest has no samples: {path}")
    return obj, samples


def _prepare_eval_samples(
    samples_all: list[dict[str, Any]],
    *,
    max_samples_head: int,
    max_s1: int,
    max_s2: int,
    sample_seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[int],
    list[dict[str, Any]],
    list[int],
    int,
    int,
    int,
]:
    if int(max_samples_head) > 0:
        samples_all = samples_all[: int(max_samples_head)]
    n_total = len(samples_all)
    _, stage1_indices = _random_subsample(samples_all, max_s1, int(sample_seed))
    if max_s2 > 0:
        samples_stage2, stage2_indices = _random_subsample(
            samples_all, max_s2, int(sample_seed) + 1
        )
    else:
        samples_stage2 = samples_all
        stage2_indices = list(range(n_total))
    return (
        samples_all,
        stage1_indices,
        samples_stage2,
        stage2_indices,
        n_total,
        len(stage1_indices),
        len(stage2_indices),
    )


def _val_alpha_score(r: ModeResult) -> float:
    """Higher is better for alpha selection on validation."""
    if r.stage2_runs <= 0:
        return -1.0
    return float(r.stage2_correct) / float(r.stage2_runs)


def _pick_best_alpha(alpha_scores: list[tuple[float, float]]) -> float:
    """``alpha_scores``: list of (alpha, score); tie-break toward smaller |alpha|."""
    if not alpha_scores:
        raise SystemExit("no alpha scores on validation")
    best_a, best_s = alpha_scores[0]
    for a, s in alpha_scores[1:]:
        if s > best_s + 1e-12:
            best_a, best_s = a, s
        elif abs(s - best_s) <= 1e-12 and abs(a) < abs(best_a):
            best_a, best_s = a, s
    return float(best_a)


def _val_grid_out_path(
    out_root: Path,
    manifest: dict[str, Any],
    args_model: str,
    layers_arg: str,
    output_category: str,
    direction_category_tag: str,
    test_category_tag: str,
    legacy_npz_category_tag: str,
) -> Path:
    mod = _safe_name(str(manifest.get("model") or args_model or "unknown"))
    ds = _safe_name(str(manifest.get("dataset") or "unknown"))
    cat = _output_category_slug(manifest, output_category)
    dir_tag = str(direction_category_tag or legacy_npz_category_tag or "").strip()
    if dir_tag:
        cat = f"{cat}__dir_{_safe_name(dir_tag)}"
    tst_tag = str(test_category_tag).strip()
    if tst_tag:
        cat = f"{cat}__test_{_safe_name(tst_tag)}"
    ls = str(layers_arg).replace(",", "_").replace(" ", "")
    out_dir = out_root / mod / ds / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"L{ls}_val_alpha_grid.json"


def _random_subsample(
    samples: list[dict[str, Any]], max_samples: int, seed: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Without-replacement random subset; returned indices are sorted ascending."""
    n = len(samples)
    if max_samples <= 0 or n <= max_samples:
        return samples, list(range(n))
    rng = random.Random(int(seed))
    idxs = sorted(rng.sample(range(n), int(max_samples)))
    return [samples[i] for i in idxs], idxs


def _run_one_mode(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    dataset_name: str,
    group_name: str,
    projection_dirs: dict[int, np.ndarray],
    projection_alpha: float,
    batch_size: int,
    mode_name: str,
    no_skip_in_stage2: bool = False,
    stage1_min_pass_rate: float | None = None,
    stage1_indices: list[int] | None = None,
) -> tuple[ModeResult, list[dict[str, Any]]]:
    layers = model.model.layers  # type: ignore[attr-defined]
    hooks = []
    for li, d in projection_dirs.items():
        if 0 <= li < len(layers):
            hooks.append(
                layers[li].register_forward_hook(
                    _layer_hook_with_vector(d, projection_alpha)
                )
            )

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
                "clean_prompt_input": s.get("clean_prompt_input"),
                "conflict_prompt_input": stage2_prompt,
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

    if stage1_indices is None:
        s1_eval_idxs = list(range(len(samples)))
    else:
        s1_eval_idxs = list(stage1_indices)

    try:
        s1_flags = [
            _default_stage1_pass_from_manifest(samples[i], group_name) for i in range(len(samples))
        ]
        s1_done = 0
        for st in range(0, len(s1_eval_idxs), batch_size):
            idxs = s1_eval_idxs[st : st + batch_size]
            chunk = [samples[i] for i in idxs]
            outs = generate_batch(model, tokenizer, [str(x["clean_prompt_input"]) for x in chunk])
            for j, row in enumerate(chunk):
                idx = idxs[j]
                raw = outs[j]
                t = _truncate_for_judge(raw)
                gold = [str(v) for v in row.get("gold_answers", [])]
                dlist = row.get("distracted_tokens")
                if not isinstance(dlist, list):
                    dlist = [str(row.get("distracted_token", ""))]
                dlist = [str(x) for x in dlist]

                if is_kre:
                    hit_gold = _answer_matches_letters(t, gold)
                else:
                    hit_gold = _answer_matches(t, gold)

                if str(group_name).strip() == "stage1_wrong_use_correct_context":
                    ok = not hit_gold
                else:
                    ok = hit_gold
                s1_flags[idx] = ok
                per_sample[idx]["stage1_output_raw"] = raw
                per_sample[idx]["stage1_output_judged"] = t
                per_sample[idx]["stage1_pass"] = bool(ok)
            s1_done += len(idxs)
            if s1_done % PROGRESS_EVERY == 0 or s1_done == len(s1_eval_idxs):
                print(
                    f"[{mode_name}] stage1 progress: {s1_done}/{len(s1_eval_idxs)} "
                    f"(manifest {len(samples)}; stage2 on pass gate over full manifest)",
                    flush=True,
                )

        s1_pass_eval = sum(1 for i in s1_eval_idxs if s1_flags[i])
        s1_rate = s1_pass_eval / max(1, len(s1_eval_idxs))
        if (
            stage1_min_pass_rate is not None
            and stage1_min_pass_rate >= 0.0
            and s1_rate < stage1_min_pass_rate
        ):
            print(
                f"[{mode_name}] early stop: stage1 pass rate {s1_rate:.3f} "
                f"< {stage1_min_pass_rate:g}, skip stage2 for all samples at this alpha",
                flush=True,
            )
            for i in range(len(samples)):
                per_sample[i]["stage2_label"] = "not_run"
                per_sample[i]["stage2_skip_reason"] = "early_stop_low_stage1_pass_rate"
            return (
                ModeResult(
                    s1_pass_eval,
                    0,
                    0,
                    0,
                    len(samples),
                    0,
                    stage2_early_stopped=True,
                    stage1_eval_n=len(s1_eval_idxs),
                ),
                per_sample,
            )

        for i in range(len(samples)):
            if not s1_flags[i]:
                per_sample[i]["stage2_label"] = "not_run"
                if str(group_name).strip() == "stage1_wrong_use_correct_context":
                    per_sample[i]["stage2_skip_reason"] = "stage1_correct_skip"
                else:
                    per_sample[i]["stage2_skip_reason"] = "stage1_fail"

        s2_correct = s2_wrong = s2_skipped = 0
        pass_idxs = [i for i, ok in enumerate(s1_flags) if ok]
        s2_done = 0
        for st in range(0, len(pass_idxs), batch_size):
            idxs = pass_idxs[st : st + batch_size]
            chunk = [samples[i] for i in idxs]
            outs = generate_batch(
                model,
                tokenizer,
                [
                    str(
                        x.get("stage2_prompt_input")
                        if x.get("stage2_prompt_input") is not None
                        else x.get("conflict_prompt_input", "")
                    )
                    for x in chunk
                ],
            )
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
                    if no_skip_in_stage2:
                        lab = "correct" if _answer_matches_letters(t, gold) else "wrong"
                    else:
                        nonempty_d = [str(d).strip() for d in dlist if str(d).strip()]
                        if nonempty_d:
                            lab = _stage2_label_from_letters(t, gold, dlist)
                        else:
                            lab = "correct" if _answer_matches_letters(t, gold) else "wrong"
                else:
                    if no_skip_in_stage2:
                        lab = "correct" if _answer_matches(t, gold) else "wrong"
                    else:
                        nonempty_d = [str(d).strip() for d in dlist if str(d).strip()]
                        if nonempty_d:
                            lab = _stage2_label_from_strings(t, gold, dlist)
                        else:
                            lab = "correct" if _answer_matches(t, gold) else "wrong"
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
                print(
                    f"[{mode_name}] stage2 progress: {s2_done}/{len(pass_idxs)} "
                    f"(manifest {len(samples)})",
                    flush=True,
                )
    finally:
        for h in hooks:
            h.remove()

    s2_runs = len(pass_idxs)
    return (
        ModeResult(
            s1_pass_eval,
            s2_correct,
            s2_wrong,
            s2_skipped,
            len(samples),
            s2_runs,
            stage1_eval_n=len(s1_eval_idxs),
        ),
        per_sample,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Residual steer: alpha grid; each alpha runs full Stage1 then Stage2 on all manifest rows."
    )
    ap.add_argument("--model", required=True, help="Model key in MODEL_PATHS or HF path.")
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset name (e.g. NQ-Swap). Used to auto-resolve --npz/--test-json when those are omitted.",
    )
    ap.add_argument(
        "--category",
        type=str,
        default="",
        help="Optional category tag (NQ-Swap always uses dev).",
    )
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="NPZ with resid_act_clean / resid_act_conflict (directions). If omitted, auto-resolved from --model/--dataset/--category.",
    )
    ap.add_argument(
        "--test-json",
        type=Path,
        default=None,
        help="internal_aware_data.json (test samples). If omitted, auto-resolved from --model/--dataset/--category.",
    )
    ap.add_argument("--gpu", type=str, required=True, help="CUDA_VISIBLE_DEVICES (e.g. 0).")
    ap.add_argument("--layers", type=str, required=True, help="Comma-separated layer indices, e.g. 10 or 27,28,29.")
    ap.add_argument("--alpha-start", type=float, default=0.0, help="First alpha (inclusive).")
    ap.add_argument("--alpha-end", type=float, default=1.5, help="Last alpha (inclusive when stepping from start).")
    ap.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Step between alphas (ignored if start==end). May be negative if alpha-start >= alpha-end.",
    )
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Alias for --max-samples-stage1: random Stage1 eval subset size (0 = all).",
    )
    ap.add_argument(
        "--max-samples-stage1",
        type=int,
        default=None,
        help="Random Stage1 rerun subset for early-stop metric (0 = all manifest rows).",
    )
    ap.add_argument(
        "--max-samples-stage2",
        type=int,
        default=0,
        help="Stage2 eval subset (0 = full manifest; >0 random subsample of manifest).",
    )
    ap.add_argument(
        "--max-samples-head",
        type=int,
        default=0,
        help="Use only the first N manifest rows (before random subsample). 0 = all.",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for Stage1/Stage2 subsampling (same draw per job).",
    )
    ap.add_argument(
        "--stage1-min-pass-rate",
        type=float,
        default=0.6,
        help="If >=0 and Stage1 subset pass rate is below this, skip Stage2 and stop alpha grid (when alpha>0).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_STEER_OUT_DIR,
        help="Output root directory for steer result json files.",
    )
    ap.add_argument(
        "--direction-category-tag",
        type=str,
        default="",
        help="Category label for NPZ / steering direction source; output gets __dir_<tag>.",
    )
    ap.add_argument(
        "--test-category-tag",
        type=str,
        default="",
        help="Category label for --test-json manifest; output gets __test_<tag> (use when NPZ and test differ).",
    )
    ap.add_argument(
        "--npz-category-tag",
        type=str,
        default="",
        help="Deprecated alias for --direction-category-tag if the latter is empty.",
    )
    ap.add_argument(
        "--output-category",
        type=str,
        default="",
        help="Output path category folder (default: parent dir of --test-json, else manifest category stem).",
    )
    ap.add_argument(
        "--eval-mode",
        choices=("legacy", "train_val_test"),
        default="legacy",
        help="legacy: sweep alphas on --test-json. train_val_test: direction from train NPZ, "
        "alpha grid on --val-json, final eval on --test-json.",
    )
    ap.add_argument(
        "--val-json",
        type=Path,
        default=None,
        help="Validation manifest (train_val_test mode). Required when --eval-mode=train_val_test.",
    )
    args = ap.parse_args()
    out_root = _abspath(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    def _tag_from_category_arg(dataset: str, category: str) -> str:
        cat = (category or "").strip()
        if not cat:
            return ""
        # If it's a path (absolute or relative), use stem.
        try:
            p = Path(cat).expanduser()
            if p.is_file():
                return p.stem
        except Exception:
            pass
        # NQ-Swap: fixed category dev
        if dataset == "NQ-Swap":
            return "dev"
        return cat

    ds = str(args.dataset or "").strip()
    tag = _tag_from_category_arg(ds, str(args.category or "").strip())

    if args.npz is None or args.test_json is None:
        if not ds or not tag:
            raise SystemExit("either provide --npz and --test-json, or provide --dataset and --category for auto-resolve")
        direction_npz = ROOT / "output" / "resid_hidden_state" / str(args.model) / tag / "resid_hidden_state.npz"
        test_manifest = ROOT / "output" / "internal_aware_data" / str(args.model) / tag / "internal_aware_data.json"
    else:
        direction_npz = _abspath(args.npz)
        test_manifest = _abspath(args.test_json)

    if not direction_npz.is_file():
        raise SystemExit(f"direction activations not found: {direction_npz}")

    eval_mode = str(args.eval_mode).strip()
    val_manifest: Path | None = _abspath(args.val_json) if args.val_json is not None else None
    if eval_mode == "train_val_test":
        if val_manifest is None or not val_manifest.is_file():
            raise SystemExit("train_val_test mode requires --val-json (validation manifest)")
        if not test_manifest.is_file():
            raise SystemExit("train_val_test mode requires --test-json (test manifest)")
    elif not test_manifest.is_file():
        raise SystemExit(f"test manifest not found: {test_manifest}")

    max_s1 = int(args.max_samples_stage1) if args.max_samples_stage1 is not None else int(args.max_samples)
    max_s2 = int(args.max_samples_stage2)
    alphas = _alphas_from_cli(args.alpha_start, args.alpha_end, args.alpha_step)

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    dirs = _load_resid_vectors(direction_npz, layers)
    used_layers = [li for li in layers if li in dirs]
    if not used_layers:
        raise SystemExit("no valid layers from direction activations NPZ")

    model, tokenizer = get_model_and_tokenizer(args.model)

    proj = {li: (dirs[li][0] - dirs[li][1]).astype(np.float32) for li in used_layers}
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is not None:
        for li, vec in proj.items():
            if int(vec.shape[0]) != int(hidden_size):
                raise SystemExit(
                    f"projection vector at layer {li} has dim {vec.shape[0]}, "
                    f"but model hidden_size={hidden_size} — NPZ/model mismatch"
                )
    steer_mode = "steer_clean_minus_conflict"
    steer_formula = (
        "v = mean(resid_act_clean) - mean(resid_act_conflict); h' = h + alpha * v "
        "(alpha>0 adds v; alpha<0 subtracts |alpha|*v)"
    )
    labeling_steer = (
        "Residual steer: v = mean(clean) - mean(conflict); h' = h + alpha * v "
        f"({direction_npz.name})."
    )

    dir_out = str(args.direction_category_tag or "").strip()
    legacy_npz = str(args.npz_category_tag or "").strip()
    tst_out = str(args.test_category_tag or "").strip()
    out_cat = str(args.output_category or "").strip()
    if not out_cat:
        out_cat = test_manifest.parent.name

    stage1_min_pass_rate: float | None = None
    if float(args.stage1_min_pass_rate) >= 0.0:
        stage1_min_pass_rate = float(args.stage1_min_pass_rate)

    def _stage2_layout(
        samples_all: list[dict[str, Any]],
        stage1_indices: list[int],
        stage2_indices: list[int],
    ) -> tuple[list[dict[str, Any]], list[int]]:
        if max_s2 > 0:
            stage2_samples = [samples_all[i] for i in stage2_indices]
            s2_index_set = set(stage2_indices)
            stage1_rel = [
                stage2_indices.index(i) for i in stage1_indices if i in s2_index_set
            ]
        else:
            stage2_samples = samples_all
            stage1_rel = stage1_indices
        return stage2_samples, stage1_rel

    def _build_result_payload(
        *,
        obj: dict[str, Any],
        manifest_path: Path,
        eval_split: str,
        manifest_group: str,
        samples_all: list[dict[str, Any]],
        stage1_indices: list[int],
        stage2_indices: list[int],
        n_total_manifest: int,
        n_stage1_eval: int,
        n_stage2_eval: int,
        r: ModeResult,
        ps: list[dict[str, Any]],
        alpha: float,
        ai: int,
    ) -> dict[str, Any]:
        s1p = r.stage1_pass
        flip_correct = r.stage2_correct
        flip_wrong = r.stage2_wrong
        flip_skip = r.stage2_skipped
        s2_runs = r.stage2_runs
        n_manifest_s2_wrong = sum(1 for s in samples_all if _manifest_stage2_is_wrong(s))
        results: dict[str, Any] = {
            "eval_split": eval_split,
            "manifest": {
                "total_samples_in_json": n_total_manifest,
                "num_stage1_eval_samples": n_stage1_eval,
                "num_stage2_eval_samples": n_stage2_eval,
                "max_samples_stage1": max_s1,
                "max_samples_stage2": max_s2,
                "sample_seed": int(args.sample_seed),
                "stage1_eval_manifest_indices": stage1_indices,
                "stage2_eval_manifest_indices": stage2_indices,
                "num_manifest_stage2_wrong": n_manifest_s2_wrong,
            },
            "stage1_all_samples": {
                "rerun_stage1_pass": s1p,
                "rerun_stage1_pass_rate": s1p / max(1, r.stage1_eval_n or n_stage1_eval),
            },
            "after_steer_stage2_on_stage1_pass": {
                "stage2_runs_with_steer": s2_runs,
                "stage2_correct_after_steer": flip_correct,
                "stage2_wrong_after_steer": flip_wrong,
                "stage2_skipped_after_steer": flip_skip,
                "stage2_early_stopped": r.stage2_early_stopped,
                "rate_correct_among_stage2_runs": flip_correct / max(1, s2_runs),
                "val_alpha_score": _val_alpha_score(r),
                "details": r.to_dict(),
            },
            "stage1_min_pass_rate_threshold": stage1_min_pass_rate,
        }
        return {
            "eval_mode": eval_mode,
            "eval_split": eval_split,
            "direction_npz": _display_path(direction_npz),
            "direction_vectors_npz": _display_path(direction_npz),
            "manifest_group": manifest_group,
            "direction_category_tag": (dir_out or legacy_npz or None),
            "test_category_tag": tst_out or None,
            "manifest_json": _display_path(manifest_path),
            "test_json": _display_path(test_manifest),
            "val_json": _display_path(val_manifest) if val_manifest else None,
            "gpu": str(args.gpu),
            "batch_size": int(args.batch_size),
            "max_samples_stage1": max_s1,
            "max_samples_stage2": max_s2,
            "sample_seed": int(args.sample_seed),
            "stage1_min_pass_rate": stage1_min_pass_rate,
            "labeling_note": (
                "Stage1 gate then Stage2 steer. "
                + labeling_steer
                + " Judgement rules match dump_activations_from_testjsonl.py."
            ),
            "steer_formula": steer_formula,
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
            "per_sample_outputs_and_judgements": {steer_mode: ps},
        }

    def _run_on_manifest(
        manifest_path: Path,
        alpha: float,
        *,
        eval_split: str,
        allow_early_stop: bool,
    ) -> tuple[ModeResult, list[dict[str, Any]], dict[str, Any]]:
        mobj, samples_all = _load_manifest(manifest_path)
        ds_name = str(mobj.get("dataset") or obj.get("dataset") or "")
        grp = str(mobj.get("group") or group_name or "")
        no_skip = grp == "stage1_wrong_use_correct_context"
        (
            samples_all,
            s1_idx,
            _s2_samples_unused,
            s2_idx,
            n_tot,
            n_s1,
            n_s2,
        ) = _prepare_eval_samples(
            samples_all,
            max_samples_head=int(args.max_samples_head),
            max_s1=max_s1,
            max_s2=max_s2,
            sample_seed=int(args.sample_seed),
        )
        s2_samples, s1_rel = _stage2_layout(samples_all, s1_idx, s2_idx)
        r, ps = _run_one_mode(
            model,
            tokenizer,
            s2_samples,
            ds_name,
            grp,
            proj,
            float(alpha),
            int(args.batch_size),
            steer_mode,
            no_skip_in_stage2=no_skip,
            stage1_min_pass_rate=stage1_min_pass_rate if allow_early_stop else None,
            stage1_indices=s1_rel,
        )
        payload = _build_result_payload(
            obj=mobj,
            manifest_path=manifest_path,
            eval_split=eval_split,
            manifest_group=grp,
            samples_all=samples_all,
            stage1_indices=s1_idx,
            stage2_indices=s2_idx,
            n_total_manifest=n_tot,
            n_stage1_eval=n_s1,
            n_stage2_eval=n_s2,
            r=r,
            ps=ps,
            alpha=float(alpha),
            ai=0,
        )
        return r, ps, payload

    obj, _ = _load_manifest(test_manifest)
    group_name = str(obj.get("group") or "")

    out_paths: list[str] = []

    if eval_mode == "train_val_test":
        assert val_manifest is not None
        print(
            f"[train_val_test] direction NPZ (train): {_display_path(direction_npz)}",
            flush=True,
        )
        print(f"[train_val_test] alpha grid on val: {_display_path(val_manifest)}", flush=True)
        print(f"[train_val_test] final test: {_display_path(test_manifest)}", flush=True)

        val_grid_path = _val_grid_out_path(
            out_root,
            obj,
            str(args.model),
            str(args.layers),
            out_cat,
            dir_out,
            tst_out,
            legacy_npz,
        )
        val_alpha_rows: list[dict[str, Any]] = []
        alpha_scores: list[tuple[float, float]] = []

        for ai, alpha in enumerate(alphas):
            print(f"[val alpha {ai + 1}/{len(alphas)}] alpha={alpha:g}", flush=True)
            r, _ps, payload = _run_on_manifest(
                val_manifest, float(alpha), eval_split="val", allow_early_stop=False
            )
            payload["alpha_grid"]["index_in_grid"] = ai
            val_alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "val_alpha_score": _val_alpha_score(r),
                    "summary": payload["results"]["after_steer_stage2_on_stage1_pass"],
                }
            )
            alpha_scores.append((float(alpha), _val_alpha_score(r)))
            if r.stage2_early_stopped and float(alpha) > 0:
                print("[val] early stop on stage1 pass rate; stopping alpha grid", flush=True)
                break

        best_alpha = _pick_best_alpha(alpha_scores)
        val_grid_doc = {
            "eval_mode": "train_val_test",
            "direction_npz": _display_path(direction_npz),
            "val_json": _display_path(val_manifest),
            "test_json": _display_path(test_manifest),
            "best_alpha": best_alpha,
            "alpha_selection_metric": "stage2_correct_rate_among_stage2_runs",
            "alphas_evaluated": val_alpha_rows,
            "model": args.model,
            "layers": str(args.layers),
        }
        val_grid_path.write_text(
            json.dumps(val_grid_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        out_paths.append(_display_path(val_grid_path))
        print(f"[train_val_test] best alpha on val: {best_alpha:g}", flush=True)

        test_out_path = _steer_out_path(
            out_root,
            obj,
            str(args.model),
            str(args.layers),
            float(best_alpha),
            output_category=out_cat,
            direction_category_tag=dir_out,
            test_category_tag=tst_out,
            legacy_npz_category_tag=legacy_npz,
        )
        if test_out_path.is_file():
            print(f"[test] skip existing -> {test_out_path.name}", flush=True)
        else:
            _r, _ps, test_payload = _run_on_manifest(
                test_manifest, float(best_alpha), eval_split="test", allow_early_stop=False
            )
            test_payload["best_alpha_from_val"] = float(best_alpha)
            test_payload["val_alpha_grid_json"] = _display_path(val_grid_path)
            test_payload["out_json"] = _display_path(test_out_path)
            test_out_path.write_text(
                json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            out_paths.append(_display_path(test_out_path))
    else:
        samples_all, stage1_indices, samples_stage2, stage2_indices, n_total_manifest, n_stage1_eval, n_stage2_eval = _prepare_eval_samples(
            list(obj.get("samples", [])),
            max_samples_head=int(args.max_samples_head),
            max_s1=max_s1,
            max_s2=max_s2,
            sample_seed=int(args.sample_seed),
        )
        dataset_name = str(obj.get("dataset") or "")
        group_name = str(obj.get("group") or "")
        no_skip_in_stage2 = group_name == "stage1_wrong_use_correct_context"
        n_manifest_s2_wrong = sum(1 for s in samples_all if _manifest_stage2_is_wrong(s))

        for ai, alpha in enumerate(alphas):
            out_path = _steer_out_path(
                out_root,
                obj,
                str(args.model),
                str(args.layers),
                float(alpha),
                output_category=out_cat,
                direction_category_tag=dir_out,
                test_category_tag=tst_out,
                legacy_npz_category_tag=legacy_npz,
            )
            if out_path.is_file():
                print(
                    f"[alpha {ai + 1}/{len(alphas)}] skip existing alpha={alpha:g} -> {out_path.name}",
                    flush=True,
                )
                continue
            print(
                f"[alpha {ai + 1}/{len(alphas)}] alpha={alpha:g} -> {out_path.name}",
                flush=True,
            )
            stage2_samples, stage1_rel = _stage2_layout(samples_all, stage1_indices, stage2_indices)
            r, ps = _run_one_mode(
                model,
                tokenizer,
                stage2_samples,
                dataset_name,
                group_name,
                proj,
                float(alpha),
                int(args.batch_size),
                steer_mode,
                no_skip_in_stage2=no_skip_in_stage2,
                stage1_min_pass_rate=stage1_min_pass_rate,
                stage1_indices=stage1_rel,
            )
            out = _build_result_payload(
                obj=obj,
                manifest_path=test_manifest,
                eval_split="test",
                manifest_group=group_name,
                samples_all=samples_all,
                stage1_indices=stage1_indices,
                stage2_indices=stage2_indices,
                n_total_manifest=n_total_manifest,
                n_stage1_eval=n_stage1_eval,
                n_stage2_eval=n_stage2_eval,
                r=r,
                ps=ps,
                alpha=float(alpha),
                ai=ai,
            )
            out["out_json"] = _display_path(out_path)
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            out_paths.append(_display_path(out_path))

            if r.stage2_early_stopped and float(alpha) > 0:
                print(
                    f"[alpha grid] stage1 pass rate < {stage1_min_pass_rate:g} at alpha={alpha:g} (>0); "
                    f"stop remaining alphas ({ai + 1}/{len(alphas)} completed in this job)",
                    flush=True,
                )
                break

    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()
