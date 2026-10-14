#!/usr/bin/env python3
"""Residual-stream steering (self-contained).

CLI: ``--model``, ``--npz``, ``--test-json``, ``--gpu``, ``--layers``,
``--alpha-start`` / ``--alpha-end`` / ``--alpha-step``, ``--batch-size``.

``--npz``: resid activations (``resid_act_clean`` and ``resid_act_conflict`` / ``resid_act_stage2``).
Per layer, default ``v = mean(resid_act_clean) - mean(resid_act_conflict)``; steer with ``h' = h + alpha * v``
(alpha sign selects direction: positive → toward clean, negative → toward conflict/context).
``--steer-toward context`` flips v so positive alpha moves toward the context (conflict) mean.
Output paths include ``__dir_<direction_category>`` / ``__test_<test_category>`` when set (via CLI).

**eval-mode=legacy:** For each alpha, rerun Stage1+Stage2 on ``--test-json``.

**eval-mode=train_val_test:** Single full dump (``--npz`` + ``--test-json``). At eval time split
manifest indices 70/15/15: train rows → direction ``v`` from NPZ; val → alpha grid; test → final metrics.

For ``stage1_wrong_use_correct_context``, Stage2 runs only if Stage1 output **does not** match gold.
For ``stage1_correct_use_wrong_context``, Stage2 runs only if Stage1 matches gold.
Labels match ``dump_activations_from_testjsonl.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from chat_prompt import format_chat_messages_for_inference, format_prompts_for_inference
from data_split import split_row_indices

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", str(ROOT / "models"))).expanduser()
DEFAULT_STEER_OUT_DIR = ROOT / "output" / "new_steer"

MODEL_PATHS: dict[str, str] = {
    # Local weights live under <MODEL_ROOT> (legacy <DATA_DIR>/model/* removed).
    "llama3-8b-it": str(MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct"),
    "llama3.1-8b-it": str(MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct"),
    "gemma2-9b-it": str(MODEL_ROOT / "Gemma2-9b-it"),
    "qwen3-8b": str(MODEL_ROOT / "Qwen3-8B"),
    "mistral-7b-v0.1": str(MODEL_ROOT / "Mistral-7B-v0.1"),
    "mistral-7b-instruct-v0.3": str(MODEL_ROOT / "Mistral-7B-Instruct-v0.3"),
    "yi-6b-chat": str(MODEL_ROOT / "Yi-6B-Chat"),
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 4096
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
    coeff: float,
    output_category: str = "",
    direction_category_tag: str = "",
    test_category_tag: str = "",
    legacy_npz_category_tag: str = "",
    coeff_name: str = "alpha",
) -> Path:
    """``.../L<layer>_<coeff_name><numeric>_steer_result.json`` under --out-root."""
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
    c_str = f"{float(coeff):g}"
    c_label = str(coeff_name or "alpha").strip() or "alpha"
    out_dir = out_root / mod / ds / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"L{ls}_{c_label}{c_str}_steer_result.json"


def _abspath(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _display_path(p: Path) -> str:
    """Path for logging; relative to repo ROOT when possible."""
    rp = p.resolve()
    try:
        return str(rp.relative_to(ROOT.resolve()))
    except ValueError:
        return str(rp)


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


def _official_judge_sample(
    *,
    dataset: str,
    category: str,
    text: str,
    gold_answers: list[str],
    distracted_tokens: list[str],
    sample: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return single-target correctness via ``official_judges``."""
    try:
        from official_judges import match_gold
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"--judge official requires src/official_judges on PYTHONPATH ({e})"
        ) from e

    sample = sample or {}
    meta = {
        "category": category or str(sample.get("category") or ""),
        "query": sample.get("question") or sample.get("query") or "",
        "question": sample.get("question") or sample.get("query") or "",
        "question_type": sample.get("question_type") or "multi-session",
        "eval_function": sample.get("eval_function") or "",
        "tool_call": sample.get("tool_call"),
    }
    target: object = sample.get("tool_call") or gold_answers
    hit = match_gold(dataset, text, target, meta=meta)
    return hit, "correct" if hit else "wrong"


def _legacy_stage1_hit(
    text: str, gold: list[str], *, is_kre: bool
) -> bool:
    if is_kre:
        return _answer_matches_letters(text, gold)
    return _answer_matches(text, gold)


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
    model_path = MODEL_PATHS.get(model_name, model_name)
    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "left"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    has_cuda = bool(torch.cuda.is_available())
    if has_cuda:
        device_map = "cuda" if (visible and "," not in visible) else "auto"
        dtype = torch.bfloat16
    else:
        # CPU fallback is useful on login/CI nodes whose installed torch sees
        # no compatible driver.  It keeps the exact same model and hooks.
        device_map = "cpu"
        dtype = torch.float32
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "seq_length") and not hasattr(config, "max_length"):
        config.max_length = int(getattr(config, "seq_length"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "eager"),
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


def _load_resid_vectors(
    activations_npz: Path,
    layers: list[int],
    *,
    row_indices: list[int] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean of last-token resid for clean vs conflict (optional row subset)."""
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
    idxs = list(row_indices) if row_indices is not None else list(range(len(rc)))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for li in layers:
        vc, vw = [], []
        for i in idxs:
            if i < 0 or i >= len(rc):
                continue
            a = np.asarray(rc[i], dtype=np.float64)
            if a.ndim == 2 and li < a.shape[0]:
                vc.append(a[li])
        for i in idxs:
            if i < 0 or i >= len(rw):
                continue
            a = np.asarray(rw[i], dtype=np.float64)
            if a.ndim == 2 and li < a.shape[0]:
                vw.append(a[li])
        dc, dw = mean_vector(vc), mean_vector(vw)
        if dc is not None and dw is not None and dc.shape == dw.shape:
            out[li] = (dc.astype(np.float32), dw.astype(np.float32))
    return out


def _layer_mean_residual_norm(
    activations_npz: Path,
    layers: list[int],
    row_indices: list[int],
) -> dict[int, float]:
    """r_l = E[||h_l(x)||_2] over clean activations at selected rows."""
    z = np.load(activations_npz, allow_pickle=True)
    rc = z["resid_act_clean"]
    z.close()
    out: dict[int, float] = {}
    for li in layers:
        norms: list[float] = []
        for i in row_indices:
            if i < 0 or i >= len(rc):
                continue
            a = np.asarray(rc[i], dtype=np.float64)
            if a.ndim == 2 and li < a.shape[0]:
                norms.append(float(np.linalg.norm(a[li])))
        if norms:
            out[li] = float(np.mean(norms))
    return out


def _resolve_direction_row_indices(
    n_npz: int,
    train_indices: list[int],
    eval_mode: str,
    subset_size: int,
    subset_seed: int,
) -> list[int]:
    """Row indices for direction v; optional random subsample from the pool."""
    pool = list(train_indices) if eval_mode == "train_val_test" else list(range(n_npz))
    if subset_size <= 0 or len(pool) <= subset_size:
        return pool
    rng = random.Random(int(subset_seed))
    return sorted(rng.sample(pool, int(subset_size)))


def _norm_match_vector(v: np.ndarray, target_norm: float) -> np.ndarray:
    """Scale v so ||v||_2 = target_norm (zero vector if target_norm <= 0)."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    tn = float(target_norm)
    if tn <= 1e-12:
        return v.astype(np.float32)
    vn = float(np.linalg.norm(v))
    if vn <= 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (v * (tn / vn)).astype(np.float32)


def _load_layer_activation_pool(
    activations_npz: Path,
    layers: list[int],
    row_indices: list[int],
) -> dict[int, list[np.ndarray]]:
    """Pool last-token residual vectors from clean and conflict rows (ignore labels)."""
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
    out: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    for i in row_indices:
        if i < 0 or i >= len(rc):
            continue
        a = np.asarray(rc[i], dtype=np.float64)
        if a.ndim == 2:
            for li in layers:
                if li < a.shape[0]:
                    out[li].append(a[li])
    for i in row_indices:
        if i < 0 or i >= len(rw):
            continue
        a = np.asarray(rw[i], dtype=np.float64)
        if a.ndim == 2:
            for li in layers:
                if li < a.shape[0]:
                    out[li].append(a[li])
    return out


def _shuffled_label_direction(
    pool: list[np.ndarray],
    true_norm: float,
    seed: int,
) -> np.ndarray | None:
    """Shuffle pooled activations, split in half, mean difference, norm-matched to true_norm."""
    if len(pool) < 2:
        return None
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(pool))
    rng.shuffle(order)
    shuffled = [pool[int(i)] for i in order]
    half = len(shuffled) // 2
    if half < 1:
        return None
    m1 = mean_vector(shuffled[:half])
    m2 = mean_vector(shuffled[half : 2 * half])
    if m1 is None or m2 is None:
        return None
    return _norm_match_vector((m1 - m2).astype(np.float64), true_norm)


def _gaussian_direction(dim: int, true_norm: float, seed: int) -> np.ndarray:
    """Sample N(0,I), norm-matched to true_norm."""
    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal(int(dim)).astype(np.float64)
    return _norm_match_vector(v, true_norm)


def _apply_direction_control(
    raw_v: dict[int, np.ndarray],
    activations_npz: Path,
    row_indices: list[int],
    *,
    direction_mode: str,
    direction_control_seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Replace true directions with shuffled-label or Gaussian controls (norm-matched)."""
    mode = str(direction_mode).strip().lower()
    if mode == "true":
        return raw_v, {"direction_mode": "true"}

    used_layers = sorted(raw_v.keys())
    pool_by_layer = _load_layer_activation_pool(activations_npz, used_layers, row_indices)
    out: dict[int, np.ndarray] = {}
    meta: dict[str, Any] = {
        "direction_mode": mode,
        "direction_control_seed": int(direction_control_seed),
        "true_v_norm_per_layer": {},
        "control_v_norm_per_layer": {},
        "cos_true_control_per_layer": {},
    }
    for li in used_layers:
        true_v = raw_v[li].astype(np.float64)
        true_norm = float(np.linalg.norm(true_v))
        meta["true_v_norm_per_layer"][str(li)] = true_norm
        if mode in {"shuffled", "random_split"}:
            ctrl = _shuffled_label_direction(
                pool_by_layer.get(li, []),
                true_norm,
                direction_control_seed,
            )
            if ctrl is None:
                ctrl = np.zeros_like(true_v, dtype=np.float32)
        elif mode == "gaussian":
            ctrl = _gaussian_direction(int(true_v.shape[0]), true_norm, direction_control_seed)
        else:
            raise ValueError(f"unknown direction_mode={direction_mode!r}; expected true|shuffled|random_split|gaussian")
        out[li] = ctrl.astype(np.float32)
        meta["control_v_norm_per_layer"][str(li)] = float(np.linalg.norm(ctrl))
        tn = true_norm
        cn = float(np.linalg.norm(ctrl))
        if tn > 1e-12 and cn > 1e-12:
            meta["cos_true_control_per_layer"][str(li)] = float(
                np.dot(true_v, np.asarray(ctrl, dtype=np.float64)) / (tn * cn)
            )
        else:
            meta["cos_true_control_per_layer"][str(li)] = None
    return out, meta


def _build_projection_dirs(
    activations_npz: Path,
    layers: list[int],
    row_indices: list[int],
    *,
    steer_coeff_mode: str,
    direction_mode: str = "true",
    direction_control_seed: int = 0,
    steer_toward: str = "clean",
) -> tuple[dict[int, np.ndarray], dict[int, float] | None, dict[str, Any]]:
    """Build per-layer hook vectors and optional residual-norm scales.

    ``steer_toward=clean``: ``v = mean(clean) - mean(conflict)`` (positive alpha → parametric).
    ``steer_toward=context``: ``v = mean(conflict) - mean(clean)`` (positive alpha → context).
    """
    dirs = _load_resid_vectors(activations_npz, layers, row_indices=row_indices)
    used_layers = [li for li in layers if li in dirs]
    if not used_layers:
        return {}, None, {"direction_row_indices": row_indices, "used_layers": []}

    toward = str(steer_toward or "clean").strip().lower()
    if toward not in {"clean", "context"}:
        raise ValueError(f"unknown steer_toward={steer_toward!r}; expected clean|context")
    raw_v: dict[int, np.ndarray] = {}
    for li in used_layers:
        clean_m, conflict_m = dirs[li]
        if toward == "context":
            raw_v[li] = (conflict_m - clean_m).astype(np.float32)
        else:
            raw_v[li] = (clean_m - conflict_m).astype(np.float32)

    raw_v, control_meta = _apply_direction_control(
        raw_v,
        activations_npz,
        row_indices,
        direction_mode=direction_mode,
        direction_control_seed=direction_control_seed,
    )

    meta: dict[str, Any] = {
        "direction_row_indices": row_indices,
        "direction_num_rows": len(row_indices),
        "used_layers": used_layers,
        "steer_coeff_mode": steer_coeff_mode,
        "steer_toward": toward,
        "v_definition": (
            "mean(conflict)-mean(clean)" if toward == "context" else "mean(clean)-mean(conflict)"
        ),
        **control_meta,
    }

    if steer_coeff_mode == "normalized_beta":
        resid_scales = _layer_mean_residual_norm(activations_npz, used_layers, row_indices)
        proj: dict[int, np.ndarray] = {}
        for li in used_layers:
            v = raw_v[li].astype(np.float64)
            norm = float(np.linalg.norm(v))
            u = (v / norm).astype(np.float32) if norm > 1e-12 else v.astype(np.float32)
            r_l = float(resid_scales.get(li, 1.0))
            proj[li] = (r_l * u).astype(np.float32)
        meta["resid_norm_per_layer"] = {str(k): float(v) for k, v in resid_scales.items()}
        meta["raw_v_norm_per_layer"] = {
            str(li): float(np.linalg.norm(raw_v[li])) for li in used_layers
        }
        return proj, resid_scales, meta

    meta["raw_v_norm_per_layer"] = {
        str(li): float(np.linalg.norm(raw_v[li])) for li in used_layers
    }
    return raw_v, None, meta


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
    stage2_correct: int | float
    stage2_wrong: int | float
    stage2_skipped: int
    total: int
    stage2_runs: int
    stage2_early_stopped: bool = False
    stage1_eval_n: int = 0
    stage2_exact_match: float | None = None
    stage2_balanced_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        denom_s2 = max(1, self.stage2_runs)
        denom_s1 = max(1, self.stage1_eval_n or self.total)
        rate = (
            self.stage2_balanced_accuracy
            if self.stage2_balanced_accuracy is not None
            else self.stage2_correct / denom_s2
        )
        out = {
            "total_samples": self.total,
            "stage1_eval_n": self.stage1_eval_n or self.total,
            "stage1_pass": self.stage1_pass,
            "stage1_pass_rate": self.stage1_pass / denom_s1,
            "stage2_runs": self.stage2_runs,
            "stage2_correct": self.stage2_correct,
            "stage2_wrong": self.stage2_wrong,
            "stage2_skipped": self.stage2_skipped,
            "stage2_early_stopped": self.stage2_early_stopped,
            "stage2_correct_rate_among_stage2_runs": rate,
            "stage2_wrong_rate_among_stage2_runs": 1.0 - rate if self.stage2_balanced_accuracy is not None else self.stage2_wrong / denom_s2,
        }
        if self.stage2_balanced_accuracy is not None:
            out["stage2_balanced_accuracy"] = self.stage2_balanced_accuracy
            out["stage2_exact_match"] = self.stage2_exact_match
        return out


@torch.inference_mode()
def generate_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    model_key: str = "",
    use_chat_template: bool = True,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
) -> list[str]:
    formatted = format_prompts_for_inference(
        tokenizer, prompts, model_key, use_chat_template=use_chat_template
    )
    enc = tokenizer(
        formatted,
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
    texts = [tokenizer.decode(out[i, w:], skip_special_tokens=True).strip() for i in range(input_ids.shape[0])]
    del enc, input_ids, attention_mask, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts


def generate_chat_completion(
    model: torch.nn.Module,
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    model_key: str = "",
    use_chat_template: bool = True,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """One OpenAI-style chat completion (used by ToolSandbox agent/user roles)."""
    formatted = format_chat_messages_for_inference(
        tokenizer,
        messages,
        model_key,
        tools=tools,
        use_chat_template=use_chat_template,
    )
    enc = tokenizer(
        formatted,
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
        max_new_tokens=max_new_tokens if max_new_tokens is not None else max(MAX_NEW_TOKENS, 400),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    w = input_ids.shape[1]
    text = tokenizer.decode(out[0, w:], skip_special_tokens=True).strip()
    del enc, input_ids, attention_mask, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def _layer_hook_with_vector(
    v_mean: np.ndarray,
    alpha: float,
    *,
    gate: dict[str, bool] | None = None,
):
    """Apply ``h' = h + alpha * v`` on the layer hidden state."""
    v = np.asarray(v_mean, dtype=np.float32).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0 or not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0.0:
        return lambda _m, _inp, out: out

    a = float(alpha)
    _dv_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def hook(_m, _inp, out):
        if gate is not None and not gate.get("enabled", True):
            return out
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
    """Higher is better for alpha selection on validation (Acc_par)."""
    if r.stage2_runs <= 0:
        return -1.0
    return float(r.stage2_correct) / float(r.stage2_runs)


def _val_alpha_score_ctx(r: ModeResult) -> float:
    """Higher is better for alpha selection on validation (Acc_ctx)."""
    if r.stage2_runs <= 0:
        return -1.0
    return float(r.stage2_wrong) / float(r.stage2_runs)


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


def _pick_best_alpha_signed(
    alpha_scores: list[tuple[float, float]], *, positive: bool
) -> float | None:
    """Pick best α among α>0 (par) or α<0 (ctx) on validation."""
    if positive:
        candidates = [(float(a), float(s)) for a, s in alpha_scores if float(a) > 0]
    else:
        candidates = [(float(a), float(s)) for a, s in alpha_scores if float(a) < 0]
    if not candidates:
        return None
    best_a, best_s = candidates[0]
    for a, s in candidates[1:]:
        if s > best_s + 1e-12:
            best_a, best_s = a, s
        elif abs(s - best_s) <= 1e-12 and abs(a) < abs(best_a):
            best_a, best_s = a, s
    return float(best_a)


def _unique_coeffs(coeffs: list[float]) -> list[float]:
    out: list[float] = []
    for c in coeffs:
        cf = float(c)
        if not any(abs(cf - x) <= 1e-9 for x in out):
            out.append(cf)
    return out


def _val_grid_out_path(
    out_root: Path,
    manifest: dict[str, Any],
    args_model: str,
    layers_arg: str,
    output_category: str,
    direction_category_tag: str,
    test_category_tag: str,
    legacy_npz_category_tag: str,
    coeff_name: str = "alpha",
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
    c_label = str(coeff_name or "alpha").strip() or "alpha"
    out_dir = out_root / mod / ds / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"L{ls}_val_{c_label}_grid.json"


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


def _matches_context_gold(
    text: str,
    gold: list[str],
    *,
    dataset_name: str,
    category_tag: str,
    sample: dict[str, Any],
    judge_mode: str,
    is_kre: bool,
) -> bool:
    """True when the generation matches the context-side gold (not the parametric proxy)."""
    gold = [str(v).strip() for v in gold if str(v).strip()]
    if not gold:
        return False
    if str(judge_mode).strip().lower() == "official":
        from official_judges.judges import match_gold

        meta = {
            "category": category_tag or str(sample.get("category") or ""),
            "question": sample.get("question") or sample.get("query") or "",
            "query": sample.get("question") or sample.get("query") or "",
            "question_type": sample.get("question_type") or "multi-session",
            "eval_function": sample.get("eval_function") or "",
            "tool_call": sample.get("tool_call"),
            "source": sample.get("source"),
            "example_id": sample.get("example_id"),
            "level": sample.get("level") or sample.get("complexity_level"),
            "target": sample.get("target"),
            "constraint": sample.get("constraint"),
            "relation_type": sample.get("relation_type"),
            "locomo_category": sample.get("locomo_category"),
            "locomo_cat5_a": sample.get("locomo_cat5_a"),
            "locomo_cat5_b": sample.get("locomo_cat5_b"),
            "iheval_task": sample.get("iheval_task"),
            "iheval_answer": sample.get("iheval_answer"),
            "iheval_prompt": sample.get("iheval_prompt"),
            "toolsandbox_scenario": sample.get("toolsandbox_scenario"),
            "toolsandbox_similarity": sample.get("toolsandbox_similarity"),
            "bfcl_id": sample.get("bfcl_id"),
            "bfcl_test_category": sample.get("bfcl_test_category"),
            "bfcl_functions": sample.get("bfcl_functions"),
            "bfcl_ground_truth": sample.get("bfcl_ground_truth"),
            "bfcl_model_name": sample.get("bfcl_model_name"),
            "ifeval_target": sample.get("ifeval_target"),
            "ifeval_prompt": sample.get("ifeval_prompt"),
            "freshqa_question": sample.get("freshqa_question"),
            "freshqa_answers": sample.get("freshqa_answers"),
        }
        return any(match_gold(dataset_name, text, g, meta=meta) for g in gold)
    if is_kre:
        return _answer_matches_letters(text, gold)
    return _answer_matches(text, gold)


def _context_item_score(
    text: str,
    gold: list[str],
    *,
    dataset_name: str,
    category_tag: str,
    sample: dict[str, Any],
    judge_mode: str,
    is_kre: bool,
) -> float:
    """Official per-item credit. LoCoMo uses mean F1; other datasets stay 0/1."""
    gold = [str(v).strip() for v in gold if str(v).strip()]
    if not gold:
        return 0.0
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip() in {"LoCoMo", "locomo"}
    ):
        from official_judges.native import locomo_official_score

        cat = (
            sample.get("locomo_category")
            or sample.get("question_type")
            or sample.get("category")
            or category_tag
        )
        return float(
            locomo_official_score(
                text,
                gold[0],
                cat,
                option_a=sample.get("locomo_cat5_a"),
                option_b=sample.get("locomo_cat5_b"),
            )
        )
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip() in {"IHEval", "iheval"}
    ):
        from official_judges.native import iheval_official_score

        task = sample.get("iheval_task") or sample.get("category") or category_tag
        answer = sample.get("iheval_answer")
        if answer is None:
            answer = gold[0]
        prompt = str(sample.get("iheval_prompt") or sample.get("question") or "")
        return float(iheval_official_score(text, answer, task, prompt=prompt))
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip().lower() == "ifeval"
    ):
        from official_judges.ifeval_scorer import ifeval_strict_score

        target = sample.get("ifeval_target")
        if target is None:
            try:
                target = json.loads(gold[0])
            except (TypeError, ValueError, json.JSONDecodeError):
                return 0.0
        prompt = str(sample.get("ifeval_prompt") or sample.get("question") or "")
        return float(ifeval_strict_score(text, target, prompt=prompt))
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip().lower() == "followbench"
    ):
        from official_judges.native import followbench_official_score

        return float(followbench_official_score(text, sample))
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip().lower() == "freshqa"
    ):
        from official_judges.freshqa_scorer import freshqa_strict_score

        return float(
            freshqa_strict_score(
                text,
                sample.get("freshqa_answers") or gold,
                question=str(sample.get("freshqa_question") or sample.get("question") or ""),
            )
        )
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip() in {"ToolSandbox", "toolsandbox"}
    ):
        raw = sample.get("toolsandbox_similarity")
        if raw is None:
            return 0.0
        return float(raw)
    if (
        str(judge_mode).strip().lower() == "official"
        and str(dataset_name).strip() in {"BFCL", "bfcl"}
    ):
        from official_judges.bfcl_scorer import bfcl_official_score

        return float(bfcl_official_score(text, sample))
    return (
        1.0
        if _matches_context_gold(
            text,
            gold,
            dataset_name=dataset_name,
            category_tag=category_tag,
            sample=sample,
            judge_mode=judge_mode,
            is_kre=is_kre,
        )
        else 0.0
    )


def _count_num(value: float) -> int | float:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-8:
        return int(rounded)
    return float(value)


def _score_label(score: float) -> str:
    """Keep partial official scores visible instead of calling them wrong."""
    if float(score) >= 1.0 - 1e-8:
        return "correct"
    if float(score) > 0.0:
        return "partial"
    return "wrong"


def _legalbench_running_ba(per_sample: list[dict[str, Any]]) -> float:
    from official_judges.native import legalbench_balanced_accuracy

    golds: list[str] = []
    hits: list[bool] = []
    for row in per_sample:
        if not row.get("stage2_evaluated"):
            continue
        gold = row.get("gold_answers") or []
        golds.append(str(gold[0]) if gold else "")
        hits.append(float(row.get("official_score") or 0.0) >= 1.0 - 1e-8)
    return legalbench_balanced_accuracy(golds, hits)


def _legalbench_mode_result(
    dataset_name: str,
    samples: list[dict[str, Any]],
    per_sample: list[dict[str, Any]],
    s2_correct: float,
    s2_wrong: float,
) -> ModeResult:
    exact = _count_num(s2_correct)
    if str(dataset_name).strip() not in {"LegalBench", "legalbench"}:
        return ModeResult(
            len(samples),
            exact,
            _count_num(s2_wrong),
            0,
            len(samples),
            len(samples),
            stage1_eval_n=0,
        )
    labels = {
        str((row.get("gold_answers") or [""])[0]).strip().lower()
        for row in per_sample
        if row.get("stage2_evaluated") and row.get("gold_answers")
    }
    # LegalBench reports balanced accuracy for binary classification tasks.
    # Extraction and multi-label tasks use ordinary exact-match accuracy.
    if len(labels) != 2:
        return ModeResult(
            len(samples),
            exact,
            _count_num(s2_wrong),
            0,
            len(samples),
            len(samples),
            stage1_eval_n=0,
        )
    ba = _legalbench_running_ba(per_sample)
    return ModeResult(
        len(samples),
        exact,
        _count_num(s2_wrong),
        0,
        len(samples),
        len(samples),
        stage1_eval_n=0,
        stage2_exact_match=float(exact),
        stage2_balanced_accuracy=ba,
    )


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
    *,
    model_key: str = "",
    use_chat_template: bool = True,
    judge_mode: str = "legacy",
    category_tag: str = "",
    context_only: bool = False,
    judge_workers: int = 8,
    max_length: int | None = None,
) -> tuple[ModeResult, list[dict[str, Any]]]:
    layers = model.model.layers  # type: ignore[attr-defined]
    hooks = []
    steer_gate = {"enabled": True}
    for li, d in projection_dirs.items():
        if 0 <= li < len(layers):
            hooks.append(
                layers[li].register_forward_hook(
                    _layer_hook_with_vector(d, projection_alpha, gate=steer_gate)
                )
            )

    per_sample: list[dict[str, Any]] = []
    is_kre = str(dataset_name).strip().lower() == "kre"
    is_toolsandbox = str(dataset_name).strip() in {"ToolSandbox", "toolsandbox"}
    is_bfcl = str(dataset_name).strip() in {"BFCL", "bfcl"}
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
        if context_only and is_toolsandbox:
            from official_judges.toolsandbox_play import play_similarity

            def _agent_complete(
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None,
            ) -> str:
                steer_gate["enabled"] = True
                return generate_chat_completion(
                    model,
                    tokenizer,
                    messages,
                    tools=tools,
                    model_key=model_key,
                    use_chat_template=use_chat_template,
                    max_new_tokens=400,
                )

            def _user_complete(
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None,
            ) -> str:
                # Official score is about the agent trajectory; do not steer the user simulator.
                steer_gate["enabled"] = False
                try:
                    return generate_chat_completion(
                        model,
                        tokenizer,
                        messages,
                        tools=tools,
                        model_key=model_key,
                        use_chat_template=use_chat_template,
                        max_new_tokens=400,
                    )
                finally:
                    steer_gate["enabled"] = True

            s2_correct = s2_wrong = 0.0
            for done, row in enumerate(samples, start=1):
                name = str(
                    row.get("toolsandbox_scenario")
                    or (row.get("gold_answers") or [""])[0]
                    or ""
                ).strip()
                score, summary = play_similarity(name, _agent_complete, _user_complete)
                lab = _score_label(score)
                idx = done - 1
                per_sample[idx]["stage1_pass"] = True
                per_sample[idx]["stage1_skip_reason"] = "context_only"
                per_sample[idx]["stage2_output_raw"] = summary
                per_sample[idx]["stage2_output_judged"] = summary
                per_sample[idx]["stage2_label"] = lab
                per_sample[idx]["stage2_evaluated"] = True
                per_sample[idx]["official_score"] = score
                per_sample[idx]["toolsandbox_similarity"] = score
                per_sample[idx]["toolsandbox_scenario"] = name
                per_sample[idx]["judge_mode"] = str(judge_mode)
                # Keep score on the sample dict for any late judge helpers.
                row["toolsandbox_similarity"] = score
                s2_correct += score
                s2_wrong += 1.0 - score
                if done % PROGRESS_EVERY == 0 or done == len(samples):
                    print(
                        f"[{mode_name}] context progress: {done}/{len(samples)} "
                        f"correct={_count_num(s2_correct)} wrong={_count_num(s2_wrong)} "
                        f"last={name} score={score:.4f}",
                        flush=True,
                    )
            return (
                _legalbench_mode_result(
                    dataset_name,
                    samples,
                    per_sample,
                    s2_correct,
                    s2_wrong,
                ),
                per_sample,
            )

        if context_only and is_bfcl:
            import json as _json

            s2_correct = s2_wrong = 0.0
            for done, row in enumerate(samples, start=1):
                messages = row.get("bfcl_messages") or []
                if isinstance(messages, str):
                    messages = _json.loads(messages)
                tools = row.get("bfcl_tools")
                if isinstance(tools, str):
                    tools = _json.loads(tools)
                steer_gate["enabled"] = True
                raw = generate_chat_completion(
                    model,
                    tokenizer,
                    list(messages),
                    # bfcl_messages already include the official BFCL system
                    # prompt and function documentation. Passing bfcl_tools a
                    # second time makes tokenizer chat templates duplicate or
                    # rewrite the tool schema, which changes the benchmark
                    # prompt and can make outputs unparsable by ast_checker.
                    tools=None,
                    model_key=model_key,
                    use_chat_template=use_chat_template,
                    max_new_tokens=400,
                )
                score = _context_item_score(
                    raw,
                    [str(v) for v in row.get("gold_answers", [])],
                    dataset_name=dataset_name,
                    category_tag=category_tag,
                    sample=row,
                    judge_mode=judge_mode,
                    is_kre=is_kre,
                )
                lab = _score_label(score)
                idx = done - 1
                per_sample[idx]["stage1_pass"] = True
                per_sample[idx]["stage1_skip_reason"] = "context_only"
                per_sample[idx]["stage2_output_raw"] = raw
                per_sample[idx]["stage2_output_judged"] = _truncate_for_judge(raw)
                per_sample[idx]["stage2_label"] = lab
                per_sample[idx]["stage2_evaluated"] = True
                per_sample[idx]["official_score"] = score
                per_sample[idx]["bfcl_id"] = row.get("bfcl_id")
                per_sample[idx]["judge_mode"] = str(judge_mode)
                s2_correct += score
                s2_wrong += 1.0 - score
                if done % PROGRESS_EVERY == 0 or done == len(samples):
                    print(
                        f"[{mode_name}] context progress: {done}/{len(samples)} "
                        f"correct={_count_num(s2_correct)} wrong={_count_num(s2_wrong)} "
                        f"last={row.get('bfcl_id')} score={score:.4f}",
                        flush=True,
                    )
            return (
                _legalbench_mode_result(
                    dataset_name,
                    samples,
                    per_sample,
                    s2_correct,
                    s2_wrong,
                ),
                per_sample,
            )

        if context_only:
            s2_correct = s2_wrong = 0.0
            done = 0
            n_workers = max(1, int(judge_workers))
            for st in range(0, len(samples), batch_size):
                idxs = list(range(st, min(st + batch_size, len(samples))))
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
                    model_key=model_key,
                    use_chat_template=use_chat_template,
                    max_length=max_length,
                )

                def _judge_one(j: int) -> tuple[int, str, str, float]:
                    raw = outs[j]
                    t = _truncate_for_judge(raw)
                    row = chunk[j]
                    gold = [str(v) for v in row.get("gold_answers", [])]
                    score = _context_item_score(
                        t,
                        gold,
                        dataset_name=dataset_name,
                        category_tag=category_tag,
                        sample=row,
                        judge_mode=judge_mode,
                        is_kre=is_kre,
                    )
                    return j, raw, t, score

                if n_workers <= 1 or len(chunk) <= 1:
                    judged = [_judge_one(j) for j in range(len(chunk))]
                else:
                    with ThreadPoolExecutor(max_workers=min(n_workers, len(chunk))) as pool:
                        judged = list(pool.map(_judge_one, range(len(chunk))))

                for j, raw, t, score in judged:
                    src_idx = idxs[j]
                    lab = _score_label(score)
                    per_sample[src_idx]["stage1_pass"] = True
                    per_sample[src_idx]["stage1_skip_reason"] = "context_only"
                    per_sample[src_idx]["stage2_output_raw"] = raw
                    per_sample[src_idx]["stage2_output_judged"] = t
                    per_sample[src_idx]["stage2_label"] = lab
                    per_sample[src_idx]["stage2_evaluated"] = True
                    per_sample[src_idx]["official_score"] = score
                    per_sample[src_idx]["judge_mode"] = str(judge_mode)
                    s2_correct += score
                    s2_wrong += 1.0 - score
                done += len(chunk)
                if done % PROGRESS_EVERY == 0 or done == len(samples):
                    extra = ""
                    if str(dataset_name).strip() in {"LegalBench", "legalbench"}:
                        extra = f" balanced_accuracy={_legalbench_running_ba(per_sample):.4f}"
                    print(
                        f"[{mode_name}] context progress: {done}/{len(samples)} "
                        f"correct={_count_num(s2_correct)} wrong={_count_num(s2_wrong)}{extra}",
                        flush=True,
                    )
            return (
                _legalbench_mode_result(
                    dataset_name,
                    samples,
                    per_sample,
                    s2_correct,
                    s2_wrong,
                ),
                per_sample,
            )

        s1_flags = [
            _default_stage1_pass_from_manifest(samples[i], group_name) for i in range(len(samples))
        ]
        s1_done = 0
        for st in range(0, len(s1_eval_idxs), batch_size):
            idxs = s1_eval_idxs[st : st + batch_size]
            chunk = [samples[i] for i in idxs]
            outs = generate_batch(
                model,
                tokenizer,
                [str(x["clean_prompt_input"]) for x in chunk],
                model_key=model_key,
                use_chat_template=use_chat_template,
                max_length=max_length,
            )
            for j, row in enumerate(chunk):
                idx = idxs[j]
                raw = outs[j]
                t = _truncate_for_judge(raw)
                gold = [str(v) for v in row.get("gold_answers", [])]
                dlist = row.get("distracted_tokens")
                if not isinstance(dlist, list):
                    dlist = [str(row.get("distracted_token", ""))]
                dlist = [str(x) for x in dlist]

                if str(row.get("internal_source") or "").strip() == "stage1_spontaneous":
                    # gold_answers is dump-time spontaneous output; Stage1 gate already
                    # applied at dump (spontaneous ≠ external). Do not require regenerating
                    # the exact same string here.
                    ok = True
                    hit_gold = True
                elif str(judge_mode).strip().lower() == "official":
                    hit_gold, _lab = _official_judge_sample(
                        dataset=dataset_name,
                        category=category_tag,
                        text=t,
                        gold_answers=gold,
                        distracted_tokens=dlist,
                        sample=row,
                    )
                    if str(group_name).strip() == "stage1_wrong_use_correct_context":
                        ok = not hit_gold
                    else:
                        ok = hit_gold
                elif is_kre:
                    hit_gold = _answer_matches_letters(t, gold)
                    if str(group_name).strip() == "stage1_wrong_use_correct_context":
                        ok = not hit_gold
                    else:
                        ok = hit_gold
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
                per_sample[idx]["judge_mode"] = str(judge_mode)
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
                model_key=model_key,
                use_chat_template=use_chat_template,
                max_length=max_length,
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
                if str(judge_mode).strip().lower() == "official":
                    _hit, lab = _official_judge_sample(
                        dataset=dataset_name,
                        category=category_tag,
                        text=t,
                        gold_answers=gold,
                        distracted_tokens=dlist,
                        sample=row,
                    )
                    if no_skip_in_stage2 and lab == "skipped":
                        lab = "correct" if _hit else "wrong"
                elif is_kre:
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
                per_sample[src_idx]["judge_mode"] = str(judge_mode)
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
        help="Optional category tag. For NQ-Swap, can be a by_wikidata_property jsonl filename or an absolute path; stem is used as tag.",
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
        "--max-length",
        type=int,
        default=TOKENIZER_MAX_LEN,
        help=(
            "Tokenizer input limit. Default 4096 for legacy compatibility; "
            "use 8192 for MemoryAgentBench 6k contexts."
        ),
    )
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
        "--judge",
        type=str,
        default="legacy",
        choices=["legacy", "official"],
        help="Use substring/letter matching or the single-target official judge.",
    )
    ap.add_argument(
        "--judge-workers",
        type=int,
        default=int(os.environ.get("JUDGE_WORKERS", "8") or 8),
        help="Parallel LLM/native judge threads after each generate batch (context-only). Default 8.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_STEER_OUT_DIR,
        help="Output root directory for steer result json files.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing per-alpha result JSON files instead of rerunning and overwriting them.",
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
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template wrapping (raw Question:/Answer: prompts).",
    )
    ap.add_argument(
        "--eval-mode",
        choices=("legacy", "train_val_test"),
        default="legacy",
        help="legacy: sweep alphas on full --test-json. train_val_test: split manifest 70/15/15 at "
        "eval time — train NPZ rows for direction, val for alpha, test for final metrics.",
    )
    ap.add_argument("--split-seed", type=int, default=42, help="RNG seed for eval-time train/val/test split.")
    ap.add_argument("--train-ratio", type=float, default=0.7, help="Train fraction for direction (eval split).")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Val fraction for alpha selection (eval split).")
    ap.add_argument(
        "--direction-subset-size",
        type=int,
        default=0,
        help="Random subsample size for direction v from NPZ train pool (0 = use all).",
    )
    ap.add_argument(
        "--direction-subset-seed",
        type=int,
        default=42,
        help="RNG seed for --direction-subset-size sampling.",
    )
    ap.add_argument(
        "--direction-mode",
        choices=("true", "shuffled", "random_split", "gaussian"),
        default="true",
        help=(
            "Direction construction: true = mean(clean)-mean(conflict); "
            "shuffled/random_split = permute pooled activations, split in half, norm-match to true; "
            "gaussian = N(0,I) norm-matched to true."
        ),
    )
    ap.add_argument(
        "--direction-control-seed",
        type=int,
        default=0,
        help="RNG seed for --direction-mode shuffled or gaussian (ignored for true).",
    )
    ap.add_argument(
        "--steer-toward",
        choices=("clean", "context"),
        default="clean",
        help=(
            "clean: v = mean(clean)-mean(conflict), positive alpha toward parametric. "
            "context: v = mean(conflict)-mean(clean), positive alpha toward context."
        ),
    )
    ap.add_argument(
        "--context-only",
        action="store_true",
        help=(
            "Skip Stage1. Run every manifest row on the context prompt and score "
            "correct iff the output matches gold_answers (the context-side answer)."
        ),
    )
    ap.add_argument(
        "--steer-coeff-mode",
        choices=("alpha", "normalized_beta"),
        default="alpha",
        help=(
            "alpha: h' = h + alpha * v (default). "
            "normalized_beta: u = v/||v||, r_l = E[||h_l||] on direction rows, "
            "h' = h + beta * r_l * u (beta passed as alpha grid value)."
        ),
    )
    ap.add_argument(
        "--coeff-selection",
        choices=("val_grid", "fixed"),
        default="val_grid",
        help="val_grid: pick best coeff on val split. fixed: use --fixed-coeff, skip val grid.",
    )
    ap.add_argument(
        "--fixed-coeff",
        type=float,
        default=None,
        help="Single fixed alpha/beta when --coeff-selection=fixed.",
    )
    ap.add_argument(
        "--fixed-coeffs",
        type=str,
        default=None,
        help="Comma-separated fixed alphas/betas (e.g. '1.0,-1.6') when --coeff-selection=fixed.",
    )
    args = ap.parse_args()
    use_chat_template = not bool(args.no_chat_template)
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
        # For NQ-Swap: allow "c1__foo.jsonl" or "c1__foo"
        if dataset == "NQ-Swap":
            if cat.endswith(".jsonl"):
                return Path(cat).stem
            return cat
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
    if not test_manifest.is_file():
        raise SystemExit(f"test manifest not found: {test_manifest}")

    max_s1 = int(args.max_samples_stage1) if args.max_samples_stage1 is not None else int(args.max_samples)
    max_s2 = int(args.max_samples_stage2)
    steer_coeff_mode = str(args.steer_coeff_mode).strip()
    coeff_selection = str(args.coeff_selection).strip()
    coeff_name = "beta" if steer_coeff_mode == "normalized_beta" else "alpha"
    if coeff_selection == "fixed":
        fixed_list: list[float] = []
        if args.fixed_coeff is not None:
            fixed_list.append(float(args.fixed_coeff))
        extra = str(args.fixed_coeffs or "").strip()
        if extra:
            fixed_list.extend(float(x.strip()) for x in extra.split(",") if x.strip())
        if not fixed_list:
            raise SystemExit("--fixed-coeff or --fixed-coeffs is required when --coeff-selection=fixed")
        coeffs = _unique_coeffs(fixed_list)
    else:
        coeffs = _alphas_from_cli(args.alpha_start, args.alpha_end, args.alpha_step)

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]

    obj, samples_all = _load_manifest(test_manifest)
    group_name = str(obj.get("group") or "")

    z_check = np.load(direction_npz, allow_pickle=True)
    n_npz = len(z_check["resid_act_clean"])
    z_check.close()

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    direction_pool_indices: list[int]
    cross_dataset_eval = False
    if eval_mode == "train_val_test":
        n_all = len(samples_all)
        split_kw = dict(
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            seed=int(args.split_seed),
        )
        if n_npz != n_all:
            cross_dataset_eval = True
            train_dir, _, _ = split_row_indices(n_npz, **split_kw)
            train_indices, val_indices, test_indices = split_row_indices(n_all, **split_kw)
            direction_pool_indices = train_dir
            print(
                f"[train_val_test] cross-dataset: direction NPZ ({n_npz} rows, source train="
                f"{len(train_dir)}) -> target manifest ({n_all} rows: "
                f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}, "
                f"ratios {args.train_ratio}/{args.val_ratio}/rest, seed={args.split_seed})",
                flush=True,
            )
        else:
            train_indices, val_indices, test_indices = split_row_indices(n_all, **split_kw)
            direction_pool_indices = train_indices
            print(
                f"[train_val_test] eval-time split from full manifest ({n_all} rows): "
                f"train={len(train_indices)} val={len(val_indices)} test={len(test_indices)} "
                f"(ratios {args.train_ratio}/{args.val_ratio}/rest, seed={args.split_seed})",
                flush=True,
            )
    else:
        direction_pool_indices = list(range(n_npz))

    direction_row_indices = _resolve_direction_row_indices(
        n_npz,
        direction_pool_indices,
        eval_mode,
        int(args.direction_subset_size),
        int(args.direction_subset_seed),
    )
    direction_mode = str(args.direction_mode).strip()
    direction_control_seed = int(args.direction_control_seed)
    print(
        f"[direction] mode={direction_mode} steer_coeff_mode={steer_coeff_mode} "
        f"rows={len(direction_row_indices)} "
        f"(subset_size={int(args.direction_subset_size)}, subset_seed={int(args.direction_subset_seed)}, "
        f"control_seed={direction_control_seed})",
        flush=True,
    )

    proj, resid_scales, dir_meta = _build_projection_dirs(
        direction_npz,
        layers,
        direction_row_indices,
        steer_coeff_mode=steer_coeff_mode,
        direction_mode=direction_mode,
        direction_control_seed=direction_control_seed,
        steer_toward=str(args.steer_toward),
    )
    used_layers = [li for li in layers if li in proj]
    if not used_layers:
        raise SystemExit("no valid layers from direction activations NPZ")

    model, tokenizer = get_model_and_tokenizer(args.model)

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is not None:
        for li, vec in proj.items():
            if int(vec.shape[0]) != int(hidden_size):
                raise SystemExit(
                    f"projection vector at layer {li} has dim {vec.shape[0]}, "
                    f"but model hidden_size={hidden_size} — NPZ/model mismatch"
                )
    steer_mode = "steer_clean_minus_conflict"
    if direction_mode == "shuffled":
        dir_desc = (
            "shuffled-label control: permute pooled clean+conflict activations, "
            "split in half, v = mean(group1)-mean(group2), norm-matched to ||v_true||"
        )
    elif direction_mode == "gaussian":
        dir_desc = "Gaussian control: v ~ N(0,I), norm-matched to ||v_true||"
    else:
        dir_desc = "v = mean(resid_act_clean) - mean(resid_act_conflict)"
    if steer_coeff_mode == "normalized_beta":
        steer_formula = (
            f"{dir_desc}; u = v/||v||; r_l = E[||h_l||] on direction rows; "
            "h' = h + beta * r_l * u (beta>0 adds; beta<0 subtracts |beta|*r_l*u)"
        )
        labeling_steer = (
            f"Normalized residual steer ({direction_mode}): h' = h + beta * r_l * u "
            f"({direction_npz.name}, {len(direction_row_indices)} direction rows)."
        )
    else:
        steer_formula = (
            f"{dir_desc}; h' = h + alpha * v "
            "(alpha>0 adds v; alpha<0 subtracts |alpha|*v)"
        )
        labeling_steer = (
            f"Residual steer ({direction_mode}): h' = h + alpha * v "
            f"({direction_npz.name}, {len(direction_row_indices)} direction rows)."
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
        coeff: float,
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
        payload = {
            "eval_mode": eval_mode,
            "eval_split": eval_split,
            "direction_npz": _display_path(direction_npz),
            "direction_vectors_npz": _display_path(direction_npz),
            "manifest_group": manifest_group,
            "direction_category_tag": (dir_out or legacy_npz or None),
            "test_category_tag": tst_out or None,
            "manifest_json": _display_path(manifest_path),
            "test_json": _display_path(test_manifest),
            "data_split": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "split_seed": int(args.split_seed),
                "train_manifest_indices": train_indices if eval_mode == "train_val_test" else None,
                "val_manifest_indices": val_indices if eval_mode == "train_val_test" else None,
                "test_manifest_indices": test_indices if eval_mode == "train_val_test" else None,
            },
            "gpu": str(args.gpu),
            "batch_size": int(args.batch_size),
            "max_samples_stage1": max_s1,
            "max_samples_stage2": max_s2,
            "sample_seed": int(args.sample_seed),
            "stage1_min_pass_rate": stage1_min_pass_rate,
            "labeling_note": (
                (
                    f"Context-only steer (judge={args.judge}): all rows on the context prompt; "
                    "correct = match context gold. "
                    if bool(args.context_only)
                    else f"Stage1 gate then Stage2 steer (judge={args.judge}). "
                )
                + labeling_steer
                + (
                    " Official judges score the prediction against one target."
                    if str(args.judge) == "official"
                    else " Judgement rules match dump_activations_from_testjsonl.py."
                )
            ),
            "steer_formula": steer_formula,
            "steer_coeff_mode": steer_coeff_mode,
            "coeff_selection": coeff_selection,
            "coeff_name": coeff_name,
            "direction_extraction": {
                **dir_meta,
                "direction_mode": direction_mode,
                "direction_control_seed": direction_control_seed,
                "direction_subset_size": int(args.direction_subset_size),
                "direction_subset_seed": int(args.direction_subset_seed),
            },
            "model": args.model,
            "coeff_grid": {
                "coeff_start": float(args.alpha_start),
                "coeff_end": float(args.alpha_end),
                "coeff_step": float(args.alpha_step),
                "coeffs": [float(x) for x in coeffs],
                "fixed_coeff": float(args.fixed_coeff) if args.fixed_coeff is not None else None,
                "index_in_grid": ai,
            },
            "alpha": float(coeff),
            "beta": float(coeff),
            "alpha_grid": {
                "alpha_start": float(args.alpha_start),
                "alpha_end": float(args.alpha_end),
                "alpha_step": float(args.alpha_step),
                "alphas": [float(x) for x in coeffs],
                "index_in_grid": ai,
            },
            "requested_resid_layers": layers,
            "used_resid_layers_with_valid_dirs": used_layers,
            "results": results,
            "per_sample_outputs_and_judgements": {steer_mode: ps},
        }
        payload[coeff_name] = float(coeff)
        return payload

    def _run_on_split_indices(
        split_indices: list[int],
        coeff: float,
        *,
        eval_split: str,
        allow_early_stop: bool,
    ) -> tuple[ModeResult, list[dict[str, Any]], dict[str, Any]]:
        subset = [samples_all[i] for i in split_indices]
        ds_name = str(obj.get("dataset") or "")
        grp = group_name
        no_skip = grp == "stage1_wrong_use_correct_context"
        (
            subset,
            s1_idx,
            _s2_samples_unused,
            s2_idx,
            n_tot,
            n_s1,
            n_s2,
        ) = _prepare_eval_samples(
            subset,
            max_samples_head=int(args.max_samples_head),
            max_s1=max_s1,
            max_s2=max_s2,
            sample_seed=int(args.sample_seed),
        )
        s2_samples, s1_rel = _stage2_layout(subset, s1_idx, s2_idx)
        r, ps = _run_one_mode(
            model,
            tokenizer,
            s2_samples,
            ds_name,
            grp,
            proj,
            float(coeff),
            int(args.batch_size),
            steer_mode,
            no_skip_in_stage2=no_skip,
            stage1_min_pass_rate=stage1_min_pass_rate if allow_early_stop else None,
            stage1_indices=s1_rel,
            model_key=str(args.model),
            use_chat_template=use_chat_template,
            judge_mode=str(args.judge),
            category_tag=str(out_cat or tst_out or ""),
            context_only=bool(args.context_only),
            judge_workers=int(args.judge_workers),
            max_length=int(args.max_length),
        )
        payload = _build_result_payload(
            obj=obj,
            manifest_path=test_manifest,
            eval_split=eval_split,
            manifest_group=grp,
            samples_all=subset,
            stage1_indices=s1_idx,
            stage2_indices=s2_idx,
            n_total_manifest=n_tot,
            n_stage1_eval=n_s1,
            n_stage2_eval=n_s2,
            r=r,
            ps=ps,
            coeff=float(coeff),
            ai=0,
        )
        return r, ps, payload

    out_paths: list[str] = []

    if eval_mode == "train_val_test":
        print(
            f"[train_val_test] direction from train rows in NPZ: {_display_path(direction_npz)}",
            flush=True,
        )
        print(
            f"[train_val_test] manifest: {_display_path(test_manifest)} "
            f"(val→alpha, test→final eval)",
            flush=True,
        )

        val_grid_path = _val_grid_out_path(
            out_root,
            obj,
            str(args.model),
            str(args.layers),
            out_cat,
            dir_out,
            tst_out,
            legacy_npz,
            coeff_name=coeff_name,
        )

        if coeff_selection == "fixed":
            coeff_str = ", ".join(f"{c:g}" for c in coeffs)
            print(
                f"[train_val_test] fixed {coeff_name}=[{coeff_str}] (skip val grid)",
                flush=True,
            )
            for ai, coeff in enumerate(coeffs):
                test_out_path = _steer_out_path(
                    out_root,
                    obj,
                    str(args.model),
                    str(args.layers),
                    float(coeff),
                    output_category=out_cat,
                    direction_category_tag=dir_out,
                    test_category_tag=tst_out,
                    legacy_npz_category_tag=legacy_npz,
                    coeff_name=coeff_name,
                )
                if args.skip_existing and test_out_path.is_file():
                    print(f"[test] skip existing -> {test_out_path.name}", flush=True)
                    continue
                _r, _ps, test_payload = _run_on_split_indices(
                    test_indices, float(coeff), eval_split="test", allow_early_stop=False
                )
                test_payload["coeff_selection"] = "fixed"
                test_payload["fixed_coeff"] = float(coeff)
                test_payload["out_json"] = _display_path(test_out_path)
                test_out_path.write_text(
                    json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                out_paths.append(_display_path(test_out_path))
        else:
            val_coeff_rows: list[dict[str, Any]] = []
            coeff_scores_par: list[tuple[float, float]] = []
            coeff_scores_ctx: list[tuple[float, float]] = []

            for ai, coeff in enumerate(coeffs):
                print(f"[val {coeff_name} {ai + 1}/{len(coeffs)}] {coeff_name}={coeff:g}", flush=True)
                r, _ps, payload = _run_on_split_indices(
                    val_indices, float(coeff), eval_split="val", allow_early_stop=False
                )
                payload["coeff_grid"]["index_in_grid"] = ai
                summary = payload["results"]["after_steer_stage2_on_stage1_pass"]
                val_coeff_rows.append(
                    {
                        coeff_name: float(coeff),
                        "val_alpha_score": _val_alpha_score(r),
                        "val_alpha_score_ctx": _val_alpha_score_ctx(r),
                        "summary": summary,
                    }
                )
                coeff_scores_par.append((float(coeff), _val_alpha_score(r)))
                coeff_scores_ctx.append((float(coeff), _val_alpha_score_ctx(r)))
                if r.stage2_early_stopped and float(coeff) > 0:
                    print(f"[val] early stop on stage1 pass rate; stopping {coeff_name} grid", flush=True)
                    break

            best_coeff_par = _pick_best_alpha_signed(coeff_scores_par, positive=True)
            best_coeff_ctx = _pick_best_alpha_signed(coeff_scores_ctx, positive=False)
            if best_coeff_par is None and best_coeff_ctx is None:
                raise SystemExit("no valid alpha on validation (need α>0 and/or α<0)")
            test_coeffs = _unique_coeffs(
                [c for c in (best_coeff_par, best_coeff_ctx) if c is not None]
            )
            val_grid_doc = {
                "eval_mode": "train_val_test",
                "steer_coeff_mode": steer_coeff_mode,
                "coeff_name": coeff_name,
                "direction_npz": _display_path(direction_npz),
                "direction_extraction": dir_meta,
                "test_json": _display_path(test_manifest),
                "data_split": {
                    "train_ratio": float(args.train_ratio),
                    "val_ratio": float(args.val_ratio),
                    "split_seed": int(args.split_seed),
                    "train_manifest_indices": train_indices,
                    "val_manifest_indices": val_indices,
                    "test_manifest_indices": test_indices,
                },
                "best_alpha_par": best_coeff_par,
                "best_alpha_ctx": best_coeff_ctx,
                f"best_{coeff_name}_par": best_coeff_par,
                f"best_{coeff_name}_ctx": best_coeff_ctx,
                f"best_{coeff_name}": best_coeff_par,
                "best_alpha": best_coeff_par,
                "alpha_selection_metric_par": "stage2_correct_rate_among_stage2_runs",
                "alpha_selection_metric_ctx": "stage2_wrong_rate_among_stage2_runs",
                f"{coeff_name}s_evaluated": val_coeff_rows,
                "alphas_evaluated": val_coeff_rows,
                "model": args.model,
                "layers": str(args.layers),
            }
            val_grid_path.write_text(
                json.dumps(val_grid_doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            out_paths.append(_display_path(val_grid_path))
            par_msg = f"{best_coeff_par:g}" if best_coeff_par is not None else "n/a"
            ctx_msg = f"{best_coeff_ctx:g}" if best_coeff_ctx is not None else "n/a"
            print(
                f"[train_val_test] best {coeff_name} on val: par={par_msg} ctx={ctx_msg}",
                flush=True,
            )

            for coeff in test_coeffs:
                test_out_path = _steer_out_path(
                    out_root,
                    obj,
                    str(args.model),
                    str(args.layers),
                    float(coeff),
                    output_category=out_cat,
                    direction_category_tag=dir_out,
                    test_category_tag=tst_out,
                    legacy_npz_category_tag=legacy_npz,
                    coeff_name=coeff_name,
                )
                if args.skip_existing and test_out_path.is_file():
                    print(f"[test] skip existing -> {test_out_path.name}", flush=True)
                    continue
                _r, _ps, test_payload = _run_on_split_indices(
                    test_indices, float(coeff), eval_split="test", allow_early_stop=False
                )
                if best_coeff_par is not None and abs(float(coeff) - float(best_coeff_par)) <= 1e-9:
                    test_payload["best_alpha_par_from_val"] = float(best_coeff_par)
                    test_payload[f"best_{coeff_name}_par_from_val"] = float(best_coeff_par)
                if best_coeff_ctx is not None and abs(float(coeff) - float(best_coeff_ctx)) <= 1e-9:
                    test_payload["best_alpha_ctx_from_val"] = float(best_coeff_ctx)
                    test_payload[f"best_{coeff_name}_ctx_from_val"] = float(best_coeff_ctx)
                test_payload["val_alpha_grid_json"] = _display_path(val_grid_path)
                test_payload["out_json"] = _display_path(test_out_path)
                test_out_path.write_text(
                    json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                out_paths.append(_display_path(test_out_path))
    else:
        samples_eval, stage1_indices, samples_stage2, stage2_indices, n_total_manifest, n_stage1_eval, n_stage2_eval = _prepare_eval_samples(
            samples_all,
            max_samples_head=int(args.max_samples_head),
            max_s1=max_s1,
            max_s2=max_s2,
            sample_seed=int(args.sample_seed),
        )
        dataset_name = str(obj.get("dataset") or "")
        group_name = str(obj.get("group") or "")
        no_skip_in_stage2 = group_name == "stage1_wrong_use_correct_context"
        n_manifest_s2_wrong = sum(1 for s in samples_eval if _manifest_stage2_is_wrong(s))

        if n_stage1_eval < n_total_manifest:
            print(
                f"[subsample] Stage1 rerun on {n_stage1_eval}/{n_total_manifest} rows "
                f"(max_samples_stage1={max_s1}, seed={int(args.sample_seed)})",
                flush=True,
            )
        else:
            print(f"[subsample] Stage1 rerun on all {n_stage1_eval} manifest rows", flush=True)
        if n_stage2_eval < n_total_manifest:
            print(
                f"[subsample] Stage2 on {n_stage2_eval}/{n_total_manifest} rows "
                f"(max_samples_stage2={max_s2})",
                flush=True,
            )
        else:
            print(f"[subsample] Stage2 on full manifest ({n_stage2_eval} rows)", flush=True)

        for ai, coeff in enumerate(coeffs):
            out_path = _steer_out_path(
                out_root,
                obj,
                str(args.model),
                str(args.layers),
                float(coeff),
                output_category=out_cat,
                direction_category_tag=dir_out,
                test_category_tag=tst_out,
                legacy_npz_category_tag=legacy_npz,
                coeff_name=coeff_name,
            )
            if args.skip_existing and out_path.is_file():
                print(
                    f"[{coeff_name} {ai + 1}/{len(coeffs)}] skip existing {coeff_name}={coeff:g} -> {out_path.name}",
                    flush=True,
                )
                continue
            print(
                f"[{coeff_name} {ai + 1}/{len(coeffs)}] {coeff_name}={coeff:g} -> {out_path.name}",
                flush=True,
            )
            stage2_samples, stage1_rel = _stage2_layout(
                samples_eval, stage1_indices, stage2_indices
            )
            r, ps = _run_one_mode(
                model,
                tokenizer,
                stage2_samples,
                dataset_name,
                group_name,
                proj,
                float(coeff),
                int(args.batch_size),
                steer_mode,
                no_skip_in_stage2=no_skip_in_stage2,
                stage1_min_pass_rate=stage1_min_pass_rate,
                stage1_indices=stage1_rel,
                model_key=str(args.model),
                use_chat_template=use_chat_template,
                judge_mode=str(args.judge),
                category_tag=str(out_cat or tst_out or ""),
                context_only=bool(args.context_only),
                judge_workers=int(args.judge_workers),
                max_length=int(args.max_length),
            )
            out = _build_result_payload(
                obj=obj,
                manifest_path=test_manifest,
                eval_split="test",
                manifest_group=group_name,
                samples_all=samples_eval,
                stage1_indices=stage1_indices,
                stage2_indices=stage2_indices,
                n_total_manifest=n_total_manifest,
                n_stage1_eval=n_stage1_eval,
                n_stage2_eval=n_stage2_eval,
                r=r,
                ps=ps,
                coeff=float(coeff),
                ai=ai,
            )
            out["out_json"] = _display_path(out_path)
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            out_paths.append(_display_path(out_path))

            if r.stage2_early_stopped and float(coeff) > 0:
                print(
                    f"[{coeff_name} grid] stage1 pass rate < {stage1_min_pass_rate:g} at {coeff_name}={coeff:g} (>0); "
                    f"stop remaining coeffs ({ai + 1}/{len(coeffs)} completed in this job)",
                    flush=True,
                )
                break

    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()
