"""Model load, residual steering, and generation helpers for IFEval ablation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BUNDLE_ROOT.parent
MODEL_ROOT = REPO_ROOT / "models"

MAX_NEW_TOKENS = 2048
TOKENIZER_MAX_LEN = 8192
PROGRESS_EVERY = 10
_LOCAL_KW = {"trust_remote_code": True, "local_files_only": True}

EARLY_STOP_METRIC_KEYS = (
    "prompt_level_strict_acc",
    "prompt_level_loose_acc",
    "instruction_level_strict_acc",
    "instruction_level_loose_acc",
)


def _load_model_paths() -> dict[str, str]:
    p = BUNDLE_ROOT / "config" / "models.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        vp = Path(str(v))
        if not vp.is_absolute():
            vp = (BUNDLE_ROOT / vp).resolve()
        out[str(k)] = str(vp)
    return out


def _safe_name(s: str) -> str:
    t = str(s).strip().replace("\\", "/")
    if "/" in t:
        t = Path(t).stem
    return t.replace(" ", "_").replace("/", "_")


def _abspath(p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (BUNDLE_ROOT / p).resolve()


def _display_path(p: Path | str) -> str:
    rp = Path(p).resolve()
    try:
        return str(rp.relative_to(BUNDLE_ROOT.resolve()))
    except ValueError:
        try:
            return str(rp.relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(rp)


def _one_line_preview(text: str, max_chars: int = 120) -> str:
    s = text.replace("\n", " ").replace("\r", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


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
    activations_npz: Path, layers: list[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    z = np.load(activations_npz, allow_pickle=True)
    rc = z["resid_act_clean"]
    if "resid_act_conflict" in z.files:
        rw = z["resid_act_conflict"]
    elif "resid_act_stage2" in z.files:
        rw = z["resid_act_stage2"]
    else:
        z.close()
        raise KeyError(
            "Neither 'resid_act_conflict' nor 'resid_act_stage2' in NPZ. "
            f"keys={list(z.files)}"
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


def _resid_direction_meta(mode: str) -> tuple[str, str]:
    m = str(mode).strip().lower()
    if m == "conflict":
        return (
            "conflict",
            "v = mean(resid_act_conflict); h' = h + alpha * v "
            "(alpha>0 adds v; alpha<0 subtracts |alpha|*v)",
        )
    if m == "clean":
        return (
            "clean",
            "v = mean(resid_act_clean); h' = h + alpha * v "
            "(alpha>0 adds v; alpha<0 subtracts |alpha|*v)",
        )
    return (
        "diff",
        "v = mean(resid_act_clean) - mean(resid_act_conflict); h' = h + alpha * v "
        "(alpha>0 adds v; alpha<0 subtracts |alpha|*v)",
    )


def _resid_projection_vectors(
    dirs: dict[int, tuple[np.ndarray, np.ndarray]],
    layers: list[int],
    mode: str = "clean_minus_conflict",
) -> tuple[dict[int, np.ndarray], str, str]:
    vec_tag, steer_formula = _resid_direction_meta(mode)
    m = str(mode).strip().lower()
    proj: dict[int, np.ndarray] = {}
    for li in layers:
        if li not in dirs:
            continue
        dc, dw = dirs[li]
        if m == "conflict":
            proj[li] = dw.astype(np.float32)
        elif m == "clean":
            proj[li] = dc.astype(np.float32)
        else:
            proj[li] = (dc - dw).astype(np.float32)
    return proj, vec_tag, steer_formula


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


def _iter_alpha_grid(a0: float, a1: float, step: float) -> list[float]:
    if step == 0:
        raise SystemExit("alpha-step must be non-zero when alpha-start != alpha-end")
    out: list[float] = []
    x = float(a0)
    n = 0
    if step > 0:
        if a0 > a1:
            raise SystemExit("alpha-start must be <= alpha-end when alpha-step > 0")
        while x <= a1 + 1e-9 and n < 1_000_000:
            out.append(round(x, 12))
            x += step
            n += 1
    else:
        if a0 < a1:
            raise SystemExit("alpha-start must be >= alpha-end when alpha-step < 0")
        while x >= a1 - 1e-9 and n < 1_000_000:
            out.append(round(x, 12))
            x += step
            n += 1
    if not out:
        raise SystemExit("empty alpha grid")
    return out


def _alphas_from_cli(a0: float, a1: float, step: float) -> list[float]:
    if abs(float(a0) - float(a1)) < 1e-12:
        return [float(a0)]
    return _iter_alpha_grid(float(a0), float(a1), float(step))


def _early_stop_metric_vector(manifest: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(manifest[k]) for k in EARLY_STOP_METRIC_KEYS)


def _all_early_stop_metrics_decreased(
    prev: tuple[float, ...], curr: tuple[float, ...]
) -> bool:
    return all(c < p - 1e-12 for p, c in zip(prev, curr))


def _alpha_run_dirname(layers_arg: str, alpha: float, vec_tag: str = "") -> str:
    ls = str(layers_arg).replace(",", "_").replace(" ", "")
    a_str = f"{float(alpha):g}".replace(".", "p")
    base = f"L{ls}_alpha{a_str}"
    vtag = _safe_name(str(vec_tag).strip())
    if vtag:
        return f"{base}__vec_{vtag}"
    return base


def _load_tokenizer(model_path: str):
    model_path = str(model_path)
    errors: list[str] = []
    try:
        return AutoTokenizer.from_pretrained(model_path, **_LOCAL_KW)
    except AttributeError as e:
        errors.append(f"AutoTokenizer (default): {e!r}")
        if "object has no attribute 'keys'" in str(e):
            try:
                return AutoTokenizer.from_pretrained(
                    model_path, extra_special_tokens={}, **_LOCAL_KW
                )
            except Exception as e2:
                errors.append(f"AutoTokenizer (extra_special_tokens={{}}): {e2!r}")
        else:
            raise
    except Exception as e:
        errors.append(f"AutoTokenizer (default): {e!r}")
    try:
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast.from_pretrained(model_path, **_LOCAL_KW)
    except Exception as e:
        errors.append(f"PreTrainedTokenizerFast.from_pretrained: {e!r}")
    tj = Path(model_path) / "tokenizer.json"
    if tj.is_file():
        try:
            from transformers import PreTrainedTokenizerFast

            return PreTrainedTokenizerFast(tokenizer_file=str(tj))
        except Exception as e:
            errors.append(f"PreTrainedTokenizerFast(tokenizer_file=...): {e!r}")
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=False, **_LOCAL_KW)
    except Exception as e:
        errors.append(f"AutoTokenizer (use_fast=False): {e!r}")
    raise RuntimeError(
        f"Could not load tokenizer from {model_path}.\n" + "\n".join(errors)
    ) from None


def _assert_local_model_dir(model_path: Path) -> None:
    if not model_path.is_dir():
        raise SystemExit(
            f"Model weights not found: {model_path}\n"
            f"  Copy checkpoints into {MODEL_ROOT}/<ModelDir>/ "
            f"(see ifeval/config/models.json). "
            f"This repo does not download weights from Hugging Face."
        )
    cfg = model_path / "config.json"
    if not cfg.is_file():
        raise SystemExit(
            f"Invalid model directory (missing config.json): {model_path}\n"
            f"  Put a local Hugging Face checkpoint under {MODEL_ROOT}/."
        )


def get_model_and_tokenizer(model_name: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    paths = _load_model_paths()
    if model_name not in paths:
        raise SystemExit(
            f"Unknown model {model_name!r}. Add it to config/models.json "
            f"(see config/models.json.example). Expected weights under {MODEL_ROOT}/."
        )
    model_path = paths[model_name]
    mp = Path(model_path)
    _assert_local_model_dir(mp)
    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "left"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device_map = "cuda" if (visible and "," not in visible) else "auto"
    config = AutoConfig.from_pretrained(model_path, **_LOCAL_KW)
    if hasattr(config, "seq_length") and not hasattr(config, "max_length"):
        config.max_length = int(getattr(config, "seq_length"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
        **_LOCAL_KW,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    max_length: int,
) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    w = input_ids.shape[1]
    return [
        tokenizer.decode(out[i, w:], skip_special_tokens=True).strip()
        for i in range(out.shape[0])
    ]


@torch.inference_mode()
def generate_batch_with_steer(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    proj_by_layer: dict[int, np.ndarray],
    alpha: float,
    *,
    max_new_tokens: int,
    max_length: int,
) -> list[str]:
    layers = model.model.layers  # type: ignore[attr-defined]
    hooks = []
    try:
        for li, vec in proj_by_layer.items():
            if 0 <= li < len(layers):
                hooks.append(
                    layers[li].register_forward_hook(_layer_hook_with_vector(vec, alpha))
                )
        return generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )
    finally:
        for h in hooks:
            h.remove()
