"""Residual-stream steering utilities (self-contained, relative paths)."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BUNDLE_ROOT.parent
MODEL_ROOT = REPO_ROOT / "models"
TOKENIZER_MAX_LEN = 50000
MAX_NEW_TOKENS = 200
_LOCAL_KW = {"trust_remote_code": True, "local_files_only": True}


def bundle_root() -> Path:
    return BUNDLE_ROOT


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
        return AutoTokenizer.from_pretrained(
            model_path, use_fast=False, **_LOCAL_KW
        )
    except Exception as e:
        errors.append(f"AutoTokenizer (use_fast=False): {e!r}")
    raise RuntimeError(
        f"Could not load tokenizer from {model_path}.\n" + "\n".join(errors)
    ) from None


def _resolve_attn_implementation() -> str:
    """ATTN_IMPL env: sdpa (default) | flash_attention_2 | eager."""
    impl = os.environ.get("ATTN_IMPL", "sdpa").strip()
    allowed = ("sdpa", "flash_attention_2", "eager")
    if impl not in allowed:
        print(f"[warn] unknown ATTN_IMPL={impl!r}, using sdpa", file=sys.stderr)
        return "sdpa"
    return impl


def _disable_torch_compile_for_hooks() -> None:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo

        torch._dynamo.config.disable = True
    except Exception:
        pass


def get_model_and_tokenizer(model_name: str):
    _disable_torch_compile_for_hooks()
    paths = _load_model_paths()
    if model_name not in paths:
        raise SystemExit(
            f"Unknown model {model_name!r}. Add it to config/models.json "
            f"(see config/models.json.example). "
            f"Expected local weights under {MODEL_ROOT}/."
        )
    model_path = paths[model_name]
    mp = Path(model_path)
    if not mp.is_dir():
        raise SystemExit(
            f"Model weights not found: {mp}\n"
            f"  Copy local checkpoints into {MODEL_ROOT}/<ModelDir>/ "
            f"(see loogle/config/models.json). "
            f"This repo does not download weights from Hugging Face."
        )
    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "left"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device_map = "cuda" if (visible and "," not in visible) else "auto"
    config = AutoConfig.from_pretrained(model_path, **_LOCAL_KW)
    if hasattr(config, "seq_length") and not hasattr(config, "max_length"):
        config.max_length = int(getattr(config, "seq_length"))
    attn_impl = _resolve_attn_implementation()
    load_kw: dict[str, Any] = dict(
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_impl,
        **_LOCAL_KW,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
    except Exception as e:
        if attn_impl == "sdpa":
            raise
        print(
            f"[warn] ATTN_IMPL={attn_impl} failed ({e!s}); falling back to sdpa",
            file=sys.stderr,
        )
        load_kw["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
    print(f"[model] attn_implementation={load_kw['attn_implementation']}", file=sys.stderr)
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
    return [
        tokenizer.decode(out[i, w:], skip_special_tokens=True).strip()
        for i in range(input_ids.shape[0])
    ]


def _layer_hook_with_vector(v_mean: np.ndarray, alpha: float):
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
