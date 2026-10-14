#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST_JSON = ROOT / "output" / "internal_aware_data" / "llama3-8b-it" / "NQ-Swap" / "internal_aware_data.json"
DEFAULT_NPZ = ROOT / "output" / "resid_hidden_state" / "llama3-8b-it" / "NQ-Swap" / "resid_hidden_state.npz"
DEFAULT_OUT_DIR = ROOT / "output" / "sample-wise"

MODEL_PATHS: dict[str, str] = {
    "llama3-8b-it": "<DATA_DIR>/model/Llama3-8b-it",
    "gemma2-9b-it": "<DATA_DIR>/model/Gemma2-9b-it",
    "qwen3-8b": "<DATA_DIR>/model/Qwen3-8B",
    "mistral-7b-v0.1": "<DATA_DIR>/model/Mistral-7B-v0.1",
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 8192


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _answer_matches(generated: str, gold_answers: list[str]) -> bool:
    g = _norm(generated)
    if not g:
        return False
    normalized = [t for t in (_norm(a) for a in gold_answers) if len(t) >= 2]
    return any(t in g for t in normalized)


def _distractor_in_text(generated: str, distracted_token: str) -> bool:
    g = _norm(generated)
    if not g:
        return False
    t = _norm(str(distracted_token or "").strip())
    if not t:
        return False
    return t in g


def _stage2_label_from_strings(raw_text: str, gold_answers: list[str], distracted_tokens: list[str]) -> str:
    if _answer_matches(raw_text, gold_answers):
        return "correct"
    if any(_distractor_in_text(raw_text, d) for d in distracted_tokens if str(d).strip()):
        return "wrong"
    return "skipped"


def _truncate_for_judge(s: str) -> str:
    if not s:
        return s
    m = re.search(r"[。！？.!?]", s)
    return s[: m.end()].rstrip() if m else s


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
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        errors.append(f"AutoTokenizer (use_fast=False): {e!r}")
    raise RuntimeError(f"Could not load tokenizer from {model_path}.\n" + "\n".join(errors))


def _get_model_and_tokenizer(model_name: str):
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
def _generate_single(model, tokenizer, prompt: str) -> str:
    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TOKENIZER_MAX_LEN,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    w = input_ids.shape[1]
    return tokenizer.decode(out[0, w:], skip_special_tokens=True).strip()


class _ReplaceLastPromptTokenHook:
    def __init__(self, vec: np.ndarray):
        self.vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        self.used = False

    def __call__(self, _m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return out
        # Apply once on prompt pass only (seq_len > 1).
        if self.used or hs.shape[1] <= 1:
            return out
        v = torch.tensor(self.vec, dtype=hs.dtype, device=hs.device)
        if v.numel() != hs.shape[-1]:
            return out
        new_hs = hs.clone()
        new_hs[:, -1, :] = v.view(1, -1)
        self.used = True
        return (new_hs,) + out[1:] if isinstance(out, tuple) else new_hs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test only manifest wrong samples; replace one layer residual with per-sample clean residual."
    )
    ap.add_argument("--model", default="llama3-8b-it")
    ap.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--gpu", type=str, default="2")
    ap.add_argument("--layer", type=int, default=10, help="Layer to replace.")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all wrong samples.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    test_json = args.test_json if args.test_json.is_absolute() else (ROOT / args.test_json)
    npz_path = args.npz if args.npz.is_absolute() else (ROOT / args.npz)
    test_json = test_json.resolve()
    npz_path = npz_path.resolve()
    if not test_json.is_file():
        raise SystemExit(f"test json not found: {test_json}")
    if not npz_path.is_file():
        raise SystemExit(f"npz not found: {npz_path}")

    manifest = json.loads(test_json.read_text(encoding="utf-8"))
    samples = list(manifest.get("samples", []))
    wrong_samples = [s for s in samples if str(s.get("stage2_label", "")).strip().lower() == "wrong"]
    if int(args.max_samples) > 0:
        wrong_samples = wrong_samples[: int(args.max_samples)]
    if not wrong_samples:
        raise SystemExit("no wrong samples in manifest")

    z = np.load(npz_path, allow_pickle=True)
    if "resid_act_clean" not in z.files or "line_idx" not in z.files:
        z.close()
        raise SystemExit("npz missing resid_act_clean or line_idx")
    rc = z["resid_act_clean"]
    li_arr = np.asarray(z["line_idx"], dtype=np.int64)
    z.close()
    idx_map = {int(li_arr[i]): i for i in range(len(li_arr))}

    model, tokenizer = _get_model_and_tokenizer(str(args.model))
    layers = model.model.layers  # type: ignore[attr-defined]
    li = int(args.layer)
    if li < 0 or li >= len(layers):
        raise SystemExit(f"invalid --layer={li}; model has {len(layers)} layers")

    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = str(manifest.get("dataset", "unknown")).strip().replace(" ", "_")
    cat = str(manifest.get("category", "unknown")).strip().replace(" ", "_")
    out_path = out_dir / f"replace_per_sample_{ds}_{cat}_L{li}.json"

    base_rows: list[dict[str, Any]] = []
    if out_path.is_file():
        try:
            old_obj = json.loads(out_path.read_text(encoding="utf-8"))
            old_rows = old_obj.get("samples", [])
            if isinstance(old_rows, list):
                base_rows = [r for r in old_rows if isinstance(r, dict)]
        except Exception:
            base_rows = []

    def _write_snapshot() -> None:
        all_rows = base_rows + out_rows
        summary = {
            "total_wrong_input": len(wrong_samples),
            "previous_rows": len(base_rows),
            "current_run_rows": len(out_rows),
            "processed_rows": len(all_rows),
            "evaluated_ok": sum(1 for r in all_rows if r.get("status") == "ok"),
            "clean_matches_gold": sum(
                1 for r in all_rows if isinstance(r.get("clean_run"), dict) and r["clean_run"].get("matches_gold") is True
            ),
            "conflict_correct": sum(
                1 for r in all_rows if isinstance(r.get("conflict_run"), dict) and r["conflict_run"].get("label") == "correct"
            ),
            "conflict_wrong": sum(
                1 for r in all_rows if isinstance(r.get("conflict_run"), dict) and r["conflict_run"].get("label") == "wrong"
            ),
            "conflict_skipped": sum(
                1 for r in all_rows if isinstance(r.get("conflict_run"), dict) and r["conflict_run"].get("label") == "skipped"
            ),
        }
        out = {
            "model": str(args.model),
            "test_json": str(test_json),
            "npz": str(npz_path),
            "layer": li,
            "summary": summary,
            "samples": all_rows,
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    out_rows: list[dict[str, Any]] = []
    _write_snapshot()
    n_done = 0
    for s in wrong_samples:
        line_idx = int(s.get("line_idx", -1))
        if line_idx not in idx_map:
            out_rows.append({"line_idx": line_idx, "status": "missing_npz_line_idx"})
            _write_snapshot()
            continue
        ridx = idx_map[line_idx]
        arr = np.asarray(rc[ridx], dtype=np.float64)
        if arr.ndim != 2 or li >= arr.shape[0]:
            out_rows.append({"line_idx": line_idx, "status": "invalid_resid_shape"})
            _write_snapshot()
            continue
        vec = arr[li].astype(np.float32)

        gold = [str(x) for x in s.get("gold_answers", [])]
        dlist = s.get("distracted_tokens")
        if not isinstance(dlist, list):
            dlist = [str(s.get("distracted_token", ""))]
        dlist = [str(x) for x in dlist]

        # Run clean prompt with replacement.
        clean_prompt = str(s.get("clean_prompt_input", ""))
        hook_obj = _ReplaceLastPromptTokenHook(vec)
        handle = layers[li].register_forward_hook(hook_obj)
        try:
            raw_clean = _generate_single(model, tokenizer, clean_prompt)
        finally:
            handle.remove()
        clean_match = _answer_matches(raw_clean, gold)
        clean_label = "correct" if clean_match else "not_match_gold"

        # Run conflict prompt with replacement.
        conflict_prompt = str(s.get("conflict_prompt_input", ""))
        hook_obj = _ReplaceLastPromptTokenHook(vec)
        handle = layers[li].register_forward_hook(hook_obj)
        try:
            raw_conflict = _generate_single(model, tokenizer, conflict_prompt)
        finally:
            handle.remove()
        conflict_label = _stage2_label_from_strings(raw_conflict, gold, dlist)

        out_rows.append(
            {
                "line_idx": line_idx,
                "orig_stage2_label": str(s.get("stage2_label", "")),
                "clean_run": {
                    "input_prompt": clean_prompt,
                    "output_raw": raw_clean,
                    "output_judged": _truncate_for_judge(raw_clean),
                    "label": clean_label,
                    "matches_gold": bool(clean_match),
                },
                "conflict_run": {
                    "input_prompt": conflict_prompt,
                    "output_raw": raw_conflict,
                    "output_judged": _truncate_for_judge(raw_conflict),
                    "label": conflict_label,
                },
                "status": "ok",
            }
        )
        _write_snapshot()
        n_done += 1
        if n_done % 10 == 0 or n_done == len(wrong_samples):
            print(f"[progress] {n_done}/{len(wrong_samples)}", flush=True)
    _write_snapshot()
    print(str(out_path.relative_to(ROOT) if str(out_path).startswith(str(ROOT)) else out_path))


if __name__ == "__main__":
    main()

