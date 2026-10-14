#!/usr/bin/env python3
"""IFEval: generate + strict/loose eval; optional residual-stream alpha ablation.

Vendored ``instruction_following_eval`` lives under ``vendor/``. Model weights load
from repo-root ``models/`` via ``config/models.json`` (local only, no HF download).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT / "src"))

import steer_eval_utils as ifg  # noqa: E402

DEFAULT_OUT_ROOT = BUNDLE_ROOT / "output" / "if_eval"
DEFAULT_IFEVAL_ROOT = BUNDLE_ROOT / "vendor"
DEFAULT_INPUT_DATA = (
    DEFAULT_IFEVAL_ROOT / "instruction_following_eval" / "data" / "input_data.jsonl"
)


def _prime_nltk_for_ifeval() -> None:
    """Newer NLTK needs ``punkt_tab`` for ``word_tokenize`` / ``sent_tokenize`` (IFEval instructions)."""
    import zipfile

    nd = BUNDLE_ROOT / ".nltk_data"
    nd.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(nd)
    import nltk  # noqa: WPS433

    nds = str(nd)
    if nds not in nltk.data.path:
        nltk.data.path.insert(0, nds)

    specs: list[tuple[str, str]] = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]
    for download_name, find_spec in specs:
        for _ in range(4):
            try:
                nltk.data.find(find_spec)
                break
            except LookupError:
                nltk.download(download_name, download_dir=nds, quiet=True, force=True)
            except (zipfile.BadZipFile, OSError, EOFError, KeyError, ValueError):
                nltk.download(download_name, download_dir=nds, quiet=True, force=True)
        else:  # pragma: no cover
            raise RuntimeError(f"NLTK resource {download_name!r} unavailable after retries.")


def _safe(s: str) -> str:
    return ifg._safe_name(s)


def _default_run_name() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_utc")


def _parse_keys(raw: str) -> list[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _split_ifeval_rows(
    rows: list[dict[str, Any]],
    *,
    cross_val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out val (alpha grid) and test (final eval); default 50/50 for cross-dataset protocol."""
    n = len(rows)
    if n < 2:
        raise SystemExit(f"need at least 2 IFEval rows for val/test split, got {n}")
    rng = random.Random(int(seed))
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val = max(1, int(round(n * float(cross_val_ratio))))
    if n_val >= n:
        n_val = max(1, n // 2)
    val_idxs = sorted(idxs[:n_val])
    test_idxs = sorted(idxs[n_val:])
    if not test_idxs:
        test_idxs = [val_idxs.pop()]
    return [rows[i] for i in val_idxs], [rows[i] for i in test_idxs]


def _rows_to_inputs(ev: Any, rows: list[dict[str, Any]], work_jsonl: Path) -> list[Any]:
    _rows_to_jsonl(work_jsonl, rows)
    inputs = ev.read_prompt_list(str(work_jsonl))
    if not inputs:
        raise SystemExit("IFEval read_prompt_list returned empty")
    return inputs


def _pick_best_alpha(manifests: list[dict[str, Any]]) -> float:
    if not manifests:
        raise SystemExit("no alpha results to pick best alpha from")
    best = max(manifests, key=lambda m: float(m["prompt_level_strict_acc"]))
    return float(best["alpha"])


def _generate_and_eval(
    ev: Any,
    model,
    tokenizer,
    inputs: list[Any],
    *,
    out_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    max_length: int,
    quiet_samples: bool,
    gpu: str,
    extra_manifest: dict[str, Any],
    proj: dict[int, Any] | None = None,
    alpha: float = 0.0,
) -> dict[str, Any]:
    n = len(inputs)
    all_texts: list[str] = []
    done = 0
    for st in range(0, n, int(batch_size)):
        chunk = inputs[st : st + int(batch_size)]
        prompts = [str(x.prompt) for x in chunk]
        if proj is None or float(alpha) == 0.0:
            chunk_texts = ifg.generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
        else:
            chunk_texts = ifg.generate_batch_with_steer(
                model,
                tokenizer,
                prompts,
                proj,
                float(alpha),
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
        for j, inp in enumerate(chunk):
            idx = st + j + 1
            all_texts.append(chunk_texts[j])
            if not quiet_samples:
                tag = f"α={alpha:g} " if proj is not None else ""
                print(
                    f"[ifeval] gen {tag}{idx}/{n} key={inp.key!r} "
                    f"preview={ifg._one_line_preview(chunk_texts[j])!r}",
                    flush=True,
                )
        if quiet_samples:
            done += len(chunk)
            if done % ifg.PROGRESS_EVERY == 0 or done == n:
                print(f"  generate {done}/{n}", flush=True)
    prompt_to_response = {str(inp.prompt): all_texts[i] for i, inp in enumerate(inputs)}
    return _run_eval_write(
        ev,
        inputs,
        prompt_to_response,
        out_dir,
        responses_path=out_dir / "responses.jsonl",
        extra_manifest=extra_manifest,
        gpu=gpu,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        quiet_samples=quiet_samples,
    )


def _load_ifeval_module(ifeval_root: Path):
    root = ifeval_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"--ifeval-root is not a directory: {root}")
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    try:
        from instruction_following_eval import evaluation_lib as ev  # noqa: WPS433
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Cannot import IFEval from {root} (need PYTHONPATH parent of package). "
            f"Install deps (PyPI uses absl-py, not absl):\n"
            f"  pip install -r {root}/instruction_following_eval/requirements.txt\n"
            f"  pip install -r {BUNDLE_ROOT}/requirements.txt\n"
            f"Original error: {e}"
        ) from e
    return ev


def _ifeval_out_dir(out_root_run: Path, model_key: str, category_stem: str) -> Path:
    mod = _safe(model_key)
    cat = _safe(category_stem)
    d = out_root_run / mod / "IFEval" / cat
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rows_to_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        for r in rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")


def _filter_by_keys(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    ks = {str(k) for k in keys}
    return [r for r in rows if str(r.get("key")) in ks]


def _per_instruction_id_stats(outputs: list[Any]) -> dict[str, dict[str, float | int]]:
    import collections

    tot: dict[str, int] = collections.defaultdict(int)
    ok: dict[str, int] = collections.defaultdict(int)
    for o in outputs:
        for iid, passed in zip(o.instruction_id_list, o.follow_instruction_list):
            tot[iid] += 1
            if passed:
                ok[iid] += 1
    out: dict[str, dict[str, float | int]] = {}
    for iid in sorted(tot.keys()):
        t = tot[iid]
        c = ok[iid]
        out[iid] = {"count": int(t), "correct": int(c), "acc": float(c) / max(1, t)}
    return out


def _run_eval_write(
    ev: Any,
    inputs: list[Any],
    prompt_to_response: dict[str, str],
    out_dir: Path,
    *,
    responses_path: Path,
    extra_manifest: dict[str, Any],
    gpu: str,
    max_new_tokens: int,
    batch_size: int,
    quiet_samples: bool,
) -> dict[str, Any]:
    inputs = copy.deepcopy(inputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_outputs = []
    loose_outputs = []
    combined_rows: list[dict[str, Any]] = []

    for i, inp in enumerate(inputs, start=1):
        p = str(inp.prompt)
        single = {p: prompt_to_response.get(p, "")}
        so = ev.test_instruction_following_strict(inp, single)
        lo = ev.test_instruction_following_loose(inp, single)
        strict_outputs.append(so)
        loose_outputs.append(lo)
        if not quiet_samples:
            print(
                f"[ifeval] eval {i}/{len(inputs)} key={inp.key!r} "
                f"strict_all={so.follow_all_instructions} loose_all={lo.follow_all_instructions}",
                flush=True,
            )
        combined_rows.append(
            {
                "key": inp.key,
                "prompt": p,
                "response": single[p],
                "strict_follow_all": so.follow_all_instructions,
                "strict_follow_list": list(so.follow_instruction_list),
                "loose_follow_all": lo.follow_all_instructions,
                "loose_follow_list": list(lo.follow_instruction_list),
                "instruction_id_list": inp.instruction_id_list,
            }
        )

    strict_path = out_dir / "eval_results_strict.jsonl"
    loose_path = out_dir / "eval_results_loose.jsonl"
    combined_path = out_dir / "eval_per_sample_strict_loose.jsonl"
    ev.write_outputs(str(strict_path), strict_outputs)
    ev.write_outputs(str(loose_path), loose_outputs)
    with combined_path.open("w", encoding="utf-8") as wf:
        for row in combined_rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    resp_rows = [{"prompt": str(inp.prompt), "response": prompt_to_response[str(inp.prompt)]} for inp in inputs]
    with responses_path.open("w", encoding="utf-8") as wf:
        for row in resp_rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 64)
    print(f"STRICT ({out_dir.name})")
    ev.print_report(strict_outputs)
    print("=" * 64)
    print(f"LOOSE ({out_dir.name})")
    ev.print_report(loose_outputs)

    n_out = len(strict_outputs)
    strict_ok = sum(1 for o in strict_outputs if o.follow_all_instructions)
    loose_ok = sum(1 for o in loose_outputs if o.follow_all_instructions)
    strict_slots = sum(len(o.instruction_id_list) for o in strict_outputs)
    strict_slot_ok = sum(sum(o.follow_instruction_list) for o in strict_outputs)
    loose_slot_ok = sum(sum(o.follow_instruction_list) for o in loose_outputs)

    manifest = {
        **extra_manifest,
        "out_dir": ifg._display_path(out_dir),
        "dataset_label": "IFEval",
        "num_samples": n_out,
        "prompt_level_strict_acc": strict_ok / max(1, n_out),
        "prompt_level_loose_acc": loose_ok / max(1, n_out),
        "instruction_level_strict_acc": strict_slot_ok / max(1, strict_slots),
        "instruction_level_loose_acc": loose_slot_ok / max(1, strict_slots),
        "strict_by_instruction_id": _per_instruction_id_stats(strict_outputs),
        "loose_by_instruction_id": _per_instruction_id_stats(loose_outputs),
        "responses_jsonl": ifg._display_path(responses_path),
        "eval_results_strict": ifg._display_path(strict_path),
        "eval_results_loose": ifg._display_path(loose_path),
        "eval_per_sample": ifg._display_path(combined_path),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "gpu": gpu,
    }
    summary_path = out_dir / "ifeval_run_summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {strict_path}\n{loose_path}\n{combined_path}\n{summary_path}", flush=True)
    return manifest


def _alpha_run_dirname(layers_arg: str, alpha: float) -> str:
    ls = str(layers_arg).replace(",", "_").replace(" ", "")
    a_str = f"{float(alpha):g}".replace(".", "p")
    return f"L{ls}_alpha{a_str}"


def _metrics_slice(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha": manifest.get("alpha"),
        "out_dir": manifest.get("out_dir"),
        "num_samples": manifest.get("num_samples"),
        "prompt_level_strict_acc": manifest.get("prompt_level_strict_acc"),
        "prompt_level_loose_acc": manifest.get("prompt_level_loose_acc"),
        "instruction_level_strict_acc": manifest.get("instruction_level_strict_acc"),
        "instruction_level_loose_acc": manifest.get("instruction_level_loose_acc"),
        "ablation": manifest.get("ablation"),
        "direction_npz": manifest.get("direction_npz"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="IFEval + optional resid ablation; outputs under ifeval/output/if_eval/."
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--gpu", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--run-name", type=str, default="", help="Subdir under --out-root; default UTC timestamp.")
    ap.add_argument(
        "--ifeval-root",
        type=Path,
        default=DEFAULT_IFEVAL_ROOT,
        help="Vendor root (parent of instruction_following_eval/).",
    )
    ap.add_argument(
        "--input-data",
        type=Path,
        default=None,
        help="IFEval input_data.jsonl; default: <ifeval-root>/instruction_following_eval/data/input_data.jsonl",
    )
    ap.add_argument("--keys", type=str, default="", help="Comma-separated keys to keep (e.g. 1000,1001).")
    ap.add_argument("--max-samples", type=int, default=0, help="Keep first N after filter (0=all).")
    ap.add_argument("--category", type=str, default="", help="Override output category stem.")
    ap.add_argument("--max-new-tokens", type=int, default=ifg.MAX_NEW_TOKENS)
    ap.add_argument("--max-length", type=int, default=ifg.TOKENIZER_MAX_LEN)
    ap.add_argument("--quiet-samples", action="store_true")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--alpha-start", type=float, default=0.0)
    ap.add_argument("--alpha-end", type=float, default=0.0)
    ap.add_argument("--alpha-step", type=float, default=0.1)
    ap.add_argument(
        "--resid-direction",
        type=str,
        default="clean_minus_conflict",
        choices=("clean_minus_conflict", "clean", "conflict"),
        help="NPZ steering vector: diff (clean-conflict), clean, or conflict mean only.",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help=(
            "Ablation: stop after N consecutive alphas where all early-stop metrics "
            "decrease vs previous alpha on the val split. 0 disables."
        ),
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for IFEval val/test split (ablation mode).",
    )
    ap.add_argument(
        "--cross-val-ratio",
        type=float,
        default=0.5,
        help="IFEval fraction for alpha grid; remainder is test (default 50/50).",
    )
    args = ap.parse_args()

    _prime_nltk_for_ifeval()
    ev = _load_ifeval_module(Path(args.ifeval_root))
    run_name = (args.run_name or "").strip() or _default_run_name()
    out_root_run = ifg._abspath(args.out_root) / _safe(run_name)
    out_root_run.mkdir(parents=True, exist_ok=True)

    ifeval_root = Path(args.ifeval_root).resolve()
    input_path = (
        Path(args.input_data).resolve()
        if args.input_data is not None
        else (ifeval_root / "instruction_following_eval" / "data" / "input_data.jsonl").resolve()
    )
    if not input_path.is_file():
        raise SystemExit(f"input_data not found: {input_path}")

    keys = _parse_keys(args.keys)
    rows = _load_jsonl_rows(input_path)
    if keys:
        rows = _filter_by_keys(rows, keys)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise SystemExit("no rows after --keys / --max-samples")

    input_stem = input_path.stem
    if str(args.category or "").strip():
        category_stem = _safe(str(args.category).strip())
    elif keys:
        category_stem = f"{_safe(input_stem)}__keys_{'_'.join(_safe(k) for k in keys[:8])}"
    else:
        category_stem = _safe(input_stem)

    model_key = str(args.model).strip()
    use_npz = args.npz is not None
    if use_npz and not str(args.layers).strip():
        raise SystemExit("--layers is required with --npz")

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    if use_npz:
        val_rows, test_rows = _split_ifeval_rows(
            rows,
            cross_val_ratio=float(args.cross_val_ratio),
            seed=int(args.split_seed),
        )
        work_jsonl = out_root_run / "_ifeval_inputs_slice.jsonl"
        val_inputs = _rows_to_inputs(ev, val_rows, work_jsonl)
        test_inputs = _rows_to_inputs(ev, test_rows, out_root_run / "_ifeval_test_slice.jsonl")
        inputs = val_inputs
        print(
            f"[ifeval ablation] alpha grid on val: n={len(val_inputs)} "
            f"(split_seed={args.split_seed}, cross_val_ratio={args.cross_val_ratio})",
            flush=True,
        )
        print(
            f"[ifeval ablation] final test split: n={len(test_inputs)}",
            flush=True,
        )
    else:
        work_jsonl = out_root_run / "_ifeval_inputs_slice.jsonl"
        inputs = _rows_to_inputs(ev, rows, work_jsonl)

    extra_base: dict[str, Any] = {
        "model": model_key,
        "category": category_stem,
        "input_data_jsonl": ifg._display_path(input_path),
        "work_jsonl": ifg._display_path(work_jsonl),
        "ifeval_root": ifg._display_path(ifeval_root),
        "run_name": run_name,
        "keys": keys or None,
        "script": "run_ifeval_prompt_ablation.py",
    }
    if use_npz:
        extra_base.update(
            {
                "split_seed": int(args.split_seed),
                "cross_val_ratio": float(args.cross_val_ratio),
                "num_val": len(val_rows),
                "num_test": len(test_rows),
            }
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    model, tokenizer = ifg.get_model_and_tokenizer(model_key)
    quiet_samples = bool(args.quiet_samples)
    per_alpha: list[dict[str, Any]] = []
    test_manifest: dict[str, Any] | None = None

    if not use_npz:
        out_dir = _ifeval_out_dir(out_root_run, model_key, category_stem)
        m = _generate_and_eval(
            ev,
            model,
            tokenizer,
            inputs,
            out_dir=out_dir,
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_new_tokens),
            max_length=int(args.max_length),
            quiet_samples=quiet_samples,
            gpu=str(args.gpu).strip(),
            extra_manifest={**extra_base, "ablation": False, "alpha": 0.0},
        )
        per_alpha.append(m)
    else:
        npz_path = ifg._abspath(args.npz)
        if not npz_path.is_file():
            raise SystemExit(f"--npz not found: {npz_path}")
        print(f"[ifeval ablation] direction NPZ (NQ-Swap full): {ifg._display_path(npz_path)}", flush=True)
        layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
        dirs = ifg._load_resid_vectors(npz_path, layers)
        used = [li for li in layers if li in dirs]
        if not used:
            raise SystemExit("no valid layers in NPZ")

        resid_mode = str(args.resid_direction).strip().lower()
        proj, vec_tag, steer_formula = ifg._resid_projection_vectors(dirs, used, resid_mode)
        layers_tag = str(args.layers).replace(",", "_").replace(" ", "")
        alphas = ifg._alphas_from_cli(float(args.alpha_start), float(args.alpha_end), float(args.alpha_step))

        base_out = _ifeval_out_dir(out_root_run, model_key, f"{category_stem}__split_val")
        ablation_root = base_out / f"ablation__{_safe(npz_path.stem)}__vec_{vec_tag}"
        ablation_root.mkdir(parents=True, exist_ok=True)
        print(f"[ifeval ablation] resid_direction={vec_tag} steer_formula={steer_formula}", flush=True)
        early_patience = int(args.early_stop_patience)
        if early_patience > 0:
            print(
                f"[ifeval ablation] early stop: halt after {early_patience} consecutive alphas "
                f"with all metrics down on val ({', '.join(ifg.EARLY_STOP_METRIC_KEYS)})",
                flush=True,
            )
        consecutive_all_down = 0
        prev_metrics: tuple[float, ...] | None = None
        val_alpha_manifests: list[dict[str, Any]] = []

        for ai, alpha in enumerate(alphas):
            run_dir = ablation_root / ifg._alpha_run_dirname(str(args.layers), alpha, vec_tag)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ifeval ablation] val alpha {ai + 1}/{len(alphas)} alpha={alpha:g} -> {run_dir}", flush=True)
            m = _generate_and_eval(
                ev,
                model,
                tokenizer,
                val_inputs,
                out_dir=run_dir,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
                max_length=int(args.max_length),
                quiet_samples=quiet_samples,
                gpu=str(args.gpu).strip(),
                extra_manifest={
                    **extra_base,
                    "ablation": True,
                    "eval_split": "val",
                    "direction_npz": ifg._display_path(npz_path),
                    "layers": layers_tag,
                    "alpha": float(alpha),
                    "alpha_grid": alphas,
                    "resid_direction": vec_tag,
                    "steer_formula": steer_formula,
                },
                proj=proj,
                alpha=float(alpha),
            )
            per_alpha.append(m)
            val_alpha_manifests.append(m)

            if early_patience > 0 and prev_metrics is not None:
                curr_metrics = ifg._early_stop_metric_vector(m)
                if ifg._all_early_stop_metrics_decreased(prev_metrics, curr_metrics):
                    consecutive_all_down += 1
                    print(
                        f"[ifeval ablation] early stop: {consecutive_all_down}/{early_patience} "
                        f"consecutive alpha(s) with all val metrics decreased",
                        flush=True,
                    )
                    if consecutive_all_down >= early_patience:
                        print(
                            f"[ifeval ablation] early stop triggered after alpha={alpha:g}",
                            flush=True,
                        )
                        break
                else:
                    consecutive_all_down = 0
                prev_metrics = ifg._early_stop_metric_vector(m)
            elif early_patience > 0:
                prev_metrics = ifg._early_stop_metric_vector(m)

        best_alpha = _pick_best_alpha(val_alpha_manifests)
        print(f"[ifeval ablation] best alpha on val: {best_alpha:g}", flush=True)

        test_out = _ifeval_out_dir(out_root_run, model_key, f"{category_stem}__split_test")
        test_run_dir = test_out / f"best_alpha__{_safe(npz_path.stem)}__vec_{vec_tag}"
        test_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ifeval ablation] final test eval alpha={best_alpha:g} -> {test_run_dir}", flush=True)
        test_manifest = _generate_and_eval(
            ev,
            model,
            tokenizer,
            test_inputs,
            out_dir=test_run_dir,
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_new_tokens),
            max_length=int(args.max_length),
            quiet_samples=quiet_samples,
            gpu=str(args.gpu).strip(),
            extra_manifest={
                **extra_base,
                "ablation": True,
                "eval_split": "test",
                "best_alpha_from_val": best_alpha,
                "direction_npz": ifg._display_path(npz_path),
                "layers": layers_tag,
                "alpha": float(best_alpha),
                "resid_direction": vec_tag,
                "steer_formula": steer_formula,
            },
            proj=proj,
            alpha=float(best_alpha),
        )

        agg = ablation_root / "ifeval_ablation_all_alphas.json"
        agg_payload: Any = {
            "per_alpha_val": [_metrics_slice(m) for m in val_alpha_manifests],
            "best_alpha": best_alpha,
            "alpha_selection_metric": "prompt_level_strict_acc",
            "test_eval": _metrics_slice(test_manifest),
        }
        if early_patience > 0:
            agg_payload["early_stop"] = {
                "enabled": True,
                "patience": early_patience,
                "metric_keys": list(ifg.EARLY_STOP_METRIC_KEYS),
            }
        agg.write_text(json.dumps(agg_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote aggregate: {agg}", flush=True)

    summary = {
        "out_root_run": ifg._display_path(out_root_run),
        "model": model_key,
        "ifeval_root": ifg._display_path(ifeval_root),
        "input_data": ifg._display_path(input_path),
        "work_jsonl": ifg._display_path(work_jsonl),
        "category_stem": category_stem,
        "num_inputs": len(inputs) if not use_npz else len(val_rows) + len(test_rows),
        "npz": ifg._display_path(ifg._abspath(args.npz)) if use_npz else None,
        "per_alpha_metrics": [_metrics_slice(m) for m in per_alpha],
        "test_metrics": _metrics_slice(test_manifest) if test_manifest else None,
    }
    sp = out_root_run / "summary_ifeval.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {sp}", flush=True)


if __name__ == "__main__":
    main()
