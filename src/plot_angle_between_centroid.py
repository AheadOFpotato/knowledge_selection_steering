#!/usr/bin/env python3
"""Plot centroid cosine similarity vs layer from angle_analyse JSON files.

For each ``angle_*.json`` under ``--angle-root`` (default: angle_analyse/),
save a figure under ``--out-root`` (default: angle_fig/) mirroring
``<model>/<dataset>/<category>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import ROOT
DEFAULT_ANGLE_ROOT = ROOT / "output" / "angle_analyse"
DEFAULT_OUT_ROOT = ROOT / "output" / "fig" / "angle"
DEFAULT_DATASET = "NQ-Swap"
DEFAULT_MODELS = ("llama3-8b-it", "qwen3-8b", "gemma2-9b-it", "yi-6b-chat")

MODEL_ALIASES: dict[str, str] = {
    "yi": "yi-6b-chat",
    "yi-6b": "yi-6b-chat",
}


def _setup_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _layers_and_cosine_sim(obj: dict[str, Any]) -> tuple[list[int], list[float]]:
    rows = list(obj.get("resid") or [])
    layers: list[int] = []
    cos_sims: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        li = r.get("layer")
        ang = r.get("between_centroid_deg")
        if li is None or ang is None:
            continue
        try:
            layers.append(int(li))
            cos_sims.append(float(np.cos(np.deg2rad(float(ang)))))
        except (TypeError, ValueError):
            continue
    if not layers:
        return [], []
    order = np.argsort(layers)
    layers_s = [layers[i] for i in order]
    cos_sims_s = [cos_sims[i] for i in order]
    return layers_s, cos_sims_s


def _resolve_model(model: str) -> str:
    m = str(model).strip()
    return MODEL_ALIASES.get(m, m)


def _normalize_models(models: list[str]) -> list[str]:
    resolved = [_resolve_model(m) for m in models if str(m).strip()]
    if "yi-6b-chat" in resolved:
        resolved = [m for m in resolved if m != "yi-6b"]
    out: list[str] = []
    seen: set[str] = set()
    for m in resolved:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _out_path_for_json(angle_json: Path, angle_root: Path, out_root: Path) -> Path:
    rel = angle_json.resolve().relative_to(angle_root.resolve())
    # rel: model/dataset/category/angle_*.json
    return out_root / rel.parent / f"{angle_json.stem}.pdf"


def _meta_from_json(angle_json: Path, angle_root: Path) -> tuple[str, str, str, str]:
    """Return (model, dataset, category, stem). Prefer layout under angle_analyse/<model>/<dataset>/..."""
    aj = angle_json.resolve()
    for root in (DEFAULT_ANGLE_ROOT.resolve(), angle_root.resolve()):
        try:
            rel = aj.relative_to(root)
            parts = rel.parts
            if len(parts) >= 4:
                return parts[0], parts[1], parts[2], aj.stem
            if len(parts) == 3:
                # angle_root = angle_analyse/<model> → rel = <dataset>/<category>/angle_*.json
                model = root.name if root.parent == DEFAULT_ANGLE_ROOT.resolve() else parts[0]
                dataset = parts[0] if root.parent == DEFAULT_ANGLE_ROOT.resolve() else parts[1]
                category = parts[1] if root.parent == DEFAULT_ANGLE_ROOT.resolve() else parts[2]
                return model, dataset, category, aj.stem
        except ValueError:
            continue
    return "unknown", "unknown", aj.parent.name, aj.stem


def plot_one(angle_json: Path, out_path: Path, angle_root: Path) -> bool:
    _setup_fonts()
    obj = json.loads(angle_json.read_text(encoding="utf-8"))
    layers, cos_sims = _layers_and_cosine_sim(obj)
    if not layers:
        print(f"[skip] no layer data: {angle_json}", file=sys.stderr)
        return False

    model, _dataset, _category, _stem = _meta_from_json(angle_json, angle_root)

    fs_label = 28
    fs_tick = 26
    fs_title = 28

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.plot(layers, cos_sims, "-o", markersize=6, linewidth=1.8, color="#2563eb")
    ax.set_xlabel("Layer", fontsize=fs_label)
    ax.set_ylabel("Cosine similarity", fontsize=fs_label)
    ax.set_title(model, fontsize=fs_title)
    ax.tick_params(labelsize=fs_tick)
    ax.grid(True, alpha=0.3)
    if len(layers) > 1:
        ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ymax = max(v for v in cos_sims if np.isfinite(v))
    ymin = min(v for v in cos_sims if np.isfinite(v))
    if np.isfinite(ymax) and np.isfinite(ymin):
        pad = max(0.02, (ymax - ymin) * 0.08)
        ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return True


def _iter_angle_jsons(
    angle_root: Path,
    *,
    models: list[str],
    dataset: str,
) -> list[Path]:
    paths: list[Path] = []
    for model in models:
        dataset_dir = angle_root / model / dataset
        if not dataset_dir.is_dir():
            print(f"[warn] missing: {dataset_dir}", file=sys.stderr)
            continue
        paths.extend(sorted(dataset_dir.rglob("angle_*.json")))
    return paths


def _display_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot centroid cosine similarity vs layer from angle_analyse JSON."
    )
    ap.add_argument(
        "--angle-root",
        type=Path,
        default=DEFAULT_ANGLE_ROOT,
        help="Root of angle JSON tree (model/dataset/category/).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output figure root (mirrors model/dataset/category/).",
    )
    ap.add_argument(
        "--angle-json",
        type=Path,
        default=None,
        help="Single JSON file (if set, only plot this file).",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Models to plot (default: 4 models).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset subdir under each model (default: NQ-Swap).",
    )
    ap.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help="Output format (default: pdf).",
    )
    args = ap.parse_args()

    angle_root = args.angle_root if args.angle_root.is_absolute() else (ROOT / args.angle_root)
    angle_root = angle_root.resolve()
    out_root = args.out_root if args.out_root.is_absolute() else (ROOT / args.out_root)
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    models = _normalize_models(list(args.models))
    dataset = str(args.dataset).strip()
    ext = args.format

    if args.angle_json is not None:
        j = args.angle_json if args.angle_json.is_absolute() else (ROOT / args.angle_json)
        j = j.resolve()
        if not j.is_file():
            raise SystemExit(f"not found: {j}")
        try:
            out_path = _out_path_for_json(j, angle_root, out_root).with_suffix(f".{ext}")
        except ValueError:
            out_path = out_root / f"{j.stem}.{ext}"
        if plot_one(j, out_path, angle_root):
            print(_display_path(out_path))
        return

    if not angle_root.is_dir():
        raise SystemExit(f"angle root not found: {angle_root}")
    if not models:
        raise SystemExit("no models specified")

    n_ok = 0
    for jpath in _iter_angle_jsons(angle_root, models=models, dataset=dataset):
        out_path = _out_path_for_json(jpath, angle_root, out_root).with_suffix(f".{ext}")
        if plot_one(jpath, out_path, angle_root):
            print(_display_path(out_path))
            n_ok += 1

    if n_ok == 0:
        raise SystemExit(f"no figures produced under {angle_root} for models={models} dataset={dataset}")
    print(f"[done] {n_ok} figures -> {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
