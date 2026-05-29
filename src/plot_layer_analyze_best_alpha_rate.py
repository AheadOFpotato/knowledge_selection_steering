#!/usr/bin/env python3
"""Per-layer best-alpha internal/external rate deltas vs α=0 from layer_analyze JSON.

For each layer, scan all α in ``L<layer>_alpha*_steer_result.json``:

  - α > 0: pick α with highest internal rate (Stage2 ``correct`` / N)
  - α < 0: pick α with highest external rate (Stage2 ``wrong`` / N)
  - Plot Δrate = rate(best α) − rate(α=0) for each line

No Stage1 filtering — only Stage2 rates decide the best α.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from project_paths import ROOT, STEER_OUT_ROOT

DEFAULT_DATA_ROOT = STEER_OUT_ROOT
DEFAULT_OUT_ROOT = ROOT / "output" / "fig" / "layer_analyze"
DEFAULT_DATASET = "NQ-Swap"
DEFAULT_CATEGORY = (
    "dev__split_test__dir_dev__split_train__test_dev__split_test"
)

DEFAULT_MODELS = ("llama3-8b-it", "qwen3-8b", "gemma2-9b-it", "yi-6b-chat")

MODEL_ALIASES: dict[str, str] = {
    "yi": "yi-6b-chat",
    "yi-6b": "yi-6b-chat",
}

MODEL_LABELS: dict[str, str] = {
    "llama3-8b-it": "Llama-3-8B-IT",
    "qwen3-8b": "Qwen3-8B",
    "gemma2-9b-it": "Gemma2-9B-IT",
    "yi-6b-chat": "Yi-6B-Chat",
}

FNAME_PAT = re.compile(r"^L(\d+)_alpha(.+)_steer_result\.json$")


def _resolve_model(model: str) -> str:
    m = str(model).strip()
    return MODEL_ALIASES.get(m, m)


def _setup_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _alpha_from_slug(slug: str) -> float:
    s = str(slug).strip()
    if not s:
        return 0.0
    if s.startswith("-"):
        return -float(s[1:].replace("p", "."))
    return float(s.replace("p", "."))


def _per_sample_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    p = obj.get("per_sample_outputs_and_judgements") or {}
    rows = p.get("steer_clean_minus_conflict")
    if rows is None:
        rows = p.get("project_to_correct_mean_direction")
    return list(rows or [])


def _rates(obj: dict[str, Any]) -> tuple[float, float, int]:
    rows = _per_sample_rows(obj)
    n_total = len(rows)
    if n_total <= 0:
        return 0.0, 0.0, 0
    internal = external = 0
    for r in rows:
        lab = str(r.get("stage2_label") or "").strip().lower()
        if lab == "correct":
            internal += 1
        elif lab == "wrong":
            external += 1
    return internal / n_total, external / n_total, n_total


def _load_by_layer(steer_dir: Path) -> dict[int, list[tuple[float, dict[str, Any]]]]:
    by_layer: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for p in sorted(steer_dir.glob("L*_alpha*_steer_result.json")):
        m = FNAME_PAT.match(p.name)
        if not m:
            continue
        layer = int(m.group(1))
        alpha = _alpha_from_slug(m.group(2))
        obj = json.loads(p.read_text(encoding="utf-8"))
        by_layer.setdefault(layer, []).append((alpha, obj))
    return by_layer


def _alpha0_obj(
    entries: list[tuple[float, dict[str, Any]]],
) -> dict[str, Any] | None:
    for alpha, obj in entries:
        if abs(alpha) < 1e-9:
            return obj
    return None


def _best_per_layer(
    by_layer: dict[int, list[tuple[float, dict[str, Any]]]],
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    layers: list[int] = []
    y_internal: list[float] = []
    y_external: list[float] = []
    best_alpha_pos: list[float] = []
    best_alpha_neg: list[float] = []

    for layer in sorted(by_layer):
        entries = by_layer[layer]
        baseline_obj = _alpha0_obj(entries)
        if baseline_obj is None:
            continue
        base_int, base_ext, _ = _rates(baseline_obj)

        pos = [(a, obj) for a, obj in entries if a > 0]
        neg = [(a, obj) for a, obj in entries if a < 0]

        if not pos or not neg:
            continue

        best_pos_alpha, best_pos_obj = max(pos, key=lambda x: _rates(x[1])[0])
        best_neg_alpha, best_neg_obj = max(neg, key=lambda x: _rates(x[1])[1])
        int_rate, _, _ = _rates(best_pos_obj)
        _, ext_rate, _ = _rates(best_neg_obj)

        layers.append(layer)
        y_internal.append(int_rate - base_int)
        y_external.append(ext_rate - base_ext)
        best_alpha_pos.append(best_pos_alpha)
        best_alpha_neg.append(best_neg_alpha)

    return layers, y_internal, y_external, best_alpha_pos, best_alpha_neg


def _steer_dir(data_root: Path, model: str, dataset: str, category: str) -> Path:
    return data_root / model / dataset / category


def plot_one_model(
    steer_dir: Path,
    out_path: Path,
    *,
    model: str,
    dataset: str,
    dpi: int,
) -> bool:
    if not steer_dir.is_dir():
        print(f"[skip] not a directory: {steer_dir}", file=sys.stderr)
        return False

    by_layer = _load_by_layer(steer_dir)
    layers, y_int, y_ext, _, _ = _best_per_layer(by_layer)
    if not layers:
        print(f"[skip] no layer points: {steer_dir}", file=sys.stderr)
        return False

    label = MODEL_LABELS.get(model, model)
    _setup_fonts()
    fs_label, fs_tick, fs_title, fs_legend = 22, 20, 20, 14

    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.plot(layers, y_int, "-o", markersize=5, linewidth=1.6, color="#1f77b4", label="internal")
    ax.plot(layers, y_ext, "-o", markersize=5, linewidth=1.6, color="#d62728", label="external")
    ax.set_title(f"{label} / {dataset}", fontsize=fs_title)
    ax.set_xlabel("Layer", fontsize=fs_label)
    ax.set_ylabel("Δ rate vs α=0", fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    if len(layers) > 1:
        ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ymax = max(max(y_int), max(y_ext))
    ymin = min(min(y_int), min(y_ext))
    pad = max(0.02, (ymax - ymin) * 0.08)
    ax.set_ylim(ymin - pad, ymax + pad)

    legend_elements = [
        Line2D(
            [0], [0], color="#1f77b4", marker="o", markersize=7, linewidth=1.8, label="internal"
        ),
        Line2D(
            [0], [0], color="#d62728", marker="o", markersize=7, linewidth=1.8, label="external"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=fs_legend, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot per-layer best-alpha Δrates vs α=0 (one figure per model)."
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="layer_analyze root (<model>/<dataset>/<category>/).",
    )
    ap.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    ap.add_argument("--category", type=str, default=DEFAULT_CATEGORY)
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Models to plot (default: 4 models, each saved separately).",
    )
    ap.add_argument(
        "--steer-dir",
        type=Path,
        default=None,
        help="Single steer dir (if set, only plot this directory).",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--out-pdf", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    dataset = args.dataset
    category = args.category
    dpi = int(args.dpi)

    _setup_fonts()

    if args.steer_dir is not None:
        steer_dir = args.steer_dir.resolve()
        model = steer_dir.parent.parent.name
        out_path = (
            args.out_pdf.resolve()
            if args.out_pdf
            else out_root / model / dataset / f"{category}_best_alpha_rate.pdf"
        )
        if plot_one_model(steer_dir, out_path, model=model, dataset=dataset, dpi=dpi):
            print(out_path)
        else:
            raise SystemExit("no figure produced")
        return

    models = [_resolve_model(m) for m in args.models if m.strip()]
    if not models:
        raise SystemExit("no models specified")

    out_files: list[Path] = []
    for model in models:
        steer_dir = _steer_dir(data_root, model, dataset, category)
        out_path = out_root / model / dataset / f"{category}_best_alpha_rate.pdf"
        if plot_one_model(steer_dir, out_path, model=model, dataset=dataset, dpi=dpi):
            out_files.append(out_path)
            print(out_path)

    if not out_files:
        raise SystemExit("no figures produced")


if __name__ == "__main__":
    main()
