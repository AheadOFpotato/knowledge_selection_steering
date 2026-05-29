#!/usr/bin/env python3
"""Internal vs external alpha curves — 4 models × 2 datasets (NQ-Swap + MacNoise).

Row 1: NQ-Swap (in-domain category)
Row 2: MacNoise (pooled: NQ_eval + TQA_eval)

Per α (only if ``rerun_stage1_pass_rate`` > ``--stage1-min-pass``):
  - Denominator N = manifest 全量 (all rows in steer JSON, pooled if multiple categories)
  - Uses stored ``stage2_label`` from JSON: ``correct`` → internal, ``wrong`` → external
  - ``skipped`` / ``not_run`` / empty: not in numerators, still in denominator N
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

FNAME_PAT = re.compile(r"^L(.+)_alpha(.+)_steer_result\.json$")

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

DEFAULT_LAYER_BY_MODEL: dict[str, str] = {
    "llama3-8b-it": "10",
    "qwen3-8b": "21",
    "gemma2-9b-it": "16",
    "yi-6b-chat": "13",
    "mistral-7b-v0.1": "16",
}

NQSWAP_DATASET = "NQ-Swap"
NQSWAP_CATEGORY = "dev__split_test__dir_dev__split_train__test_dev__split_test"

MACNOISE_DATASET = "macnoise"
def _resolve_model(model: str) -> str:
    m = str(model).strip()
    return MODEL_ALIASES.get(m, m)


MACNOISE_CATEGORIES = (
    "NQ_eval_longpre_test_fix",
    "TQA_eval_gpt4_dev_256_new_fix",
)

DATASET_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("NQ-Swap", NQSWAP_DATASET, (NQSWAP_CATEGORY,)),
)
MACNOISE_DATASET_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("MacNoise", MACNOISE_DATASET, MACNOISE_CATEGORIES),
)


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
    return float(s.replace("p", ".")) if s else 0.0


def _per_sample_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    p = obj.get("per_sample_outputs_and_judgements") or {}
    rows = p.get("steer_clean_minus_conflict")
    if rows is None:
        rows = p.get("project_to_correct_mean_direction")
    return list(rows or [])


def _stage2_label(row: dict[str, Any]) -> str:
    return str(row.get("stage2_label") or "").strip().lower()


def _stage1_pass_rate_obj(obj: dict[str, Any]) -> float:
    r = obj.get("results") or {}
    s1 = r.get("stage1_all_samples") or {}
    if s1.get("rerun_stage1_pass_rate") is not None:
        return float(s1.get("rerun_stage1_pass_rate") or 0.0)
    m = r.get("manifest") or {}
    passed = float(s1.get("rerun_stage1_pass", 0) or 0)
    total = float(m.get("num_stage1_eval_samples", 0) or 0)
    if total <= 0:
        total = float(m.get("total_samples_in_json", 0) or 0)
    if total <= 0:
        total = float(len(_per_sample_rows(obj)))
    if total <= 0:
        return 0.0
    return passed / total


def _load_steer_dir(steer_dir: Path, layer: str | None) -> dict[float, dict[str, Any]]:
    table: dict[float, dict[str, Any]] = {}
    if not steer_dir.is_dir():
        return table
    for p in sorted(steer_dir.glob("L*_alpha*_steer_result.json")):
        m = FNAME_PAT.match(p.name)
        if not m:
            continue
        layer_key = str(m.group(1))
        if layer is not None and layer_key != layer:
            continue
        alpha = _alpha_from_slug(m.group(2))
        table[alpha] = json.loads(p.read_text(encoding="utf-8"))
    return table


def _pool_per_alpha(
    category_tables: list[dict[float, dict[str, Any]]],
) -> dict[float, dict[str, Any]]:
    """Merge multiple category steer tables by α (concat rows; sum Stage1 stats)."""
    all_alphas: set[float] = set()
    for t in category_tables:
        all_alphas |= set(t.keys())
    pooled: dict[float, dict[str, Any]] = {}
    for a in sorted(all_alphas):
        objs = [t[a] for t in category_tables if a in t]
        if not objs:
            continue
        rows: list[dict[str, Any]] = []
        s1_pass = 0.0
        n_stage1_eval = 0.0
        n_manifest = 0.0
        for obj in objs:
            rows.extend(_per_sample_rows(obj))
            r = obj.get("results") or {}
            s1 = r.get("stage1_all_samples") or {}
            m = r.get("manifest") or {}
            s1_pass += float(s1.get("rerun_stage1_pass", 0) or 0)
            n_stage1 = float(m.get("num_stage1_eval_samples", 0) or 0)
            if n_stage1 <= 0 and s1.get("rerun_stage1_pass_rate") is not None:
                n_stage1 = float(s1.get("rerun_stage1_pass", 0) or 0) / max(
                    float(s1.get("rerun_stage1_pass_rate") or 0.0), 1e-12
                )
            n_stage1_eval += n_stage1
            n_manifest += float(m.get("total_samples_in_json", len(_per_sample_rows(obj))) or 0)
        pooled[a] = {
            "_rows": rows,
            "_stage1_pass_rate": (s1_pass / n_stage1_eval) if n_stage1_eval > 0 else 0.0,
            "_n_manifest": int(n_manifest) if n_manifest > 0 else len(rows),
        }
    return pooled


def _load_dataset_per_alpha(
    data_root: Path,
    model: str,
    dataset: str,
    categories: tuple[str, ...],
    layer: str,
) -> dict[float, dict[str, Any]]:
    tables = [
        _load_steer_dir(data_root / model / dataset / cat, layer) for cat in categories
    ]
    if len(categories) == 1:
        return tables[0]
    return _pool_per_alpha(tables)


def _build_series_manifest(
    per_alpha: dict[float, dict[str, Any]],
    stage1_min_pass: float,
) -> tuple[list[float], list[float], list[float], float, float, int] | None:
    alphas_sorted = sorted(per_alpha.keys())
    xvals: list[float] = []
    y_int: list[float] = []
    y_ext: list[float] = []
    last_n = 0
    for a in alphas_sorted:
        entry = per_alpha[a]
        if "_rows" in entry:
            rows = entry["_rows"]
            s1_rate = float(entry.get("_stage1_pass_rate", 0.0))
        else:
            rows = _per_sample_rows(entry)
            s1_rate = _stage1_pass_rate_obj(entry)
        if s1_rate <= stage1_min_pass:
            continue
        n_total = len(rows)
        if n_total <= 0:
            continue
        internal = external = 0
        for r in rows:
            lab = _stage2_label(r)
            if lab == "correct":
                internal += 1
            elif lab == "wrong":
                external += 1
        xvals.append(a)
        y_int.append(internal / n_total)
        y_ext.append(external / n_total)
        last_n = n_total

    if not xvals:
        return None

    return xvals, y_int, y_ext, min(xvals), max(xvals), last_n


def _steer_dir(data_root: Path, model: str, dataset: str, category: str) -> Path:
    return data_root / model / dataset / category


def _default_layer(model: str, data_root: Path, dataset: str, categories: tuple[str, ...]) -> str:
    if model in DEFAULT_LAYER_BY_MODEL:
        return DEFAULT_LAYER_BY_MODEL[model]
    counts: dict[str, int] = {}
    for cat in categories:
        steer_dir = _steer_dir(data_root, model, dataset, cat)
        for p in steer_dir.glob("L*_alpha*_steer_result.json"):
            m = FNAME_PAT.match(p.name)
            if m:
                counts[str(m.group(1))] = counts.get(str(m.group(1)), 0) + 1
    if not counts:
        return "16"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _apply_shared_xaxis(ax: plt.Axes, x_lo: float, x_hi: float, *, fs_tick: int) -> None:
    ax.set_xlim(x_lo, x_hi)
    ax.axvline(0.0, color="0.80", linestyle="--", linewidth=1.0, zorder=0)
    span = x_hi - x_lo
    if span <= 0:
        ax.set_xticks([0.0])
    else:
        step = 0.5 if span <= 3.5 else 1.0
        start = step * (int(x_lo / step) + (0 if x_lo % step == 0 else 1))
        if start < x_lo:
            start += step
        ticks = [t for t in _frange(start, x_hi + step * 0.01, step) if x_lo - 1e-9 <= t <= x_hi + 1e-9]
        if x_lo <= 0.0 <= x_hi:
            ticks = sorted(set(ticks) | {0.0})
        if not ticks:
            ticks = [x_lo, 0.0, x_hi] if x_lo <= 0.0 <= x_hi else [x_lo, x_hi]
        ax.set_xticks(ticks)
    ax.tick_params(labelsize=fs_tick)


def _frange(start: float, stop: float, step: float) -> list[float]:
    out: list[float] = []
    x = start
    while x <= stop + step * 1e-9:
        out.append(round(x, 10))
        x += step
    return out


def _plot_panel(
    ax: plt.Axes,
    x: list[float],
    y_int: list[float],
    y_ext: list[float],
    x_lo: float,
    x_hi: float,
    *,
    show_ylabel: bool,
    fs_label: int,
    fs_tick: int,
    y_headroom: float = 0.12,
) -> None:
    if x:
        ax.plot(x, y_int, "-o", markersize=5, linewidth=1.6, color="#1f77b4", label="internal")
        ax.plot(x, y_ext, "-o", markersize=5, linewidth=1.6, color="#d62728", label="external")
    _apply_shared_xaxis(ax, x_lo, x_hi, fs_tick=fs_tick)
    ymax = max(y_int + y_ext, default=[0.0])
    ax.set_ylim(0.0, ymax + max(y_headroom, ymax * 0.10) + 0.02)
    ax.set_xlabel("α", fontsize=fs_label)
    if show_ylabel:
        ax.set_ylabel("Rate", fontsize=fs_label)
    ax.grid(True, alpha=0.25)


def plot_grid(
    models: list[str],
    data_root: Path,
    out_path: Path,
    *,
    dataset_rows: tuple[tuple[str, str, tuple[str, ...]], ...],
    layer_by_model: dict[str, str],
    stage1_min_pass: float,
    dpi: int,
) -> bool:
    _setup_fonts()
    fs_label = 22
    fs_tick = 20
    fs_title = 20
    fs_row = 20
    fs_legend = 18
    n_rows = len(dataset_rows)
    n_cols = len(models)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.2 * n_rows),
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    plotted = 0
    for ri, (row_label, dataset, categories) in enumerate(dataset_rows):
        for ci, model in enumerate(models):
            ax = axes[ri][ci]
            label = MODEL_LABELS.get(model, model)
            layer = layer_by_model.get(model) or _default_layer(
                model, data_root, dataset, categories
            )
            per_alpha = _load_dataset_per_alpha(
                data_root, model, dataset, categories, layer
            )
            if not per_alpha:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, f"{label}\n(missing)", ha="center", va="center", transform=ax.transAxes
                )
                print(
                    f"[warn] missing {row_label} {model}: {dataset} {categories}",
                    file=sys.stderr,
                )
                continue

            built = _build_series_manifest(per_alpha, stage1_min_pass)
            if built is None:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, f"{label}\n(no data)", ha="center", va="center", transform=ax.transAxes
                )
                print(f"[warn] no series: {row_label} {model} L{layer}", file=sys.stderr)
                continue

            x, y_int, y_ext, x_lo, x_hi, n_total = built
            _plot_panel(
                ax,
                x,
                y_int,
                y_ext,
                x_lo,
                x_hi,
                show_ylabel=(ci == 0),
                fs_label=fs_label,
                fs_tick=fs_tick,
                y_headroom=0.14 if ri == 1 else 0.10,
            )
            if ri == 0:
                ax.set_title(label, fontsize=fs_title)
            plotted += 1
            print(
                f"  [{row_label}] {model}: L{layer} N={n_total} "
                f"α∈[{x_lo:g},{x_hi:g}] ({len(x)} points)",
                file=sys.stderr,
            )

        # Row label on leftmost panel
        axes[ri][0].text(
            -0.34,
            0.5,
            row_label,
            transform=axes[ri][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=fs_row,
            fontweight="bold",
        )

    if plotted == 0:
        plt.close(fig)
        return False

    legend_elements = [
        Line2D(
            [0], [0], color="#1f77b4", marker="o", markersize=8, linewidth=2.0, label="internal"
        ),
        Line2D(
            [0], [0], color="#d62728", marker="o", markersize=8, linewidth=2.0, label="external"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=fs_legend,
        frameon=False,
    )

    fig.tight_layout(rect=[0.10, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path, format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Alpha curves: 4 models × NQ-Swap (optional MacNoise row)."
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=STEER_OUT_ROOT,
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output" / "fig" / "nqswap")
    ap.add_argument(
        "--out-name",
        type=str,
        default="alpha_curves_nqswap_4models",
    )
    ap.add_argument("--format", choices=("pdf", "png"), default="pdf")
    ap.add_argument(
        "--include-macnoise",
        action="store_true",
        help="Also plot MacNoise row (default: NQ-Swap only).",
    )
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument(
        "--stage1-min-pass",
        type=float,
        default=0.7,
        help="Plot α only when pooled Stage1 pass rate > this (default 0.7)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    models = [_resolve_model(m) for m in args.models if m.strip()]
    if not models:
        raise SystemExit("no models specified")

    layer_by_model = {m: DEFAULT_LAYER_BY_MODEL[m] for m in models if m in DEFAULT_LAYER_BY_MODEL}
    out_path = args.out_dir.resolve() / f"{args.out_name}.{args.format}"
    dataset_rows = DATASET_ROWS + (
        MACNOISE_DATASET_ROWS if args.include_macnoise else ()
    )

    print(f"[plot] data_root={data_root}", file=sys.stderr)
    ok = plot_grid(
        models,
        data_root,
        out_path,
        dataset_rows=dataset_rows,
        layer_by_model=layer_by_model,
        stage1_min_pass=float(args.stage1_min_pass),
        dpi=int(args.dpi),
    )
    if not ok:
        raise SystemExit("no figure produced")
    print(out_path)


if __name__ == "__main__":
    main()
