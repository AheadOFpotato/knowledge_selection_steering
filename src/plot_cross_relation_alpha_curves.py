#!/usr/bin/env python3
"""Plot 6×6 internal vs external alpha curves for cross-relation steer results.

Data layout: ``{STEER_ROOT}/<model>/counterfact/<test_cat>__dir_<dir_cat>__test_<test_cat>/``.

Each subplot (row = direction knowledge type, col = test knowledge type):
  - x: steer strength α (sorted ascending)
  - y: share in [0, 1]
  - internal = Stage2 correct / N, external = Stage2 wrong / N
    (N = manifest 总样本数，含 Stage1 未通过、未跑 Stage2 的样本；
     skipped/not_run 等不计入分子，故 y_int + y_ext 可 < 1)
  - x-axis: tight to valid α only — left = most negative α with Stage1 pass ≥
    ``--stage1-min-pass``; right = most positive such α (no side padding)
  - skip plotting α when Stage1 pass rate is below threshold

One combined figure per model under ``--out-dir`` (default: ``fig_cross_relation``).
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from project_paths import KNOWLEDGE_TYPE_STEER_ROOT, ROOT

RELATION_IDS = ("P27", "P176", "P106", "P140", "P641", "P449")
# Wikidata property labels (English); order matches RELATION_IDS.
PROPERTY_NAMES: dict[str, str] = {
    "P27": "country of citizenship",
    "P176": "manufacturer",
    "P106": "occupation",
    "P140": "religion",
    "P641": "sport",
    "P449": "original broadcaster",
}
FNAME_PAT = re.compile(r"^L(.+)_alpha(.+)_steer_result\.json$")
DIR_NAME_PAT = re.compile(
    r"^(?P<test_prefix>.+)__dir_(?P<dir_cat>.+)__test_(?P<test_cat>.+)$"
)

DEFAULT_LAYER_BY_MODEL: dict[str, str] = {
    "llama3-8b-it": "10",
    "qwen3-8b": "21",
    "gemma2-9b-it": "16",
    "yi-6b-chat": "13",
    "mistral-7b-v0.1": "16",
}

MODEL_ALIASES: dict[str, str] = {
    "yi": "yi-6b-chat",
    "yi-6b": "yi-6b-chat",
}


def _resolve_model(model: str) -> str:
    m = str(model).strip()
    return MODEL_ALIASES.get(m, m)


def _normalize_model_list(models: list[str]) -> list[str]:
    """Resolve aliases; prefer yi-6b-chat over yi-6b when both appear."""
    resolved = [_resolve_model(m) for m in models]
    if "yi-6b-chat" in resolved:
        resolved = [m for m in resolved if m != "yi-6b"]
    out: list[str] = []
    seen: set[str] = set()
    for m in resolved:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _alpha_from_slug(slug: str) -> float:
    s = str(slug).strip()
    return float(s.replace("p", ".")) if s else 0.0


def _property_label(pid: str) -> str:
    return PROPERTY_NAMES.get(pid, pid)


def _property_label_wrapped(pid: str) -> str:
    """1 word: one line; 2 words: one per line; 3+ words: prefix on line 1, last word on line 2."""
    words = _property_label(pid).split()
    if len(words) <= 1:
        return words[0] if words else pid
    if len(words) == 2:
        return "\n".join(words)
    return " ".join(words[:-1]) + "\n" + words[-1]


def _pid_from_category(cat: str) -> str | None:
    low = cat.lower()
    for pid in RELATION_IDS:
        if f"_{pid.lower()}_" in low:
            return pid
    return None


def _stage1_pass_rate(obj: dict[str, Any]) -> float:
    r = obj.get("results") or {}
    s1 = r.get("stage1_all_samples") or {}
    return float(s1.get("rerun_stage1_pass_rate", 0.0) or 0.0)


def _per_sample_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    p = obj.get("per_sample_outputs_and_judgements") or {}
    rows = p.get("steer_clean_minus_conflict")
    if rows is None:
        rows = p.get("project_to_correct_mean_direction")
    return list(rows or [])


def _line_stage2_label(row: dict[str, Any]) -> str:
    return str(row.get("stage2_label") or "").strip().lower()


def _load_steer_dir(steer_dir: Path, layer: str | None) -> dict[float, dict[str, Any]]:
    table: dict[float, dict[str, Any]] = {}
    for p in sorted(steer_dir.glob("L*_alpha*_steer_result.json")):
        m = FNAME_PAT.match(p.name)
        if not m:
            continue
        layer_key = str(m.group(1))
        if layer is not None and layer_key != layer:
            continue
        alpha = _alpha_from_slug(m.group(2))
        obj = json.loads(p.read_text(encoding="utf-8"))
        table[alpha] = obj
    return table


def _build_series_all_evaluated(
    per_alpha: dict[float, dict[str, Any]],
    stage1_min_pass: float,
) -> tuple[list[float], list[float], list[float], float, float, int] | None:
    """Per α: correct/N and wrong/N; N = len(all manifest rows). Returns xlim (lo, hi)."""
    alphas_sorted = sorted(per_alpha.keys())
    xvals: list[float] = []
    y_int: list[float] = []
    y_ext: list[float] = []
    last_n = 0
    for a in alphas_sorted:
        obj = per_alpha[a]
        if _stage1_pass_rate(obj) < stage1_min_pass:
            continue
        rows = _per_sample_rows(obj)
        n_total = len(rows)
        if n_total <= 0:
            continue
        internal = external = 0
        for r in rows:
            lab = _line_stage2_label(r)
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

    # Tight x: left = most negative valid α; right = most positive valid α.
    x_lo, x_hi = min(xvals), max(xvals)
    return xvals, y_int, y_ext, x_lo, x_hi, last_n


def _parse_combo_dirname(name: str) -> tuple[str, str] | None:
    m = DIR_NAME_PAT.match(name)
    if not m:
        return None
    dir_cat = m.group("dir_cat")
    test_cat = m.group("test_cat")
    if m.group("test_prefix") != test_cat:
        pass
    return dir_cat, test_cat


def _discover_categories(data_root: Path) -> dict[str, str]:
    """pid -> full category slug (first seen)."""
    pid_to_cat: dict[str, str] = {}
    cf = data_root / "counterfact"
    if not cf.is_dir():
        return pid_to_cat
    for d in cf.iterdir():
        if not d.is_dir():
            continue
        parsed = _parse_combo_dirname(d.name)
        if parsed is None:
            continue
        dir_cat, test_cat = parsed
        for cat in (dir_cat, test_cat):
            pid = _pid_from_category(cat)
            if pid and pid not in pid_to_cat:
                pid_to_cat[pid] = cat
    return pid_to_cat


def _combo_dir(data_root: Path, dir_cat: str, test_cat: str) -> Path:
    return data_root / "counterfact" / f"{test_cat}__dir_{dir_cat}__test_{test_cat}"


def _default_layer(model: str, data_root: Path) -> str:
    model = _resolve_model(model)
    if model in DEFAULT_LAYER_BY_MODEL:
        return DEFAULT_LAYER_BY_MODEL[model]
    cf = data_root / "counterfact"
    if not cf.is_dir():
        return "16"
    counts: dict[str, int] = {}
    for combo in cf.iterdir():
        if not combo.is_dir():
            continue
        for p in combo.glob("L*_alpha*_steer_result.json"):
            m = FNAME_PAT.match(p.name)
            if m:
                counts[str(m.group(1))] = counts.get(str(m.group(1)), 0) + 1
    if not counts:
        return "16"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _plot_subplot(
    ax: plt.Axes,
    x: list[float],
    y_int: list[float],
    y_ext: list[float],
    x_lo: float,
    x_hi: float,
    *,
    show_yticks: bool,
) -> tuple[Any, Any]:
    line_int = ax.plot(
        x, y_int, "-o", markersize=2.5, linewidth=1.1, color="#1f77b4", label="internal"
    )[0]
    line_ext = ax.plot(
        x, y_ext, "-o", markersize=2.5, linewidth=1.1, color="#d62728", label="external"
    )[0]
    ax.set_xlim(x_lo, x_hi)
    y_top = max(y_int + y_ext) if (y_int and y_ext) else 1.0
    ax.set_ylim(0.0, min(1.0, y_top * 1.08 + 0.02))
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)
    if not show_yticks:
        ax.set_yticklabels([])
    return line_int, line_ext


def plot_model(
    model: str,
    data_root: Path,
    out_dir: Path,
    *,
    layer: str | None,
    stage1_min_pass: float,
    dpi: int,
) -> Path | None:
    pid_to_cat = _discover_categories(data_root)
    if not pid_to_cat:
        print(f"[skip] {model}: no counterfact combos under {data_root}", file=sys.stderr)
        return None

    use_layer = layer or _default_layer(model, data_root)
    n = len(RELATION_IDS)
    # Wide, flat plot cells; extra margin for row/column property labels.
    fig = plt.figure(figsize=(2.05 * n + 2.4, 0.88 * n + 1.35))
    gs_outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.14, 1.0],
        left=0.16,
        right=0.96,
        top=0.90,
        bottom=0.07,
        hspace=0.06,
    )
    col_width_ratios = [0.50] + [1.0] * n
    gs_header = GridSpecFromSubplotSpec(
        1,
        n + 1,
        subplot_spec=gs_outer[0],
        width_ratios=col_width_ratios,
        wspace=0.22,
    )
    gs_body = GridSpecFromSubplotSpec(
        n,
        n + 1,
        subplot_spec=gs_outer[1],
        width_ratios=col_width_ratios,
        hspace=0.16,
        wspace=0.22,
    )

    # Column headers: same grid as plots (skip col 0 = row-label gutter)
    for j, test_pid in enumerate(RELATION_IDS):
        ax_h = fig.add_subplot(gs_header[0, j + 1])
        ax_h.axis("off")
        ax_h.text(
            0.5,
            0.02,
            _property_label_wrapped(test_pid),
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color="black",
        )

    # Row headers: figure coords, right-aligned just left of plot grid
    row_label_axes: list[Any] = []
    for i, dir_pid in enumerate(RELATION_IDS):
        ax_h = fig.add_subplot(gs_body[i, 0])
        ax_h.axis("off")
        row_label_axes.append((ax_h, dir_pid))
    fig.canvas.draw()
    for ax_h, dir_pid in row_label_axes:
        bbox = ax_h.get_position()
        fig.text(
            bbox.x0 + 0.028,
            bbox.y0 + bbox.height / 2,
            _property_label_wrapped(dir_pid),
            ha="right",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="black",
            transform=fig.transFigure,
        )

    fig.text(
        0.52, 0.915, "test knowledge type (column)", ha="center", fontsize=17, fontweight="bold", color="black"
    )
    fig.text(
        0.025,
        0.50,
        "direction knowledge type (row)",
        ha="left",
        va="center",
        rotation=90,
        fontsize=17,
        fontweight="bold",
        color="black",
    )

    plotted = 0
    for i, dir_pid in enumerate(RELATION_IDS):
        dir_cat = pid_to_cat.get(dir_pid)
        for j, test_pid in enumerate(RELATION_IDS):
            ax = fig.add_subplot(gs_body[i, j + 1])
            show_yticks = j == 0
            if dir_cat is None:
                ax.axis("off")
                continue
            test_cat = pid_to_cat.get(test_pid)
            if test_cat is None:
                ax.axis("off")
                continue
            combo = _combo_dir(data_root, dir_cat, test_cat)
            if not combo.is_dir():
                ax.axis("off")
                continue
            per_alpha = _load_steer_dir(combo, use_layer)
            built = _build_series_all_evaluated(per_alpha, stage1_min_pass)
            if built is None:
                ax.axis("off")
                continue
            x, y_int, y_ext, x_lo, x_hi, _n0 = built
            _plot_subplot(
                ax,
                x,
                y_int,
                y_ext,
                x_lo,
                x_hi,
                show_yticks=show_yticks,
            )
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"[skip] {model}: no subplots drawn", file=sys.stderr)
        return None

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#1f77b4",
            marker="o",
            markersize=6,
            linewidth=1.8,
            label="internal",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            marker="o",
            markersize=6,
            linewidth=1.8,
            label="external",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.78, 0.96),
        fontsize=15,
        frameon=False,
        borderpad=0.25,
        labelspacing=0.4,
        handlelength=2.2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_cross_relation_alpha_6x6.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot 6×6 cross-relation internal/external alpha curves per model.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=KNOWLEDGE_TYPE_STEER_ROOT,
        help="Root with <model>/counterfact/... combo dirs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "fig" / "knowledge_type",
        help="Output directory for PNG figures",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to plot (default: all subdirs of --data-root)",
    )
    ap.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Residual layer tag in filenames (default: auto per model)",
    )
    ap.add_argument(
        "--stage1-min-pass",
        type=float,
        default=0.8,
        help="Do not plot α when Stage1 pass rate is below this (default 0.8)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"not a directory: {data_root}")

    if args.models:
        models = _normalize_model_list(list(args.models))
    else:
        models = _normalize_model_list(
            sorted(
                p.name
                for p in data_root.iterdir()
                if p.is_dir() and (p / "counterfact").is_dir()
            )
        )

    if not models:
        raise SystemExit(f"no models under {data_root}")

    out_files: list[Path] = []
    for model in models:
        model_key = _resolve_model(model)
        model_root = data_root / model_key
        path = plot_model(
            model_key,
            model_root,
            out_dir,
            layer=args.layer,
            stage1_min_pass=float(args.stage1_min_pass),
            dpi=int(args.dpi),
        )
        if path is not None:
            out_files.append(path)
            print(path)

    if not out_files:
        raise SystemExit("no figures produced")


if __name__ == "__main__":
    main()
