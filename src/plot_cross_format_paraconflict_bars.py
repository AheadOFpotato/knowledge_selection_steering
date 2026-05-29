#!/usr/bin/env python3
"""Bar charts: ParaConflict steer — unsteer vs best steer per category.

Default: **in-domain** — each category uses its own direction on its own test data
(``<cat>__dir_<cat>__test_<cat>``). Optional ``--dir-category`` fixes one direction for all.

For each test category, compare unsteer (α=0) vs best steer:
  - **Internal panel** (α > 0): y = Stage2 correct / N_manifest (all rows)
  - **External panel** (α < 0): y = Stage2 wrong / N_manifest (all rows)
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
import numpy as np

from project_paths import CROSS_FORMAT_STEER_ROOT, ROOT

FNAME_PAT = re.compile(r"^L(.+)_alpha(.+)_steer_result\.json$")

PARACONFLICT_CATEGORIES = [
    "Athelete Sport",
    "Book Author",
    "Company Founder",
    "Company Headquarter",
    "Official Language",
    "World Capital",
]

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

def _safe_name(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def _alpha_from_slug(slug: str) -> float:
    s = str(slug).strip()
    return float(s.replace("p", ".")) if s else 0.0


def _display_category(slug: str) -> str:
    return str(slug).replace("_", " ")


def _per_sample_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    p = obj.get("per_sample_outputs_and_judgements") or {}
    rows = p.get("steer_clean_minus_conflict")
    if rows is None:
        rows = p.get("project_to_correct_mean_direction")
    return list(rows or [])


def _line_stage2_label(row: dict[str, Any]) -> str:
    return str(row.get("stage2_label") or "").strip().lower()


def _internal_external_rates(obj: dict[str, Any]) -> tuple[float, float, int]:
    """Return (correct/N, wrong/N, N) over all manifest rows."""
    rows = _per_sample_rows(obj)
    n_total = len(rows)
    if n_total <= 0:
        return 0.0, 0.0, 0
    internal = external = 0
    for r in rows:
        lab = _line_stage2_label(r)
        if lab == "correct":
            internal += 1
        elif lab == "wrong":
            external += 1
    return internal / n_total, external / n_total, n_total


def _load_per_alpha(steer_dir: Path, layer: str | None) -> dict[float, dict[str, Any]]:
    table: dict[float, dict[str, Any]] = {}
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


def _find_alpha_zero(per_alpha: dict[float, dict[str, Any]]) -> float | None:
    for a in per_alpha:
        if abs(float(a)) < 1e-9:
            return float(a)
    return None


def _combo_dir(data_root: Path, dir_cat: str, test_cat: str) -> Path:
    ds = _safe_name(test_cat)
    dd = _safe_name(dir_cat)
    return data_root / f"{ds}__dir_{dd}__test_{ds}"


def _pick_best_alpha(
    per_alpha: dict[float, dict[str, Any]],
    *,
    sign: str,
    metric: str,
) -> tuple[float | None, float]:
    """sign: 'pos' | 'neg'; metric: 'internal' | 'external'."""
    best_a: float | None = None
    best_v = -1.0
    for a, obj in per_alpha.items():
        if sign == "pos" and a <= 0:
            continue
        if sign == "neg" and a >= 0:
            continue
        y_int, y_ext, _ = _internal_external_rates(obj)
        v = y_int if metric == "internal" else y_ext
        if v > best_v:
            best_v = v
            best_a = float(a)
    return best_a, best_v


def collect_category_stats(
    data_root: Path,
    test_categories: list[str],
    layer: str,
    *,
    fixed_dir_cat: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_cat in test_categories:
        dir_cat = fixed_dir_cat if fixed_dir_cat else test_cat
        combo = _combo_dir(data_root, dir_cat, test_cat)
        rec: dict[str, Any] = {
            "test_category": test_cat,
            "direction_category": dir_cat,
            "combo_dir": str(combo),
            "found": combo.is_dir(),
        }
        if not combo.is_dir():
            rows.append(rec)
            continue

        per_alpha = _load_per_alpha(combo, layer)
        rec["n_alphas"] = len(per_alpha)
        a0 = _find_alpha_zero(per_alpha)
        if a0 is None:
            rec["error"] = "no alpha=0 file"
            rows.append(rec)
            continue

        y_int0, y_ext0, n_total = _internal_external_rates(per_alpha[a0])
        rec["n_total"] = n_total
        rec["unsteer_alpha"] = a0
        rec["unsteer_internal"] = y_int0
        rec["unsteer_external"] = y_ext0

        best_a_int, best_v_int = _pick_best_alpha(
            per_alpha, sign="pos", metric="internal"
        )
        best_a_ext, best_v_ext = _pick_best_alpha(
            per_alpha, sign="neg", metric="external"
        )
        rec["best_internal_alpha"] = best_a_int
        rec["best_internal_rate"] = best_v_int if best_a_int is not None else None
        rec["best_external_alpha"] = best_a_ext
        rec["best_external_rate"] = best_v_ext if best_a_ext is not None else None
        rows.append(rec)
    return rows


def _setup_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def plot_bars(
    stats: list[dict[str, Any]],
    *,
    out_path: Path,
    dpi: int,
) -> None:
    _setup_fonts()
    fs_label = 28
    fs_tick = 26
    fs_legend = 24

    labels = [_display_category(_safe_name(s["test_category"])) for s in stats]
    x = np.arange(len(stats))
    width = 0.36

    unsteer_int = [float(s.get("unsteer_internal") or 0.0) for s in stats]
    steer_int = [
        float(s["best_internal_rate"]) if s.get("best_internal_rate") is not None else 0.0
        for s in stats
    ]
    unsteer_ext = [float(s.get("unsteer_external") or 0.0) for s in stats]
    steer_ext = [
        float(s["best_external_rate"]) if s.get("best_external_rate") is not None else 0.0
        for s in stats
    ]

    y_ticks = np.arange(0.0, 1.21, 0.2)

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 12.0), sharex=True)
    x_margin = 0.65

    # --- Internal (α > 0), top panel ---
    ax0 = axes[0]
    ax0.bar(x - width / 2, unsteer_int, width, label="unsteer (α=0)", color="#9e9e9e")
    ax0.bar(x + width / 2, steer_int, width, label="steer (best α>0)", color="#1f77b4")
    ax0.set_ylabel("Internal rate", fontsize=fs_label)
    ax0.set_xticks(x)
    ax0.tick_params(axis="x", labelbottom=False)
    ax0.set_ylim(0.0, 1.2)
    ax0.set_yticks(y_ticks)
    ax0.tick_params(axis="y", labelsize=fs_tick)
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="upper left", fontsize=fs_legend, frameon=False)

    # --- External (α < 0), bottom panel ---
    ax1 = axes[1]
    ax1.bar(x - width / 2, unsteer_ext, width, label="unsteer (α=0)", color="#9e9e9e")
    ax1.bar(x + width / 2, steer_ext, width, label="steer (best α<0)", color="#d62728")
    ax1.set_ylabel("External rate", fontsize=fs_label)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=fs_tick)
    ax1.set_ylim(0.0, 1.2)
    ax1.set_yticks(y_ticks)
    ax1.tick_params(axis="x", pad=8)
    ax1.tick_params(axis="y", labelsize=fs_tick)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=fs_legend, frameon=False)

    for ax in axes:
        ax.set_xlim(-x_margin, len(stats) - 1 + x_margin)

    fig.tight_layout(pad=1.4, h_pad=2.0)
    fig.subplots_adjust(bottom=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ParaConflict cross-format bar plot: unsteer vs best steer per test category."
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=CROSS_FORMAT_STEER_ROOT / "llama3-8b-it" / "ParaConfilct",
        help="ParaConfilct result root (combo subdirs)",
    )
    ap.add_argument("--model", type=str, default="llama3-8b-it")
    ap.add_argument(
        "--dir-category",
        type=str,
        default=None,
        help="Fixed direction for all categories (default: each category uses its own)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "fig" / "cross_format",
        help="Output directory for PDF and summary JSON",
    )
    ap.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer tag in L<layer>_alpha* filenames (default: per-model preset)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"not a directory: {data_root}")

    model = _resolve_model(args.model)
    layer = str(args.layer or DEFAULT_LAYER_BY_MODEL.get(model, "10"))
    fixed_dir = str(args.dir_category).strip() if args.dir_category else None
    in_domain = fixed_dir is None
    stats = collect_category_stats(
        data_root,
        PARACONFLICT_CATEGORIES,
        layer,
        fixed_dir_cat=fixed_dir,
    )

    missing = [s for s in stats if not s.get("found")]
    if missing:
        for s in missing:
            print(f"[warn] missing combo: {s['test_category']}", file=sys.stderr)

    usable = [s for s in stats if s.get("found") and "unsteer_internal" in s]
    if not usable:
        raise SystemExit("no usable category results (check paths and alpha≈0 files)")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "same_category" if in_domain else f"dir_{_safe_name(fixed_dir)}"
    pdf_path = out_dir / f"{model}_{slug}_internal_external_bars.pdf"
    json_path = out_dir / f"{model}_{slug}_internal_external_bars.json"

    plot_bars(stats, out_path=pdf_path, dpi=int(args.dpi))

    summary = {
        "model": model,
        "mode": "in_domain" if in_domain else "fixed_direction",
        "direction_category": fixed_dir,
        "layer": layer,
        "data_root": str(data_root),
        "denominator": "all manifest rows (Stage2 correct/wrong only in numerator)",
        "categories": stats,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pdf_path)
    print(json_path)
    for s in usable:
        d = s.get("direction_category", s["test_category"])
        print(
            f"  {s['test_category']} (dir={d}): "
            f"int unsteer={s['unsteer_internal']:.3f} bestα={s.get('best_internal_alpha')}→{s.get('best_internal_rate', 0):.3f} | "
            f"ext unsteer={s['unsteer_external']:.3f} bestα={s.get('best_external_alpha')}→{s.get('best_external_rate', 0):.3f}"
        )


if __name__ == "__main__":
    main()
