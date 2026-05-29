#!/usr/bin/env python3
"""Plot internal vs external knowledge vs alpha, relative to wrong@α=0.

Baseline: samples with Stage2 label **wrong** when **alpha = 0** (must exist).

For each alpha: among those line_idx only, plot
  - **internal** = correct / N  (gold match; internal knowledge)
  - **external** = wrong / N   (distractor / conflict; external)

N = count wrong at α=0. Skipped labels count toward neither numerator; y_int + y_ext + y_skip/N = 1.

Stops when Stage1 fail rate exceeds ``--fail-stop`` (default 10%).
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

from project_paths import ROOT

FNAME_PAT = re.compile(r"^L(.+)_alpha(.+)_steer_result\.json$")


def _alpha_from_slug(slug: str) -> float:
    s = str(slug).strip()
    return float(s.replace("p", ".")) if s else 0.0


def _stage1_fail_ratio(obj: dict[str, Any]) -> float:
    r = obj.get("results") or {}
    m = r.get("manifest") or {}
    s1 = r.get("stage1_all_samples") or {}
    total = float(m.get("total_samples_in_json", 0) or 0)
    if total <= 0:
        return 0.0
    passed = float(s1.get("rerun_stage1_pass", 0) or 0)
    return max(0.0, (total - passed) / total)


def _per_sample_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    p = obj.get("per_sample_outputs_and_judgements") or {}
    return list(p.get("project_to_correct_mean_direction") or [])


def _line_stage2_label(row: dict[str, Any]) -> str:
    return str(row.get("stage2_label") or "").strip().lower()


def _load_steer_dir(steer_dir: Path) -> dict[str, dict[float, dict[str, Any]]]:
    table: dict[str, dict[float, dict[str, Any]]] = {}
    for p in sorted(steer_dir.glob("L*_alpha*_steer_result.json")):
        m = FNAME_PAT.match(p.name)
        if not m:
            continue
        layer_key = str(m.group(1))
        alpha = _alpha_from_slug(m.group(2))
        obj = json.loads(p.read_text(encoding="utf-8"))
        table.setdefault(layer_key, {})[alpha] = obj
    return table


def _find_baseline_alpha(alphas: list[float]) -> float | None:
    for a in alphas:
        if abs(float(a)) < 1e-12:
            return float(a)
    return None


def _build_series_wrong_at_zero(
    per_alpha: dict[float, dict[str, Any]],
    fail_stop: float,
) -> tuple[list[float], list[float], list[float], int] | None:
    """Returns (x, y_internal, y_external, n_baseline_wrong) or None if no baseline."""
    alphas_sorted = sorted(per_alpha.keys())
    a_base = _find_baseline_alpha(alphas_sorted)
    if a_base is None:
        print(
            "plot_steer_alpha_curves: no alpha≈0 file; need L*_alpha0_* or L*_alpha0.0_* for wrong@α=0 baseline",
            file=sys.stderr,
        )
        return None

    rows0 = _per_sample_rows(per_alpha[a_base])
    baseline_wrong: set[int] = set()
    for r in rows0:
        if _line_stage2_label(r) == "wrong":
            baseline_wrong.add(int(r.get("line_idx", -1)))
    n0 = len(baseline_wrong)
    if n0 == 0:
        print(
            "plot_steer_alpha_curves: no Stage2 wrong at alpha=0; skip layer",
            file=sys.stderr,
        )
        return None

    xvals: list[float] = []
    y_int: list[float] = []
    y_ext: list[float] = []
    for a in alphas_sorted:
        obj = per_alpha[a]
        if _stage1_fail_ratio(obj) > fail_stop:
            break
        rows = _per_sample_rows(obj)
        by_line = {int(r["line_idx"]): _line_stage2_label(r) for r in rows}
        internal = external = 0
        for lid in baseline_wrong:
            lab = by_line.get(lid, "")
            if lab == "correct":
                internal += 1
            elif lab == "wrong":
                external += 1
        xvals.append(a)
        y_int.append(internal / n0)
        y_ext.append(external / n0)
    return xvals, y_int, y_ext, n0


def _tag_from_path(steer_dir: Path) -> str:
    parts = steer_dir.resolve().parts
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return steer_dir.name


def _default_out_fig_dir(steer_dir: Path) -> Path:
    p = steer_dir.resolve()
    parts = p.parts
    fig_root = ROOT / "output" / "fig"
    try:
        i = parts.index("steer_result")
        if i + 3 < len(parts):
            model, dataset, cat = parts[i + 1], parts[i + 2], parts[i + 3]
            return fig_root / model / dataset / cat
    except ValueError:
        pass
    return fig_root / p.name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot internal (correct) vs external (wrong) among samples wrong at alpha=0."
    )
    ap.add_argument(
        "--steer-dir",
        type=Path,
        default=ROOT
        / "output"
        / "steer_result"
        / "llama3-8b-it"
        / "ParaConfilct"
        / "World_Capital",
        help="Directory containing L*_alpha*_steer_result.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="PNG directory (default: output/fig/<model>/<dataset>/<category> inferred from --steer-dir)",
    )
    ap.add_argument(
        "--fail-stop",
        type=float,
        default=0.10,
        help="Stop curve when Stage1 fail rate exceeds this",
    )
    args = ap.parse_args()

    steer_dir = args.steer_dir.resolve()
    if not steer_dir.is_dir():
        raise SystemExit(f"not a directory: {steer_dir}")

    out_dir = args.out_dir.resolve() if args.out_dir is not None else _default_out_fig_dir(steer_dir)

    table = _load_steer_dir(steer_dir)
    if not table:
        raise SystemExit(f"no L*_alpha*_steer_result.json under {steer_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _tag_from_path(steer_dir).replace("/", "_")
    fail_pct = int(round(float(args.fail_stop) * 100))
    out_files: list[str] = []

    for layer_key in sorted(table.keys(), key=lambda s: (len(s), s)):
        per_alpha = table[layer_key]
        built = _build_series_wrong_at_zero(per_alpha, fail_stop=float(args.fail_stop))
        if built is None:
            continue
        x, y_int, y_ext, n0 = built
        if not x:
            continue

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.plot(
            x,
            y_int,
            "-o",
            markersize=3,
            linewidth=1.6,
            label="internal (correct) / N",
        )
        ax.plot(
            x,
            y_ext,
            "-o",
            markersize=3,
            linewidth=1.6,
            label="external (wrong) / N",
        )
        ax.set_ylim(0.0, 1.0)
        x_max = max(x)
        ax.set_xlim(-0.02, x_max + 0.02)
        ax.set_xlabel("Steer strength α")
        ax.set_ylabel("Share among wrong@α=0 (N=%d)" % n0)
        ax.set_title(
            f"{tag} — L{layer_key}  wrong@α=0 → internal vs external  (stop if Stage1 fail > {fail_pct}%)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        safe_layer = str(layer_key).replace(",", "_")
        out_path = out_dir / f"L{safe_layer}_alpha_curve.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_files.append(str(out_path))

    if not out_files:
        raise SystemExit("no figure produced (need alpha≈0 JSON and nonzero wrong@α=0)")

    for p in out_files:
        print(p)


if __name__ == "__main__":
    main()
