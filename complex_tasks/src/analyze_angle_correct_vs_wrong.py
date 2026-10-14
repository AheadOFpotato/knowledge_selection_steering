#!/usr/bin/env python3
"""Angle between clean vs conflict residual-stream centroids (per layer).

CLI: ``--npz`` (input ``resid_hidden_state.npz``), ``--out-dir`` (default
``<project>/output/angle_analyze``, i.e. ``<HOME_DIR>/conflict_steer/output/angle_analyze``).

Reads ``resid_act_clean`` / ``resid_act_conflict`` from the NPZ.

Per layer: ``between_centroid_deg`` (clean vs conflict mean directions); each class
``clean_intra_deg`` / ``conflict_intra_deg`` = mean/min/max of (sample unit vector vs
that class mean direction). Layers sorted by ``between_centroid_deg`` descending.
``summary`` adds the same mean/min/max pooled over all layers × samples per class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output" / "angle_analyze"


def _safe_name(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def _load_vectors(path: Path, key: str) -> list[np.ndarray] | None:
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)
    if key not in z.files:
        z.close()
        return None
    raw = z[key]
    z.close()
    return [np.asarray(raw[i], dtype=np.float64) for i in range(len(raw))]


def unitvec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(v))
    return None if n < eps else (v / n).astype(np.float64)


def mean_direction(vectors: list[np.ndarray]) -> np.ndarray | None:
    us: list[np.ndarray] = []
    for v in vectors:
        u = unitvec(v)
        if u is not None:
            us.append(u)
    if not us:
        return None
    return unitvec(np.mean(np.stack(us, axis=0), axis=0))


def angle_deg(u: np.ndarray | None, v: np.ndarray | None) -> float:
    if u is None or v is None:
        return float("nan")
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _f(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _intra_angles_to_centroid(
    vectors: list[np.ndarray], centroid: np.ndarray | None
) -> list[float]:
    """Each sample's unit vector vs class mean direction (degrees)."""
    if centroid is None:
        return []
    out: list[float] = []
    for v in vectors:
        u = unitvec(np.asarray(v, dtype=np.float64).reshape(-1))
        if u is None:
            continue
        a = angle_deg(u, centroid)
        if np.isfinite(a):
            out.append(float(a))
    return out


def _intra_stats(angles: list[float]) -> dict[str, float | None]:
    if not angles:
        return {"mean_deg": None, "min_deg": None, "max_deg": None}
    arr = np.asarray(angles, dtype=np.float64)
    return {
        "mean_deg": float(np.mean(arr)),
        "min_deg": float(np.min(arr)),
        "max_deg": float(np.max(arr)),
    }


def analyze_resid(vecs_c: list[np.ndarray], vecs_w: list[np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrs_c = [np.asarray(v, dtype=np.float64) for v in vecs_c if np.asarray(v).ndim == 2]
    arrs_w = [np.asarray(v, dtype=np.float64) for v in vecs_w if np.asarray(v).ndim == 2]
    if not arrs_c or not arrs_w:
        return [], {}
    L = min(min(x.shape[0] for x in arrs_c), min(x.shape[0] for x in arrs_w))
    D = min(min(x.shape[1] for x in arrs_c), min(x.shape[1] for x in arrs_w))
    Xc = np.stack([x[:L, :D] for x in arrs_c], axis=0)
    Xw = np.stack([x[:L, :D] for x in arrs_w], axis=0)
    out: list[dict[str, Any]] = []
    all_clean_intra: list[float] = []
    all_conflict_intra: list[float] = []
    for li in range(L):
        vc = [Xc[i, li] for i in range(Xc.shape[0])]
        vw = [Xw[i, li] for i in range(Xw.shape[0])]
        cc, cw = mean_direction(vc), mean_direction(vw)
        between = _f(angle_deg(cc, cw))
        ca = _intra_angles_to_centroid(vc, cc)
        wa = _intra_angles_to_centroid(vw, cw)
        all_clean_intra.extend(ca)
        all_conflict_intra.extend(wa)
        out.append(
            {
                "layer": li,
                "between_centroid_deg": between,
                "clean_intra_deg": _intra_stats(ca),
                "conflict_intra_deg": _intra_stats(wa),
            }
        )

    def _sort_key(r: dict[str, Any]) -> float:
        v = r.get("between_centroid_deg")
        if v is None or not np.isfinite(v):
            return float("-inf")
        return float(v)

    out.sort(key=_sort_key, reverse=True)

    summary = {
        "clean_intra_deg_global": _intra_stats(all_clean_intra),
        "conflict_intra_deg_global": _intra_stats(all_conflict_intra),
        "sort_by": "between_centroid_deg descending",
    }
    return out, summary


def _default_json_name(npz_path: Path) -> str:
    """``angle_<group>.json`` when NPZ is ``resid_hidden_state__<group>.npz``."""
    stem = npz_path.stem
    if stem.startswith("resid_hidden_state__"):
        group = stem[len("resid_hidden_state__") :]
        return f"angle_{_safe_name(group)}.json"
  # legacy flat name
    p = npz_path.resolve()
    if len(p.parts) >= 2:
        parent = _safe_name(p.parent.name)
        if len(p.parts) >= 3:
            gp = _safe_name(p.parent.parent.name)
            return f"{gp}__{parent}_angle_resid.json"
        return f"{parent}__{p.stem}_angle_resid.json"
    return f"{p.stem}_angle_resid.json"


def _display_path(p: Path) -> str:
    rp = p.resolve()
    try:
        return str(rp.relative_to(ROOT.resolve()))
    except ValueError:
        return str(rp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resid-stream angle: clean vs conflict centroids per layer.")
    ap.add_argument("--npz", type=Path, default="<DATA_DIR>/conflict_steer/llama3-8b-it/macnoise/NQ_eval_longpre_test_fix/resid_hidden_state__stage1_correct_use_wrong_context.npz", help="Input resid_hidden_state.npz")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Directory for output JSON (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Full output JSON path (overrides --out-dir + default filename).",
    )
    args = ap.parse_args()

    npz_path = args.npz if args.npz.is_absolute() else (ROOT / args.npz)
    npz_path = npz_path.resolve()
    if not npz_path.is_file():
        raise SystemExit(f"npz not found: {npz_path}")

    if args.out_json is not None:
        out_path = args.out_json if args.out_json.is_absolute() else (ROOT / args.out_json)
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
        out_dir = out_dir if out_dir.is_absolute() else (ROOT / out_dir)
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / _default_json_name(npz_path)

    resid_c = _load_vectors(npz_path, "resid_act_clean")
    resid_w_key = "resid_act_conflict"
    resid_w = _load_vectors(npz_path, resid_w_key)
    if resid_w is None:
        resid_w_key = "resid_act_stage2"
        resid_w = _load_vectors(npz_path, resid_w_key)
    if resid_c is None or resid_w is None:
        raise SystemExit(
            f"missing resid_act_clean and/or ({'resid_act_conflict' or 'resid_act_stage2'}) in {npz_path}"
        )

    layers, summary = analyze_resid(resid_c, resid_w)
    report: dict[str, Any] = {
        "npz": str(npz_path),
        "out_json": str(out_path),
        "keys_used": {"clean": "resid_act_clean", "other": resid_w_key},
        "summary": summary,
        "resid": layers,
    }

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_display_path(out_path))


if __name__ == "__main__":
    main()
