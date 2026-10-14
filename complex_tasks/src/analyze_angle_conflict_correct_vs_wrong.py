#!/usr/bin/env python3
"""Angle between *conflict(stage2)* residual-stream centroids: correct vs wrong (per layer).

This script is analogous to analyze_angle_correct_vs_wrong.py, but instead of comparing
clean vs conflict, it analyzes ONLY the conflict/stage2 residual stream and splits samples
by stage2_label (correct vs wrong).

Inputs:
- NPZ: produced by dump_activations_from_testjsonl.py group outputs.
  Expected keys:
    - resid_act_stage2 (preferred) OR resid_act_conflict (legacy)
    - line_idx (used to align with manifest json samples)
    - resid_act_clean is ignored here
- Manifest JSON: internal_aware_data__<group>.json that contains samples[].stage2_label
  and samples[].line_idx.

Output JSON format intentionally mirrors analyze_angle_correct_vs_wrong.py:
- top-level: npz, out_json, keys_used, summary, resid
- resid: per-layer between_centroid_deg + intra-angle stats for each class
  sorted by between_centroid_deg descending.
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


def _intra_angles_to_centroid(vectors: list[np.ndarray], centroid: np.ndarray | None) -> list[float]:
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


def _load_npz_vectors(npz_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    try:
        if "resid_act_stage2" in z.files:
            vec_key = "resid_act_stage2"
        elif "resid_act_conflict" in z.files:
            vec_key = "resid_act_conflict"
        else:
            raise SystemExit(
                f"NPZ missing resid vectors key. Expected resid_act_stage2 or resid_act_conflict. keys={z.files}"
            )
        if "line_idx" not in z.files:
            raise SystemExit(f"NPZ missing line_idx. keys={z.files}")
        vecs = z[vec_key]
        line_idx = z["line_idx"].astype(np.int64)
        if len(vecs) != len(line_idx):
            raise SystemExit(f"NPZ length mismatch: len(vecs)={len(vecs)} len(line_idx)={len(line_idx)}")
        return vec_key, vecs, line_idx
    finally:
        z.close()


def _infer_manifest_path(npz_path: Path) -> Path | None:
    # Try to infer internal_aware_data__*.json in same directory.
    parent = npz_path.parent
    name = npz_path.name
    if name.startswith("resid_hidden_state__") and name.endswith(".npz"):
        group = name[len("resid_hidden_state__") : -len(".npz")]
        cand = parent / f"internal_aware_data__{group}.json"
        if cand.is_file():
            return cand
    # fallback: none
    return None


def _load_labels(manifest_path: Path) -> dict[int, str]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = obj.get("samples", [])
    if not isinstance(samples, list):
        raise SystemExit("manifest has no samples list")
    out: dict[int, str] = {}
    for s in samples:
        if not isinstance(s, dict):
            continue
        li = s.get("line_idx")
        try:
            li_int = int(li)
        except Exception:
            continue
        lab = str(s.get("stage2_label", "")).strip().lower()
        if lab:
            out[li_int] = lab
    return out


def analyze_conflict_correct_wrong(
    vecs: list[np.ndarray], labels: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrs = [np.asarray(v, dtype=np.float64) for v in vecs if np.asarray(v).ndim == 2]
    if not arrs:
        return [], {}
    L = min(x.shape[0] for x in arrs)
    D = min(x.shape[1] for x in arrs)
    X = np.stack([x[:L, :D] for x in arrs], axis=0)  # [N, L, D]

    # Split indices by labels (strictly correct/wrong)
    idx_correct = [i for i, lab in enumerate(labels) if lab == "correct"]
    idx_wrong = [i for i, lab in enumerate(labels) if lab == "wrong"]

    out: list[dict[str, Any]] = []
    all_correct_intra: list[float] = []
    all_wrong_intra: list[float] = []
    for li in range(L):
        vc = [X[i, li] for i in idx_correct]
        vw = [X[i, li] for i in idx_wrong]
        mc, mw = mean_direction(vc), mean_direction(vw)
        ca = _intra_angles_to_centroid(vc, mc)
        wa = _intra_angles_to_centroid(vw, mw)
        all_correct_intra.extend(ca)
        all_wrong_intra.extend(wa)
        out.append(
            {
                "layer": li,
                "between_centroid_deg": _f(angle_deg(mc, mw)),
                "correct_intra_deg": _intra_stats(ca),
                "wrong_intra_deg": _intra_stats(wa),
            }
        )

    def _sort_key(r: dict[str, Any]) -> float:
        v = r.get("between_centroid_deg")
        if v is None or not np.isfinite(v):
            return float("-inf")
        return float(v)

    out.sort(key=_sort_key, reverse=True)
    summary = {
        "correct_intra_deg_global": _intra_stats(all_correct_intra),
        "wrong_intra_deg_global": _intra_stats(all_wrong_intra),
        "sort_by": "between_centroid_deg descending",
    }
    return out, summary


def _default_json_name(npz_path: Path) -> str:
    p = npz_path.resolve()
    parent = _safe_name(p.parent.name)
    gp = _safe_name(p.parent.parent.name) if p.parent.parent else "unknown"
    return f"{gp}__{parent}__conflict_correct_vs_wrong_angle_resid.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Conflict(stage2) resid angle: correct vs wrong centroids per layer.")
    ap.add_argument("--npz", type=Path, default="<DATA_DIR>/conflict_steer/llama3-8b-it/NQ-Swap/dev_dedup_by_question_only_p_no_table_li/resid_hidden_state__stage1_correct_use_wrong_context.npz", help="Input NPZ with resid_act_stage2/resid_act_conflict and line_idx.")
    ap.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="internal_aware_data__<group>.json. If omitted, inferred from --npz path.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Directory for output JSON (default: {DEFAULT_OUT_DIR})",
    )
    args = ap.parse_args()

    npz_path = args.npz if args.npz.is_absolute() else (ROOT / args.npz)
    npz_path = npz_path.resolve()
    if not npz_path.is_file():
        raise SystemExit(f"npz not found: {npz_path}")

    manifest_path = args.manifest_json
    if manifest_path is None:
        inferred = _infer_manifest_path(npz_path)
        if inferred is None:
            raise SystemExit("manifest-json not provided and could not be inferred from npz filename")
        manifest_path = inferred
    manifest_path = manifest_path if manifest_path.is_absolute() else (ROOT / manifest_path)
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"manifest json not found: {manifest_path}")

    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    out_dir = out_dir if out_dir.is_absolute() else (ROOT / out_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _default_json_name(npz_path)

    vec_key, raw_vecs, line_idx = _load_npz_vectors(npz_path)
    labels_by_line = _load_labels(manifest_path)

    vecs: list[np.ndarray] = []
    labs: list[str] = []
    missing = 0
    for i in range(len(line_idx)):
        li = int(line_idx[i])
        lab = labels_by_line.get(li, "")
        if lab not in ("correct", "wrong"):
            missing += 1
            continue
        vecs.append(np.asarray(raw_vecs[i], dtype=np.float64))
        labs.append(lab)

    layers, summary = analyze_conflict_correct_wrong(vecs, labs)
    report: dict[str, Any] = {
        "npz": str(npz_path),
        "out_json": str(out_path),
        "keys_used": {"conflict": vec_key},
        "summary": summary,
        "resid": layers,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()

