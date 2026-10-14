#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "output" / "resid_hidden_state" / "llama3-8b-it" / "NQ-Swap" / "resid_hidden_state.npz"
DEFAULT_OUT_DIR = ROOT / "output" / "probe"


def _safe_name(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def _load_npz_arrays(npz_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    z = np.load(npz_path, allow_pickle=True)
    if "resid_act_clean" not in z.files or "resid_act_conflict" not in z.files:
        z.close()
        raise SystemExit(f"missing resid_act_clean/resid_act_conflict in {npz_path}")
    rc = z["resid_act_clean"]
    rw = z["resid_act_conflict"]
    z.close()

    clean = [np.asarray(rc[i], dtype=np.float64) for i in range(len(rc))]
    conflict = [np.asarray(rw[i], dtype=np.float64) for i in range(len(rw))]
    return clean, conflict


def _infer_ld(clean: list[np.ndarray], conflict: list[np.ndarray]) -> tuple[int, int]:
    all_arr = [a for a in (clean + conflict) if a.ndim == 2]
    if not all_arr:
        raise SystemExit("no valid 2D residual arrays in npz")
    L = min(a.shape[0] for a in all_arr)
    D = min(a.shape[1] for a in all_arr)
    if L <= 0 or D <= 0:
        raise SystemExit("invalid residual shape")
    return int(L), int(D)


def _build_layer_matrix(
    clean: list[np.ndarray],
    conflict: list[np.ndarray],
    layer: int,
    dim: int,
    max_samples_per_class: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    c = [a[layer, :dim] for a in clean if a.ndim == 2 and layer < a.shape[0] and dim <= a.shape[1]]
    w = [a[layer, :dim] for a in conflict if a.ndim == 2 and layer < a.shape[0] and dim <= a.shape[1]]
    if max_samples_per_class is not None:
        m = int(max_samples_per_class)
        if len(c) > m:
            idx = rng.choice(len(c), size=m, replace=False)
            c = [c[i] for i in idx]
        if len(w) > m:
            idx = rng.choice(len(w), size=m, replace=False)
            w = [w[i] for i in idx]
    if not c or not w:
        return np.empty((0, dim), dtype=np.float64), np.empty((0,), dtype=np.int64)

    Xc = np.stack(c, axis=0).astype(np.float64)
    Xw = np.stack(w, axis=0).astype(np.float64)
    y = np.concatenate(
        [np.zeros(Xc.shape[0], dtype=np.int64), np.ones(Xw.shape[0], dtype=np.int64)], axis=0
    )
    X = np.concatenate([Xc, Xw], axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def _split_indices(n: int, test_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_ratio)))
    n_test = min(n_test, n - 1)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    steps: int = 800,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    yf = y.astype(np.float64)
    for _ in range(int(steps)):
        p = _sigmoid(X @ w + b)
        err = (p - yf)
        gw = (X.T @ err) / n + l2 * w
        gb = float(np.mean(err))
        w -= lr * gw
        b -= lr * gb
    return w, b


def _acc_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    pred = (logits >= 0.0).astype(np.int64)
    return float(np.mean(pred == y))


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return None
    return (v / n).astype(np.float64)


def _probe_mean_direction(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray
) -> np.ndarray:
    c0 = X_train[y_train == 0]
    c1 = X_train[y_train == 1]
    if c0.shape[0] == 0 or c1.shape[0] == 0:
        return np.zeros(X_eval.shape[0], dtype=np.int64)
    m0 = np.mean(c0, axis=0)
    m1 = np.mean(c1, axis=0)
    u0 = _unit(m0)
    u1 = _unit(m1)
    if u0 is None or u1 is None:
        return np.zeros(X_eval.shape[0], dtype=np.int64)
    s0 = X_eval @ u0
    s1 = X_eval @ u1
    return (s1 > s0).astype(np.int64)


def _run_layer_probes(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    train_idx, test_idx = _split_indices(X.shape[0], test_ratio, rng)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mu, sd = _standardize_fit(X_train)
    X_train_z = _standardize_apply(X_train, mu, sd)
    X_test_z = _standardize_apply(X_test, mu, sd)

    # Probe 1: logistic regression
    w, b = _fit_logistic_regression(X_train_z, y_train)
    lr_train_acc = _acc_from_logits(X_train_z @ w + b, y_train)
    lr_test_acc = _acc_from_logits(X_test_z @ w + b, y_test)

    # Probe 2: mean-direction by cosine / dot score
    md_train_pred = _probe_mean_direction(X_train_z, y_train, X_train_z)
    md_test_pred = _probe_mean_direction(X_train_z, y_train, X_test_z)
    md_train_acc = float(np.mean(md_train_pred == y_train))
    md_test_acc = float(np.mean(md_test_pred == y_test))

    return {
        "n_total": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "probe_lr": {
            "train_acc": lr_train_acc,
            "test_acc": lr_test_acc,
        },
        "probe_mean_direction": {
            "train_acc": md_train_acc,
            "test_acc": md_test_acc,
        },
    }


def _parse_layers(s: str, max_layer: int) -> list[int]:
    t = str(s).strip().lower()
    if t in ("all", "*", ""):
        return list(range(max_layer))
    out: list[int] = []
    for p in str(s).split(","):
        q = p.strip()
        if not q:
            continue
        li = int(q)
        if 0 <= li < max_layer:
            out.append(li)
    if not out:
        raise SystemExit(f"no valid layers in --layers={s!r}; valid range: [0, {max_layer - 1}]")
    return sorted(set(out))


def _default_out_name(npz: Path) -> str:
    p = npz.resolve()
    parent = _safe_name(p.parent.name)
    gp = _safe_name(p.parent.parent.name) if p.parent.parent != p.parent else "unknown"
    return f"{gp}__{parent}_probe.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe residual stream by layer (LR and mean-direction)."
    )
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Path to resid_hidden_state.npz")
    ap.add_argument("--layers", type=str, default="all", help="Layer ids, e.g. '10,20' or 'all'")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio in (0,1)")
    ap.add_argument("--max-samples-per-class", type=int, default=0, help="0 means all; otherwise downsample each class")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = ap.parse_args()

    if not (0.0 < float(args.test_ratio) < 1.0):
        raise SystemExit("--test-ratio must be in (0, 1)")

    npz_path = args.npz if args.npz.is_absolute() else (ROOT / args.npz)
    npz_path = npz_path.resolve()
    if not npz_path.is_file():
        raise SystemExit(f"npz not found: {npz_path}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _default_out_name(npz_path)

    clean, conflict = _load_npz_arrays(npz_path)
    L, D = _infer_ld(clean, conflict)
    layers = _parse_layers(str(args.layers), L)

    max_samp = int(args.max_samples_per_class)
    max_samp = None if max_samp <= 0 else max_samp
    rng = np.random.default_rng(int(args.seed))

    per_layer: list[dict[str, Any]] = []
    for li in layers:
        X, y = _build_layer_matrix(clean, conflict, li, D, max_samp, rng)
        if X.shape[0] < 4:
            continue
        r = _run_layer_probes(X, y, float(args.test_ratio), rng)
        r["layer"] = int(li)
        per_layer.append(r)

    if not per_layer:
        raise SystemExit("no valid layer results")

    lr_tests = [x["probe_lr"]["test_acc"] for x in per_layer]
    md_tests = [x["probe_mean_direction"]["test_acc"] for x in per_layer]
    summary = {
        "num_layers": len(per_layer),
        "lr_test_acc_mean": float(np.mean(lr_tests)),
        "lr_test_acc_max": float(np.max(lr_tests)),
        "mean_direction_test_acc_mean": float(np.mean(md_tests)),
        "mean_direction_test_acc_max": float(np.max(md_tests)),
        "best_layer_by_lr_test_acc": int(per_layer[int(np.argmax(lr_tests))]["layer"]),
        "best_layer_by_mean_direction_test_acc": int(per_layer[int(np.argmax(md_tests))]["layer"]),
    }

    out = {
        "npz": str(npz_path),
        "out_json": str(out_path),
        "settings": {
            "layers": layers,
            "test_ratio": float(args.test_ratio),
            "max_samples_per_class": None if max_samp is None else int(max_samp),
            "seed": int(args.seed),
            "feature_dim_used": int(D),
        },
        "summary": summary,
        "per_layer": per_layer,
        "label_note": "class 0 = clean, class 1 = conflict",
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()

