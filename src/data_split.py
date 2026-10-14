"""Runtime train/val/test splits for steering (not applied at dump time)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def split_indices_train_val_test(
    n: int,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 70/15/15-style partition (disjoint, covers all rows)."""
    if n <= 0:
        return [], [], []
    rng = random.Random(int(seed))
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = max(1, int(round(n * float(train_ratio)))) if n >= 3 else max(1, n - 2)
    n_val = max(1, int(round(n * float(val_ratio)))) if n >= 3 else (1 if n > 1 else 0)
    if n_train + n_val >= n:
        n_val = max(0, min(n_val, n - n_train - 1))
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    train_idx = sorted(idx[:n_train])
    val_idx = sorted(idx[n_train : n_train + n_val])
    test_idx = sorted(idx[n_train + n_val :])
    if not test_idx and n > 1:
        if val_idx:
            moved = val_idx.pop()
            test_idx = [moved]
        elif train_idx:
            moved = train_idx.pop()
            test_idx = [moved]
    return train_idx, val_idx, test_idx


def split_indices_eval_val_test(
    n: int,
    *,
    val_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split eval data into val (alpha grid) and test (final metrics). Default 50/50."""
    if n < 2:
        raise ValueError(f"need at least 2 rows for eval split, got {n}")
    rng = random.Random(int(seed))
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(round(n * float(val_ratio))))
    if n_val >= n:
        n_val = max(1, n // 2)
    val_idx = sorted(idx[:n_val])
    test_idx = sorted(idx[n_val:])
    if not test_idx:
        test_idx = [val_idx.pop()]
    return val_idx, test_idx


def subset_manifest(
    base: dict[str, Any],
    indices: list[int],
    *,
    split_name: str = "",
) -> dict[str, Any]:
    samples = list(base.get("samples") or [])
    picked = [samples[i] for i in indices]
    out = {k: v for k, v in base.items() if k != "samples"}
    out["samples"] = picked
    out["num_kept"] = len(picked)
    if split_name:
        out["runtime_split"] = split_name
        out["runtime_split_indices"] = list(indices)
    return out


def write_manifest_slice(
    base: dict[str, Any],
    indices: list[int],
    path: Path,
    *,
    split_name: str = "",
) -> Path:
    obj = subset_manifest(base, indices, split_name=split_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
