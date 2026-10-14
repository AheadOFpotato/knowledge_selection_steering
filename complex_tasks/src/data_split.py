"""Train/val/test index splits (used at evaluation time, not during extraction)."""
from __future__ import annotations

import random


def split_row_indices(
    n: int, *, train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic train/val/test index partition (disjoint, covers all rows)."""
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
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    if not test_idx and n > 1:
        test_idx = [val_idx.pop()] if val_idx else [train_idx.pop()]
    return train_idx, val_idx, test_idx
