#!/usr/bin/env python3
"""Plot clean-vs-conflict residual centroid cosine similarity by layer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FILTERED_SOURCES = {
    "Llama-3-8B-Instruct": Path(
        "<DATA_DIR>/conflict_taxonomy_steer_500/llama3.1-8b-it/NQ-Swap/"
        "dev_dedup_p_only/resid_hidden_state__stage1_correct_use_wrong_context.npz"
    ),
    "Qwen3-8B": Path(
        "<DATA_DIR>/conflict_taxonomy_eval/dumps/qwen3-8b/NQ-Swap/"
        "dev_dedup_p_only/resid_hidden_state__stage1_correct_use_wrong_context.npz"
    ),
    "Gemma2-9B-it": Path(
        "<DATA_DIR>/conflict_taxonomy_eval/dumps/gemma2-9b-it/NQ-Swap/"
        "dev_dedup_p_only/resid_hidden_state__stage1_correct_use_wrong_context.npz"
    ),
    "Mistral-7B-Instruct-v0.3": Path(
        "<DATA_DIR>/conflict_taxonomy_eval/dumps/mistral-7b-instruct-v0.3/NQ-Swap/"
        "dev_dedup_p_only/resid_hidden_state__stage1_correct_use_wrong_context.npz"
    ),
}
OUT = Path(__file__).resolve().parents[1] / "figures" / "nqswap_filtered_context_centroid_cosines.png"


def centroid_cosines(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as archive:
        clean = archive["resid_act_clean"]
        conflict = archive[
            "resid_act_conflict" if "resid_act_conflict" in archive.files else "resid_act_stage2"
        ]
        clean_sum = np.zeros(clean[0].shape, dtype=np.float64)
        conflict_sum = np.zeros(conflict[0].shape, dtype=np.float64)
        for clean_row, conflict_row in zip(clean, conflict):
            clean_sum += clean_row
            conflict_sum += conflict_row

    clean_mean = clean_sum / len(clean)
    conflict_mean = conflict_sum / len(conflict)
    denom = np.linalg.norm(clean_mean, axis=1) * np.linalg.norm(conflict_mean, axis=1)
    return np.clip(np.einsum("ij,ij->i", clean_mean, conflict_mean) / denom, -1.0, 1.0)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for axis, (name, source) in zip(axes.flat, FILTERED_SOURCES.items()):
        cosine = centroid_cosines(source)
        layers = np.arange(len(cosine))
        min_layer = int(np.argmin(cosine))
        angle = float(np.degrees(np.arccos(cosine[min_layer])))

        axis.plot(layers, cosine, color="#1f77b4", linewidth=2)
        axis.scatter([min_layer], [cosine[min_layer]], color="#d62728", zorder=3)
        axis.annotate(
            f"L{min_layer}: {cosine[min_layer]:.3f}\n{angle:.1f} deg",
            (min_layer, cosine[min_layer]),
            xytext=(7, -28),
            textcoords="offset points",
            fontsize=9,
            color="#8b1a1a",
        )
        axis.set_title(name, fontsize=12)
        axis.set_xlabel("Transformer layer (0-based)")
        axis.set_ylabel("Centroid cosine similarity")
        axis.set_ylim(0.78, 1.005)
        axis.grid(alpha=0.25)

    fig.suptitle("NQ-Swap Filtered: Clean vs Conflict Residual Centroid Cosine by Layer", fontsize=14)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
