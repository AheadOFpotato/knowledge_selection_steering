# Knowledge Conflict Steering

Steer large language models between **parameter memory (internal)** and **context (external)** answers under knowledge conflicts, using **residual-stream direction vectors**.

This repository provides the full experiment pipeline: activation dump → direction estimation → α grid search → test evaluation, plus three cross-domain steering experiments and plotting scripts.

---

## Repository layout

```
knowledge_conflict_steering/
├── data/                    # Five datasets (see below)
├── models/                  # Local model weights (or set MODEL_ROOT)
├── src/                     # Core Python scripts
├── run/                     # Shell batch entry points
│   └── scripts/             # Shared helpers for cross experiments
└── output/
    ├── dump/                # Activations and manifests
    ├── steer_result/        # In-domain steering results
    ├── cross_format/        # ParaConflict cross-format
    ├── knowledge_type/      # CounterFact cross knowledge type
    ├── cross_dataset/       # NQ-Swap direction → other datasets
    └── fig/                 # Figure outputs
```

All paths can be overridden via environment variables (see `run/common.sh`, `src/project_paths.py`). Defaults are **repo-relative**.

---

## Environment

- Python 3.9+
- PyTorch + CUDA
- `transformers>=4.46`, `tokenizers`, `numpy`

```bash
pip install torch transformers tokenizers numpy matplotlib
```

Place models under `models/<ModelName>/`, or point `MODEL_ROOT` elsewhere. Supported model keys:

| Key | Local dir name | Default steering layer |
|-----|----------------|------------------------|
| `llama3-8b-it` | `Llama3-8b-it` | 10 |
| `qwen3-8b` | `Qwen3-8B` | 21 |
| `gemma2-9b-it` | `Gemma2-9b-it` | 16 |
| `yi-6b-chat` | `Yi-6B-Chat` | 13 |

---

## Datasets

Only these five datasets are supported:

| Dataset | Path | Notes |
|---------|------|-------|
| **NQ-Swap** | `data/NQ-Swap/dev.jsonl` | Fixed to `dev.jsonl`, category=`dev` |
| **macnoise** | `data/macnoise/*.json` | One category per JSON file |
| **ParaConfilct** | `data/ParaConfilct/test.jsonl` | Six relation-type categories |
| **memotrap** | `data/memotrap/*.csv` | One category per CSV file |
| **counterfact** | `data/counterfact/counterfact_relation_*.json` | Six knowledge types: P27 P176 P106 P140 P641 P449 |

**ParaConflict categories:** Athelete Sport, Book Author, Company Founder, Company Headquarter, Official Language, World Capital

**Splits:** train / val / test = **70% / 15% / 15%** (reproducible with `--split-seed 42`). Output dirs use suffixes `__split_train`, `__split_val`, `__split_test`.

---

## Evaluation protocol (train → val → test)

```
train split  →  estimate steering direction v = mean(clean) − mean(conflict) from NPZ
val   split  →  sweep α grid; pick best α by val Stage2 correct rate
test  split  →  final evaluation at best α
```

Implemented as `--eval-mode train_val_test` in `run_resid_projection_ablation.py`.

**Behaviour groups (dump outputs):**

- `stage1_correct_use_wrong_context` — Stage1 answers with parameter knowledge; Stage2 injects wrong context (default steering group)
- `stage1_wrong_use_correct_context` — Stage1 wrong; Stage2 injects correct context

Each group produces:

- `internal_aware_data__<group>.json` — Stage1/Stage2 manifest and labels
- `resid_hidden_state__<group>.npz` — per-layer residual activations (`resid_act_clean` / `resid_act_conflict`)

---

## Quick start

### 1. Dump activations (all datasets × three splits)

```bash
cd knowledge_conflict_steering

# Default: llama3-8b-it, all five datasets, train+val+test
GPU=0 ./run/run_dump_all_datasets.sh

# Subset / debug
MODELS=llama3-8b-it GPU=0 DATASETS=NQ-Swap,counterfact SPLITS=train,val DRY_RUN=1 ./run/run_dump_all_datasets.sh
```

Output: `output/dump/<model>/<dataset>/<category>__split_<split>/`

### 2. In-domain steering (single category example: NQ-Swap dev)

```bash
GPU=0 MODEL=llama3-8b-it CATEGORY=dev LAYERS=10 ./run/run_resid_projection_ablation.sh
```

### 3. Batch steering (all datasets)

```bash
MODEL=llama3-8b-it LAYERS=10 GPU=0 SKIP_EXISTING=1 ./run/run_resid_projection_ablation_all_final.sh
```

Output: `output/steer_result/<model>/<dataset>/<category>__split_test/`

---

## Three cross-domain experiments

All use the `train_val_test` protocol (`run/scripts/cross_steer_common.sh`).

| Experiment | Direction source (train) | Evaluation target (val/test) | Script |
|------------|--------------------------|------------------------------|--------|
| **Cross-format** | ParaConflict **Substitution** conflict | ParaConflict **Coherent** per category | `run/run_resid_projection_paraconflict_cross_format.sh` |
| **Knowledge type** | CounterFact relation A | CounterFact relation B (6×6) | `run/run_resid_projection_knowledge_type_cross_steer.sh` |
| **Cross-dataset** | NQ-Swap `dev` | ParaConflict / macnoise / memotrap / counterfact | `run/run_resid_projection_nqswap_dir_cross_datasets.sh` |

Cross-format requires dumping Substitution-conflict activations first:

```bash
GPU=0 ./run/run_dump_paraconflict_substitution_cross_format.sh
# → output/dump/cross_format/<model>/ParaConfilct/<cat>__split_*/
```

Then run cross-format steering:

```bash
GPU=0 ./run/run_resid_projection_paraconflict_cross_format.sh
# → output/cross_format/...
```

```bash
MODEL=llama3-8b-it GPU=0 ./run/run_resid_projection_knowledge_type_cross_steer.sh
# → output/knowledge_type/...

MODEL=llama3-8b-it GPU=0 ./run/run_resid_projection_nqswap_dir_cross_datasets.sh
# → output/cross_dataset/...
```

---

## Plotting

| Script | Purpose |
|--------|---------|
| `src/plot_steer_alpha_curves.py` | Single-experiment α curves (needs legacy mode or multi-α artifacts) |
| `src/plot_nqswap_alpha_curves_multi_model.py` | Multi-model NQ-Swap α curves |
| `src/plot_layer_analyze_best_alpha_rate.py` | Best α and correct rate per layer |
| `src/plot_angle_between_centroid.py` | Angle between clean/conflict centroids |
| `src/plot_cross_format_paraconflict_bars.py` | Cross-format bar charts |
| `src/plot_cross_relation_alpha_curves.py` | Knowledge-type cross-relation α curves |

```bash
./run/run_plot_angle_fig_all.sh
# → output/fig/angle/
```

> **Note:** In `train_val_test` mode, test evaluation uses only the **single** best α from validation. For full α curves, use `--eval-mode legacy` or plot from val-grid artifacts separately.

---

## Core Python CLI

### `src/dump_activations_from_testjsonl.py`

```bash
python3 src/dump_activations_from_testjsonl.py \
  --model llama3-8b-it \
  --dataset NQ-Swap \
  --category dev \
  --input-jsonl data/NQ-Swap/dev.jsonl \
  --data-split train \
  --out-root output/dump \
  --gpu 0 \
  --batch-size 64 \
  --save-resid true
```

### `src/run_resid_projection_ablation.py`

```bash
python3 src/run_resid_projection_ablation.py \
  --eval-mode train_val_test \
  --model llama3-8b-it \
  --npz output/dump/llama3-8b-it/NQ-Swap/dev__split_train/resid_hidden_state__stage1_correct_use_wrong_context.npz \
  --val-json output/dump/llama3-8b-it/NQ-Swap/dev__split_val/internal_aware_data__stage1_correct_use_wrong_context.json \
  --test-json output/dump/llama3-8b-it/NQ-Swap/dev__split_test/internal_aware_data__stage1_correct_use_wrong_context.json \
  --out-root output/steer_result \
  --gpu 0 \
  --layers 10 \
  --alpha-start 0 --alpha-end 3 --alpha-step 0.2 \
  --batch-size 64
```

Steering: `h' = h + α · v`. The sign of α selects `+v` or `−v`. Positive α biases toward parameter answers; negative α toward context answers (consistent with `v = mean(clean) − mean(conflict)`).

---

## Output JSON (brief)

Steering result JSON includes:

- `summary_metrics` — internal / external match rates, Stage1/Stage2 pass rates
- `best_alpha` — α selected on val (`train_val_test` mode)
- `eval_mode` — `train_val_test` or `legacy`

Path fields prefer **repo-relative** paths (`project_paths.display_path`).

---

## Related baseline

The SpARE (SAE-based representation engineering) baseline lives under `baseline/SAE-based-representation-engineering-main/` in the parent project. It uses the same Stage2 manifest protocol; metrics `UseM_*` / `UseC_*` correspond to internal / external.

---

## Citation

If you use this pipeline, please cite the relevant steering / knowledge-conflict papers and the original dataset papers.
