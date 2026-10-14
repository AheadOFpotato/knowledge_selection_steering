#!/usr/bin/env bash
# Single-category residual steering with runtime train/val/test split.
#
# Full dump NPZ  -> direction (train 70% when same_dataset)
# Full manifest  -> val alpha grid + test eval (split at use time)
#
# Usage:
#   GPU=0 ./run/run_resid_projection_ablation.sh
#   MODEL=llama3-8b-it CATEGORY=dev LAYERS=10 GPU=0 ./run/run_resid_projection_ablation.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"

PY="${ROOT}/src/run_resid_projection_ablation.py"

MODEL="${MODEL:-llama3-8b-it}"
DATASET="${DATASET:-NQ-Swap}"
CATEGORY="${CATEGORY:-dev}"
GROUP="${GROUP:-stage1_correct_use_wrong_context}"

DUMP_ROOT="${DUMP_ROOT}"
STEER_OUT_ROOT="${STEER_OUT_ROOT}"

LAYERS="${LAYERS:-10}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"

SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"

BASE_DIR="${DUMP_ROOT}/${MODEL}/${DATASET}/${CATEGORY}"
NPZ="${BASE_DIR}/resid_hidden_state__${GROUP}.npz"
MANIFEST="${BASE_DIR}/internal_aware_data__${GROUP}.json"

echo "[run] direction NPZ: ${NPZ}" >&2
echo "[run] eval manifest: ${MANIFEST} (runtime 70/15/15 split)" >&2

python3 "${PY}" \
  --eval-mode train_val_test \
  --split-protocol same_dataset \
  --split-seed "${SPLIT_SEED}" \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --model "${MODEL}" \
  --npz "${NPZ}" \
  --test-json "${MANIFEST}" \
  --out-root "${STEER_OUT_ROOT}" \
  --output-category "${CATEGORY}" \
  --direction-category-tag "${CATEGORY}" \
  --test-category-tag "${CATEGORY}" \
  --gpu "${GPU}" \
  --layers "${LAYERS}" \
  --alpha-start "${ALPHA_START}" \
  --alpha-end "${ALPHA_END}" \
  --alpha-step "${ALPHA_STEP}" \
  --batch-size "${BATCH_SIZE}" \
  --max-samples-stage1 0 \
  --max-samples-stage2 0 \
  --stage1-min-pass-rate -1
