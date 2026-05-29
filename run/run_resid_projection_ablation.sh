#!/usr/bin/env bash
# Single-category residual steering with train/val/test protocol.
#
# train NPZ  -> steering direction
# val JSON   -> alpha grid
# test JSON  -> final metrics at best alpha
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

BASE_DIR="${DUMP_ROOT}/${MODEL}/${DATASET}/${CATEGORY}"
TRAIN_NPZ="${BASE_DIR}__split_train/resid_hidden_state__${GROUP}.npz"
VAL_JSON="${BASE_DIR}__split_val/internal_aware_data__${GROUP}.json"
TEST_JSON="${BASE_DIR}__split_test/internal_aware_data__${GROUP}.json"

echo "[run] train NPZ: ${TRAIN_NPZ}" >&2
echo "[run] val JSON:  ${VAL_JSON}" >&2
echo "[run] test JSON: ${TEST_JSON}" >&2

python3 "${PY}" \
  --eval-mode train_val_test \
  --model "${MODEL}" \
  --npz "${TRAIN_NPZ}" \
  --val-json "${VAL_JSON}" \
  --test-json "${TEST_JSON}" \
  --out-root "${STEER_OUT_ROOT}" \
  --output-category "${CATEGORY}__split_test" \
  --direction-category-tag "${CATEGORY}__split_train" \
  --test-category-tag "${CATEGORY}__split_test" \
  --gpu "${GPU}" \
  --layers "${LAYERS}" \
  --alpha-start "${ALPHA_START}" \
  --alpha-end "${ALPHA_END}" \
  --alpha-step "${ALPHA_STEP}" \
  --batch-size "${BATCH_SIZE}" \
  --max-samples-stage1 0 \
  --max-samples-stage2 0 \
  --stage1-min-pass-rate -1
