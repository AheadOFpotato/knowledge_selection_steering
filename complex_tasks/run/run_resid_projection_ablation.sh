#!/usr/bin/env bash
# Single-category residual steering: fixed five-point alpha sweep on the full manifest.
#
# Usage:
#   GPU=0 ./run/run_resid_projection_ablation.sh
#   MODEL=llama3-8b-it CATEGORY=dev_dedup_by_question_only_p_no_table_li DATASET=NQ-Swap LAYERS=10 GPU=0 ./run/run_resid_projection_ablation.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/src/run_resid_projection_ablation.py"

MODEL="${MODEL:-llama3-8b-it}"
DATASET="${DATASET:-NQ-Swap}"
CATEGORY="${CATEGORY:-dev_dedup_p_only}"
DIR_GROUP="${DIR_GROUP:-unfiltered_all_pairs}"

DUMP_ROOT="${DUMP_ROOT:-<DATA_DIR>/conflict_taxonomy_steer}"
STEER_OUT_ROOT="${STEER_OUT_ROOT:-<DATA_DIR>/conflict_taxonomy_steer_result}"

LAYERS="${LAYERS:-10}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-2}"
ALPHA_STEP="${ALPHA_STEP:-0.5}"

BASE_DIR="${DUMP_ROOT}/${MODEL}/${DATASET}/${CATEGORY}"
NPZ="${BASE_DIR}/resid_hidden_state__${DIR_GROUP}.npz"
MANIFEST="${BASE_DIR}/internal_aware_data__context_all.json"

echo "[run] NPZ:      ${NPZ}" >&2
echo "[run] manifest: ${MANIFEST}" >&2

python3 "${PY}" \
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
  --stage1-min-pass-rate -1 \
  --steer-toward context \
  --context-only \
  --judge official
