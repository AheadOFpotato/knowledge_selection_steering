#!/usr/bin/env bash
# LooGLE residual steering — Llama configuration
#
#   GPU=0 ./run_llama.sh
#   GPU=0 TASK=longdep_qa MAX_SAMPLES=50 ./run_llama.sh

set -u
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_ROOT}"
# shellcheck source=run_common.sh
source "${BUNDLE_ROOT}/run_common.sh"

PY="${BUNDLE_ROOT}/src/run_loogle_resid_projection_ablation.py"

MODEL="${MODEL:-llama3-8b-it}"
TASK="${TASK:-longdep_qa}"
# shortdep_qa | shortdep_cloze | longdep_qa | longdep_summarization

DATA_ROOT="${DATA_ROOT:-data}"
STEER_OUT_ROOT="${STEER_OUT_ROOT:-output/new_steer}"
NPZ="${NPZ:-npz/llama3-8b-it__NQ-Swap__dev_dedup__stage1_correct_use_wrong_context.npz}"
DIRECTION_TAG="${DIRECTION_TAG:-dev_dedup}"

LAYERS="${LAYERS:-10}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_LENGTH="${MAX_LENGTH:-${MAX_PROMPT_LEN}}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
DATA_JSON="${DATA_JSON:-}"

ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"

if [[ "${NPZ}" = /* ]]; then
  NPZ_PATH="${NPZ}"
else
  NPZ_PATH="${BUNDLE_ROOT}/${NPZ}"
fi

if [[ ! -f "${NPZ_PATH}" ]]; then
  echo "[error] NPZ not found: ${NPZ_PATH}" >&2
  echo "  Put steering NPZ under npz/ or set NPZ=..." >&2
  exit 1
fi

if [[ ! -f "${BUNDLE_ROOT}/config/models.json" ]]; then
  echo "[error] config/models.json missing. Copy from config/models.json.example" >&2
  exit 1
fi

check_model_weights "${MODEL}" || exit 1

echo "[run] BUNDLE_ROOT=${BUNDLE_ROOT}" >&2
echo "[run] MODEL=${MODEL} TASK=${TASK} NPZ=${NPZ}" >&2

cmd=(
  python3 "${PY}"
  --model "${MODEL}"
  --npz "${NPZ}"
  --task "${TASK}"
  --out-root "${STEER_OUT_ROOT}"
  --data-root "${DATA_ROOT}"
  --gpu "${GPU}"
  --layers "${LAYERS}"
  --alpha-start "${ALPHA_START}"
  --alpha-end "${ALPHA_END}"
  --alpha-step "${ALPHA_STEP}"
  --batch-size "${BATCH_SIZE}"
  --max-prompt-len "${MAX_PROMPT_LEN}"
  --max-length "${MAX_LENGTH}"
  --max-samples "${MAX_SAMPLES}"
  --sample-seed "${SAMPLE_SEED}"
  --direction-category-tag "${DIRECTION_TAG}"
)

if [[ -n "${DATA_JSON}" ]]; then
  cmd+=(--data-json "${DATA_JSON}")
fi

"${cmd[@]}"
