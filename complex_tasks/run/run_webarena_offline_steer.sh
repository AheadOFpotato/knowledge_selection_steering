#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-<DATA_DIR>/.venv/bin/python}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-llama3.1-8b-it}"
GPU="${GPU:-0}"
LAYERS="${LAYERS:-10}"
DATA="${DATA:-${ROOT}/data/policy_web/webarena_offline.jsonl}"
DIR_NPZ="${DIR_NPZ:-<DATA_DIR>/conflict_taxonomy_steer_500/llama3.1-8b-it/NQ-Swap/dev_dedup_p_only/resid_hidden_state__unfiltered_all_pairs.npz}"
OUT_ROOT="${OUT_ROOT:-<DATA_DIR>/conflict_taxonomy_steer_result_500/WebArenaOffline}"
ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-2}"
ALPHA_STEP="${ALPHA_STEP:-0.5}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DIRECTION_MODE="${DIRECTION_MODE:-true}"
DIRECTION_SEED="${DIRECTION_SEED:-0}"

# Keep normal results in the existing location; controls receive their own tree.
if [[ "${DIRECTION_MODE}" != "true" ]]; then
  OUT_ROOT="${OUT_ROOT}/${DIRECTION_MODE}"
fi

args=(
  "${PYTHON}" "${ROOT}/src/run_webarena_offline_steer.py"
  --model "${MODEL}" --gpu "${GPU}" --layers "${LAYERS}"
  --data "${DATA}" --npz "${DIR_NPZ}" --out-root "${OUT_ROOT}"
  --alpha-start "${ALPHA_START}" --alpha-end "${ALPHA_END}" --alpha-step "${ALPHA_STEP}"
  --max-samples "${MAX_SAMPLES}" --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}" --max-new-tokens "${MAX_NEW_TOKENS}"
  --direction-mode "${DIRECTION_MODE}" --direction-control-seed "${DIRECTION_SEED}"
)
[[ "${SKIP_EXISTING}" == "1" ]] && args+=(--skip-existing)

echo "[webarena_offline] GPU=${GPU} MODEL=${MODEL} samples=${MAX_SAMPLES} batch=${BATCH_SIZE}"
echo "[webarena_offline] direction=${DIRECTION_MODE} seed=${DIRECTION_SEED} alpha=${ALPHA_START}..${ALPHA_END} step=${ALPHA_STEP}"
exec "${args[@]}"
