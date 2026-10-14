#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-llama3.1-8b-it}"
GPU="${GPU:-0}"
ENV="${ENV:-retail}"
LAYERS="${LAYERS:-10}"
DIR_NPZ="${DIR_NPZ:-${ROOT}/output/dump_taxonomy/llama3.1-8b-it/NQ-Swap/dev_dedup_p_only/resid_hidden_state__unfiltered_all_pairs.npz}"
TAU_ROOT="${TAU_ROOT:-${ROOT}/data/policy_api/tau-bench}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/steer_taxonomy/tau-bench}"
ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-2}"
ALPHA_STEP="${ALPHA_STEP:-0.5}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_STEPS="${MAX_STEPS:-30}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
DIRECTION_MODE="${DIRECTION_MODE:-true}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export ATTN_IMPLEMENTATION
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "${PYTHON}" "${ROOT}/src/run_taubench_steer.py"
  --model "${MODEL}" --gpu "${GPU}" --env "${ENV}" --layers "${LAYERS}"
  --npz "${DIR_NPZ}" --tau-root "${TAU_ROOT}" --out-root "${OUT_ROOT}"
  --alpha-start "${ALPHA_START}" --alpha-end "${ALPHA_END}" --alpha-step "${ALPHA_STEP}"
  --start-index "${START_INDEX}" --end-index "${END_INDEX}"
  --max-samples "${MAX_SAMPLES}" --max-steps "${MAX_STEPS}"
  --context-length "${CONTEXT_LENGTH}"
  --direction-mode "${DIRECTION_MODE}"
)
[[ "${SKIP_EXISTING}" == "1" ]] && cmd+=(--skip-existing)
[[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)

echo "[tau] ENV=${ENV} MODEL=${MODEL} GPU=${GPU} tasks=${START_INDEX}:${END_INDEX} max=${MAX_SAMPLES}" >&2
echo "[tau] alpha=${ALPHA_START}..${ALPHA_END} step=${ALPHA_STEP}" >&2
echo "[tau] CONTEXT_LENGTH=${CONTEXT_LENGTH}" >&2
echo "[tau] OUT_ROOT=${OUT_ROOT}" >&2
exec "${cmd[@]}"
