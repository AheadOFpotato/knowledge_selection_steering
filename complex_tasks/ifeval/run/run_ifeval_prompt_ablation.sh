#!/usr/bin/env bash
# IFEval (Google instruction_following_eval): generate + strict/loose eval; optional resid ablation.
#
# Usage:
#   ./run/run_ifeval_prompt_ablation.sh
#   MODEL_FAMILY=qwen GPU=0 ./run/run_ifeval_prompt_ablation.sh
#   RUN_MODE=baseline MODEL_FAMILY=llama GPU=0 ./run/run_ifeval_prompt_ablation.sh
#
# Models (pick one via MODEL_FAMILY or MODEL):
#   llama | qwen
#
# Modes:
#   RUN_MODE=baseline  — no steering hooks
#   RUN_MODE=ablation  — h' = h + α·v, v = mean(clean) − mean(conflict); default α 0 → -1 step -0.1

set -u

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BUNDLE_ROOT}"
# shellcheck source=run_common.sh
source "${BUNDLE_ROOT}/run/run_common.sh"

PY="${BUNDLE_ROOT}/src/run_ifeval_prompt_ablation.py"

MODEL="${MODEL:-}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"

apply_model_defaults() {
  case "${MODEL_FAMILY}" in
    llama|Llama|LLAMA)
      MODEL=llama3-8b-it
      LAYERS="${LAYERS:-10}"
      ;;
    qwen|Qwen|QWEN)
      MODEL=qwen3-8b
      LAYERS="${LAYERS:-21}"
      ;;
    *)
      echo "[error] MODEL_FAMILY must be llama|qwen (got: ${MODEL_FAMILY})" >&2
      exit 1
      ;;
  esac
}

if [[ -n "${MODEL// /}" ]]; then
  case "${MODEL}" in
    llama3-8b-it) LAYERS="${LAYERS:-10}" ;;
    qwen3-8b) LAYERS="${LAYERS:-21}" ;;
    *) echo "[error] Unknown MODEL=${MODEL}; supported: llama3-8b-it, qwen3-8b" >&2; exit 1 ;;
  esac
else
  apply_model_defaults
fi

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

IFEVAL_ROOT="${IFEVAL_ROOT:-${BUNDLE_ROOT}/vendor}"
INPUT_DATA="${INPUT_DATA:-${IFEVAL_ROOT}/instruction_following_eval/data/input_data.jsonl}"

IF_PROMPT_OUT_ROOT="${IF_PROMPT_OUT_ROOT:-output/if_eval}"
RUN_NAME="${RUN_NAME:-}"

KEYS="${KEYS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CATEGORY="${CATEGORY:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RUN_MODE="${RUN_MODE:-ablation}"

REPO_ROOT="$(cd "${BUNDLE_ROOT}/.." && pwd)"
DUMP_ROOT="${DUMP_ROOT:-${REPO_ROOT}/output/dump}"
DATASET="${DATASET:-NQ-Swap}"
NQ_CATEGORY="${NQ_CATEGORY:-dev}"
GROUP="${GROUP:-stage1_correct_use_wrong_context}"
RESID_DIRECTION="${RESID_DIRECTION:-clean_minus_conflict}"
ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:--1}"
ALPHA_STEP="${ALPHA_STEP:--0.1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
SPLIT_SEED="${SPLIT_SEED:-42}"
CROSS_VAL_RATIO="${CROSS_VAL_RATIO:-0.5}"

# Steering direction v: full NQ-Swap dev NPZ (cross-dataset protocol)
TRAIN_NPZ="${DUMP_ROOT}/${MODEL}/${DATASET}/${NQ_CATEGORY}/resid_hidden_state__${GROUP}.npz"
NPZ_REL="${NPZ:-npz/${MODEL}/${DATASET}/${NQ_CATEGORY}/resid_hidden_state__${GROUP}.npz}"
if [[ -f "${TRAIN_NPZ}" ]]; then
  NPZ="${TRAIN_NPZ}"
else
  NPZ="$(resolve_npz_path "${MODEL}" "${NPZ_REL}")"
fi

if [[ ! -f "${BUNDLE_ROOT}/config/models.json" ]]; then
  echo "[error] config/models.json missing. Copy from config/models.json.example" >&2
  exit 1
fi

check_model_weights "${MODEL}" || exit 1

RUN_NAME_ARGS=()
[[ -n "${RUN_NAME}" ]] && RUN_NAME_ARGS=(--run-name "${RUN_NAME}")

KEYS_ARGS=()
[[ -n "${KEYS}" ]] && KEYS_ARGS=(--keys "${KEYS}")

MAX_SAMPLES_ARGS=()
[[ "${MAX_SAMPLES}" != "0" ]] && MAX_SAMPLES_ARGS=(--max-samples "${MAX_SAMPLES}")

CATEGORY_ARGS=()
[[ -n "${CATEGORY}" ]] && CATEGORY_ARGS=(--category "${CATEGORY}")

COMMON=(
  python3 "${PY}"
  --model "${MODEL}"
  --gpu "${GPU}"
  --batch-size "${BATCH_SIZE}"
  --ifeval-root "${IFEVAL_ROOT}"
  --input-data "${INPUT_DATA}"
  --out-root "${IF_PROMPT_OUT_ROOT}"
  "${RUN_NAME_ARGS[@]}"
  "${KEYS_ARGS[@]}"
  "${MAX_SAMPLES_ARGS[@]}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  "${CATEGORY_ARGS[@]}"
)

echo "[run_ifeval_prompt_ablation] MODEL_FAMILY=${MODEL_FAMILY} -> MODEL=${MODEL} LAYERS=${LAYERS} RUN_MODE=${RUN_MODE}" >&2

if [[ "${RUN_MODE}" == "baseline" ]]; then
  echo "[run_ifeval_prompt_ablation] baseline: NPZ/LAYERS/ALPHA_* ignored." >&2
  echo "[run_ifeval_prompt_ablation] INPUT_DATA=${INPUT_DATA}" >&2
  "${COMMON[@]}"
elif [[ "${RUN_MODE}" == "ablation" ]]; then
  echo "[run_ifeval_prompt_ablation] steer: ${RESID_DIRECTION}; alpha ${ALPHA_START}..${ALPHA_END} step ${ALPHA_STEP}; early_stop_patience=${EARLY_STOP_PATIENCE}" >&2
  echo "[run_ifeval_prompt_ablation] direction NPZ (NQ-Swap full): ${NPZ}" >&2
  echo "[run_ifeval_prompt_ablation] alpha grid: IFEval 50/50 split (seed=${SPLIT_SEED})" >&2
  if [[ ! -f "${NPZ}" ]]; then
    echo "[error] NPZ not found: ${NPZ}" >&2
    echo "  Run ../run/run_dump_all_datasets.sh first, or copy NPZ to:" >&2
    echo "    npz/<model>/NQ-Swap/dev/resid_hidden_state__${GROUP}.npz" >&2
    exit 1
  fi
  "${COMMON[@]}" \
    --npz "${NPZ}" \
    --layers "${LAYERS}" \
    --resid-direction "${RESID_DIRECTION}" \
    --alpha-start "${ALPHA_START}" \
    --alpha-end "${ALPHA_END}" \
    --alpha-step "${ALPHA_STEP}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --split-seed "${SPLIT_SEED}" \
    --cross-val-ratio "${CROSS_VAL_RATIO}"
else
  echo "[error] RUN_MODE must be baseline or ablation" >&2
  exit 1
fi
