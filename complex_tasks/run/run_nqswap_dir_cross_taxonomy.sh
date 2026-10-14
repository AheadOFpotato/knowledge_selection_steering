#!/usr/bin/env bash
# Cross-dataset steering: NQ-Swap direction, toward context only.
#
# Test set: every processed row, prompt = sub_context, gold = sub_answer.
# No Stage1 split. Alpha grid is 0, 0.5, 1, 1.5, 2 (positive = toward context).
#
# Direction NPZ (default):
#   ${DUMP_ROOT}/<model>/NQ-Swap/dev_dedup_p_only/
#     resid_hidden_state__unfiltered_all_pairs.npz
#
# Test manifests (built by scripts/build_context_manifests.py):
#   ${DUMP_ROOT}/<model>/<dataset>/<category>/internal_aware_data__context_all.json
#
# Usage:
#   DATASETS=RGB ./run/run_nqswap_dir_cross_taxonomy.sh
# Defaults are the isolated 500-sample run; normally only DATASETS is needed.

set -u
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
[[ "${STOP_ON_ERROR}" == "1" ]] && set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/src/run_resid_projection_ablation.py"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-llama3.1-8b-it}"
LAYERS="${LAYERS:-10}"
GPU="${GPU:-}"
DATASETS="${DATASETS:-RGB}"

if [[ ",${DATASETS}," == *",ToolBench,"* ]]; then
  echo "[error] ToolBench must use run/run_toolbench_official_tooleval.sh." >&2
  echo "        ToolEval requires executed multi-step agent trajectories; this runner is one-shot." >&2
  exit 2
fi

# MemoryAgentBench 6k prompts are ~6.1k Llama tokens.  Use SDPA to avoid the
# quadratic eager-attention matrix and batch one to keep long-context memory
# bounded. Explicit caller values still take precedence.
if [[ ",${DATASETS}," == *",MemoryAgentBench,"* ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  MAX_LENGTH="${MAX_LENGTH:-8192}"
  export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
else
  BATCH_SIZE="${BATCH_SIZE:-4}"
  MAX_LENGTH="${MAX_LENGTH:-4096}"
fi
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
DIR_GROUP="${DIR_GROUP:-unfiltered_all_pairs}"
GROUP="${GROUP:-context_all}"
JUDGE="${JUDGE:-official}"
DIRECTION_MODE="${DIRECTION_MODE:-true}"

DUMP_ROOT="${DUMP_ROOT:-${ROOT}/output/dump_taxonomy}"
STEER_OUT_ROOT="${STEER_OUT_ROOT:-${ROOT}/output/steer_taxonomy}"
PYTHON="${PYTHON:-python3}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${ROOT}/data/processed}"
export PROCESSED_ROOT

# Use only the isolated taxonomy direction; never silently fall back to legacy.
if [[ -z "${DIR_NPZ:-}" ]]; then
  _cand="${DUMP_ROOT}/${MODEL}/NQ-Swap/dev_dedup_p_only/resid_hidden_state__${DIR_GROUP}.npz"
  DIR_NPZ="${_cand}"
  DIR_TAG="${DIR_TAG:-NQ-Swap__dev_dedup_p_only}"
fi
DIR_TAG="${DIR_TAG:-NQ-Swap__dev_dedup_p_only}"
if [[ "${DIRECTION_MODE}" != "true" && "${DIR_TAG}" != *"__${DIRECTION_MODE}"* && "${DIR_TAG}" != *"__${DIRECTION_MODE}__"* ]]; then
  DIR_TAG="${DIR_TAG}__${DIRECTION_MODE}"
fi
if [[ "${DIR_TAG}" != *__toward_context ]]; then
  DIR_TAG="${DIR_TAG}__toward_context"
fi

ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-2}"
ALPHA_STEP="${ALPHA_STEP:-0.5}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
STAGE1_MIN_PASS_RATE="${STAGE1_MIN_PASS_RATE:--1}"  # -1: do not early-stop on Stage1 rate
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${GPU}" ]]; then
  case "${DATASETS}" in
    RGB|LongMemEval) GPU=0 ;;
    CRAG|Mem2ActBench) GPU=1 ;;
    *) GPU=0 ;;
  esac
fi
# Optional bash glob matched against category directory names. Empty means every category.
CATEGORY_GLOB="${CATEGORY_GLOB:-}"

dataset_enabled() {
  local ds="$1" x
  IFS=',' read -r -a _f <<< "${DATASETS}"
  for x in "${_f[@]}"; do
    x="${x// /}"
    [[ "${x}" == "${ds}" ]] && return 0
  done
  return 1
}

cross_out_dir() {
  local test_dataset="$1" test_cat="$2"
  printf '%s/%s/%s/%s__dir_%s__test_%s' \
    "${STEER_OUT_ROOT}" "${MODEL}" "${test_dataset}" "${test_cat}" "${DIR_TAG}" "${test_cat}"
}

already_done() {
  local out_dir="$1"
  python3 - <<'PY' "${out_dir}" "${LAYERS}" "${ALPHA_START}" "${ALPHA_END}" "${ALPHA_STEP}"
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
layers_tag = str(sys.argv[2]).replace(",", "_").replace(" ", "")
a0, a1, step = map(float, sys.argv[3:6])
def grid(a0, a1, step):
    if abs(a0 - a1) < 1e-12:
        return [a0]
    out, x, n = [], float(a0), 0
    if step > 0:
        while x <= a1 + 1e-9 and n < 1_000_000:
            out.append(round(x, 12)); x += step; n += 1
    else:
        while x >= a1 - 1e-9 and n < 1_000_000:
            out.append(round(x, 12)); x += step; n += 1
    return out
for a in grid(a0, a1, step):
    if not (out_dir / f"L{layers_tag}_alpha{float(a):g}_steer_result.json").is_file():
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_one() {
  local test_dataset="$1" test_cat="$2" test_json="$3"
  local out_dir
  out_dir="$(cross_out_dir "${test_dataset}" "${test_cat}")"
  if [[ "${SKIP_EXISTING}" == "1" ]] && already_done "${out_dir}"; then
    echo "[skip] done: ${out_dir}" >&2
    return 0
  fi
  if [[ ! -f "${test_json}" ]]; then
    echo "[skip] missing manifest: ${test_json}" >&2
    return 0
  fi
  local -a cmd=(
    "${PYTHON}" "${PY}"
    --model "${MODEL}"
    --npz "${DIR_NPZ}"
    --test-json "${test_json}"
    --out-root "${STEER_OUT_ROOT}"
    --output-category "${test_cat}"
    --direction-category-tag "${DIR_TAG}"
    --test-category-tag "${test_cat}"
    --gpu "${GPU}"
    --layers "${LAYERS}"
    --alpha-start "${ALPHA_START}"
    --alpha-end "${ALPHA_END}"
    --alpha-step "${ALPHA_STEP}"
    --batch-size "${BATCH_SIZE}"
    --max-length "${MAX_LENGTH}"
    --judge-workers "${JUDGE_WORKERS}"
    --max-samples-stage1 0
    --max-samples-stage2 "${MAX_SAMPLES}"
    --stage1-min-pass-rate "${STAGE1_MIN_PASS_RATE}"
    --steer-toward context
    --direction-mode "${DIRECTION_MODE}"
    --context-only
    --judge "${JUDGE}"
  )
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi
  echo "[run] NQ-Swap dir -> ${test_dataset}/${test_cat}  alpha ${ALPHA_START}..${ALPHA_END} step ${ALPHA_STEP}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo >&2
    return 0
  fi
  "${cmd[@]}"
}

cd "${ROOT}" || exit 1
mkdir -p "${STEER_OUT_ROOT}"

if [[ "${DRY_RUN}" != "1" && ! -f "${DIR_NPZ}" ]]; then
  echo "[error] NQ-Swap direction NPZ missing: ${DIR_NPZ}" >&2
  echo "Tip: set DIR_NPZ=... or dump NQ-Swap first under conflict_steer." >&2
  exit 1
fi

echo "[nqswap_cross_taxonomy] MODEL=${MODEL} LAYERS=${LAYERS} GPU=${GPU}" >&2
echo "[nqswap_cross_taxonomy] DIR_NPZ=${DIR_NPZ}" >&2
echo "[nqswap_cross_taxonomy] DUMP_ROOT=${DUMP_ROOT}" >&2
echo "[nqswap_cross_taxonomy] STEER_OUT_ROOT=${STEER_OUT_ROOT}" >&2
echo "[nqswap_cross_taxonomy] DATASETS=${DATASETS} GROUP=${GROUP} JUDGE=${JUDGE}" >&2
echo "[nqswap_cross_taxonomy] DIRECTION_MODE=${DIRECTION_MODE}" >&2
echo "[nqswap_cross_taxonomy] BATCH_SIZE=${BATCH_SIZE} MAX_LENGTH=${MAX_LENGTH}" >&2
echo "[nqswap_cross_taxonomy] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}" >&2
echo "[nqswap_cross_taxonomy] alphas ${ALPHA_START}..${ALPHA_END} step ${ALPHA_STEP} toward context" >&2

if [[ "${DRY_RUN}" != "1" && "${BUILD_MANIFESTS:-1}" == "1" ]]; then
  missing_processed=()
  IFS=',' read -r -a _build_ds_list <<< "${DATASETS}"
  for _ds in "${_build_ds_list[@]}"; do
    _ds="${_ds// /}"
    [[ -z "${_ds}" ]] && continue
    if [[ ! -d "${PROCESSED_ROOT}/${_ds}" ]] || [[ -z "$(find "${PROCESSED_ROOT}/${_ds}" -maxdepth 1 -name '*.jsonl' -print -quit 2>/dev/null)" ]]; then
      missing_processed+=("${_ds}")
    fi
  done
  if [[ "${#missing_processed[@]}" -gt 0 ]]; then
    _missing_csv="$(IFS=,; echo "${missing_processed[*]}")"
    "${PYTHON}" "${ROOT}/scripts/build_processed_jsonl.py" \
      --datasets "${_missing_csv}"
  fi
  "${PYTHON}" "${ROOT}/scripts/build_context_manifests.py" \
    --model "${MODEL}" \
    --datasets "${DATASETS}" \
    --out-root "${DUMP_ROOT}"
fi

n_jobs=0
IFS=',' read -r -a DS_LIST <<< "${DATASETS}"
for test_dataset in "${DS_LIST[@]}"; do
  test_dataset="${test_dataset// /}"
  ds_dir="${DUMP_ROOT}/${MODEL}/${test_dataset}"
  if [[ ! -d "${ds_dir}" ]]; then
    echo "[warn] no dump dir (run run_dump_taxonomy.sh first): ${ds_dir}" >&2
    continue
  fi
  echo "[nqswap_cross_taxonomy] === ${test_dataset} ===" >&2
  shopt -s nullglob
  if [[ -n "${CATEGORY_GLOB}" ]]; then
    cat_dirs=( "${ds_dir}"/${CATEGORY_GLOB}/ )
  else
    cat_dirs=( "${ds_dir}"/*/ )
  fi
  for cat_dir in "${cat_dirs[@]}"; do
    [[ -d "${cat_dir}" ]] || continue
    test_cat="$(basename "${cat_dir}")"
    test_json="${cat_dir}internal_aware_data__${GROUP}.json"
    run_one "${test_dataset}" "${test_cat}" "${test_json}"
    n_jobs=$((n_jobs + 1))
  done
  shopt -u nullglob
done

echo "[nqswap_cross_taxonomy] done (${n_jobs} category jobs)." >&2
