#!/usr/bin/env bash
# ParaConflict cross-format dump: Substitution Conflict as Stage2 wrong context.
# Output: output/dump/cross_format/<model>/ParaConfilct/<Category>__split_<train|val|test>/
#
# Usage:
#   ./run/run_dump_paraconflict_substitution_cross_format.sh
#   MODELS=llama3-8b-it GPU=0 DRY_RUN=1 ./run/run_dump_paraconflict_substitution_cross_format.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"

PY="${ROOT}/src/dump_activations_from_testjsonl.py"
OUT_ROOT="${OUT_ROOT:-${CROSS_FORMAT_DUMP_ROOT}}"

MODELS="${MODELS:-llama3-8b-it}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SAVE_RESID="${SAVE_RESID:-true}"
DATASET="${DATASET:-ParaConfilct}"
PARACONFLICT_JSONL="${PARACONFLICT_JSONL:-${DATA_ROOT}/ParaConfilct/test.jsonl}"
PARACONFLICT_CONFLICT_STYLE="${PARACONFLICT_CONFLICT_STYLE:-substitution}"
SPLITS="${SPLITS:-train,val,test}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
CATEGORIES="${CATEGORIES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

PARACONFLICT_CATEGORIES=(
  "Athelete Sport"
  "Book Author"
  "Company Founder"
  "Company Headquarter"
  "Official Language"
  "World Capital"
)

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
IFS=',' read -r -a SPLIT_LIST <<< "${SPLITS}"

category_enabled() {
  local cat="$1"
  [[ -z "${CATEGORIES}" ]] && return 0
  local x
  IFS=',' read -r -a _f <<< "${CATEGORIES}"
  for x in "${_f[@]}"; do
    [[ "${x// /}" == "${cat}" ]] && return 0
  done
  return 1
}

out_dir_for() {
  local model="$1" category="$2" split="$3"
  local cat_out="${category}"
  if [[ -n "${split}" ]]; then
    cat_out="${category}__split_${split}"
  fi
  printf '%s/%s/%s/%s' "${OUT_ROOT}" "${model}" "${DATASET}" "${cat_out}"
}

already_dumped() {
  local out_dir="$1"
  [[ -f "${out_dir}/internal_aware_data__stage1_correct_use_wrong_context.json" ]] \
    && [[ -f "${out_dir}/internal_aware_data__stage1_wrong_use_correct_context.json" ]]
}

run_dump() {
  local model="$1" category="$2" split="$3"
  local out_dir
  out_dir="$(out_dir_for "${model}" "${category}" "${split}")"

  if [[ "${SKIP_EXISTING}" == "1" ]] && already_dumped "${out_dir}"; then
    echo "[skip] ${model} ${category} split=${split}" >&2
    return 0
  fi

  local -a cmd=(
    python3 "${PY}"
    --model "${model}"
    --dataset "${DATASET}"
    --category "${category}"
    --out-root "${OUT_ROOT}"
    --input-jsonl "${PARACONFLICT_JSONL}"
    --paraconflict-conflict-style "${PARACONFLICT_CONFLICT_STYLE}"
    --data-split "${split}"
    --split-seed "${SPLIT_SEED}"
    --train-ratio "${TRAIN_RATIO}"
    --val-ratio "${VAL_RATIO}"
    --gpu "${GPU}"
    --batch-size "${BATCH_SIZE}"
    --max-samples "${MAX_SAMPLES}"
    --save-resid "${SAVE_RESID}"
  )

  echo "[run] ${model} | ${category} | split=${split} | style=${PARACONFLICT_CONFLICT_STYLE}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo >&2
    return 0
  fi
  "${cmd[@]}"
}

cd "${ROOT}" || exit 1
mkdir -p "${OUT_ROOT}"
[[ -f "${PY}" ]] || { echo "[error] not found: ${PY}" >&2; exit 1; }
[[ -f "${PARACONFLICT_JSONL}" ]] || { echo "[error] not found: ${PARACONFLICT_JSONL}" >&2; exit 1; }

echo "[cross_format dump] OUT_ROOT=${OUT_ROOT} SPLITS=${SPLITS}" >&2

for model in "${MODEL_LIST[@]}"; do
  model="${model// /}"
  [[ -z "${model}" ]] && continue
  for cat in "${PARACONFLICT_CATEGORIES[@]}"; do
    category_enabled "${cat}" || continue
    for sp in "${SPLIT_LIST[@]}"; do
      sp="${sp// /}"
      [[ -n "${sp}" ]] || continue
      run_dump "${model}" "${cat}" "${sp}"
    done
  done
done

echo "[cross_format dump] done." >&2
