#!/usr/bin/env bash
# ParaConflict cross-format dump: Substitution Conflict as Stage2 wrong context.
# Output: output/dump/cross_format/<model>/ParaConfilct/<Category>/  (one combined dump)
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
  local model="$1" category="$2"
  printf '%s/%s/%s/%s' "${OUT_ROOT}" "${model}" "${DATASET}" "${category}"
}

already_dumped() {
  local out_dir="$1"
  [[ -f "${out_dir}/internal_aware_data__stage1_correct_use_wrong_context.json" ]] \
    && [[ -f "${out_dir}/internal_aware_data__stage1_wrong_use_correct_context.json" ]]
}

run_dump() {
  local model="$1" category="$2"
  local out_dir
  out_dir="$(out_dir_for "${model}" "${category}")"

  if [[ "${SKIP_EXISTING}" == "1" ]] && already_dumped "${out_dir}"; then
    echo "[skip] ${model} ${category}" >&2
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
    --gpu "${GPU}"
    --batch-size "${BATCH_SIZE}"
    --max-samples "${MAX_SAMPLES}"
    --save-resid "${SAVE_RESID}"
  )

  echo "[run] ${model} | ${category} | style=${PARACONFLICT_CONFLICT_STYLE}" >&2
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

echo "[cross_format dump] OUT_ROOT=${OUT_ROOT} (combined dumps, split at steer time)" >&2

for model in "${MODEL_LIST[@]}"; do
  model="${model// /}"
  [[ -z "${model}" ]] && continue
  for cat in "${PARACONFLICT_CATEGORIES[@]}"; do
    category_enabled "${cat}" || continue
    run_dump "${model}" "${cat}"
  done
done

echo "[cross_format dump] done." >&2
