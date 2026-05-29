#!/usr/bin/env bash
# Batch dump: resid_hidden_state__*.npz + internal_aware_data__*.json
# Datasets: macnoise, NQ-Swap, ParaConfilct, memotrap, counterfact only.
# Train/val/test splits via --data-split (see src/dump_activations_from_testjsonl.py).
#
# Usage:
#   ./run/run_dump_all_datasets.sh
#   MODELS=llama3-8b-it GPU=0 ./run/run_dump_all_datasets.sh
#   DATASETS=NQ-Swap,counterfact SPLITS=train,val DRY_RUN=1 ./run/run_dump_all_datasets.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"

PY="${ROOT}/src/dump_activations_from_testjsonl.py"
OUT_ROOT="${OUT_ROOT:-${DUMP_ROOT}}"

MODELS="${MODELS:-llama3-8b-it}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SAVE_RESID="${SAVE_RESID:-true}"
DATASETS="${DATASETS:-}"
SPLITS="${SPLITS:-train,val,test}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

# NQ-Swap: fixed dev.jsonl only
NQ_SWAP_JSONL="${DATA_ROOT}/NQ-Swap/dev.jsonl"
NQ_SWAP_CATEGORY="dev"

PARACONFLICT_CATEGORIES=(
  "Athelete Sport"
  "Book Author"
  "Company Founder"
  "Company Headquarter"
  "Official Language"
  "World Capital"
)

COUNTERFACT_RELATION_IDS=(P27 P176 P106 P140 P641 P449)

# Only these five datasets are supported.
ALLOWED_DATASETS=(macnoise NQ-Swap ParaConfilct memotrap counterfact)

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
IFS=',' read -r -a SPLIT_LIST <<< "${SPLITS}"

dataset_known() {
  local ds="$1" a
  for a in "${ALLOWED_DATASETS[@]}"; do
    [[ "${a}" == "${ds}" ]] && return 0
  done
  return 1
}

dataset_enabled() {
  local ds="$1"
  dataset_known "${ds}" || {
    echo "[error] unknown dataset '${ds}'; allowed: ${ALLOWED_DATASETS[*]}" >&2
    return 1
  }
  if [[ -z "${DATASETS}" ]]; then
    return 0
  fi
  local x
  IFS=',' read -r -a _ds_filter <<< "${DATASETS}"
  for x in "${_ds_filter[@]}"; do
    x="${x// /}"
    if [[ "${x}" == "${ds}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "${DATASETS}" ]]; then
  for _ds_name in $(echo "${DATASETS}" | tr ',' ' '); do
    _ds_name="${_ds_name// /}"
    [[ -z "${_ds_name}" ]] && continue
    dataset_known "${_ds_name}" || exit 1
  done
fi

counterfact_category_enabled() {
  local category="$1"
  local pid tag
  local cat_lc="${category,,}"
  for pid in "${COUNTERFACT_RELATION_IDS[@]}"; do
    tag="_$(echo "${pid}" | tr '[:upper:]' '[:lower:]')_"
    if [[ "${cat_lc}" == *"${tag}"* ]]; then
      return 0
    fi
  done
  return 1
}

out_dir_for() {
  local model="$1"
  local dataset="$2"
  local category="$3"
  local split="$4"
  local cat_out="${category}"
  if [[ -n "${split}" && "${split}" != "none" ]]; then
    cat_out="${category}__split_${split}"
  fi
  printf '%s/%s/%s/%s' "${OUT_ROOT}" "${model}" "${dataset}" "${cat_out}"
}

already_dumped() {
  local out_dir="$1"
  [[ -f "${out_dir}/internal_aware_data__stage1_correct_use_wrong_context.json" ]] \
    && [[ -f "${out_dir}/internal_aware_data__stage1_wrong_use_correct_context.json" ]]
}

run_dump() {
  local model="$1"
  local dataset="$2"
  local category="$3"
  local split="$4"
  local input_jsonl="${5:-}"

  local out_dir
  out_dir="$(out_dir_for "${model}" "${dataset}" "${category}" "${split}")"

  if [[ "${SKIP_EXISTING}" == "1" ]] && already_dumped "${out_dir}"; then
    echo "[skip] ${model} ${dataset} ${category} split=${split} (${out_dir})" >&2
    return 0
  fi

  local -a cmd=(
    python3 "${PY}"
    --model "${model}"
    --dataset "${dataset}"
    --category "${category}"
    --out-root "${OUT_ROOT}"
    --gpu "${GPU}"
    --batch-size "${BATCH_SIZE}"
    --max-samples "${MAX_SAMPLES}"
    --save-resid "${SAVE_RESID}"
    --split-seed "${SPLIT_SEED}"
    --train-ratio "${TRAIN_RATIO}"
    --val-ratio "${VAL_RATIO}"
  )
  if [[ -n "${split}" && "${split}" != "none" ]]; then
    cmd+=(--data-split "${split}")
  fi
  if [[ -n "${input_jsonl}" ]]; then
    cmd+=(--input-jsonl "${input_jsonl}")
  fi

  echo "[run] ${model} | ${dataset} | ${category} | split=${split}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo >&2
    return 0
  fi
  "${cmd[@]}"
}

run_all_splits() {
  local model="$1"
  local dataset="$2"
  local category="$3"
  local input_jsonl="${4:-}"
  local sp
  for sp in "${SPLIT_LIST[@]}"; do
    sp="${sp// /}"
    [[ -z "${sp}" ]] && continue
    run_dump "${model}" "${dataset}" "${category}" "${sp}" "${input_jsonl}"
  done
}

cd "${ROOT}" || exit 1
mkdir -p "${OUT_ROOT}"

if [[ ! -f "${PY}" ]]; then
  echo "[error] not found: ${PY}" >&2
  exit 1
fi

echo "[run_dump_all_datasets] ROOT=${ROOT} OUT_ROOT=${OUT_ROOT}" >&2
echo "[run_dump_all_datasets] SPLITS=${SPLITS} seed=${SPLIT_SEED} train=${TRAIN_RATIO} val=${VAL_RATIO}" >&2
echo "[run_dump_all_datasets] NQ-Swap: ${NQ_SWAP_JSONL} (category=${NQ_SWAP_CATEGORY})" >&2

for model in "${MODEL_LIST[@]}"; do
  model="${model// /}"
  [[ -z "${model}" ]] && continue

  if dataset_enabled macnoise; then
    shopt -s nullglob
    for f in "${DATA_ROOT}/macnoise"/*.json; do
      cat="$(basename "${f}" .json)"
      run_all_splits "${model}" macnoise "${cat}"
    done
    shopt -u nullglob
  fi

  if dataset_enabled memotrap; then
    shopt -s nullglob
    for f in "${DATA_ROOT}/memotrap"/*.csv; do
      cat="$(basename "${f}" .csv)"
      run_all_splits "${model}" memotrap "${cat}"
    done
    shopt -u nullglob
  fi

  if dataset_enabled ParaConfilct; then
    for cat in "${PARACONFLICT_CATEGORIES[@]}"; do
      run_all_splits "${model}" ParaConfilct "${cat}"
    done
  fi

  if dataset_enabled counterfact; then
    shopt -s nullglob
    for f in "${DATA_ROOT}/counterfact"/counterfact_relation_*.json; do
      base="$(basename "${f}")"
      [[ "${base}" == *manifest* ]] && continue
      cat="$(basename "${f}" .json)"
      counterfact_category_enabled "${cat}" || continue
      run_all_splits "${model}" counterfact "${cat}"
    done
    shopt -u nullglob
  fi

  if dataset_enabled NQ-Swap; then
    if [[ ! -f "${NQ_SWAP_JSONL}" ]]; then
      echo "[error] NQ-Swap dev.jsonl not found: ${NQ_SWAP_JSONL}" >&2
      exit 1
    fi
    run_all_splits "${model}" NQ-Swap "${NQ_SWAP_CATEGORY}" "${NQ_SWAP_JSONL}"
  fi
done

echo "[run_dump_all_datasets] done." >&2
