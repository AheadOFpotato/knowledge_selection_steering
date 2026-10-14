#!/usr/bin/env bash
# Batch steering: runtime split (same_dataset 70/15/15 per category).
#
# Expects one combined dump per category under DUMP_ROOT (see run_dump_all_datasets.sh).
#
# Usage:
#   MODEL=llama3-8b-it LAYERS=10 GPU=0 ./run/run_resid_projection_ablation_all_final.sh
#   DATASETS=NQ-Swap,counterfact SKIP_EXISTING=1 DRY_RUN=1 ./run/run_resid_projection_ablation_all_final.sh

set -u
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
[[ "${STOP_ON_ERROR}" == "1" ]] && set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"

PY="${ROOT}/src/run_resid_projection_ablation.py"

MODEL="${MODEL:-}"
MODELS="${MODELS:-}"
LAYERS="${LAYERS:-}"
GPU="${GPU:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GROUP="${GROUP:-stage1_correct_use_wrong_context}"
DATASETS="${DATASETS:-}"
CATEGORIES="${CATEGORIES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"

ALPHA_START="${ALPHA_START:-0.2}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"

ALLOWED_DATASETS=(macnoise NQ-Swap ParaConfilct memotrap counterfact)
DEFAULT_DATASET_ORDER=("${ALLOWED_DATASETS[@]}")

build_dataset_order() {
  DATASET_ORDER=()
  if [[ -n "${DATASETS// /}" ]]; then
    local x
    IFS=',' read -r -a _ds <<< "${DATASETS}"
    for x in "${_ds[@]}"; do
      x="${x// /}"
      [[ -n "${x}" ]] && DATASET_ORDER+=("${x}")
    done
    return 0
  fi
  DATASET_ORDER=("${DEFAULT_DATASET_ORDER[@]}")
}

build_model_list() {
  MODEL_LIST=()
  if [[ -n "${MODEL// /}" ]]; then
    MODEL_LIST=("${MODEL// /}")
    return 0
  fi
  if [[ -n "${MODELS// /}" ]]; then
    local m
    IFS=',' read -r -a _ms <<< "${MODELS}"
    for m in "${_ms[@]}"; do
      m="${m// /}"
      [[ -n "${m}" ]] && MODEL_LIST+=("${m}")
    done
    return 0
  fi
  echo "[error] set MODEL or MODELS" >&2
  exit 1
}

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
  [[ -z "${DATASETS}" ]] && return 0
  local x
  IFS=',' read -r -a _ds <<< "${DATASETS}"
  for x in "${_ds[@]}"; do
    [[ "${x// /}" == "${ds}" ]] && return 0
  done
  return 1
}

category_enabled() {
  local category="$1"
  [[ -z "${CATEGORIES}" ]] && return 0
  local pid tag cat_lc="${category,,}"
  IFS=',' read -r -a _cf <<< "${CATEGORIES}"
  for pid in "${_cf[@]}"; do
    pid="${pid// /}"
    [[ -z "${pid}" ]] && continue
    tag="_$(echo "${pid}" | tr '[:upper:]' '[:lower:]')_"
    [[ "${cat_lc}" == *"${tag}"* ]] && return 0
  done
  return 1
}

out_cat_slug() {
  local base="$1"
  printf '%s__dir_%s__test_%s' "${base}" "${base}" "${base}"
}

steer_already_done() {
  local model="$1"
  local dataset="$2"
  local base_cat="$3"
  local out_dir="${STEER_OUT_ROOT}/${model}/${dataset}/$(out_cat_slug "${base_cat}")"
  local ls_tag="${LAYERS//,/_}"
  [[ -f "${out_dir}/L${ls_tag}_val_alpha_grid.json" ]] \
    && compgen -G "${out_dir}/L${ls_tag}_alpha"*.json >/dev/null 2>&1
}

run_category() {
  local model="$1"
  local dataset="$2"
  local base_cat="$3"

  local npz="${DUMP_ROOT}/${model}/${dataset}/${base_cat}/resid_hidden_state__${GROUP}.npz"
  local manifest="${DUMP_ROOT}/${model}/${dataset}/${base_cat}/internal_aware_data__${GROUP}.json"

  if [[ ! -f "${npz}" || ! -f "${manifest}" ]]; then
    echo "[skip] missing dump for ${model}/${dataset}/${base_cat}" >&2
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && steer_already_done "${model}" "${dataset}" "${base_cat}"; then
    echo "[skip] steer results exist for ${model}/${dataset}/${base_cat}" >&2
    return 0
  fi

  local -a cmd=(
    python3 "${PY}"
    --eval-mode train_val_test
    --split-protocol same_dataset
    --split-seed "${SPLIT_SEED}"
    --train-ratio "${TRAIN_RATIO}"
    --val-ratio "${VAL_RATIO}"
    --model "${model}"
    --npz "${npz}"
    --test-json "${manifest}"
    --out-root "${STEER_OUT_ROOT}"
    --output-category "${base_cat}"
    --direction-category-tag "${base_cat}"
    --test-category-tag "${base_cat}"
    --gpu "${GPU}"
    --layers "${LAYERS}"
    --alpha-start "${ALPHA_START}"
    --alpha-end "${ALPHA_END}"
    --alpha-step "${ALPHA_STEP}"
    --batch-size "${BATCH_SIZE}"
    --max-samples-stage1 0
    --max-samples-stage2 0
    --stage1-min-pass-rate -1
  )

  echo "[run] ${model} | ${dataset} | ${base_cat} (runtime 70/15/15)" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo >&2
    return 0
  fi
  "${cmd[@]}"
}

cd "${ROOT}" || exit 1
mkdir -p "${STEER_OUT_ROOT}"

[[ -n "${LAYERS// /}" ]] || { echo "[error] LAYERS required" >&2; exit 1; }
[[ -n "${GPU// /}" ]] || { echo "[error] GPU required" >&2; exit 1; }

build_model_list
build_dataset_order

echo "[batch] DUMP_ROOT=${DUMP_ROOT} STEER_OUT_ROOT=${STEER_OUT_ROOT}" >&2
echo "[batch] models=${MODEL_LIST[*]} LAYERS=${LAYERS} alpha=${ALPHA_START}..${ALPHA_END}" >&2

for model in "${MODEL_LIST[@]}"; do
  model="${model// /}"
  model_dir="${DUMP_ROOT}/${model}"
  [[ -d "${model_dir}" ]] || { echo "[warn] skip missing ${model_dir}" >&2; continue; }

  for dataset in "${DATASET_ORDER[@]}"; do
    dataset_enabled "${dataset}" || continue
    ds_dir="${model_dir}/${dataset}"
    [[ -d "${ds_dir}" ]] || continue

    shopt -s nullglob
    for cat_dir in "${ds_dir}"/*/; do
      [[ -d "${cat_dir}" ]] || continue
      base_cat="$(basename "${cat_dir}")"
      [[ "${base_cat}" == _runtime_split ]] && continue
      [[ "${base_cat}" == *__split_* ]] && continue
      category_enabled "${base_cat}" || continue
      run_category "${model}" "${dataset}" "${base_cat}"
    done
    shopt -u nullglob
  done
done

echo "[batch] done." >&2
