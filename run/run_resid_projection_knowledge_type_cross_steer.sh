#!/usr/bin/env bash
# CounterFact six knowledge types (P27 P176 P106 P140 P641 P449) cross steering 6×6:
#   - Direction: <DUMP_ROOT>/<model>/counterfact/<dir_category>__split_train/
#   - Val / test: same dataset, test category splits
#   - train → direction, val → alpha, test → metrics
#
# Output:
#   ${STEER_OUT_ROOT}/<model>/counterfact/<test>__split_test__dir_<dir>__split_train__test_<test>__split_test/
#
# Usage:
#   ./run/run_resid_projection_knowledge_type_cross_steer.sh
#   MODEL=llama3-8b-it LAYERS=10 GPU=0 DRY_RUN=1 ./run/run_resid_projection_knowledge_type_cross_steer.sh

set -u
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
[[ "${STOP_ON_ERROR}" == "1" ]] && set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"
# shellcheck source=scripts/cross_steer_common.sh
source "${ROOT}/run/scripts/cross_steer_common.sh"

STEER_OUT_ROOT="${STEER_OUT_ROOT:-${KNOWLEDGE_TYPE_STEER_ROOT}}"

ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-llama3-8b-it}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DATASET="${DATASET:-counterfact}"
KNOWLEDGE_TYPE_IDS=(P27 P176 P106 P140 P641 P449)

LAYERS=""
apply_model_layers_default "${MODEL}"

resolve_category_for_type() {
  local model="$1" pid="$2"
  local tag ds_dir
  tag="_$(echo "${pid}" | tr '[:upper:]' '[:lower:]')_"
  ds_dir="${DUMP_ROOT}/${model}/${DATASET}"
  [[ -d "${ds_dir}" ]] || return 1
  local name
  for name in "${ds_dir}"/*__split_train/; do
    [[ -d "${name}" ]] || continue
    name="$(basename "${name}" __split_train)"
    if [[ "${name,,}" == *"${tag}"* ]]; then
      printf '%s' "${name}"
      return 0
    fi
  done
  return 1
}

cd "${ROOT}" || exit 1
mkdir -p "${STEER_OUT_ROOT}"

if [[ ! -d "${DUMP_ROOT}/${MODEL}/${DATASET}" ]]; then
  echo "[error] dump dir not found: ${DUMP_ROOT}/${MODEL}/${DATASET}" >&2
  echo "Run ./run/run_dump_all_datasets.sh first." >&2
  exit 1
fi

CATEGORY_LIST=()
for pid in "${KNOWLEDGE_TYPE_IDS[@]}"; do
  if cat_slug="$(resolve_category_for_type "${MODEL}" "${pid}")"; then
    CATEGORY_LIST+=("${cat_slug}")
    echo "[resolve] knowledge_type ${pid} -> ${cat_slug}" >&2
  else
    echo "[warn] ${pid}: no __split_train category under ${DUMP_ROOT}/${MODEL}/${DATASET}" >&2
  fi
done

if [[ "${#CATEGORY_LIST[@]}" -eq 0 ]]; then
  echo "[error] no categories resolved for ${MODEL}" >&2
  exit 1
fi

echo "[knowledge_type] MODEL=${MODEL} LAYERS=${LAYERS} GPU=${GPU}" >&2
echo "[knowledge_type] jobs=$(( ${#CATEGORY_LIST[@]} * ${#CATEGORY_LIST[@]} ))" >&2
echo "[knowledge_type] DUMP_ROOT=${DUMP_ROOT} STEER_OUT_ROOT=${STEER_OUT_ROOT}" >&2

for dir_cat in "${CATEGORY_LIST[@]}"; do
  dir_npz="$(train_npz_path "${DUMP_ROOT}" "${MODEL}" "${DATASET}" "${dir_cat}")"
  dir_tag="$(split_train_cat "${dir_cat}")"
  echo "[knowledge_type] === direction from ${dir_cat} ===" >&2
  for test_cat in "${CATEGORY_LIST[@]}"; do
    val_json="$(val_json_path "${DUMP_ROOT}" "${MODEL}" "${DATASET}" "${test_cat}")"
    test_json="$(test_json_path "${DUMP_ROOT}" "${MODEL}" "${DATASET}" "${test_cat}")"
    run_cross_train_val_test \
      "${MODEL}" "${STEER_OUT_ROOT}" "${DATASET}" \
      "${dir_npz}" "${val_json}" "${test_json}" \
      "${test_cat}" "${dir_tag}" "${LAYERS}"
  done
done

echo "[knowledge_type] done." >&2
