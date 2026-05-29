#!/usr/bin/env bash
# NQ-Swap (dev) train direction → steer on ParaConfilct / macnoise / memotrap / counterfact:
#   - Direction: <DUMP_ROOT>/<model>/NQ-Swap/dev__split_train/
#   - Val / test: target dataset category __split_{val,test}/
#   - counterfact: only six knowledge types (P27 P176 P106 P140 P641 P449)
#
# Output:
#   ${STEER_OUT_ROOT}/<model>/<test_dataset>/<test_cat>__split_test__dir_<dir_tag>__test_<test_cat>__split_test/
#
# Usage:
#   ./run/run_resid_projection_nqswap_dir_cross_datasets.sh
#   MODEL=llama3-8b-it LAYERS=10 GPU=0 DRY_RUN=1 ./run/run_resid_projection_nqswap_dir_cross_datasets.sh

set -u
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
[[ "${STOP_ON_ERROR}" == "1" ]] && set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"
# shellcheck source=scripts/cross_steer_common.sh
source "${ROOT}/run/scripts/cross_steer_common.sh"

STEER_OUT_ROOT="${STEER_OUT_ROOT:-${CROSS_DATASET_STEER_ROOT}}"
MODEL="${MODEL:-llama3-8b-it}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DIR_DATASET="${DIR_DATASET:-NQ-Swap}"
DIR_CATEGORY="${DIR_CATEGORY:-dev}"

ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

KNOWLEDGE_TYPE_IDS=(P27 P176 P106 P140 P641 P449)
TEST_DATASETS=(ParaConfilct macnoise memotrap counterfact)

DIR_TAG="$(split_train_cat "${DIR_CATEGORY}")"
DIR_NPZ="$(train_npz_path "${DUMP_ROOT}" "${MODEL}" "${DIR_DATASET}" "${DIR_CATEGORY}")"

LAYERS=""
apply_model_layers_default "${MODEL}"

counterfact_type_enabled() {
  local category="$1"
  local pid tag cat_lc="${category,,}"
  for pid in "${KNOWLEDGE_TYPE_IDS[@]}"; do
    tag="_$(echo "${pid}" | tr '[:upper:]' '[:lower:]')_"
    [[ "${cat_lc}" == *"${tag}"* ]] && return 0
  done
  return 1
}

cd "${ROOT}" || exit 1
mkdir -p "${STEER_OUT_ROOT}"

if [[ ! -f "${DIR_NPZ}" ]]; then
  echo "[error] NQ-Swap train NPZ not found: ${DIR_NPZ}" >&2
  echo "Run: DATASETS=NQ-Swap ./run/run_dump_all_datasets.sh" >&2
  exit 1
fi

echo "[cross_dataset] MODEL=${MODEL} LAYERS=${LAYERS} GPU=${GPU}" >&2
echo "[cross_dataset] direction: ${DIR_DATASET}/${DIR_TAG}" >&2
echo "[cross_dataset] DIR_NPZ=${DIR_NPZ}" >&2
echo "[cross_dataset] STEER_OUT_ROOT=${STEER_OUT_ROOT}" >&2

n_jobs=0
for test_dataset in "${TEST_DATASETS[@]}"; do
  ds_dir="${DUMP_ROOT}/${MODEL}/${test_dataset}"
  if [[ ! -d "${ds_dir}" ]]; then
    echo "[warn] skip dataset (no dump): ${ds_dir}" >&2
    continue
  fi

  echo "[cross_dataset] === test dataset: ${test_dataset} ===" >&2
  shopt -s nullglob
  for val_dir in "${ds_dir}"/*__split_val/; do
    [[ -d "${val_dir}" ]] || continue
    test_cat="$(basename "${val_dir}" __split_val)"

    if [[ "${test_dataset}" == "counterfact" ]]; then
      counterfact_type_enabled "${test_cat}" || continue
    fi

    val_json="$(val_json_path "${DUMP_ROOT}" "${MODEL}" "${test_dataset}" "${test_cat}")"
    test_json="$(test_json_path "${DUMP_ROOT}" "${MODEL}" "${test_dataset}" "${test_cat}")"
    run_cross_train_val_test \
      "${MODEL}" "${STEER_OUT_ROOT}" "${test_dataset}" \
      "${DIR_NPZ}" "${val_json}" "${test_json}" \
      "${test_cat}" "${DIR_DATASET}__${DIR_TAG}" "${LAYERS}"
    n_jobs=$((n_jobs + 1))
  done
  shopt -u nullglob
done

echo "[cross_dataset] done (${n_jobs} category jobs)." >&2
