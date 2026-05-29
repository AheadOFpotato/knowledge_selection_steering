#!/usr/bin/env bash
# ParaConflict 6×6 cross-format steering (substitution direction × coherent test):
#   - Direction (train NPZ): cross_format/<model>/ParaConfilct/<dir_category>__split_train/
#   - Val / test manifest:   <DUMP_ROOT>/<test_model>/ParaConfilct/<test_category>__split_{val,test}/
#   - train → direction, val → alpha, test → metrics
#
# Output:
#   ${STEER_OUT_ROOT}/<model>/ParaConfilct/<test>__split_test__dir_<dir>__split_train__test_<test>__split_test/
#
# Usage:
#   ./run/run_resid_projection_paraconflict_cross_format.sh
#   MODELS=llama3-8b-it,gemma2-9b-it GPU=0 DRY_RUN=1 ./run/run_resid_projection_paraconflict_cross_format.sh

set -u
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
[[ "${STOP_ON_ERROR}" == "1" ]] && set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"
# shellcheck source=scripts/cross_steer_common.sh
source "${ROOT}/run/scripts/cross_steer_common.sh"

DIRECTION_ROOT="${DIRECTION_ROOT:-${CROSS_FORMAT_DUMP_ROOT}}"
STEER_OUT_ROOT="${STEER_OUT_ROOT:-${CROSS_FORMAT_STEER_ROOT}}"
MODELS="${MODELS:-llama3-8b-it}"
TEST_MODEL="${TEST_MODEL:-}"
DATASET="${DATASET:-ParaConfilct}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ALPHA_START="${ALPHA_START:-0}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"
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

cd "${ROOT}" || exit 1
mkdir -p "${STEER_OUT_ROOT}"

echo "[cross_format] DIRECTION_ROOT=${DIRECTION_ROOT} DUMP_ROOT=${DUMP_ROOT}" >&2
echo "[cross_format] STEER_OUT_ROOT=${STEER_OUT_ROOT} MODELS=${MODELS}" >&2

for model in "${MODEL_LIST[@]}"; do
  model="${model// /}"
  [[ -z "${model}" ]] && continue

  LAYERS=""
  apply_model_layers_default "${model}"
  test_model="${TEST_MODEL:-${model}}"

  if [[ ! -d "${DIRECTION_ROOT}/${model}/${DATASET}" ]]; then
    echo "[warn] skip ${model}: no ${DIRECTION_ROOT}/${model}/${DATASET}" >&2
    continue
  fi
  if [[ ! -d "${DUMP_ROOT}/${test_model}/${DATASET}" ]]; then
    echo "[warn] skip ${model}: no ${DUMP_ROOT}/${test_model}/${DATASET}" >&2
    continue
  fi

  n_cat="${#PARACONFLICT_CATEGORIES[@]}"
  echo "[cross_format] === ${model} LAYERS=${LAYERS} manifest@${test_model} jobs=$(( n_cat * n_cat )) ===" >&2

  for dir_cat in "${PARACONFLICT_CATEGORIES[@]}"; do
    dir_npz="$(train_npz_path "${DIRECTION_ROOT}" "${model}" "${DATASET}" "${dir_cat}")"
    dir_tag="$(split_train_cat "${dir_cat}")"
    echo "[cross_format] --- direction ${dir_cat} (${dir_tag}) ---" >&2
    for test_cat in "${PARACONFLICT_CATEGORIES[@]}"; do
      val_json="$(val_json_path "${DUMP_ROOT}" "${test_model}" "${DATASET}" "${test_cat}")"
      test_json="$(test_json_path "${DUMP_ROOT}" "${test_model}" "${DATASET}" "${test_cat}")"
      run_cross_train_val_test \
        "${model}" "${STEER_OUT_ROOT}" "${DATASET}" \
        "${dir_npz}" "${val_json}" "${test_json}" \
        "${test_cat}" "${dir_tag}" "${LAYERS}"
    done
  done
done

echo "[cross_format] done." >&2
