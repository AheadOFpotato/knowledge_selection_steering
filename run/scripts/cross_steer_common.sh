# shellcheck shell=bash
# Shared helpers for cross-format / knowledge-type / cross-dataset steering.

_RUN_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${_RUN_SCRIPTS}/../common.sh"
unset _RUN_SCRIPTS

PY="${PY:-${ROOT}/src/run_resid_projection_ablation.py}"
GROUP="${GROUP:-stage1_correct_use_wrong_context}"
SPLIT_SEED="${SPLIT_SEED:-42}"

safe_slug() {
  printf '%s' "${1// /_}"
}

dump_npz_path() {
  local root="$1" model="$2" dataset="$3" category="$4"
  printf '%s/%s/%s/%s/resid_hidden_state__%s.npz' \
    "${root}" "${model}" "${dataset}" "${category}" "${GROUP}"
}

dump_manifest_path() {
  local root="$1" model="$2" dataset="$3" category="$4"
  printf '%s/%s/%s/%s/internal_aware_data__%s.json' \
    "${root}" "${model}" "${dataset}" "${category}" "${GROUP}"
}

cross_steer_out_dir() {
  local steer_root="$1" model="$2" dataset="$3" test_cat="$4" dir_cat="$5"
  printf '%s/%s/%s/%s__dir_%s__test_%s' \
    "${steer_root}" "${model}" "${dataset}" \
    "$(safe_slug "${test_cat}")" "$(safe_slug "${dir_cat}")" "$(safe_slug "${test_cat}")"
}

cross_steer_already_done() {
  local out_dir="$1" layers="$2"
  local ls_tag="${layers//,/_}"
  [[ -f "${out_dir}/L${ls_tag}_val_alpha_grid.json" ]] \
    && compgen -G "${out_dir}/L${ls_tag}_alpha"*.json >/dev/null 2>&1
}

apply_model_layers_default() {
  local model="$1"
  case "${model}" in
    llama3-8b-it) LAYERS="${LAYERS:-10}" ;;
    qwen3-8b) LAYERS="${LAYERS:-21}" ;;
    gemma2-9b-it) LAYERS="${LAYERS:-16}" ;;
    yi-6b|yi-6b-chat) LAYERS="${LAYERS:-13}" ;;
    *) LAYERS="${LAYERS:-16}" ;;
  esac
}

run_cross_train_val_test() {
  local model="$1"
  local steer_out_root="$2"
  local dataset="$3"
  local dir_npz="$4"
  local eval_json="$5"
  local test_cat="$6"
  local dir_cat="$7"
  local layers="$8"
  local split_protocol="${9:-cross_dataset}"

  local out_dir
  out_dir="$(cross_steer_out_dir "${steer_out_root}" "${model}" "${dataset}" "${test_cat}" "${dir_cat}")"

  if [[ "${SKIP_EXISTING:-1}" == "1" ]] && cross_steer_already_done "${out_dir}" "${layers}"; then
    echo "[skip] done: ${out_dir}" >&2
    return 0
  fi
  if [[ ! -f "${dir_npz}" ]]; then
    echo "[skip] missing direction NPZ: ${dir_npz}" >&2
    return 0
  fi
  if [[ ! -f "${eval_json}" ]]; then
    echo "[skip] missing eval manifest: ${eval_json}" >&2
    return 0
  fi

  local -a cmd=(
    python3 "${PY}"
    --eval-mode train_val_test
    --split-protocol "${split_protocol}"
    --split-seed "${SPLIT_SEED}"
    --model "${model}"
    --npz "${dir_npz}"
    --test-json "${eval_json}"
    --out-root "${steer_out_root}"
    --output-category "${test_cat}"
    --direction-category-tag "${dir_cat}"
    --test-category-tag "${test_cat}"
    --gpu "${GPU}"
    --layers "${layers}"
    --alpha-start "${ALPHA_START}"
    --alpha-end "${ALPHA_END}"
    --alpha-step "${ALPHA_STEP}"
    --batch-size "${BATCH_SIZE}"
    --max-samples-stage1 0
    --max-samples-stage2 0
    --stage1-min-pass-rate -1
  )

  echo "[run] dir=${dir_cat} -> test=${test_cat} | protocol=${split_protocol} | alpha ${ALPHA_START}..${ALPHA_END}" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo >&2
    return 0
  fi
  if ! "${cmd[@]}"; then
    echo "[error] failed: dataset=${dataset} dir=${dir_cat} test=${test_cat}" >&2
    return 1
  fi
}
