#!/usr/bin/env bash
# Full-manifest fixed-alpha evaluation for the four NQ-Swap A/B/C/D directions.
# Reuses legacy Llama NQ-Swap -> OOD optima from
# /ssd/lyb/final_result/nqswap_dir_cross; it never runs an alpha sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/run/common.sh"

MODEL="${MODEL:-llama3-8b-it}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LAYERS="${LAYERS:-10}"
DUMP_ROOT="${DUMP_ROOT:-${ROOT}/output/dump}"
CONTROL_ROOT="${CONTROL_ROOT:-${ROOT}/output/four_condition_control}"
OUT_ROOT="${FOUR_CONDITION_FIXED_FULL_ROOT:-${ROOT}/output/four_condition_fixed_full}"
GROUP="${GROUP:-stage1_correct_use_wrong_context}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DIRECTIONS="${DIRECTIONS:-A-C,B-C,A-B,A-D}"
RUN_BASELINE="${RUN_BASELINE:-1}"
PY="${ROOT}/src/run_resid_projection_ablation.py"

# Values selected from the prior Llama NQ-Swap cross-dataset alpha curves.
legacy_positive_alpha() {
  case "$1" in
    "Athelete Sport") echo 1.2 ;; "Book Author") echo 1.6 ;;
    "Company Founder") echo 2.2 ;; "Company Headquarter") echo 1.6 ;;
    "Official Language") echo 1.6 ;; "World Capital") echo 1.4 ;;
    "NQ_eval_longpre_test_fix") echo 1.6 ;; "TQA_eval_gpt4_dev_256_new_fix") echo 1.8 ;;
    "1-proverb-ending") echo 2.8 ;;
    "counterfact_relation_002_p27_3f9dda33") echo 3.0 ;;
    "counterfact_relation_005_p176_75ab4900") echo 2.6 ;;
    "counterfact_relation_013_p106_7d9485da") echo 3.0 ;;
    "counterfact_relation_014_p449_349af58e") echo 0.4 ;;
    "counterfact_relation_027_p140_5fa45f36") echo 0.0 ;;
    "counterfact_relation_029_p641_5e29ef20") echo 0.8 ;;
    *) return 1 ;;
  esac
}

legacy_negative_alpha() {
  case "$1" in
    "Athelete Sport") echo -0.8 ;; "Book Author") echo -1.0 ;;
    "Company Founder") echo -0.8 ;; "Company Headquarter") echo -0.2 ;;
    "Official Language") echo 0.0 ;; "World Capital") echo -0.6 ;;
    "NQ_eval_longpre_test_fix") echo -1.4 ;; "TQA_eval_gpt4_dev_256_new_fix") echo -1.8 ;;
    "1-proverb-ending") echo -2.0 ;;
    "counterfact_relation_002_p27_3f9dda33") echo -2.0 ;;
    "counterfact_relation_005_p176_75ab4900") echo -0.6 ;;
    "counterfact_relation_013_p106_7d9485da") echo -1.0 ;;
    "counterfact_relation_014_p449_349af58e") echo -1.4 ;;
    "counterfact_relation_027_p140_5fa45f36") echo 0.0 ;;
    "counterfact_relation_029_p641_5e29ef20") echo -2.0 ;;
    *) return 1 ;;
  esac
}

alpha_tag() { printf '%g' "$1"; }
result_path() {
  local dataset="$1" category="$2" dir="$3" alpha="$4"
  local category_slug="${category// /_}"
  printf '%s/%s/%s/%s__dir_%s__test_%s/L%s_alpha%s_steer_result.json' \
    "${OUT_ROOT}" "${MODEL}" "${dataset}" "${category_slug}" "${dir}" "${category_slug}" \
    "${LAYERS//,/_}" "$(alpha_tag "${alpha}")"
}

run_one() {
  local dataset="$1" category="$2" manifest="$3" dir="$4" npz="$5" alpha="$6"
  local out
  out="$(result_path "${dataset}" "${category}" "${dir}" "${alpha}")"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
    echo "[skip] ${dataset} | ${category} | ${dir} | alpha=${alpha}" >&2
    return
  fi
  echo "[run] ${dataset} | ${category} | ${dir} | alpha=${alpha}" >&2
  python3 "${PY}" --eval-mode legacy --model "${MODEL}" --npz "${npz}" --test-json "${manifest}" \
    --out-root "${OUT_ROOT}" --output-category "${category}" \
    --direction-category-tag "${dir}" --test-category-tag "${category}" \
    --gpu "${GPU}" --layers "${LAYERS}" --alpha-start "${alpha}" --alpha-end "${alpha}" \
    --alpha-step 0.2 --batch-size "${BATCH_SIZE}" --max-samples-stage1 0 --max-samples-stage2 0 \
    --stage1-min-pass-rate -1
}

datasets=(ParaConfilct macnoise memotrap counterfact)
IFS=',' read -r -a direction_list <<< "${DIRECTIONS}"
mkdir -p "${OUT_ROOT}"

# Each unsteered baseline is identical across directions, so parallel jobs
# should set RUN_BASELINE=1 for exactly one worker and 0 for the rest.
if [[ "${RUN_BASELINE}" == "1" ]]; then
  for dataset in "${datasets[@]}"; do
    while IFS= read -r manifest; do
      category="$(basename "$(dirname "${manifest}")")"
      baseline_npz="${CONTROL_ROOT}/${MODEL}/direction_A-C.npz"
      run_one "${dataset}" "${category}" "${manifest}" "unsteered" "${baseline_npz}" 0.0
    done < <(find "${DUMP_ROOT}/${MODEL}/${dataset}" -type f -name "internal_aware_data__${GROUP}.json" -print | sort)
  done
elif [[ "${RUN_BASELINE}" != "0" ]]; then
  echo "invalid RUN_BASELINE=${RUN_BASELINE}; use 0 or 1" >&2
  exit 1
fi

for direction in "${direction_list[@]}"; do
  direction="${direction// /}"
  case "${direction}" in A-C|B-C|A-B|A-D) ;; *) echo "invalid direction: ${direction}" >&2; exit 1 ;; esac
  npz="${CONTROL_ROOT}/${MODEL}/direction_${direction}.npz"
  [[ -f "${npz}" ]] || { echo "missing direction NPZ: ${npz}" >&2; exit 1; }
  for dataset in "${datasets[@]}"; do
    while IFS= read -r manifest; do
      category="$(basename "$(dirname "${manifest}")")"
      positive="$(legacy_positive_alpha "${category}")"
      negative="$(legacy_negative_alpha "${category}")"
      [[ "$(alpha_tag "${positive}")" == "0" ]] || run_one "${dataset}" "${category}" "${manifest}" "${direction}" "${npz}" "${positive}"
      # Do not duplicate alpha=0: the single unsteered baseline above is canonical.
      [[ "$(alpha_tag "${negative}")" == "0" ]] || run_one "${dataset}" "${category}" "${manifest}" "${direction}" "${npz}" "${negative}"
    done < <(find "${DUMP_ROOT}/${MODEL}/${dataset}" -type f -name "internal_aware_data__${GROUP}.json" -print | sort)
  done
done

echo "[done] fixed full-manifest evaluation: ${OUT_ROOT}/${MODEL}" >&2
