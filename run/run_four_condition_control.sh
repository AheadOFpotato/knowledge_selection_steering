#!/usr/bin/env bash
# Build A/B/C/D directions from NQ-Swap, then evaluate each on four OOD datasets.
# Defaults: Llama-3-8B-Instruct, layer 10, GPU 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/run/common.sh"

MODEL="${MODEL:-llama3-8b-it}"
GPU="${GPU:-1}"
LAYERS="${LAYERS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ALPHA_START="${ALPHA_START:--2}"
ALPHA_END="${ALPHA_END:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONTROL_ROOT="${CONTROL_ROOT:-${ROOT}/output/four_condition_control}"
STEER_OUT_ROOT="${FOUR_CONDITION_STEER_ROOT:-${ROOT}/output/four_condition_steer_v2}"
DUMP_ROOT="${DUMP_ROOT:-${ROOT}/output/dump}"
GROUP="${GROUP:-stage1_correct_use_wrong_context}"
INPUT_JSONL="${INPUT_JSONL:-/home/lyb/conflict_steer/data/NQ-Swap/dev_dedup_by_question_only_p_no_table_li.jsonl}"
SKIP_DIRECTION="${SKIP_DIRECTION:-0}"
DIRECTIONS="${DIRECTIONS:-A-C,B-C,A-B,A-D}"
ALPHA_MODE="${ALPHA_MODE:-sweep}"

# Best non-negative L10 alpha from the prior Llama NQ-Swap -> OOD transfer
# sweep in /ssd/lyb/final_result/nqswap_dir_cross.  This is an optional
# speed-oriented setting for the four-condition controls, not a replacement
# for per-direction validation-based alpha selection.
legacy_nqswap_alpha() {
  case "$1" in
    "Athelete Sport") echo 1.2 ;;
    "Book Author") echo 1.6 ;;
    "Company Founder") echo 2.2 ;;
    "Company Headquarter") echo 1.6 ;;
    "Official Language") echo 1.6 ;;
    "World Capital") echo 1.4 ;;
    "NQ_eval_longpre_test_fix") echo 1.6 ;;
    "TQA_eval_gpt4_dev_256_new_fix") echo 1.8 ;;
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

if [[ "${SKIP_DIRECTION}" == "1" ]]; then
  echo "[skip] reusing existing direction NPZs under ${CONTROL_ROOT}/${MODEL}"
elif [[ "${DRY_RUN}" == "1" ]]; then
  echo "python3 ${ROOT}/src/run_four_condition_control.py --model ${MODEL} --gpu ${GPU} --batch-size ${BATCH_SIZE} --max-samples ${MAX_SAMPLES} --layers ${LAYERS} --input ${INPUT_JSONL} --out-root ${CONTROL_ROOT}"
else
  python3 "${ROOT}/src/run_four_condition_control.py" \
    --model "${MODEL}" --gpu "${GPU}" --batch-size "${BATCH_SIZE}" \
    --max-samples "${MAX_SAMPLES}" --layers "${LAYERS}" --input "${INPUT_JSONL}" \
    --out-root "${CONTROL_ROOT}"
fi

DIR_ROOT="${CONTROL_ROOT}/${MODEL}"
PY="${ROOT}/src/run_resid_projection_ablation.py"
datasets=(ParaConfilct macnoise memotrap counterfact)
IFS=',' read -r -a direction_list <<< "${DIRECTIONS}"

for direction in "${direction_list[@]}"; do
  direction="${direction// /}"
  case "${direction}" in
    A-C|B-C|A-B|A-D) ;;
    *) echo "invalid direction: ${direction}; use A-C,B-C,A-B,A-D" >&2; exit 1 ;;
  esac
  npz="${DIR_ROOT}/direction_${direction}.npz"
  if [[ "${DRY_RUN}" != "1" && ! -f "${npz}" ]]; then
    echo "missing ${npz}" >&2
    exit 1
  fi
  for dataset in "${datasets[@]}"; do
    ds_dir="${DUMP_ROOT}/${MODEL}/${dataset}"
    if [[ ! -d "${ds_dir}" ]]; then
      [[ "${DRY_RUN}" == "1" ]] && echo "[dry-run] find ${ds_dir}/**/internal_aware_data__${GROUP}.json"
      continue
    fi
    while IFS= read -r manifest; do
      [[ -f "${manifest}" ]] || continue
      category="$(basename "$(dirname "${manifest}")")"
      job_alpha_start="${ALPHA_START}"
      job_alpha_end="${ALPHA_END}"
      job_alpha_step="${ALPHA_STEP}"
      if [[ "${ALPHA_MODE}" == "legacy_nqswap_fixed" ]]; then
        fixed_alpha="$(legacy_nqswap_alpha "${category}")" || {
          echo "[warn] no legacy NQ-Swap alpha for ${category}; skip" >&2
          continue
        }
        job_alpha_start="${fixed_alpha}"
        job_alpha_end="${fixed_alpha}"
        job_alpha_step="0.2"
      elif [[ "${ALPHA_MODE}" != "sweep" ]]; then
        echo "invalid ALPHA_MODE=${ALPHA_MODE}; use sweep or legacy_nqswap_fixed" >&2
        exit 1
      fi
      out_dir="${STEER_OUT_ROOT}/${MODEL}/${dataset}/${category}__dir_${direction}"
      if [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out_dir}/L${LAYERS}_alpha*.json" >/dev/null 2>&1; then
        continue
      fi
      cmd=(python3 "${PY}" --eval-mode train_val_test --split-protocol cross_dataset \
        --split-seed 42 --model "${MODEL}" --npz "${npz}" --test-json "${manifest}" \
        --out-root "${STEER_OUT_ROOT}" --output-category "${category}" \
        --direction-category-tag "${direction}" --test-category-tag "${category}" \
        --gpu "${GPU}" --layers "${LAYERS}" --alpha-start "${job_alpha_start}" \
        --alpha-end "${job_alpha_end}" --alpha-step "${job_alpha_step}" \
        --batch-size "${BATCH_SIZE}" --max-samples-stage1 0 --max-samples-stage2 0 \
        --stage1-min-pass-rate -1)
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%q ' "${cmd[@]}"; echo
      else
        "${cmd[@]}"
      fi
    done < <(find "${ds_dir}" -type f -name "internal_aware_data__${GROUP}.json" -print)
  done
done

echo "four-condition control evaluation complete: ${STEER_OUT_ROOT}/${MODEL}" >&2
