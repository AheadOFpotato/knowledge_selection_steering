#!/usr/bin/env bash
# Dump taxonomy processed JSONL → manifests (+ optional resid NPZ).
#
# Output:
#   ${DUMP_ROOT}/<model>/<dataset>/<category>/
#     resid_hidden_state__unfiltered_all_pairs.npz (if SAVE_RESID=true)
#
# Usage:
#   ./run/run_dump_taxonomy.sh
#   MODEL=llama3-8b-it GPU=0 DATASETS=RGB,GuideBench MAX_SAMPLES=200 ./run/run_dump_taxonomy.sh
#   DRY_RUN=1 ./run/run_dump_taxonomy.sh

set -u
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
[[ "${STOP_ON_ERROR}" == "1" ]] && set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/src/dump_activations_from_testjsonl.py"
PROCESSED="${PROCESSED_ROOT:-${ROOT}/data/processed}"
export PROCESSED_ROOT="${PROCESSED}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-llama3.1-8b-it}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
SAVE_RESID="${SAVE_RESID:-true}"
DUMP_ROOT="${DUMP_ROOT:-${ROOT}/output/dump_taxonomy}"
DATASETS="${DATASETS:-NQ-Swap,RGB,CRAG,GuideBench,ContractNLI}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"

dataset_enabled() {
  local ds="$1" x
  IFS=',' read -r -a _f <<< "${DATASETS}"
  for x in "${_f[@]}"; do
    x="${x// /}"
    [[ "${x}" == "${ds}" ]] && return 0
  done
  return 1
}

already_dumped() {
  local out_dir="$1"
  [[ -f "${out_dir}/resid_hidden_state__unfiltered_all_pairs.npz" ]]
}

run_one() {
  local dataset="$1" category="$2" jsonl="$3"
  if [[ ! -f "${jsonl}" ]]; then
    echo "[skip] missing jsonl: ${jsonl}" >&2
    return 0
  fi
  local nlines
  nlines="$(wc -l < "${jsonl}" | tr -d ' ')"
  if [[ "${nlines}" -eq 0 ]]; then
    echo "[skip] empty jsonl: ${jsonl}" >&2
    return 0
  fi
  local out_dir="${DUMP_ROOT}/${MODEL}/${dataset}/${category}"
  if [[ "${SKIP_EXISTING}" == "1" ]] && already_dumped "${out_dir}"; then
    echo "[skip] dump exists: ${out_dir}" >&2
    return 0
  fi
  local -a cmd=(
    "${PYTHON}" "${PY}"
    --model "${MODEL}"
    --dataset "${dataset}"
    --category "${category}"
    --input-jsonl "${jsonl}"
    --out-root "${DUMP_ROOT}"
    --gpu "${GPU}"
    --batch-size "${BATCH_SIZE}"
    --max-samples "${MAX_SAMPLES}"
    --save-resid "${SAVE_RESID}"
  )
  cmd+=(--save-unfiltered-pairs true)
  echo "[dump] ${dataset}/${category} <- ${jsonl} (n=${nlines})" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo >&2
    return 0
  fi
  mkdir -p "${out_dir}"
  "${cmd[@]}"
}

cd "${ROOT}" || exit 1
[[ -f "${PY}" ]] || { echo "missing ${PY}"; exit 1; }

# Ensure processed JSONL exists
if [[ "${DRY_RUN}" != "1" && ! -d "${PROCESSED}/RGB" ]]; then
  echo "[info] building processed JSONL..." >&2
  python3 "${ROOT}/scripts/build_processed_jsonl.py"
fi

echo "[dump_taxonomy] MODEL=${MODEL} GPU=${GPU} DUMP_ROOT=${DUMP_ROOT}" >&2
echo "[dump_taxonomy] DATASETS=${DATASETS}" >&2

n=0
# NQ-Swap (direction source)
if dataset_enabled NQ-Swap; then
  jl="${PROCESSED}/NQ-Swap/dev_dedup_p_only.jsonl"
  [[ -f "${jl}" ]] && run_one "NQ-Swap" "dev_dedup_p_only" "${jl}" && n=$((n+1))
fi
# RGB
if dataset_enabled RGB; then
  jl="${PROCESSED}/RGB/en_fact.jsonl"
  [[ -f "${jl}" ]] && run_one "RGB" "en_fact" "${jl}" && n=$((n+1))
fi
# CRAG
if dataset_enabled CRAG; then
  jl="${PROCESSED}/CRAG/task12_dev.jsonl"
  [[ -f "${jl}" ]] && run_one "CRAG" "task12_dev" "${jl}" && n=$((n+1))
fi
# GuideBench categories
if dataset_enabled GuideBench; then
  shopt -s nullglob
  for jl in "${PROCESSED}/GuideBench"/*.jsonl; do
    cat="$(basename "${jl}" .jsonl)"
    run_one "GuideBench" "${cat}" "${jl}"
    n=$((n+1))
  done
  shopt -u nullglob
fi
# ContractNLI tasks
if dataset_enabled ContractNLI; then
  shopt -s nullglob
  for jl in "${PROCESSED}/ContractNLI"/*.jsonl; do
    cat="$(basename "${jl}" .jsonl)"
    run_one "ContractNLI" "${cat}" "${jl}"
    n=$((n+1))
  done
  shopt -u nullglob
fi

# Policy: API-Bank
if dataset_enabled API-Bank; then
  shopt -s nullglob
  for jl in "${PROCESSED}/API-Bank"/*.jsonl; do
    cat="$(basename "${jl}" .jsonl)"
    run_one "API-Bank" "${cat}" "${jl}"
    n=$((n+1))
  done
  shopt -u nullglob
fi
# Policy: LongMemEval
if dataset_enabled LongMemEval; then
  jl="${PROCESSED}/LongMemEval/oracle.jsonl"
  [[ -f "${jl}" ]] && run_one "LongMemEval" "oracle" "${jl}" && n=$((n+1))
fi
# Policy: Mem2ActBench
if dataset_enabled Mem2ActBench; then
  jl="${PROCESSED}/Mem2ActBench/qa.jsonl"
  [[ -f "${jl}" ]] && run_one "Mem2ActBench" "qa" "${jl}" && n=$((n+1))
fi
# Policy: MemoryAgentBench FactConsolidation categories
if dataset_enabled MemoryAgentBench; then
  shopt -s nullglob
  for jl in "${PROCESSED}/MemoryAgentBench"/*.jsonl; do
    cat="$(basename "${jl}" .jsonl)"
    if [[ -n "${CATEGORY_GLOB:-}" && ! "${cat}" == ${CATEGORY_GLOB} ]]; then
      continue
    fi
    run_one "MemoryAgentBench" "${cat}" "${jl}"
    n=$((n+1))
  done
  shopt -u nullglob
fi

echo "[dump_taxonomy] scheduled/finished ${n} jobs." >&2

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" "${ROOT}/scripts/build_context_manifests.py" \
    --model "${MODEL}" --datasets "${DATASETS}" --out-root "${DUMP_ROOT}"
fi
