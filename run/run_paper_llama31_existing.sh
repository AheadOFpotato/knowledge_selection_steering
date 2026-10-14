#!/usr/bin/env bash
# Re-run/check only the datasets named in the paper.
# Large assets can be mounted elsewhere by overriding the variables below.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL_ROOT="${MODEL_ROOT:-${ROOT}/models}"
export MODEL="${MODEL:-llama3.1-8b-it}"
export GPU="${GPU:-0}"
export DUMP_ROOT="${DUMP_ROOT:-${ROOT}/output/dump_taxonomy}"
export STEER_OUT_ROOT="${STEER_OUT_ROOT:-${ROOT}/output/steer_taxonomy}"
export COMPLEX_RAW_ROOT="${COMPLEX_RAW_ROOT:-${ROOT}/data}"
export PROCESSED_ROOT="${PROCESSED_ROOT:-${COMPLEX_RAW_ROOT}/processed}"
# Reuse the evaluation project's OpenAI-compatible judge endpoint when it is
# available, without copying credentials into this repository.
JUDGE_SETTINGS="${JUDGE_SETTINGS:-${ROOT}/llmapi/llm_settings.py}"
if [[ -f "${JUDGE_SETTINGS}" ]]; then
  export LLM_BASE_URL="${LLM_BASE_URL:-$(python3 -c 'import runpy,sys; print(runpy.run_path(sys.argv[1])["LLM_BASE_URL"])' "${JUDGE_SETTINGS}")}"
  export LLM_API_KEY="${LLM_API_KEY:-$(python3 -c 'import runpy,sys; print(runpy.run_path(sys.argv[1])["LLM_API_KEY"])' "${JUDGE_SETTINGS}")}"
  export LLM_MODEL="${LLM_MODEL:-$(python3 -c 'import runpy,sys; print(runpy.run_path(sys.argv[1])["LLM_MODEL"])' "${JUDGE_SETTINGS}")}"
fi
export DATASETS="RGB,CRAG,LoCoMo,LoCoMo-Plus,LongMemEval,IFEval,FollowBench"
export BUILD_MANIFESTS="${BUILD_MANIFESTS:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export ALPHA_START="${ALPHA_START:-0}"
export ALPHA_END="${ALPHA_END:-2}"
export ALPHA_STEP="${ALPHA_STEP:-0.5}"

exec "${ROOT}/complex_tasks/run/run_nqswap_dir_cross_taxonomy.sh"
