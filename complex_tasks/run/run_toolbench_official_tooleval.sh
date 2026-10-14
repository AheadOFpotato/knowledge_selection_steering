#!/usr/bin/env bash
# Run ToolBench's released ToolEval Pass Rate evaluator unchanged.
#
# Input must be genuine multi-step ToolBench trajectories produced by an agent
# that executed the released ToolBench tools.  A one-shot steering completion
# is not a valid ToolEval answer and is rejected here.
#
# Required layout under RAW_ANSWER_PATH/MODEL_NAME:
#   G1_instruction/, G1_category/, G1_tool/, G2_instruction/, G2_category/,
#   G3_instruction/   (raw ToolBench agent trajectory JSON files)
#
# Required environment variables:
#   RAW_ANSWER_PATH, MODEL_NAME, METHOD, TEST_IDS, API_POOL_FILE
# Optional: TOOLBENCH_PYTHON, CONVERTED_ANSWER_PATH, SAVE_PATH,
#   EVALUATE_TIMES, MAX_EVAL_THREADS, EVALUATOR.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLBENCH_ROOT="${TOOLBENCH_ROOT:-${ROOT}/data/policy_api/ToolBench}"
TOOLBENCH_PYTHON="${TOOLBENCH_PYTHON:-<DATA_DIR>/venvs/toolbench-official/bin/python}"
RAW_ANSWER_PATH="${RAW_ANSWER_PATH:?Set RAW_ANSWER_PATH to the directory containing model trajectory folders}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the trajectory subdirectory name}"
METHOD="${METHOD:?Set METHOD to the official trajectory method, for example DFS_woFilter_w2}"
TEST_IDS="${TEST_IDS:?Set TEST_IDS to ToolBench's released data/test_ids directory}"
API_POOL_FILE="${API_POOL_FILE:?Set API_POOL_FILE to ToolEval's OpenAI-compatible evaluator key JSON}"
CONVERTED_ANSWER_PATH="${CONVERTED_ANSWER_PATH:-${RAW_ANSWER_PATH}_converted}"
SAVE_PATH="${SAVE_PATH:-${ROOT}/results/toolbench_official/${MODEL_NAME}}"
EVALUATE_TIMES="${EVALUATE_TIMES:-4}"
MAX_EVAL_THREADS="${MAX_EVAL_THREADS:-8}"
EVALUATOR="${EVALUATOR:-tooleval_gpt-3.5-turbo_default}"

if [[ ! -x "${TOOLBENCH_PYTHON}" ]]; then
  echo "[error] ToolBench official Python missing: ${TOOLBENCH_PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${TOOLBENCH_ROOT}/toolbench/tooleval" ]]; then
  echo "[error] ToolBench checkout missing: ${TOOLBENCH_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${TEST_IDS}" ]]; then
  echo "[error] Official ToolEval test_ids missing: ${TEST_IDS}" >&2
  exit 2
fi
if [[ ! -f "${API_POOL_FILE}" ]]; then
  echo "[error] ToolEval API pool missing: ${API_POOL_FILE}" >&2
  exit 2
fi

test_sets=(G1_instruction G1_category G1_tool G2_category G2_instruction G3_instruction)
for test_set in "${test_sets[@]}"; do
  if [[ ! -d "${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}" ]]; then
    echo "[error] Missing raw official trajectory directory: ${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}" >&2
    exit 2
  fi
  if [[ ! -f "${TEST_IDS}/${test_set}.json" ]]; then
    echo "[error] Missing official test IDs: ${TEST_IDS}/${test_set}.json" >&2
    exit 2
  fi
done

mkdir -p "${CONVERTED_ANSWER_PATH}/${MODEL_NAME}" "${SAVE_PATH}"
cd "${TOOLBENCH_ROOT}/toolbench/tooleval"
export API_POOL_FILE

for test_set in "${test_sets[@]}"; do
  "${TOOLBENCH_PYTHON}" convert_to_answer_format.py \
    --answer_dir "${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}" \
    --method "${METHOD}" \
    --output "${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json"
done

"${TOOLBENCH_PYTHON}" eval_pass_rate.py \
  --converted_answer_path "${CONVERTED_ANSWER_PATH}" \
  --save_path "${SAVE_PATH}" \
  --reference_model "${MODEL_NAME}" \
  --test_ids "${TEST_IDS}" \
  --evaluator "${EVALUATOR}" \
  --max_eval_threads "${MAX_EVAL_THREADS}" \
  --evaluate_times "${EVALUATE_TIMES}"
