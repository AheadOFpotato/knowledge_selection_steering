# shellcheck shell=bash
# Shared repo paths for run/*.sh (all relative to project root).
#
#   source "${ROOT}/run/common.sh"
# or from run/scripts/*.sh:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

if [[ -z "${ROOT:-}" ]]; then
  _common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "${_common_dir}" == */run/scripts ]]; then
    ROOT="$(cd "${_common_dir}/../.." && pwd)"
  else
    ROOT="$(cd "${_common_dir}/.." && pwd)"
  fi
  unset _common_dir
fi

export ROOT

DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
DUMP_ROOT="${DUMP_ROOT:-${ROOT}/output/dump}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/models}"
STEER_OUT_ROOT="${STEER_OUT_ROOT:-${ROOT}/output/steer_result}"
CROSS_FORMAT_DUMP_ROOT="${CROSS_FORMAT_DUMP_ROOT:-${DUMP_ROOT}/cross_format}"
CROSS_FORMAT_STEER_ROOT="${CROSS_FORMAT_STEER_ROOT:-${ROOT}/output/cross_format}"
KNOWLEDGE_TYPE_STEER_ROOT="${KNOWLEDGE_TYPE_STEER_ROOT:-${ROOT}/output/knowledge_type}"
CROSS_DATASET_STEER_ROOT="${CROSS_DATASET_STEER_ROOT:-${ROOT}/output/cross_dataset}"

export DATA_ROOT DUMP_ROOT MODEL_ROOT STEER_OUT_ROOT
export CROSS_FORMAT_DUMP_ROOT CROSS_FORMAT_STEER_ROOT KNOWLEDGE_TYPE_STEER_ROOT CROSS_DATASET_STEER_ROOT
