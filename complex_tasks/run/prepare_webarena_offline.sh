#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SOURCE="${SOURCE:-${ROOT}/data/policy_web/webarena_expert_test.parquet}"
OUTPUT="${OUTPUT:-${ROOT}/data/policy_web/webarena_offline.jsonl}"

"${PYTHON}" "${ROOT}/scripts/build_webarena_expert_offline.py" \
  --source "${SOURCE}" \
  --output "${OUTPUT}"

echo "[prepare_webarena_offline] ready: ${OUTPUT}"
