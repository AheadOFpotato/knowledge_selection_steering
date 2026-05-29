#!/usr/bin/env bash
# Plot NQ-Swap centroid cosine similarity vs layer (4 models) → angle_fig/ as PDF
#
#   ./run/run_plot_angle_fig_all.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/run/common.sh"

PY="${ROOT}/src/plot_angle_between_centroid.py"

ANGLE_ROOT="${ANGLE_ROOT:-${ROOT}/output/angle_analyse}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/fig/angle}"
MODELS="${MODELS:-llama3-8b-it,qwen3-8b,gemma2-9b-it,yi-6b-chat}"
DATASET="${DATASET:-NQ-Swap}"

cd "${ROOT}" || exit 1
IFS=',' read -r -a _MODEL_ARR <<< "${MODELS}"
python3 "${PY}" \
  --angle-root "${ANGLE_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --dataset "${DATASET}" \
  --models "${_MODEL_ARR[@]}" \
  --format pdf
