#!/usr/bin/env bash
# IFEval ablation — llama3-8b-it (layer 10)
#   GPU=0 ./run/run_ifeval_prompt_ablation_llama.sh
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_FAMILY=llama
export GPU="${GPU:-0}"
exec bash "${BUNDLE_ROOT}/run/run_ifeval_prompt_ablation.sh" "$@"
