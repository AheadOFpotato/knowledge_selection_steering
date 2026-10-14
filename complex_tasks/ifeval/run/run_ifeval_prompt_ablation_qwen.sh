#!/usr/bin/env bash
# IFEval ablation — qwen3-8b (layer 21)
#   GPU=0 ./run/run_ifeval_prompt_ablation_qwen.sh
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_FAMILY=qwen
export GPU="${GPU:-0}"
exec bash "${BUNDLE_ROOT}/run/run_ifeval_prompt_ablation.sh" "$@"
