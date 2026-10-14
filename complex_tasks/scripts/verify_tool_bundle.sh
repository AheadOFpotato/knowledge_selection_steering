#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
required=(
  "$ROOT/src/run_resid_projection_ablation.py"
  "$ROOT/src/official_judges/bfcl_scorer.py"
  "$ROOT/src/official_judges/toolsandbox_play.py"
  "$ROOT/run/run_dump_taxonomy.sh"
  "$ROOT/run/run_toolbench_official_tooleval.sh"
  "$ROOT/data/policy_api/API-Bank/repo"
  "$ROOT/data/policy_api/BFCL/bfcl_eval"
  "$ROOT/data/policy_api/ToolBench/toolbench/tooleval"
  "$ROOT/data/policy_api/ToolSandbox/tool_sandbox"
  "$ROOT/data/policy_api/tau-bench/tau_bench"
)

for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done

echo "tool bundle verified: $ROOT"
