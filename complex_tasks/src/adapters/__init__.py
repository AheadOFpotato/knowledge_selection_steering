"""Adapters: convert taxonomy datasets → conflict_steer dump JSONL schema.

Target row fields (compatible with dump_activations_from_testjsonl.py):
  - question / clean prompt fields
  - org_context / sub_context  (or Clean Prompt / Coherent Conflict)
  - org_answer / sub_answer

Each adapter should emit jsonl under data/processed/<dataset>/...
"""

from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("COMPLEX_RAW_ROOT", str(ROOT / "data"))).expanduser()

TAXONOMY = {
    "content_factual": {
        "NQ-Swap": DATA / "content_factual" / "NQ-Swap",
        "FreshQA": DATA / "content_factual" / "FreshQA",
    },
    "content_rag": {
        "CRAG": DATA / "content_rag" / "CRAG",
        "RGB": DATA / "content_rag" / "RGB",
    },
    "rule_guideline": {
        "GuideBench": DATA / "rule_guideline" / "GuideBench",
    },
    "rule_instruction": {
        "FollowBench": DATA / "rule_instruction" / "FollowBench",
        "IFEval": DATA / "rule_instruction" / "IFEval",
    },
    "rule_legal": {
        "LegalBench": DATA / "rule_legal" / "LegalBench",
        "ContractNLI": DATA / "rule_legal" / "ContractNLI",
    },
    "policy_api": {
        "ToolBench": DATA / "policy_api" / "ToolBench",
        "tau-bench": DATA / "policy_api" / "tau-bench",
        "BFCL": DATA / "policy_api" / "BFCL",
        "ToolSandbox": DATA / "policy_api" / "ToolSandbox",
        "API-Bank": DATA / "policy_api" / "API-Bank",
    },
    "policy_personalization": {
        "MemoryAgentBench": DATA / "policy_personalization" / "MemoryAgentBench",
        "PersonaChat": DATA / "policy_personalization" / "PersonaChat",
        "LongMemEval": DATA / "policy_personalization" / "LongMemEval",
        "Mem2ActBench": DATA / "policy_personalization" / "Mem2ActBench",
    },
}


def inventory() -> dict:
    report = {}
    for axis, ds_map in TAXONOMY.items():
        report[axis] = {}
        for name, path in ds_map.items():
            exists = path.exists()
            size = 0
            nfiles = 0
            if exists:
                for p in path.rglob("*"):
                    if p.is_file() and ".git" not in p.parts:
                        nfiles += 1
                        try:
                            size += p.stat().st_size
                        except OSError:
                            pass
            report[axis][name] = {
                "path": str(path),
                "exists": exists,
                "n_files": nfiles,
                "size_mb": round(size / 1e6, 2),
            }
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(inventory(), indent=2))
