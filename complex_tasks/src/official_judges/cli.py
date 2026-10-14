#!/usr/bin/env python3
"""CLI: score one prediction with an official (native) judge.

Examples:
  python3 -m official_judges.cli --dataset RGB \\
    --prediction "Tampa, Florida" --gold "Tampa, Florida"

  # CRAG needs LLM key in official_judges/llm_settings.py
  python3 -m official_judges.cli --dataset CRAG \\
    --prediction "..." --gold "..." --query "..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python3 -m official_judges.cli` from src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from official_judges.judges import judge_prediction  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Official dataset judge CLI")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--gold", required=True, help="Official target; JSON is accepted for tool calls.")
    ap.add_argument("--query", default="")
    ap.add_argument("--question", default="")
    ap.add_argument("--category", default="")
    ap.add_argument("--question-type", default="multi-session")
    ap.add_argument("--eval-function", default="")
    ap.add_argument("--json-meta", default="", help="Extra meta JSON object")
    args = ap.parse_args()

    meta: dict = {
        "query": args.query,
        "question": args.question or args.query,
        "category": args.category,
        "question_type": args.question_type,
    }
    if args.eval_function:
        meta["eval_function"] = args.eval_function
    if args.json_meta.strip():
        meta.update(json.loads(args.json_meta))

    gold: object = args.gold
    if args.gold.strip().startswith(("{", "[")):
        try:
            gold = json.loads(args.gold)
        except json.JSONDecodeError:
            pass

    res = judge_prediction(
        args.dataset,
        args.prediction,
        gold,
        meta=meta,
    )
    print(
        json.dumps(
            {
                "label": res.label,
                "official_score": res.official_score,
                "meta": res.meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
