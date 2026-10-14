#!/usr/bin/env python3
"""Build all-sample context manifests for steering.

Every processed JSONL row is kept. The prompt is the injected context
(``sub_context``). ``gold_answers`` is the injected external/context answer
(``sub_answer``), including the counterfactual target in NQ-Swap and RGB.
No Stage1 filter.

Writes:
  ${out_root}/<model>/<dataset>/<category>/internal_aware_data__context_all.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path(os.environ.get("PROCESSED_ROOT", str(ROOT / "data" / "processed"))).expanduser()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _context_prompt(question: str, context: str, *, dataset: str = "") -> str:
    ds = str(dataset).strip()
    # LongMemEval oracle rows already embed History Chats + Current Date + Question.
    if ds in {"LongMemEval", "longmemeval"} and "History Chats:" in context:
        return f"{context.strip()}\nAnswer:"
    # Mem2Act official prompt already ends with Request + JSON tool call cue.
    if ds in {"Mem2ActBench", "mem2act"} and context.strip().startswith("Conversation history:"):
        return context.strip()
    # ToolSandbox plays the official scenario; the prompt field is only a label.
    if ds in {"ToolSandbox", "toolsandbox"}:
        return context.strip()
    # BFCL uses chat messages + tools at inference; sub_context is the system prompt.
    if ds in {"BFCL", "bfcl"}:
        return context.strip()
    # These adapters already end with the official answer cue.
    if context.rstrip().endswith("Answer:"):
        return context.strip()
    return f"{context.strip()}\n\nQuestion: {question.strip()}\nAnswer:"


def _official_target(row: dict[str, Any], dataset: str) -> list[str]:
    """Select the external/context answer used as the steering target."""
    del dataset
    return _as_str_list(row.get("sub_answer"))


def build_one(jsonl: Path, *, model: str, out_root: Path) -> Path | None:
    dataset = jsonl.parent.name
    category = jsonl.stem
    samples: list[dict[str, Any]] = []
    with jsonl.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question") or "").strip()
            context = str(row.get("sub_context") or "").strip()
            gold = _official_target(row, str(row.get("dataset") or dataset))
            if not question or not context or not gold:
                continue
            sample: dict[str, Any] = {
                "line_idx": i,
                "model": model,
                "dataset": str(row.get("dataset") or dataset),
                "category": str(row.get("category") or category),
                "stage2_prompt_input": _context_prompt(
                    question, context, dataset=str(row.get("dataset") or dataset)
                ),
                "gold_answers": gold,
                "distracted_tokens": [],
                "question": question,
            }
            if row.get("question_type"):
                sample["question_type"] = row.get("question_type")
            if row.get("question_date"):
                sample["question_date"] = row.get("question_date")
            if row.get("complexity_level"):
                sample["complexity_level"] = row.get("complexity_level")
            if row.get("complexity_name"):
                sample["complexity_name"] = row.get("complexity_name")
            if row.get("domain"):
                sample["domain"] = row.get("domain")
            for extra in (
                "source",
                "example_id",
                "level",
                "target",
                "constraint",
                "relation_type",
                "locomo_category",
                "locomo_cat5_a",
                "locomo_cat5_b",
                "iheval_task",
                "iheval_answer",
                "iheval_prompt",
                "toolsandbox_scenario",
                "toolsandbox_categories",
                "bfcl_id",
                "bfcl_test_category",
                "bfcl_functions",
                "bfcl_ground_truth",
                "bfcl_messages",
                "bfcl_tools",
                "bfcl_model_name",
                "ifeval_target",
                "ifeval_prompt",
                "freshqa_question",
                "freshqa_answers",
                "freshqa_effective_year",
                "freshqa_false_premise",
                "freshqa_num_hops",
                "freshqa_fact_type",
                "official_metric",
                "context_length_tag",
                "followbench_evolve_instructions",
                "followbench_rule_based",
            ):
                if row.get(extra) not in (None, ""):
                    sample[extra] = row.get(extra)
            samples.append(sample)
    if not samples:
        print(f"[skip] no context rows: {jsonl}")
        return None
    out = (
        out_root
        / model
        / dataset
        / category
        / "internal_aware_data__context_all.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "dataset": dataset,
        "category": category,
        "group": "context_all",
        "num_kept": len(samples),
        "samples": samples,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[context_all] {dataset}/{category} n={len(samples)} -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", required=True, help="Comma-separated dataset names under data/processed.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("<DATA_DIR>/conflict_taxonomy_steer"),
    )
    args = ap.parse_args()
    names = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    n_files = 0
    for name in names:
        ds_dir = PROCESSED / name
        if not ds_dir.is_dir():
            raise SystemExit(f"missing processed dir: {ds_dir}")
        for jsonl in sorted(ds_dir.glob("*.jsonl")):
            if build_one(jsonl, model=str(args.model), out_root=args.out_root) is not None:
                n_files += 1
    print(f"[context_all] wrote {n_files} manifests")


if __name__ == "__main__":
    main()
