#!/usr/bin/env python3
"""Batch LLM eval for datasets **without** an official eval script.

Uses ``src/official_judges/llm_settings.py`` (no env vars).

Supported datasets: NQ-Swap, Mem2ActBench, or generic QA via --dataset generic.

Examples:
  # single pair
  PYTHONPATH=src python3 scripts/run_llm_eval.py \\
    --dataset NQ-Swap --prediction "Paris" --gold "Paris" --question "Capital of France?"

  # Mem2Act
  PYTHONPATH=src python3 scripts/run_llm_eval.py \\
    --dataset Mem2ActBench --prediction '{"name":"/foo","arguments":{}}' \\
    --gold '{"name":"/foo","arguments":{}}'

  # score dump manifest samples (stage2 outputs)
  PYTHONPATH=src python3 scripts/run_llm_eval.py \\
    --dataset Mem2ActBench \\
    --manifest <DATA_DIR>/conflict_taxonomy_steer/llama3.1-8b-it/Mem2ActBench/qa/internal_aware_data__stage1_wrong_use_correct_context.json \\
    --out <TMP_DIR>/mem2act_llm_eval.jsonl \\
    --max-samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from official_judges.llm_eval import (  # noqa: E402
    llm_judge_dispatch,
    llm_judge_generic_qa,
    llm_judge_nqswap,
)
from official_judges.native import mem2act_match  # noqa: E402
from official_judges.llm_client import LLMConfig, print_cost_summary  # noqa: E402


def _parse_gold(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _score_one(dataset: str, prediction: str, gold, question: str) -> dict:
    ds = dataset.strip()
    if ds.lower() in {"nq-swap", "nqswap"}:
        ok = llm_judge_nqswap(prediction=prediction, gold_answers=gold, question=question)
        backend = "llm_judge_nqswap"
    elif ds.lower() in {"mem2actbench", "mem2act"}:
        if not isinstance(gold, dict):
            gold = _parse_gold(str(gold)) if not isinstance(gold, dict) else gold
        if not isinstance(gold, dict):
            gold = {"name": str(gold), "arguments": {}}
        ok = mem2act_match(prediction, gold, query=question)
        backend = "mem2act_match"
    else:
        ok = llm_judge_generic_qa(prediction=prediction, gold=gold, question=question)
        backend = "llm_judge_generic_qa"
    return {"match": bool(ok), "backend": backend, "dataset": ds}


def _iter_manifest(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    samples = obj.get("samples") or []
    for s in samples:
        pred = s.get("stage2_output_judged") or s.get("stage2_output_raw") or s.get("stage1_output_judged") or ""
        gold = s.get("sub_answer") or s.get("distracted_tokens") or s.get("tool_call")
        if isinstance(gold, list) and len(gold) == 1:
            g0 = gold[0]
            if isinstance(g0, str) and g0.strip().startswith("{"):
                try:
                    gold = json.loads(g0)
                except Exception:
                    gold = g0
            else:
                gold = g0 if not isinstance(s.get("tool_call"), dict) else s.get("tool_call")
        if s.get("tool_call") and isinstance(s["tool_call"], dict):
            gold = s["tool_call"]
        q = s.get("question") or ""
        yield {
            "line_idx": s.get("line_idx"),
            "prediction": pred,
            "gold": gold,
            "question": q,
            "sample": s,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM API eval for datasets without official eval")
    ap.add_argument("--dataset", required=True, help="NQ-Swap | Mem2ActBench | generic")
    ap.add_argument("--prediction", default="")
    ap.add_argument("--gold", default="")
    ap.add_argument("--question", default="")
    ap.add_argument("--manifest", type=Path, default=None, help="Dump internal_aware_data JSON")
    ap.add_argument("--out", type=Path, default=None, help="Write JSONL results")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    cfg = LLMConfig.from_settings()
    print(f"[llm_eval] model={cfg.model} base={cfg.base_url}", flush=True)

    rows_out = []
    if args.manifest:
        n_ok = 0
        n = 0
        for item in _iter_manifest(args.manifest):
            if args.max_samples and n >= args.max_samples:
                break
            r = _score_one(args.dataset, str(item["prediction"]), item["gold"], str(item["question"]))
            r["line_idx"] = item["line_idx"]
            r["prediction"] = item["prediction"]
            r["gold"] = item["gold"]
            rows_out.append(r)
            n += 1
            n_ok += int(r["match"])
            print(f"[{n}] line={item['line_idx']} match={r['match']}", flush=True)
        print(f"[done] {n_ok}/{n} = {n_ok / max(1, n):.3f}", flush=True)
    else:
        if not args.prediction:
            raise SystemExit("provide --prediction or --manifest")
        gold = _parse_gold(args.gold)
        r = _score_one(args.dataset, args.prediction, gold, args.question)
        rows_out.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[wrote] {args.out}", flush=True)
    print_cost_summary()


if __name__ == "__main__":
    main()
