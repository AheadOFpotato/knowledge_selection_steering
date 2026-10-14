#!/usr/bin/env python3
"""Audit Stage-2 string matching against an OpenAI-compatible LLM judge.

The default input is the fixed-alpha Llama OOD evaluation.  For each of the
four OOD datasets, the script samples existing Stage-2 outputs and compares
the repository's substring matcher with a four-way LLM judgement:

  correct_only, distractor_only, both, neither

The API configuration is read from the locally configured judge settings file;
credentials are never copied into this repository or written to the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import runpy
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ParaConfilct", "macnoise", "memotrap", "counterfact")
LABELS = ("correct_only", "distractor_only", "both", "neither")
DEFAULT_INPUT_ROOT = ROOT / "output" / "four_condition_fixed_full" / "llama3-8b-it"
DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "llmapi" / "llm_settings.py"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _matches(text: str, candidates: list[str]) -> bool:
    normalized = _norm(text)
    return bool(normalized) and any(
        candidate in normalized
        for candidate in (_norm(item) for item in candidates)
        if len(candidate) >= 2
    )


def string_label(answer: str, gold: list[str], distractors: list[str]) -> str:
    """Four-way version of the current substring judgement rule."""
    hit_gold = _matches(answer, gold)
    hit_distractor = _matches(answer, distractors)
    if hit_gold and hit_distractor:
        return "both"
    if hit_gold:
        return "correct_only"
    if hit_distractor:
        return "distractor_only"
    return "neither"


def _sample_id(source: Path, line_idx: int) -> str:
    digest = hashlib.sha256(f"{source}:{line_idx}".encode()).hexdigest()
    return digest[:16]


def load_samples(input_root: Path, direction: str, alpha_sign: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASETS}
    for path in sorted(input_root.rglob("*steer_result.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if str(doc.get("direction_category_tag")) != direction:
            continue
        alpha = float(doc.get("alpha", 0.0))
        if alpha_sign == "positive" and alpha <= 0:
            continue
        if alpha_sign == "negative" and alpha >= 0:
            continue
        dataset = path.relative_to(input_root).parts[0]
        if dataset not in grouped:
            continue
        for row in doc.get("per_sample_outputs_and_judgements", {}).get("steer_clean_minus_conflict", []):
            answer = str(row.get("stage2_output_raw") or "").strip()
            if not row.get("stage2_evaluated") or not answer:
                continue
            gold = [str(item) for item in row.get("gold_answers", []) if str(item).strip()]
            distractors = [str(item) for item in row.get("distracted_tokens", []) if str(item).strip()]
            if not distractors and str(row.get("distracted_token") or "").strip():
                distractors = [str(row["distracted_token"])]
            grouped[dataset].append(
                {
                    "sample_id": _sample_id(path, int(row.get("line_idx", -1))),
                    "source_result": str(path.relative_to(ROOT)),
                    "dataset": dataset,
                    "line_idx": int(row.get("line_idx", -1)),
                    "prompt": str(row.get("conflict_prompt_input") or ""),
                    "model_answer": answer,
                    "gold_answers": gold,
                    "distracted_tokens": distractors,
                    "string_label": string_label(answer, gold, distractors),
                    "existing_three_way_label": str(row.get("stage2_label") or ""),
                }
            )
    return grouped


def choose_samples(grouped: dict[str, list[dict[str, Any]]], per_dataset: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidates = grouped[dataset]
        if len(candidates) < per_dataset:
            raise ValueError(f"{dataset}: only {len(candidates)} judged Stage-2 outputs; need {per_dataset}")
        rng = random.Random(f"{seed}:{dataset}")
        selected.extend(rng.sample(candidates, per_dataset))
    return selected


def judge_prompt(sample: dict[str, Any]) -> str:
    return f"""You are evaluating an answer to a factual QA task. Decide which ONE label applies.

correct_only: the answer supports at least one gold answer and does not support a distractor.
distractor_only: the answer supports a distractor and does not support a gold answer.
both: the answer supports at least one gold answer and at least one distractor.
neither: the answer supports neither, is irrelevant, refuses, or is too ambiguous to determine.

Use only the supplied gold answers, distractors, and model answer. Names or answers may occur in explanations; classify based on what the model answer asserts, not merely a quoted question or an incidental token. Return JSON only: {{"label":"correct_only|distractor_only|both|neither"}}.

Prompt:
{sample['prompt']}

Gold answers: {json.dumps(sample['gold_answers'], ensure_ascii=False)}
Distractors: {json.dumps(sample['distracted_tokens'], ensure_ascii=False)}
Model answer:
{sample['model_answer']}
"""


def parse_label(text: str) -> str | None:
    try:
        value = json.loads(text)
        label = str(value.get("label", "")).strip()
        return label if label in LABELS else None
    except json.JSONDecodeError:
        return None


def judge_one(client: OpenAI, model: str, sample: dict[str, Any], retries: int) -> str:
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt(sample)}],
                temperature=0,
                max_tokens=32,
                response_format={"type": "json_object"},
            )
            label = parse_label(response.choices[0].message.content or "")
            if label:
                return label
            last_error = f"invalid JSON label: {(response.choices[0].message.content or '')[:300]!r}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            # Some OpenAI-compatible endpoints do not implement response_format.
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": judge_prompt(sample)}],
                    temperature=0,
                    max_tokens=32,
                )
                label = parse_label(response.choices[0].message.content or "")
                if label:
                    return label
                last_error = f"invalid fallback JSON label: {(response.choices[0].message.content or '')[:300]!r}"
            except Exception as fallback_exc:
                last_error = f"{type(fallback_exc).__name__}: {str(fallback_exc)[:300]}"
        if attempt < retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"LLM judge returned no valid label for {sample['sample_id']}: {last_error}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset and row.get("llm_label") in LABELS]
        by_dataset[dataset] = _summary(subset)
    return {"overall": _summary([row for row in rows if row.get("llm_label") in LABELS]), "by_dataset": by_dataset}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {expected: {actual: 0 for actual in LABELS} for expected in LABELS}
    for row in rows:
        matrix[row["llm_label"]][row["string_label"]] += 1
    matches = sum(row["llm_label"] == row["string_label"] for row in rows)
    return {
        "n": len(rows),
        "agreement": matches / len(rows) if rows else None,
        "agreement_count": matches,
        "llm_label_counts": dict(Counter(row["llm_label"] for row in rows)),
        "string_label_counts": dict(Counter(row["string_label"] for row in rows)),
        "confusion_matrix_rows_llm_columns_string": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--direction", default="A-C")
    parser.add_argument("--alpha-sign", choices=("positive", "negative"), default="positive")
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "string_match_llm_audit")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    settings = runpy.run_path(str(args.settings))
    base_url = str(settings.get("LLM_BASE_URL") or "").rstrip("/")
    api_key = str(settings.get("LLM_API_KEY") or "")
    model = str(settings.get("LLM_MODEL") or "")
    if not (base_url and api_key and model):
        raise RuntimeError(f"incomplete judge settings: {args.settings}")

    grouped = load_samples(args.input_root, args.direction, args.alpha_sign)
    selected = choose_samples(grouped, args.per_dataset, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.direction}_alpha-{args.alpha_sign}_n{args.per_dataset}_seed{args.seed}"
    jsonl_path = args.out_dir / f"{stem}.jsonl"
    canonical_jsonl_path = args.out_dir / f"{stem}_canonical.jsonl"
    summary_path = args.out_dir / f"{stem}_summary.json"

    completed: dict[str, dict[str, Any]] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("llm_label") in LABELS:
                completed[str(row["sample_id"])] = row

    pending = [sample for sample in selected if sample["sample_id"] not in completed]

    def judge_sample(sample: dict[str, Any]) -> dict[str, Any]:
        # Separate clients avoid sharing connection state across worker threads.
        client = OpenAI(base_url=base_url, api_key=api_key)
        sample["llm_label"] = judge_one(client, model, sample, args.retries)
        return sample

    with jsonl_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(judge_sample, sample) for sample in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                sample = future.result()
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.flush()
                completed[sample["sample_id"]] = sample
                print(f"judged {index}/{len(pending)}: {sample['dataset']}", flush=True)

    rows = [completed[sample["sample_id"]] for sample in selected]
    canonical_jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "input_root": str(args.input_root),
        "direction": args.direction,
        "alpha_sign": args.alpha_sign,
        "per_dataset": args.per_dataset,
        "seed": args.seed,
        "judge_model": model,
        "labels": list(LABELS),
        "summary": summarize(rows),
        "resume_jsonl": str(jsonl_path),
        "canonical_jsonl": str(canonical_jsonl_path),
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overall = report["summary"]["overall"]
    print(f"agreement: {overall['agreement_count']}/{overall['n']} = {overall['agreement']:.1%}")
    print(summary_path)


if __name__ == "__main__":
    main()
