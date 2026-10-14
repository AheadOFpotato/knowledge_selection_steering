#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from chat_prompt import format_prompts_for_inference

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(ROOT / "data"))).expanduser()
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DUMP_ROOT", str(ROOT / "output" / "dump"))).expanduser()
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", str(ROOT / "models"))).expanduser()

MODEL_PATHS: dict[str, str] = {
    # Local weights live under <MODEL_ROOT> (legacy <DATA_DIR>/model/* removed).
    # llama3-8b-it aliases to 3.1 Instruct when 3.0 weights are unavailable.
    "llama3-8b-it": str(MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct"),
    "llama3.1-8b-it": str(MODEL_ROOT / "Meta-Llama-3.1-8B-Instruct"),
    "gemma2-9b-it": str(MODEL_ROOT / "Gemma2-9b-it"),
    "qwen3-8b": str(MODEL_ROOT / "Qwen3-8B"),
    "mistral-7b-v0.1": str(MODEL_ROOT / "Mistral-7B-v0.1"),
    "mistral-7b-instruct-v0.3": str(MODEL_ROOT / "Mistral-7B-Instruct-v0.3"),
    "yi-6b-chat": str(MODEL_ROOT / "Yi-6B-Chat"),
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 4096
DEFAULT_BATCH_SIZE = 4

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _truncate_for_judge(s: str) -> str:
    if not s:
        return s
    m = re.search(r"[。！？.!?]", s)
    return s[: m.end()].rstrip() if m else s


def _normalized_answers(gold_answers: list[str]) -> list[str]:
    # Keep the previous rule: only use answers with length >= 2.
    return [t for t in (_norm(a) for a in gold_answers) if len(t) >= 2]


def _answer_matches(generated: str, gold_answers: list[str]) -> bool:
    g = _norm(generated)
    if not g:
        return False
    return any(t in g for t in _normalized_answers(gold_answers))


def _distractor_in_text(generated: str, distracted_token: str) -> bool:
    g = _norm(generated)
    if not g:
        return False
    t = _norm(str(distracted_token or "").strip())
    if not t:
        return False
    return t in g


def _stage2_label_from_strings(
    raw_text: str, gold_answers: list[str], distracted_tokens: list[str]
) -> str:
    """Gold substring anywhere -> correct; else any distractor substring -> wrong; else skipped."""
    if _answer_matches(raw_text, gold_answers):
        return "correct"
    if any(_distractor_in_text(raw_text, d) for d in distracted_tokens if str(d).strip()):
        return "wrong"
    return "skipped"


def _first_option_letter(generated: str) -> str:
    g = str(generated or "")
    m = re.search(r"[A-Za-z]", g)
    return m.group(0).upper() if m else ""


def _answer_matches_letters(generated: str, letters: list[str]) -> bool:
    first = _first_option_letter(generated)
    if not first:
        return False
    allowed = {str(l).strip().upper() for l in letters if str(l).strip()}
    return first in allowed


def _stage2_label_from_letters(raw_text: str, gold_letters: list[str], wrong_letters: list[str]) -> str:
    if _answer_matches_letters(raw_text, gold_letters):
        return "correct"
    if _answer_matches_letters(raw_text, wrong_letters):
        return "wrong"
    return "skipped"


def _official_judge_sample(
    *,
    dataset: str,
    category: str,
    text: str,
    gold_answers: list[str],
    distracted_tokens: list[str],
    sample: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    try:
        from official_judges import match_gold
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"--judge official requires src/official_judges on PYTHONPATH ({e})"
        ) from e

    sample = sample or {}
    meta = {
        "category": category or str(sample.get("category") or ""),
        "query": sample.get("question") or sample.get("query") or "",
        "question": sample.get("question") or sample.get("query") or "",
        "question_type": sample.get("question_type") or "multi-session",
        "eval_function": sample.get("eval_function") or "",
        "tool_call": sample.get("tool_call"),
        "bfcl_id": sample.get("bfcl_id"),
        "bfcl_test_category": sample.get("bfcl_test_category"),
        "bfcl_functions": sample.get("bfcl_functions"),
        "bfcl_ground_truth": sample.get("bfcl_ground_truth"),
        "bfcl_model_name": sample.get("bfcl_model_name"),
    }
    target: object = sample.get("tool_call") or sample.get("sub_answer") or distracted_tokens or gold_answers
    hit = match_gold(dataset, text, target, meta=meta)
    return hit, "correct" if hit else "wrong"


def _resolve_dataset_file(dataset_dir: Path) -> Path:
    for name in ("test.jsonl", "dev.jsonl", "train.jsonl"):
        p = dataset_dir / name
        if p.is_file():
            return p
    csv_candidates = sorted(dataset_dir.glob("*.csv"))
    if csv_candidates:
        return csv_candidates[0]
    raise FileNotFoundError(f"no test/dev/train jsonl or csv under {dataset_dir}")


def _infer_row_format(sample: dict[str, Any]) -> str:
    if "Clean Prompt" in sample:
        return "paraconflict"
    if "org_context" in sample and "sub_context" in sample:
        return "nq_swap"
    if "question" in sample and "context1" in sample and "context2" in sample and "answer1" in sample and "answer2" in sample:
        return "wiki_contradict"
    if "question" in sample and "choices" in sample and "answer" in sample and "negative_context" in sample:
        return "kre"
    if "question" in sample and "context" in sample and "obj" in sample and "replace_name" in sample:
        return "dynamicqa"
    if "prompt" in sample and "classes" in sample and "answer_index" in sample:
        return "memotrap"
    if "requested_rewrite" in sample:
        if isinstance(sample.get("requested_rewrite"), list) and "questions" in sample and "new_answer" in sample:
            return "mquake"
        return "counterfact"
    if "question" in sample and "cf_context" in sample and "cf_answer" in sample and "orig_answer" in sample:
        return "confiqa"
    if "question" in sample and "answers" in sample and "ctxs" in sample and "answer_replace" in sample:
        return "macnoise"
    raise ValueError(
        "unrecognized jsonl row schema: expected ParaConflict fields (Clean Prompt) "
        "or NQ-Swap fields (org_context, sub_context) or WikiContradict fields "
        "(question, context1, context2, answer1, answer2) or KRE fields "
        "(question, choices, answer, negative_context) or dynamicqa fields "
        "(question, context, obj, replace_name) or memotrap fields "
        "(prompt, classes, answer_index) or counterfact fields (requested_rewrite) "
        "or ConFiQA fields (question, cf_context, cf_answer, orig_answer) "
        "or MQuAKE fields (requested_rewrite:list, questions, new_answer) "
        "or macnoise fields (question, answers, ctxs, answer_replace)"
    )


def _load_rows_paraconflict(jsonl_path: Path, category: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("Category", "")).strip() == category.strip():
                rows.append((i, row))
    return rows


def _load_rows_nq_swap(jsonl_path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append((i, json.loads(line)))
    return rows


def _load_rows_wiki_contradict(csv_path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append((i, dict(row)))
    return rows


def _resolve_dynamicqa_csv(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip()
    if not cat:
        raise FileNotFoundError("dynamicqa requires --category to choose csv (static/disputable)")
    p = dataset_dir / cat
    if p.suffix.lower() != ".csv":
        p = dataset_dir / f"{cat}.csv"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"dynamicqa csv not found for category={category!r}. Expected one of: static.csv, disputable.csv"
    )


def _load_rows_csv_dict(csv_path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append((i, dict(row)))
    return rows


def _resolve_memotrap_csv(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip()
    if not cat:
        raise FileNotFoundError("memotrap requires --category to choose csv (e.g. 1-proverb-ending)")
    p = dataset_dir / cat
    if p.suffix.lower() != ".csv":
        p = dataset_dir / f"{cat}.csv"
    if p.is_file():
        return p
    raise FileNotFoundError(f"memotrap csv not found for category={category!r} under {dataset_dir}")


def _resolve_kre_json(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip()
    if not cat:
        raise FileNotFoundError("KRE requires --category to choose json (cose/ecare/musique/squad)")
    p = dataset_dir / cat
    if p.suffix.lower() != ".json":
        p = dataset_dir / f"{cat}.json"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"KRE json not found for category={category!r}. Expected one of: cose.json, ecare.json, musique.json, squad.json"
    )


def _resolve_counterfact_json(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip()
    if not cat:
        raise FileNotFoundError("counterfact requires --category to choose json filename")
    p = dataset_dir / cat
    if p.suffix.lower() != ".json":
        p = dataset_dir / f"{cat}.json"
    if p.is_file():
        return p
    raise FileNotFoundError(f"counterfact json not found for category={category!r} under {dataset_dir}")


def _load_rows_kre(json_path: Path) -> list[tuple[int, dict[str, Any]]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"KRE file is not a JSON list: {json_path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(obj):
        if isinstance(row, dict):
            rows.append((i, row))
    return rows


def _load_rows_counterfact(json_path: Path) -> list[tuple[int, dict[str, Any]]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    recs = obj.get("records", []) if isinstance(obj, dict) else []
    if not isinstance(recs, list):
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(recs):
        if isinstance(row, dict):
            rows.append((i, row))
    return rows


def _resolve_macnoise_json(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip()
    if cat and not cat.startswith("pert_lbls"):
        p = dataset_dir / cat
        if p.suffix.lower() != ".json":
            p = dataset_dir / f"{cat}.json"
        if p.is_file():
            return p
    json_candidates = sorted(dataset_dir.glob("*.json"))
    if json_candidates:
        return json_candidates[0]
    raise FileNotFoundError(f"macnoise json not found under {dataset_dir}")


def _load_rows_macnoise(json_path: Path) -> list[tuple[int, dict[str, Any]]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"macnoise file is not a JSON list: {json_path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(obj):
        if isinstance(row, dict):
            rows.append((i, row))
    return rows


def _resolve_confiqa_json(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip().lower()
    mapping = {
        "qa": "ConFiQA-QA.json",
        "mr": "ConFiQA-MR.json",
        "mc": "ConFiQA-MC.json",
    }
    if cat in mapping:
        p = dataset_dir / mapping[cat]
        if p.is_file():
            return p
    if cat:
        p = dataset_dir / category
        if p.suffix.lower() != ".json":
            p = dataset_dir / f"{category}.json"
        if p.is_file():
            return p
    default_p = dataset_dir / "ConFiQA-QA.json"
    if default_p.is_file():
        return default_p
    json_candidates = sorted(dataset_dir.glob("*.json"))
    if json_candidates:
        return json_candidates[0]
    raise FileNotFoundError(f"ConFiQA json not found under {dataset_dir}")


def _resolve_mquake_json(dataset_dir: Path, category: str) -> Path:
    cat = str(category).strip()
    if cat:
        p = dataset_dir / cat
        if p.suffix.lower() != ".json":
            p = dataset_dir / f"{cat}.json"
        if p.is_file():
            return p
    default_p = dataset_dir / "MQuAKE-CF-3k-v2.json"
    if default_p.is_file():
        return default_p
    json_candidates = sorted(dataset_dir.glob("*.json"))
    if json_candidates:
        return json_candidates[0]
    raise FileNotFoundError(f"MQuAKE json not found under {dataset_dir}")


def _load_rows_json_list(json_path: Path) -> list[tuple[int, dict[str, Any]]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"json file is not a JSON list: {json_path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(obj):
        if isinstance(row, dict):
            rows.append((i, row))
    return rows


def _load_rows_nq_pair(orig_json: Path, conflict_json: Path) -> list[tuple[int, dict[str, Any]]]:
    orig_obj = json.loads(orig_json.read_text(encoding="utf-8"))
    conflict_obj = json.loads(conflict_json.read_text(encoding="utf-8"))
    if not isinstance(orig_obj, list) or not isinstance(conflict_obj, list):
        raise ValueError("nq files must be JSON lists")
    n = min(len(orig_obj), len(conflict_obj))
    rows: list[tuple[int, dict[str, Any]]] = []
    for i in range(n):
        oe = orig_obj[i] if isinstance(orig_obj[i], dict) else {}
        ce = conflict_obj[i] if isinstance(conflict_obj[i], dict) else {}
        rows.append(
            (
                i,
                {
                    "question": ce.get("question", ""),
                    "org_context": oe.get("context", ""),
                    "sub_context": ce.get("context", ""),
                    "org_answer": oe.get("answer", []),
                    "sub_answer": ce.get("answer", ""),
                },
            )
        )
    return rows


def _nq_swap_prompt(question: str, context: str) -> str:
    q = str(question).strip()
    c = str(context).strip()
    return f"{c}\n\nQuestion: {q}\nAnswer:"


def _nq_swap_question_only_prompt(question: str) -> str:
    q = str(question).strip()
    return f"Question: {q}\nAnswer:"


def _qa_prompt(question: str, context: str) -> str:
    q = str(question).strip()
    c = str(context).strip()
    return f"{c}\n\nQuestion: {q}\nAnswer:"


def _memotrap_clean_prompt(full_prompt: str) -> str:
    """
    Example:
      'Write a quote that ends in the word ""down"": He who does not advance goes'
    -> 'Write a quote : He who does not advance goes'
    """
    s = str(full_prompt)
    # Accept both csv-escaped ""word"" and plain "word" forms.
    s = re.sub(
        r'^Write a quote\s+that ends in the word\s+(?:""[^"]+""|"[^"]+")\s*:',
        "Write a quote :",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _memotrap_parse_classes(s: Any) -> list[str]:
    # In csv it's a python list literal string, e.g. "[' down.', ' backwards.']"
    try:
        v = ast.literal_eval(str(s))
    except Exception:
        v = None
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            t = str(x).strip()
            if not t:
                continue
            # User requirement: do not keep trailing '.' in answers.
            t = t.rstrip().rstrip(".").rstrip()
            out.append(t)
        return out
    return [str(s).strip()] if str(s).strip() else []


def _stage2_label_memotrap(raw_text: str, wrong_answers: list[str]) -> str:
    # Per user rule: if conflict output matches answer_index answer -> wrong; else correct.
    return "wrong" if _answer_matches(raw_text, wrong_answers) else "correct"


def _mcq_prompt(question: str, choices: list[str], *, context: str = "") -> str:
    q = str(question).strip()
    lines = ["Please output only the option letter (e.g., A, B, C).", f"Question: {q}"]
    if context.strip():
        lines.insert(0, context.strip())
    if choices:
        lines.append("Choices:")
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, c in enumerate(choices):
            lab = labels[i] if i < len(labels) else str(i + 1)
            lines.append(f"{lab}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def _letter_index(s: str) -> int | None:
    t = str(s).strip().upper()
    if len(t) == 1 and "A" <= t <= "Z":
        return ord(t) - ord("A")
    return None


def _gold_list_from_row_field(ans: Any) -> list[str]:
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(v).strip() for v in ans if str(v).strip()]
    sv = str(ans).strip()
    return [sv] if sv else []


def _answers_from_wikicontradict_field(ans: Any) -> list[str]:
    vals = _gold_list_from_row_field(ans)
    out: list[str] = []
    for v in vals:
        for p in str(v).split("|"):
            sp = p.strip()
            if sp:
                out.append(sp)
    return out


def _distractor_strings_paraconflict(row: dict[str, Any]) -> list[str]:
    t = str(row.get("Distracted Token", "")).strip()
    return [t] if t else []


def _paraconflict_correct_context_prompt(
    conflict_p: str, gold: list[str], dstrs: list[str], *, clean_fallback: str
) -> str:
    """ParaConflict has no separate gold passage in jsonl; mirror conflict structure with distractor→gold."""
    if not conflict_p.strip() or not gold:
        return clean_fallback.strip()
    correct_ans = str(gold[0]).strip()
    if not correct_ans:
        return clean_fallback.strip()
    out = conflict_p
    replaced = False
    for ds in dstrs:
        s = str(ds).strip()
        if not s:
            continue
        new_out, n = re.subn(re.escape(s), correct_ans, out, flags=re.IGNORECASE)
        if n > 0:
            replaced = True
            out = new_out
    if not replaced:
        return clean_fallback.strip()
    return out.strip()


def _distractor_strings_nq_swap(row: dict[str, Any]) -> list[str]:
    return _gold_list_from_row_field(row.get("sub_answer"))


def _macnoise_context(row: dict[str, Any], *, use_pert: bool, pert_key: str) -> str:
    ctxs = row.get("ctxs", [])
    if not isinstance(ctxs, list):
        return ""
    labels = row.get(pert_key, [])
    if not isinstance(labels, list):
        labels = []
    chunks: list[str] = []
    for i, c in enumerate(ctxs):
        if not isinstance(c, dict):
            continue
        if use_pert:
            use_this_pert = True
            if i < len(labels):
                try:
                    use_this_pert = int(labels[i]) == 1
                except Exception:
                    use_this_pert = bool(labels[i])
            t = str(c.get("text_pert", "") if use_this_pert else c.get("text", "")).strip()
        else:
            t = str(c.get("text", "")).strip()
        if t:
            chunks.append(t)
    return "\n\n".join(chunks).strip()


def _macnoise_first_pert100_index(row: dict[str, Any]) -> int:
    """Pick first perturbed index; prefer pert_lbls_100, fallback to pert_lbls."""
    ctxs = row.get("ctxs", [])
    labs = row.get("pert_lbls_100", [])
    if not isinstance(labs, list) or not labs:
        labs = row.get("pert_lbls", [])
    if not isinstance(ctxs, list) or not isinstance(labs, list):
        return -1
    n = min(len(ctxs), len(labs))
    for i in range(n):
        try:
            if int(labs[i]) == 1:
                return i
        except Exception:
            if bool(labs[i]):
                return i
    return -1


def _macnoise_ctx_at_index(row: dict[str, Any], idx: int, *, use_pert: bool) -> str:
    """Use text/text_pert from a specific ctx index."""
    ctxs = row.get("ctxs", [])
    if not isinstance(ctxs, list) or idx < 0 or idx >= len(ctxs):
        return ""
    c = ctxs[idx]
    if not isinstance(c, dict):
        return ""
    key = "text_pert" if use_pert else "text"
    return str(c.get(key, "")).strip()


def _paraconflict_conflict_prompt(
    row: dict[str, Any], conflict_style: str = "auto"
) -> str:
    """Pick Stage2 wrong-context prompt for ParaConflict rows."""
    style = str(conflict_style).strip().lower()
    if style in ("substitution", "sub"):
        return str(row.get("Substitution Conflict", "")).strip()
    if style in ("coherent", "coh"):
        return str(row.get("Coherent Conflict", "")).strip()
    # auto: legacy default (prefer coherent, else substitution)
    return str(row.get("Coherent Conflict", "")).strip() or str(
        row.get("Substitution Conflict", "")
    ).strip()


def _row_uniform(
    row: dict[str, Any],
    fmt: str,
    macnoise_pert_key: str = "pert_lbls",
    *,
    paraconflict_conflict_style: str = "auto",
) -> tuple[str, str, list[str], list[str]]:
    """Returns (clean_prompt, conflict_prompt, gold_answers, distractor_strings)."""
    if fmt == "paraconflict":
        gold = _gold_list_from_row_field(row.get("Answer"))
        conflict = _paraconflict_conflict_prompt(row, paraconflict_conflict_style)
        clean = str(row.get("Clean Prompt", "")).strip()
        dstrs = _distractor_strings_paraconflict(row)
        return clean, conflict, gold, dstrs
    if fmt == "nq_swap":
        gold = _gold_list_from_row_field(row.get("org_answer"))
        dstrs = _distractor_strings_nq_swap(row)
        q = str(row.get("question", "")).strip()
        clean = _nq_swap_question_only_prompt(q)
        conflict = _nq_swap_prompt(q, str(row.get("sub_context", "")))
        return clean, conflict, gold, dstrs
    if fmt == "kre":
        q = str(row.get("question", "")).strip()
        raw_choices = row.get("choices", [])
        if isinstance(raw_choices, list):
            choices = [str(x).strip() for x in raw_choices]
        else:
            choices = []
        ai = _letter_index(str(row.get("answer", "")))
        gold: list[str] = []
        if ai is not None and 0 <= ai < len(choices):
            gold.append(chr(ord("A") + ai))
        dstrs: list[str] = []
        ci = _letter_index(str(row.get("candidate", "")))
        if ci is not None and 0 <= ci < len(choices):
            dstrs.append(chr(ord("A") + ci))
        clean = _mcq_prompt(q, choices)
        conflict = _mcq_prompt(q, choices, context=str(row.get("negative_context", "")))
        return clean, conflict, gold, dstrs
    if fmt == "dynamicqa":
        q = str(row.get("question", "")).strip()
        ctx = str(row.get("context", ""))
        obj = str(row.get("obj", "")).strip()
        rep = str(row.get("replace_name", "")).strip()
        conflict_ctx = ctx.replace("[ENTITY]", rep)
        clean = _nq_swap_question_only_prompt(q)
        conflict = _qa_prompt(q, conflict_ctx)
        gold = [obj] if obj else []
        dstrs = [rep] if rep else []
        return clean, conflict, gold, dstrs
    if fmt == "memotrap":
        prompt = str(row.get("prompt", "")).strip()
        classes = _memotrap_parse_classes(row.get("classes"))
        try:
            ai = int(str(row.get("answer_index", "")).strip())
        except Exception:
            ai = -1
        if not classes or ai not in (0, 1) or ai >= len(classes):
            return _nq_swap_question_only_prompt(prompt), "", [], []
        wrong_ans = str(classes[ai]).strip()
        other_ans = str(classes[1 - ai]).strip()
        clean = _memotrap_clean_prompt(prompt)
        conflict = prompt
        gold = [other_ans] if other_ans else []
        dstrs = [wrong_ans] if wrong_ans else []
        return clean, conflict, gold, dstrs
    if fmt == "counterfact":
        rw = row.get("requested_rewrite", {})
        if not isinstance(rw, dict):
            return "", "", [], []
        subj = str(rw.get("subject", "")).strip()
        ptempl = str(rw.get("prompt", "{} is located in")).strip()
        # User requirement: only process templates with '{}', mapping it to subject.
        if "{}" not in ptempl:
            return "", "", [], []
        base_q = ptempl.format(subj)
        t_new = rw.get("target_new", {})
        t_true = rw.get("target_true", {})
        new_s = str(t_new.get("str", "")).strip() if isinstance(t_new, dict) else str(t_new).strip()
        true_s = str(t_true.get("str", "")).strip() if isinstance(t_true, dict) else str(t_true).strip()
        # Clean: keep original behavior (do not wrap).
        clean = base_q

        # Conflict context: sample one paraphrase prompt prefix, then append target_new at the end.
        # Example paraphrase prompt: "Pidgeon Island is in" -> "Pidgeon Island is in Asia."
        pps = row.get("paraphrase_prompts", [])
        prefix = ""
        if isinstance(pps, list):
            cand = [str(x).strip() for x in pps if str(x).strip()]
            if cand:
                # Deterministic-ish randomness per row to avoid run-to-run drift.
                seed = int(row.get("case_id", 0)) if str(row.get("case_id", "")).strip().isdigit() else 0
                rng = random.Random(seed)
                prefix = rng.choice(cand)
        if not prefix:
            prefix = base_q
        conflict_ctx = f"{prefix.rstrip()} {new_s}".strip() if new_s else prefix.strip()
        if conflict_ctx and conflict_ctx[-1] not in ".!?。！？":
            conflict_ctx = conflict_ctx + "."
        # Conflict: context then append the question text (no Question:/Answer: wrapper)
        conflict = f"{conflict_ctx}\n\n{base_q}".strip()
        gold = [true_s] if true_s else []
        dstrs = [new_s] if new_s else []
        return clean, conflict, gold, dstrs
    if fmt == "macnoise":
        q = str(row.get("question", "")).strip()
        clean = _nq_swap_question_only_prompt(q)
        # Must choose from positions with pert_lbls_100 == 1.
        use_idx = _macnoise_first_pert100_index(row)
        conflict_ctx = _macnoise_ctx_at_index(row, use_idx, use_pert=True)
        if use_idx < 0 or not conflict_ctx:
            return clean, "", [], []
        conflict = _qa_prompt(q, conflict_ctx) if conflict_ctx else clean
        gold = _gold_list_from_row_field(row.get("answers"))
        dstrs: list[str] = []
        ars = row.get("answer_replace", [])
        if isinstance(ars, list):
            if ars:
                it = ars[0]
                if isinstance(it, dict):
                    t = str(it.get("text", "")).strip()
                else:
                    t = str(it).strip()
                if t:
                    dstrs.append(t)
        return clean, conflict, gold, dstrs
    if fmt == "confiqa":
        q = str(row.get("question", "")).strip()
        cf_ctx = str(row.get("cf_context", "")).strip()
        clean = _nq_swap_question_only_prompt(q)
        conflict = _qa_prompt(q, cf_ctx)
        gold: list[str] = []
        cf_answer = str(row.get("cf_answer", "")).strip()
        if cf_answer:
            gold.append(cf_answer)
        cf_alias = row.get("cf_alias", [])
        if isinstance(cf_alias, list):
            gold.extend(str(x).strip() for x in cf_alias if str(x).strip())
        dstrs: list[str] = []
        orig_answer = str(row.get("orig_answer", "")).strip()
        if orig_answer:
            dstrs.append(orig_answer)
        orig_alias = row.get("orig_alias", [])
        if isinstance(orig_alias, list):
            dstrs.extend(str(x).strip() for x in orig_alias if str(x).strip())
        return clean, conflict, gold, dstrs
    if fmt == "mquake":
        questions = row.get("questions", [])
        q = str(questions[0]).strip() if isinstance(questions, list) and questions else ""
        clean = f"Question: {q}\nAnswer: "
        rw_list = row.get("requested_rewrite", [])
        new_fact = ""
        if isinstance(rw_list, list):
            for r in rw_list:
                if not isinstance(r, dict):
                    continue
                prompt_t = str(r.get("prompt", "")).strip()
                subj = str(r.get("subject", "")).strip()
                t_new = r.get("target_new", {})
                t_new_s = str(t_new.get("str", "")).strip() if isinstance(t_new, dict) else str(t_new).strip()
                if prompt_t:
                    try:
                        fact_head = prompt_t.format(subj)
                    except Exception:
                        fact_head = f"{prompt_t} {subj}".strip()
                    new_fact += f"{fact_head} {t_new_s}. "
        conflict = f"Question: {q}\nEdit Knowledge: {new_fact}\nAnswer: ".strip()
        gold: list[str] = []
        new_answer = str(row.get("new_answer", "")).strip()
        if new_answer:
            gold.append(new_answer)
        na_alias = row.get("new_answer_alias", [])
        if isinstance(na_alias, list):
            gold.extend(str(x).strip() for x in na_alias if str(x).strip())
        dstrs: list[str] = []
        orig_answer = str(row.get("answer", "")).strip()
        if orig_answer:
            dstrs.append(orig_answer)
        oa_alias = row.get("answer_alias", [])
        if isinstance(oa_alias, list):
            dstrs.extend(str(x).strip() for x in oa_alias if str(x).strip())
        return clean, conflict, gold, dstrs
    raise ValueError(f"unknown format {fmt!r}")


def _row_uniform_wiki_contradict(
    row: dict[str, Any], stage1_judged: str
) -> tuple[str, str, list[str], list[str], str]:
    """
    Returns (clean_prompt, conflict_prompt, gold_answers, distractor_strings, stage1_choice).
    stage1_choice is one of: answer1, answer2.
    """
    q = str(row.get("question", "")).strip()
    clean = _nq_swap_question_only_prompt(q)
    ans1 = _answers_from_wikicontradict_field(row.get("answer1"))
    ans2 = _answers_from_wikicontradict_field(row.get("answer2"))
    hit1 = _answer_matches(stage1_judged, ans1)
    hit2 = _answer_matches(stage1_judged, ans2)
    # Keep only unambiguous rows: exactly one side is matched at stage1.
    if hit1 == hit2:
        return clean, "", [], [], ""
    if hit1:
        conflict = _nq_swap_prompt(q, str(row.get("context2", "")))
        return clean, conflict, ans1, ans2, "answer1"
    conflict = _nq_swap_prompt(q, str(row.get("context1", "")))
    return clean, conflict, ans2, ans1, "answer2"


def _counterfact_context_with_target(row: dict[str, Any], target_str: str) -> str:
    rw = row.get("requested_rewrite", {})
    if not isinstance(rw, dict):
        return ""
    subj = str(rw.get("subject", "")).strip()
    ptempl = str(rw.get("prompt", "{} is located in")).strip()
    if "{}" not in ptempl:
        return ""
    base_q = ptempl.format(subj)
    pps = row.get("paraphrase_prompts", [])
    prefix = ""
    if isinstance(pps, list):
        cand = [str(x).strip() for x in pps if str(x).strip()]
        if cand:
            seed = int(row.get("case_id", 0)) if str(row.get("case_id", "")).strip().isdigit() else 0
            rng = random.Random(seed)
            prefix = rng.choice(cand)
    if not prefix:
        prefix = base_q
    ctx = f"{prefix.rstrip()} {str(target_str).strip()}".strip() if str(target_str).strip() else prefix.strip()
    if ctx and ctx[-1] not in ".!?。！？":
        ctx = ctx + "."
    return ctx


def _load_tokenizer(model_path: str):
    """Align with conflict/new/activation_collect: Qwen tokenizer_config bug -> extra_special_tokens={{}}."""
    model_path = str(model_path)
    errors: list[str] = []
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as e:
        errors.append(f"AutoTokenizer (default): {e!r}")
        # Same workaround as activation_collect.get_model_and_tokenizer for Qwen2/Qwen3
        if "object has no attribute 'keys'" in str(e):
            try:
                return AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, extra_special_tokens={}
                )
            except Exception as e2:
                errors.append(f"AutoTokenizer (extra_special_tokens={{}}): {e2!r}")
        else:
            raise
    except Exception as e:
        errors.append(f"AutoTokenizer (default): {e!r}")

    try:
        return PreTrainedTokenizerFast.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        errors.append(f"PreTrainedTokenizerFast.from_pretrained: {e!r}")

    tj = Path(model_path) / "tokenizer.json"
    if tj.is_file():
        try:
            return PreTrainedTokenizerFast(tokenizer_file=str(tj))
        except Exception as e:
            errors.append(f"PreTrainedTokenizerFast(tokenizer_file=...): {e!r}")

    try:
        return AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    except Exception as e:
        errors.append(f"AutoTokenizer (use_fast=False): {e!r}")

    raise RuntimeError(
        f"Could not load tokenizer from {model_path}. Tried AutoTokenizer, "
        f"PreTrainedTokenizerFast (tokenizer.json), then slow AutoTokenizer.\n"
        + "\n".join(errors)
        + "\nTip: pip install -U 'transformers>=4.46' tokenizers"
    )


def _get_model_and_tokenizer(model_key: str):
    model_path = MODEL_PATHS.get(model_key, model_key)
    tokenizer = _load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    tokenizer.padding_side = "left"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device_map = "cuda" if (visible and "," not in visible) else "auto"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "seq_length") and not hasattr(config, "max_length"):
        config.max_length = int(getattr(config, "seq_length"))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    model_key: str = "",
    use_chat_template: bool = True,
    max_new_tokens: int | None = None,
) -> list[str]:
    if not prompts:
        return []
    mnt = int(max_new_tokens) if max_new_tokens is not None else int(MAX_NEW_TOKENS)
    formatted = format_prompts_for_inference(
        tokenizer, prompts, model_key, use_chat_template=use_chat_template
    )
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TOKENIZER_MAX_LEN,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=mnt,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    w = input_ids.shape[1]
    return [tokenizer.decode(out[i, w:], skip_special_tokens=True).strip() for i in range(input_ids.shape[0])]


@torch.inference_mode()
def _collect_resid_last_token_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    model_key: str = "",
    use_chat_template: bool = True,
) -> list[np.ndarray]:
    if not prompts:
        return []
    formatted = format_prompts_for_inference(
        tokenizer, prompts, model_key, use_chat_template=use_chat_template
    )
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TOKENIZER_MAX_LEN,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(model.device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    if hs is None or len(hs) < 2:
        raise RuntimeError("model did not return hidden_states")
    bsz = input_ids.shape[0]
    idxs = []
    for b in range(bsz):
        nz = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        idxs.append(int(nz[-1].item()) if nz.numel() > 0 else 0)
    n_layer = len(hs) - 1
    out_rows: list[np.ndarray] = []
    for b in range(bsz):
        li_last = idxs[b]
        per_layer = [hs[li + 1][b, li_last].float().cpu().numpy().astype(np.float16) for li in range(n_layer)]
        out_rows.append(np.stack(per_layer, axis=0))
    return out_rows


def _abspath(p: Path) -> Path:
    p = Path(p)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _parse_bool(s: str) -> bool:
    sl = str(s).strip().lower()
    if sl in ("1", "true", "t", "yes", "y"):
        return True
    if sl in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump two files only: resid_hidden_state.npz and internal_aware_data.json"
    )
    ap.add_argument("--model", default="llama3-8b-it")
    ap.add_argument(
        "--input-jsonl",
        type=str,
        default="",
        help="Optional: directly specify an input .jsonl/.csv file path. When set, overrides --dataset file resolution.",
    )
    ap.add_argument(
        "--dataset",
        default="counterfact",
        # When --input-jsonl is set, any name is allowed (taxonomy adapters → NQ-Swap schema).
        # Otherwise prefer the classic conflict_steer names below.
        help=(
            "Output path tag under <out-root>/<model>/<dataset>/<category>/. "
            "Classic names: ParaConfilct, NQ-Swap, WikiContradict, KRE, dynamicqa, "
            "memotrap, counterfact, macnoise, nq, ConFiQA, MQuAKE. "
            "Taxonomy adapters (RGB/CRAG/GuideBench/ContractNLI) require --input-jsonl."
        ),
    )
    ap.add_argument(
        "--category",
        default="counterfact_relation_018_p364_8fddb749",
        help="ParaConflict: row Category filter. NQ-Swap/WikiContradict: output path tag. KRE: json name (cose/ecare/musique/squad). dynamicqa: csv name (static/disputable). memotrap: csv name (e.g. 1-proverb-ending). counterfact: json filename under data/counterfact. macnoise: json filename or perturbation key (e.g. pert_lbls, pert_lbls_25).",
    )
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--gpu", type=str, default="7")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=f"Inference batch size for stage1/stage2/resid (default: {DEFAULT_BATCH_SIZE}).",
    )
    ap.add_argument(
        "--save-resid",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="Write resid_hidden_state.npz: true or false (default: true).",
    )
    ap.add_argument(
        "--paraconflict-conflict-style",
        default="auto",
        choices=("auto", "coherent", "substitution"),
        help=(
            "ParaConflict wrong-context field: auto=Coherent then Substitution fallback; "
            "coherent=Coherent Conflict only; substitution=Substitution Conflict only."
        ),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root; writes to <out-root>/<model>/<dataset>/<category>/ (default: <DATA_DIR>/final).",
    )
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template wrapping (raw Question:/Answer: prompts).",
    )
    ap.add_argument(
        "--save-unfiltered-pairs",
        type=_parse_bool,
        default=False,
        metavar="BOOL",
        help=(
            "Also write unfiltered_all_pairs: all clean/conflict (wrong-context) pairs "
            "without stage1 correctness filtering (default: false)."
        ),
    )
    ap.add_argument(
        "--judge",
        type=str,
        default="legacy",
        choices=["legacy", "official"],
        help="Legacy dump-only labeling: substring match or the single-target official judge.",
    )
    ap.add_argument(
        "--internal-from-stage1",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help=(
            "If true (default), set internal/gold_answers to the model's Stage1 "
            "spontaneous answer (no context), not dataset org_answer proxies. "
            "External/distractors stay dataset sub_answer. "
            "Primary group = Stage1≠external, Stage2 uses conflict/external context."
        ),
    )
    args = ap.parse_args()
    use_chat_template = not bool(args.no_chat_template)
    save_resid = bool(args.save_resid)
    save_unfiltered_pairs = bool(args.save_unfiltered_pairs)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise SystemExit(f"--batch-size must be > 0, got {batch_size}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    macnoise_pert_key = "pert_lbls"
    try:
        input_path = str(args.input_jsonl).strip()
        if input_path:
            dataset_file = Path(input_path).expanduser()
            if not dataset_file.is_file():
                raise SystemExit(f"--input-jsonl not found: {dataset_file}")
            if dataset_file.suffix.lower() == ".csv":
                with dataset_file.open("r", encoding="utf-8", newline="") as _peek:
                    _reader = csv.DictReader(_peek)
                    _first_row = next(iter(_reader), None)
                    if _first_row is None:
                        raise SystemExit(f"empty csv: {dataset_file}")
                    row_fmt = _infer_row_format(dict(_first_row))
            else:
                with dataset_file.open("r", encoding="utf-8") as _peek:
                    _first = ""
                    for _first in _peek:
                        if _first.strip():
                            break
                    if not _first.strip():
                        raise SystemExit(f"empty jsonl: {dataset_file}")
                row_fmt = _infer_row_format(json.loads(_first.strip()))
        else:
            dataset_dir = DATA_ROOT / str(args.dataset)
            if not dataset_dir.is_dir():
                raise SystemExit(f"dataset directory not found: {dataset_dir}")
            if str(args.dataset) == "KRE":
                dataset_file = _resolve_kre_json(dataset_dir, str(args.category))
                row_fmt = "kre"
            elif str(args.dataset) == "dynamicqa":
                dataset_file = _resolve_dynamicqa_csv(dataset_dir, str(args.category))
                row_fmt = "dynamicqa"
            elif str(args.dataset) == "memotrap":
                dataset_file = _resolve_memotrap_csv(dataset_dir, str(args.category))
                row_fmt = "memotrap"
            elif str(args.dataset) == "counterfact":
                dataset_file = _resolve_counterfact_json(dataset_dir, str(args.category))
                row_fmt = "counterfact"
            elif str(args.dataset) == "macnoise":
                cat = str(args.category).strip()
                if cat.startswith("pert_lbls"):
                    macnoise_pert_key = cat
                dataset_file = _resolve_macnoise_json(dataset_dir, cat)
                row_fmt = "macnoise"
            elif str(args.dataset) == "ConFiQA":
                dataset_file = _resolve_confiqa_json(dataset_dir, str(args.category))
                row_fmt = "confiqa"
            elif str(args.dataset) == "MQuAKE":
                dataset_file = _resolve_mquake_json(dataset_dir, str(args.category))
                row_fmt = "mquake"
            elif str(args.dataset) == "nq":
                dataset_file = dataset_dir
                row_fmt = "nq_swap"
            else:
                # NQ-Swap special: allow --category to point to a split jsonl under data/NQ-Swap/by_wikidata_property/
                cat = str(args.category).strip()
                if str(args.dataset) == "NQ-Swap" and cat:
                    cat_path = Path(cat).expanduser()
                    if cat_path.is_absolute() and cat_path.is_file():
                        dataset_file = cat_path
                    else:
                        by_prop_dir = dataset_dir / "by_wikidata_property"
                        # allow passing either "c1__foo.jsonl" or "c1__foo"
                        cand1 = by_prop_dir / cat
                        cand2 = by_prop_dir / f"{cat}.jsonl"
                        if cand1.is_file():
                            dataset_file = cand1
                        elif cand2.is_file():
                            dataset_file = cand2
                        else:
                            dataset_file = _resolve_dataset_file(dataset_dir)
                else:
                    dataset_file = _resolve_dataset_file(dataset_dir)
                if dataset_file.suffix.lower() == ".csv":
                    with dataset_file.open("r", encoding="utf-8", newline="") as _peek:
                        _reader = csv.DictReader(_peek)
                        _first_row = next(iter(_reader), None)
                        if _first_row is None:
                            raise SystemExit(f"empty csv: {dataset_file}")
                        row_fmt = _infer_row_format(dict(_first_row))
                else:
                    with dataset_file.open("r", encoding="utf-8") as _peek:
                        _first = ""
                        for _first in _peek:
                            if _first.strip():
                                break
                        if not _first.strip():
                            raise SystemExit(f"empty jsonl: {dataset_file}")
                    row_fmt = _infer_row_format(json.loads(_first.strip()))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    # Output tag: by default use --category; if empty and --input-jsonl is set, use filename stem.
    out_tag = str(args.category).strip()
    if not out_tag and str(args.input_jsonl).strip():
        out_tag = Path(str(args.input_jsonl)).stem
    # If category is used to point to a jsonl path, tag output by the file stem.
    if out_tag and ("/" in out_tag or out_tag.endswith(".jsonl") or out_tag.endswith(".csv")):
        try:
            p = Path(out_tag).expanduser()
            if p.is_file():
                out_tag = p.stem
        except Exception:
            pass

    model, tokenizer = _get_model_and_tokenizer(str(args.model))
    if row_fmt == "paraconflict":
        rows = _load_rows_paraconflict(dataset_file, str(args.category))
    elif row_fmt == "nq_swap":
        rows = _load_rows_nq_swap(dataset_file)
    elif row_fmt == "wiki_contradict":
        rows = _load_rows_wiki_contradict(dataset_file)
    elif row_fmt == "kre":
        rows = _load_rows_kre(dataset_file)
    elif row_fmt == "dynamicqa":
        rows = _load_rows_csv_dict(dataset_file)
    elif row_fmt == "memotrap":
        rows = _load_rows_csv_dict(dataset_file)
    elif row_fmt == "counterfact":
        rows = _load_rows_counterfact(dataset_file)
    elif row_fmt == "macnoise":
        rows = _load_rows_macnoise(dataset_file)
    elif row_fmt == "confiqa":
        rows = _load_rows_json_list(dataset_file)
    elif row_fmt == "mquake":
        rows = _load_rows_json_list(dataset_file)
    else:
        if str(args.dataset) == "nq":
            orig_path = dataset_file / "orig_dev_filtered.json"
            conflict_path = dataset_file / "conflict_dev_filtered.json"
            if not orig_path.is_file() or not conflict_path.is_file():
                raise SystemExit(
                    f"nq requires files under {dataset_file}: orig_dev_filtered.json and conflict_dev_filtered.json"
                )
            rows = _load_rows_nq_pair(orig_path, conflict_path)
        else:
            raise SystemExit(f"unsupported row format: {row_fmt}")

    out_root = _abspath(args.out_root) / str(args.model) / str(args.dataset) / out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _get_model_and_tokenizer(str(args.model))
    stage1_correct_meta: list[dict[str, Any]] = []
    stage1_wrong_meta: list[dict[str, Any]] = []
    stage1_correct_clean_prompts: list[str] = []
    stage1_wrong_clean_prompts: list[str] = []
    stage1_correct_stage2_prompts: list[str] = []
    stage1_wrong_stage2_prompts: list[str] = []
    unfiltered_meta: list[dict[str, Any]] = []
    unfiltered_clean_prompts: list[str] = []
    unfiltered_conflict_prompts: list[str] = []
    total_kept = 0
    for st in range(0, len(rows), batch_size):
        if int(args.max_samples) > 0 and total_kept >= int(args.max_samples):
            break
        chunk = rows[st : st + batch_size]
        clean_prompts_chunk = []
        for _, r in chunk:
            if row_fmt == "wiki_contradict":
                c_p = _nq_swap_question_only_prompt(str(r.get("question", "")).strip())
            else:
                c_p, _, _, _ = _row_uniform(
                    r,
                    row_fmt,
                    macnoise_pert_key,
                    paraconflict_conflict_style=str(args.paraconflict_conflict_style),
                )
            clean_prompts_chunk.append(c_p)
        stage1_raw = _generate_batch(
            model,
            tokenizer,
            clean_prompts_chunk,
            model_key=str(args.model),
            use_chat_template=use_chat_template,
        )
        for j, (line_idx, row) in enumerate(chunk):
            if int(args.max_samples) > 0 and total_kept >= int(args.max_samples):
                break
            s1_full = str(stage1_raw[j] or "").strip()
            s1 = _truncate_for_judge(s1_full)
            if row_fmt == "wiki_contradict":
                clean_p, conflict_p, gold, dstrs, s1_choice = _row_uniform_wiki_contradict(row, s1)
            else:
                clean_p, conflict_p, gold, dstrs = _row_uniform(
                    row,
                    row_fmt,
                    macnoise_pert_key,
                    paraconflict_conflict_style=str(args.paraconflict_conflict_style),
                )
                s1_choice = ""

            # Stage1: spontaneous answer on clean prompt.
            # Default: internal := Stage1 output; external := dataset sub_answer/distractors.
            if not conflict_p:
                continue

            dataset_gold = [str(x) for x in gold]
            external_answers = [str(x) for x in dstrs]
            internal_from_s1 = bool(args.internal_from_stage1)

            def _matches_external(text: str) -> bool:
                if not external_answers:
                    return False
                if str(args.judge).strip().lower() == "official":
                    from official_judges import match_gold

                    return match_gold(
                        str(args.dataset),
                        text,
                        external_answers,
                        meta={
                            "category": str(args.category),
                            "query": (row.get("question") if isinstance(row, dict) else "") or "",
                            "question": (row.get("question") if isinstance(row, dict) else "") or "",
                            "question_type": (row.get("question_type") if isinstance(row, dict) else "")
                            or "multi-session",
                            "eval_function": (row.get("eval_function") if isinstance(row, dict) else "")
                            or "",
                            "tool_call": row.get("tool_call") if isinstance(row, dict) else None,
                        },
                    )
                if row_fmt == "kre":
                    return _answer_matches_letters(text, external_answers)
                return _answer_matches(text, external_answers)

            if internal_from_s1:
                # Group by whether spontaneous already equals external gold.
                s1_matches_ext = _matches_external(s1_full) if external_answers else False
                # Primary conflict set: spontaneous ≠ external → Stage2 gets external/conflict ctx.
                # Map to stage1_correct_use_wrong_context (default GROUP in cross-steer scripts).
                s1_correct = not s1_matches_ext
                # Prefer compact label if GuideBench-style letter/number can be extracted
                from official_judges.native import guidebench_extract

                if str(args.dataset).strip() in {"GuideBench", "guidebench"}:
                    extracted = guidebench_extract(
                        s1_full, category=str(args.category), gold=external_answers[0] if external_answers else None
                    )
                    gold = [extracted] if extracted else [s1_full[:500]]
                else:
                    gold = [s1_full[:500] if s1_full else s1]  # internal = spontaneous (untruncated)
            else:
                if str(args.judge).strip().lower() == "official":
                    s1_correct, _ = _official_judge_sample(
                        dataset=str(args.dataset),
                        category=str(args.category),
                        text=s1,
                        gold_answers=dataset_gold,
                        distracted_tokens=external_answers,
                        sample=row if isinstance(row, dict) else {},
                    )
                elif row_fmt == "kre":
                    s1_correct = _answer_matches_letters(s1, dataset_gold)
                else:
                    s1_correct = _answer_matches(s1, dataset_gold)
                gold = dataset_gold

            # Build explicit correct/wrong contexts for stage2 routing.
            correct_p = clean_p
            wrong_p = conflict_p
            if row_fmt == "nq_swap":
                q = str(row.get("question", "")).strip()
                correct_p = _nq_swap_prompt(q, str(row.get("org_context", "")))
                wrong_p = _nq_swap_prompt(q, str(row.get("sub_context", "")))
            elif row_fmt == "dynamicqa":
                q = str(row.get("question", "")).strip()
                ctx = str(row.get("context", ""))
                obj = str(row.get("obj", "")).strip()
                rep = str(row.get("replace_name", "")).strip()
                correct_p = _qa_prompt(q, ctx.replace("[ENTITY]", obj))
                wrong_p = _qa_prompt(q, ctx.replace("[ENTITY]", rep))
            elif row_fmt == "counterfact":
                rw = row.get("requested_rewrite", {})
                if isinstance(rw, dict):
                    subj = str(rw.get("subject", "")).strip()
                    ptempl = str(rw.get("prompt", "{} is located in")).strip()
                    if "{}" in ptempl:
                        base_q = ptempl.format(subj)
                        t_new = rw.get("target_new", {})
                        t_true = rw.get("target_true", {})
                        new_s = str(t_new.get("str", "")).strip() if isinstance(t_new, dict) else str(t_new).strip()
                        true_s = str(t_true.get("str", "")).strip() if isinstance(t_true, dict) else str(t_true).strip()
                        correct_ctx = _counterfact_context_with_target(row, true_s)
                        wrong_ctx = _counterfact_context_with_target(row, new_s)
                        correct_p = f"{correct_ctx}\n\n{base_q}".strip() if correct_ctx else base_q
                        wrong_p = f"{wrong_ctx}\n\n{base_q}".strip() if wrong_ctx else base_q
            elif row_fmt == "macnoise":
                q = str(row.get("question", "")).strip()
                # Must use a ctx index where pert_lbls_100 == 1.
                use_idx = _macnoise_first_pert100_index(row)
                correct_ctx = _macnoise_ctx_at_index(row, use_idx, use_pert=False)
                wrong_ctx = _macnoise_ctx_at_index(row, use_idx, use_pert=True)
                correct_p = _qa_prompt(q, correct_ctx) if correct_ctx else clean_p
                wrong_p = _qa_prompt(q, wrong_ctx) if wrong_ctx else clean_p
            elif row_fmt == "paraconflict":
                correct_p = _paraconflict_correct_context_prompt(
                    conflict_p, dataset_gold, dstrs, clean_fallback=clean_p
                )

            d0 = external_answers[0] if external_answers else ""
            m = {
                "line_idx": int(line_idx),
                "model": str(args.model),
                "dataset": str(args.dataset),
                "category": str(args.category),
                "clean_prompt_input": clean_p,
                "stage2_prompt_input": wrong_p if s1_correct else correct_p,
                "stage2_context_type": "wrong_context" if s1_correct else "correct_context",
                "gold_answers": gold,
                "distracted_token": d0,
                "distracted_tokens": list(external_answers),
                "stage1_output_judged": s1,
                "stage1_choice": s1_choice,
                "stage1_is_correct": bool(s1_correct),
                "internal_source": "stage1_spontaneous" if internal_from_s1 else "dataset_org_answer",
                "dataset_org_answer": dataset_gold,
                "question": row.get("question") if isinstance(row, dict) else None,
                "org_answer": gold if internal_from_s1 else (row.get("org_answer") if isinstance(row, dict) else None),
                "sub_answer": row.get("sub_answer") if isinstance(row, dict) else external_answers,
                "tool_call": row.get("tool_call") if isinstance(row, dict) else None,
                "eval_function": row.get("eval_function") if isinstance(row, dict) else None,
                "question_type": row.get("question_type") if isinstance(row, dict) else None,
            }
            if s1_correct:
                stage1_correct_meta.append(m)
                stage1_correct_clean_prompts.append(clean_p)
                stage1_correct_stage2_prompts.append(wrong_p)
            else:
                stage1_wrong_meta.append(m)
                stage1_wrong_clean_prompts.append(clean_p)
                stage1_wrong_stage2_prompts.append(correct_p)
            if save_unfiltered_pairs:
                unfiltered_meta.append(
                    {
                        **m,
                        "stage2_prompt_input": wrong_p,
                        "stage2_context_type": "wrong_context",
                        "unfiltered_conflict_prompt_input": wrong_p,
                    }
                )
                unfiltered_clean_prompts.append(clean_p)
                unfiltered_conflict_prompts.append(wrong_p)
            total_kept += 1

        print(
            f"[stage1] batch rows [{st}, {st + len(chunk)}), kept={total_kept}/"
            f"{'all' if int(args.max_samples) <= 0 else args.max_samples}",
            flush=True,
        )

    def _run_stage2_for_group(meta_list: list[dict[str, Any]], prompts: list[str]) -> None:
        stage2_raw: list[str] = []
        for st2 in range(0, len(prompts), batch_size):
            stage2_raw.extend(
                _generate_batch(
                    model,
                    tokenizer,
                    prompts[st2 : st2 + batch_size],
                    model_key=str(args.model),
                    use_chat_template=use_chat_template,
                )
            )
        for i2, m2 in enumerate(meta_list):
            raw2 = stage2_raw[i2] if i2 < len(stage2_raw) else ""
            m2["stage2_output_raw"] = raw2
            judged2 = _truncate_for_judge(raw2)
            m2["stage2_output_judged"] = judged2
            dlist = m2.get("distracted_tokens")
            if not isinstance(dlist, list):
                dlist = [str(m2.get("distracted_token", ""))]
            dlist = [str(x) for x in dlist]
            if str(args.judge).strip().lower() == "official":
                _hit, lab = _official_judge_sample(
                    dataset=str(args.dataset),
                    category=str(args.category),
                    text=judged2,
                    gold_answers=[str(x) for x in m2.get("gold_answers", [])],
                    distracted_tokens=dlist,
                    sample=m2,
                )
                m2["stage2_label"] = lab
            elif row_fmt == "kre":
                m2["stage2_label"] = _stage2_label_from_letters(
                    judged2,
                    [str(x) for x in m2.get("gold_answers", [])],
                    dlist,
                )
            elif row_fmt == "memotrap":
                m2["stage2_label"] = _stage2_label_memotrap(judged2, dlist)
            elif row_fmt == "macnoise" and len(dlist) == 0:
                # For macnoise files without answer_replace: binary rule.
                # If output is not correct answer, count it as wrong (no skipped).
                m2["stage2_label"] = (
                    "correct"
                    if _answer_matches(judged2, [str(x) for x in m2.get("gold_answers", [])])
                    else "wrong"
                )
            else:
                m2["stage2_label"] = _stage2_label_from_strings(
                    judged2,
                    [str(x) for x in m2.get("gold_answers", [])],
                    dlist,
                )
            m2["judge_mode"] = str(args.judge)

    _run_stage2_for_group(stage1_correct_meta, stage1_correct_stage2_prompts)
    _run_stage2_for_group(stage1_wrong_meta, stage1_wrong_stage2_prompts)
    print(
        f"[stage2] labeled stage1_correct={len(stage1_correct_meta)}, stage1_wrong={len(stage1_wrong_meta)}",
        flush=True,
    )

    def _save_group(
        name: str,
        meta_list: list[dict[str, Any]],
        clean_prompts: list[str],
        stage2_prompts: list[str],
        *,
        conflict_key: str = "resid_act_stage2",
    ) -> tuple[Path, Path | None]:
        aware_json = out_root / f"internal_aware_data__{name}.json"
        aware_obj = {
            "model": str(args.model),
            "dataset": str(args.dataset),
            "category": str(args.category),
            "group": name,
            "max_samples": int(args.max_samples),
            "num_kept": len(meta_list),
            "samples": meta_list,
        }
        aware_json.write_text(json.dumps(aware_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        resid_npz: Path | None = None
        if save_resid:
            resid_npz = out_root / f"resid_hidden_state__{name}.npz"
            clean_resid: list[np.ndarray] = []
            stage2_resid: list[np.ndarray] = []
            for st3 in range(0, len(clean_prompts), batch_size):
                clean_resid.extend(
                    _collect_resid_last_token_batch(
                        model,
                        tokenizer,
                        clean_prompts[st3 : st3 + batch_size],
                        model_key=str(args.model),
                        use_chat_template=use_chat_template,
                    )
                )
            for st3 in range(0, len(stage2_prompts), batch_size):
                stage2_resid.extend(
                    _collect_resid_last_token_batch(
                        model,
                        tokenizer,
                        stage2_prompts[st3 : st3 + batch_size],
                        model_key=str(args.model),
                        use_chat_template=use_chat_template,
                    )
                )
            arr_clean = np.empty(len(clean_resid), dtype=object)
            arr_stage2 = np.empty(len(stage2_resid), dtype=object)
            for i3, v3 in enumerate(clean_resid):
                arr_clean[i3] = v3
            for i3, v3 in enumerate(stage2_resid):
                arr_stage2[i3] = v3
            np.savez_compressed(
                resid_npz,
                resid_act_clean=arr_clean,
                **{conflict_key: arr_stage2},
                line_idx=np.array([m["line_idx"] for m in meta_list], dtype=np.int64),
                category=np.array([m["category"] for m in meta_list], dtype=object),
                stage2_context_type=np.array([m.get("stage2_context_type", "") for m in meta_list], dtype=object),
            )
        return aware_json, resid_npz

    aware_correct, resid_correct = _save_group(
        "stage1_correct_use_wrong_context",
        stage1_correct_meta,
        stage1_correct_clean_prompts,
        stage1_correct_stage2_prompts,
    )
    aware_wrong, resid_wrong = _save_group(
        "stage1_wrong_use_correct_context",
        stage1_wrong_meta,
        stage1_wrong_clean_prompts,
        stage1_wrong_stage2_prompts,
    )
    aware_unfiltered: Path | None = None
    resid_unfiltered: Path | None = None
    if save_unfiltered_pairs:
        aware_unfiltered, resid_unfiltered = _save_group(
            "unfiltered_all_pairs",
            unfiltered_meta,
            unfiltered_clean_prompts,
            unfiltered_conflict_prompts,
            conflict_key="resid_act_conflict",
        )
        print(
            f"[unfiltered] kept {len(unfiltered_meta)} clean/conflict pairs (no stage1 filter)",
            flush=True,
        )

    if resid_correct is not None:
        print(str(resid_correct))
    if resid_wrong is not None:
        print(str(resid_wrong))
    print(str(aware_correct))
    print(str(aware_wrong))
    if aware_unfiltered is not None:
        print(str(aware_unfiltered))
    if resid_unfiltered is not None:
        print(str(resid_unfiltered))


if __name__ == "__main__":
    main()
