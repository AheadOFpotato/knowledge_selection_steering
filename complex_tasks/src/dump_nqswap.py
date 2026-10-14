#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
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

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
NQ_SPLIT_ROOT = ROOT / "data" / "NQ-Swap" / "by_question_first_word"
NQ_OUT_ROOT = ROOT / "output" / "nq_swap"

MODEL_PATHS: dict[str, str] = {
    "llama3-8b-it": "<DATA_DIR>/model/Llama3-8b-it",
    "gemma2-9b-it": "<DATA_DIR>/model/Gemma2-9b-it",
    "qwen3-8b": "<DATA_DIR>/model/Qwen3-8B",
    "mistral-7b-v0.1": "<DATA_DIR>/model/Mistral-7B-v0.1",
}

MAX_NEW_TOKENS = 200
TOKENIZER_MAX_LEN = 8192
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
    raise ValueError(
        "unrecognized jsonl row schema: expected ParaConflict fields (Clean Prompt) "
        "or NQ-Swap fields (org_context, sub_context) or WikiContradict fields "
        "(question, context1, context2, answer1, answer2) or KRE fields "
        "(question, choices, answer, negative_context) or dynamicqa fields "
        "(question, context, obj, replace_name) or memotrap fields "
        "(prompt, classes, answer_index)"
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


def _load_rows_kre(json_path: Path) -> list[tuple[int, dict[str, Any]]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"KRE file is not a JSON list: {json_path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    for i, row in enumerate(obj):
        if isinstance(row, dict):
            rows.append((i, row))
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


def _distractor_strings_nq_swap(row: dict[str, Any]) -> list[str]:
    return _gold_list_from_row_field(row.get("sub_answer"))


def _row_uniform(
    row: dict[str, Any], fmt: str
) -> tuple[str, str, list[str], list[str]]:
    """Returns (clean_prompt, conflict_prompt, gold_answers, distractor_strings)."""
    if fmt == "paraconflict":
        gold = _gold_list_from_row_field(row.get("Answer"))
        conflict = str(row.get("Coherent Conflict", "")).strip() or str(
            row.get("Substitution Conflict", "")
        ).strip()
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
    max_new_tokens: int | None = None,
) -> list[str]:
    if not prompts:
        return []
    mnt = int(max_new_tokens) if max_new_tokens is not None else int(MAX_NEW_TOKENS)
    enc = tokenizer(
        prompts,
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
def _collect_resid_last_token_batch(model, tokenizer, prompts: list[str]) -> list[np.ndarray]:
    if not prompts:
        return []
    enc = tokenizer(
        prompts,
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
        "--split-file",
        default="who.jsonl",
        help="Filename under data/NQ-Swap/by_question_first_word (e.g. who.jsonl, how.jsonl).",
    )
    ap.add_argument(
        "--category",
        default="",
        help="Output path tag. Empty means use split-file stem.",
    )
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--gpu", type=str, default="1")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=f"Inference batch size for stage1/stage2/resid (default: {DEFAULT_BATCH_SIZE}).",
    )
    ap.add_argument(
        "--save-resid",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="Write resid_hidden_state.npz: true or false (default: true).",
    )
    args = ap.parse_args()
    save_resid = bool(args.save_resid)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise SystemExit(f"--batch-size must be > 0, got {batch_size}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    dataset_file = NQ_SPLIT_ROOT / str(args.split_file)
    if dataset_file.suffix.lower() != ".jsonl":
        dataset_file = dataset_file.with_suffix(".jsonl")
    if not dataset_file.is_file():
        raise SystemExit(f"split file not found: {dataset_file}")
    row_fmt = "nq_swap"
    if not str(args.category).strip():
        args.category = dataset_file.stem

    aware_dir = NQ_OUT_ROOT / "internal_aware_data" / str(args.model) / str(args.category)
    aware_dir.mkdir(parents=True, exist_ok=True)
    aware_json = aware_dir / "internal_aware_data.json"
    resid_npz: Path | None = None
    if save_resid:
        resid_dir = NQ_OUT_ROOT / "resid_hidden_state" / str(args.model) / str(args.category)
        resid_dir.mkdir(parents=True, exist_ok=True)
        resid_npz = resid_dir / "resid_hidden_state.npz"

    model, tokenizer = _get_model_and_tokenizer(str(args.model))
    if row_fmt == "nq_swap":
        rows = _load_rows_nq_swap(dataset_file)
    else:
        raise SystemExit(f"unsupported row format: {row_fmt}")

    kept_meta: list[dict[str, Any]] = []
    kept_clean_prompts: list[str] = []
    kept_conflict_prompts: list[str] = []
    for st in range(0, len(rows), batch_size):
        if len(kept_meta) >= int(args.max_samples):
            break
        chunk = rows[st : st + batch_size]
        clean_prompts_chunk = []
        for _, r in chunk:
            if row_fmt == "wiki_contradict":
                c_p = _nq_swap_question_only_prompt(str(r.get("question", "")).strip())
            else:
                c_p, _, _, _ = _row_uniform(r, row_fmt)
            clean_prompts_chunk.append(c_p)
        stage1_raw = _generate_batch(model, tokenizer, clean_prompts_chunk)
        for j, (line_idx, row) in enumerate(chunk):
            if len(kept_meta) >= int(args.max_samples):
                break
            s1 = _truncate_for_judge(stage1_raw[j])
            if row_fmt == "wiki_contradict":
                clean_p, conflict_p, gold, dstrs, s1_choice = _row_uniform_wiki_contradict(row, s1)
            else:
                clean_p, conflict_p, gold, dstrs = _row_uniform(row, row_fmt)
                s1_choice = ""

            # Stage1 filtering
            if row_fmt == "kre":
                if not _answer_matches_letters(s1, [str(x) for x in gold]):
                    continue
            elif row_fmt != "wiki_contradict" and not _answer_matches(s1, gold):
                continue
            if not conflict_p:
                continue
            d0 = dstrs[0] if dstrs else ""
            kept_meta.append(
                {
                    "line_idx": int(line_idx),
                    "model": str(args.model),
                    "dataset": "NQ-Swap",
                    "split_file": str(args.split_file),
                    "category": str(args.category),
                    "clean_prompt_input": clean_p,
                    "conflict_prompt_input": conflict_p,
                    "gold_answers": gold,
                    "distracted_token": d0,
                    "distracted_tokens": list(dstrs),
                    "stage1_output_judged": s1,
                    "stage1_choice": s1_choice,
                }
            )
            kept_clean_prompts.append(clean_p)
            kept_conflict_prompts.append(conflict_p)

        print(
            f"[stage1] batch rows [{st}, {st + len(chunk)}), kept={len(kept_meta)}/{args.max_samples}",
            flush=True,
        )

    # Stage2 generation and string-match labeling (full raw output).
    stage2_raw: list[str] = []
    for st in range(0, len(kept_conflict_prompts), batch_size):
        chunk_prompts = kept_conflict_prompts[st : st + batch_size]
        stage2_raw.extend(_generate_batch(model, tokenizer, chunk_prompts))
    for i, m in enumerate(kept_meta):
        raw2 = stage2_raw[i] if i < len(stage2_raw) else ""
        m["stage2_output_raw"] = raw2
        m["stage2_output_judged"] = _truncate_for_judge(raw2)
        dlist = m.get("distracted_tokens")
        if not isinstance(dlist, list):
            dlist = [str(m.get("distracted_token", ""))]
        dlist = [str(x) for x in dlist]
        if row_fmt == "kre":
            m["stage2_label"] = _stage2_label_from_letters(
                raw2,
                [str(x) for x in m.get("gold_answers", [])],
                dlist,
            )
        elif row_fmt == "memotrap":
            m["stage2_label"] = _stage2_label_memotrap(raw2, dlist)
        else:
            m["stage2_label"] = _stage2_label_from_strings(
                raw2,
                [str(x) for x in m.get("gold_answers", [])],
                dlist,
            )
    print(f"[stage2] labeled {len(kept_meta)} samples (substring rules)", flush=True)

    if save_resid and resid_npz is not None:
        clean_resid: list[np.ndarray] = []
        conflict_resid: list[np.ndarray] = []
        for st in range(0, len(kept_clean_prompts), batch_size):
            clean_resid.extend(
                _collect_resid_last_token_batch(
                    model, tokenizer, kept_clean_prompts[st : st + batch_size]
                )
            )
        for st in range(0, len(kept_conflict_prompts), batch_size):
            conflict_resid.extend(
                _collect_resid_last_token_batch(
                    model, tokenizer, kept_conflict_prompts[st : st + batch_size]
                )
            )
        arr_clean = np.empty(len(clean_resid), dtype=object)
        arr_conflict = np.empty(len(conflict_resid), dtype=object)
        for i, v in enumerate(clean_resid):
            arr_clean[i] = v
        for i, v in enumerate(conflict_resid):
            arr_conflict[i] = v
        np.savez_compressed(
            resid_npz,
            resid_act_clean=arr_clean,
            resid_act_conflict=arr_conflict,
            line_idx=np.array([m["line_idx"] for m in kept_meta], dtype=np.int64),
            category=np.array([m["category"] for m in kept_meta], dtype=object),
        )

    aware_obj = {
        "model": str(args.model),
        "dataset": "NQ-Swap",
        "split_file": str(args.split_file),
        "category": str(args.category),
        "max_samples": int(args.max_samples),
        "num_kept": len(kept_meta),
        "samples": kept_meta,
    }
    aware_json.write_text(json.dumps(aware_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if save_resid and resid_npz is not None:
        print(str(resid_npz.relative_to(ROOT) if str(resid_npz).startswith(str(ROOT)) else resid_npz))
    print(str(aware_json.relative_to(ROOT) if str(aware_json).startswith(str(ROOT)) else aware_json))


if __name__ == "__main__":
    main()

