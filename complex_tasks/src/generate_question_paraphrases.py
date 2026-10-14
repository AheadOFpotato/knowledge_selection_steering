#!/usr/bin/env python3
"""Generate semantic paraphrases of NQ-Swap questions for parametric-knowledge verification."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chat_prompt import format_prompt_for_inference  # noqa: E402
from stress_test_common import read_jsonl, write_jsonl  # noqa: E402
from dump_activations_from_testjsonl import _get_model_and_tokenizer  # noqa: E402

_PARAPHRASE_SYSTEM = (
    "You rewrite questions without changing their meaning or answer. "
    "Keep the same entity names, numbers, and temporal references. "
    "Output exactly one paraphrase per line. No numbering, no bullets, no explanation."
)


def _paraphrase_user_prompt(question: str, n: int) -> str:
    return (
        f"Rewrite the question below into {n} different paraphrases.\n"
        f"Each line must be a standalone question with identical meaning.\n\n"
        f"Original: {question.strip()}\n"
    )


def _parse_paraphrase_lines(text: str, *, original: str) -> list[str]:
    orig_norm = re.sub(r"\s+", " ", original.lower().strip())
    out: list[str] = []
    seen: set[str] = {orig_norm}
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*[\d\-*.)]+\s*", "", line.strip())
        line = line.strip(" \"'")
        if not line or not line.endswith("?"):
            if line and "?" in line:
                line = line[: line.rfind("?") + 1]
            else:
                continue
        norm = re.sub(r"\s+", " ", line.lower().strip())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(line)
    return out


@torch.inference_mode()
def generate_paraphrases_for_question(
    model,
    tokenizer,
    question: str,
    *,
    n: int = 5,
    model_key: str = "",
    use_chat_template: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> list[str]:
    user = _paraphrase_user_prompt(question, n)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": _PARAPHRASE_SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{_PARAPHRASE_SYSTEM}\n\n{user}\nParaphrases:\n"

    formatted = format_prompt_for_inference(
        tokenizer, prompt, model_key, use_chat_template=False
    )
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
    parsed = _parse_paraphrase_lines(gen, original=question)
    return parsed[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate question paraphrases for NQ-Swap rows.")
    ap.add_argument("--input", type=Path, required=True, help="NQ-Swap jsonl")
    ap.add_argument("--output", type=Path, required=True, help="jsonl with paraphrases field")
    ap.add_argument("--model", default="llama3.1-8b-it")
    ap.add_argument("--num-paraphrases", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip rows already in output")
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template for downstream eval (paraphrase gen uses its own template).",
    )
    args = ap.parse_args()

    rows_in = read_jsonl(args.input.resolve())
    if args.limit > 0:
        rows_in = rows_in[: args.limit]

    existing: dict[int, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        for r in read_jsonl(args.output):
            if "line_idx" in r:
                existing[int(r["line_idx"])] = r

    todo = [(i, r) for i, r in enumerate(rows_in) if i not in existing]
    model, tokenizer = _get_model_and_tokenizer(str(args.model))

    out_rows: list[dict[str, Any]] = list(existing.values())
    for i, row in todo:
        q = str(row.get("question", "")).strip()
        paras = generate_paraphrases_for_question(
            model,
            tokenizer,
            q,
            n=args.num_paraphrases,
            model_key=str(args.model),
            use_chat_template=not args.no_chat_template,
            temperature=args.temperature,
        )
        rec = dict(row)
        rec["line_idx"] = i
        rec["paraphrases"] = paras
        rec["paraphrase_meta"] = {
            "model": args.model,
            "num_requested": args.num_paraphrases,
            "num_generated": len(paras),
        }
        out_rows.append(rec)
        print(f"[paraphrase] {len(out_rows)}/{len(rows_in)} line={i} n={len(paras)}", flush=True)

    out_rows.sort(key=lambda r: int(r.get("line_idx", 0)))
    write_jsonl(args.output.resolve(), out_rows)
    print(f"wrote {len(out_rows)} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
