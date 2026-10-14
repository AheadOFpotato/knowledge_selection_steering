#!/usr/bin/env python3
"""Convert taxonomy datasets into NQ-Swap-compatible JSONL.

Output schema (per line), readable by ``dump_activations_from_testjsonl.py``:
  question, org_context, sub_context, org_answer, sub_answer

Semantics (source-choice):
  - org_*  = parametric / internal (or "default" side)
  - sub_*  = contextual / external (conflict side we inject at Stage2)

Usage:
  python3 scripts/build_processed_jsonl.py
  python3 scripts/build_processed_jsonl.py --datasets RGB,GuideBench
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("COMPLEX_RAW_ROOT", str(ROOT / "data"))).expanduser()
OUT = Path(os.environ.get("PROCESSED_ROOT", str(RAW / "processed"))).expanduser()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _join_passages(xs: Any, limit: int = 3, max_chars: int = 2500) -> str:
    if not isinstance(xs, list):
        return str(xs or "").strip()
    parts: list[str] = []
    for x in xs[:limit]:
        t = str(x).strip()
        if t:
            parts.append(t)
    text = "\n\n".join(parts)
    return text[:max_chars]


def adapt_rgb_en_fact() -> Path:
    """RGB counterfactual robustness: positive vs positive_wrong."""
    src = RAW / "content_rag" / "RGB" / "en_fact.json"
    out = OUT / "RGB" / "en_fact.jsonl"
    rows: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ans = str(obj.get("answer") or "").strip()
            fake = str(obj.get("fakeanswer") or "").strip()
            q = str(obj.get("query") or "").strip()
            if not (q and ans and fake):
                continue
            rows.append(
                {
                    "question": q,
                    "org_context": _join_passages(obj.get("positive")),
                    "sub_context": _join_passages(obj.get("positive_wrong")),
                    "org_answer": [ans],
                    "sub_answer": [fake],
                    "source_id": str(obj.get("id", i)),
                    "taxonomy": "content_rag",
                    "dataset": "RGB",
                }
            )
    n = _write_jsonl(out, rows)
    print(f"[RGB/en_fact] {n} -> {out}")
    return out


def _strip_html(text: str, max_chars: int = 2000) -> str:
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def adapt_crag_with_alt() -> Path:
    """CRAG task1/2 official RAG: search snippets as context, field ``answer`` as gold.

    Does **not** use ``alt_ans`` as gold (that was a conflict-injection proxy and
    often a label like ``0`` / ``invalid question``, not the official target).
    Context is truncated from the head of ranked snippets to fit the tokenizer.
    """
    import os

    src = RAW / "content_rag" / "CRAG" / "crag_task_1_and_2_dev_v4.jsonl"
    out = OUT / "CRAG" / "task12_dev.jsonl"
    max_ctx = int(os.environ.get("CRAG_MAX_CONTEXT_CHARS", "12000"))
    rows: list[dict[str, Any]] = []
    n_drop = 0
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            ans = str(obj.get("answer") if obj.get("answer") is not None else "").strip()
            q = str(obj.get("query") or "").strip()
            if not (q and ans):
                n_drop += 1
                continue
            snippets: list[str] = []
            for sr in obj.get("search_results") or []:
                if not isinstance(sr, dict):
                    continue
                title = str(sr.get("page_name") or "").strip()
                sn = _strip_html(str(sr.get("page_snippet") or ""), max_chars=4000)
                if not sn:
                    continue
                snippets.append(f"{title}\n{sn}" if title else sn)
            if not snippets:
                n_drop += 1
                continue
            packed: list[str] = []
            used = 0
            for sn in snippets:
                add = len(sn) + (2 if packed else 0)
                if packed and used + add > max_ctx:
                    break
                packed.append(sn)
                used += add
                if used >= max_ctx:
                    break
            sub_ctx = "\n\n".join(packed)
            if len(sub_ctx) > max_ctx:
                sub_ctx = sub_ctx[:max_ctx]
            rows.append(
                {
                    "question": q,
                    "org_context": "",
                    "sub_context": sub_ctx,
                    "org_answer": ["I don't know"],
                    "sub_answer": [ans],
                    "source_id": str(obj.get("interaction_id", i)),
                    "taxonomy": "content_rag",
                    "dataset": "CRAG",
                    "category": "task12_dev",
                    "domain": obj.get("domain"),
                    "question_type": obj.get("question_type"),
                }
            )
    n = _write_jsonl(out, rows)
    stale = OUT / "CRAG" / "with_alt_ans.jsonl"
    if stale.is_file():
        stale.unlink()
        print(f"[CRAG] removed stale {stale}")
    print(f"[CRAG/task12_dev] {n} -> {out} (dropped={n_drop}, max_ctx={max_ctx})")
    return out


def adapt_guidebench() -> list[Path]:
    """Guideline conflict: clean = Instruction+Context; conflict = Guidelines+Context."""
    src_dir = RAW / "rule_guideline" / "GuideBench" / "data"
    outs: list[Path] = []
    for src in sorted(src_dir.glob("*_tasks.json")):
        cat = src.stem  # e.g. math_tasks
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        rows: list[dict[str, Any]] = []
        for i, obj in enumerate(data):
            instruction = str(obj.get("Instruction") or "").strip()
            context = str(obj.get("Context") or "").strip()
            gt = obj.get("Groundtruth") or {}
            ref = gt.get("ReferenceAnswer")
            if ref is None:
                ref = gt.get("OptimalOption")
            if ref is None:
                continue
            ref_s = str(ref).strip()
            if not ref_s:
                continue
            guidelines = obj.get("Guidelines") or []
            rule_lines = []
            for g in guidelines:
                if isinstance(g, dict):
                    rule_lines.append(
                        f"- [{g.get('type','rule')}] {g.get('rule_text','')}".strip()
                    )
                else:
                    rule_lines.append(f"- {g}")
            rules_block = "\n".join(rule_lines)
            question = instruction or "Answer according to the task."
            # Multiple-choice GuideBench: options may be a multi-line string or a list
            opts = obj.get("MultipleOptions")
            opt_block = ""
            if isinstance(opts, str) and opts.strip():
                opt_block = opts.strip()
            elif isinstance(opts, list) and opts:
                opt_lines = []
                for o in opts:
                    if isinstance(o, dict):
                        opt_lines.append(
                            f"{o.get('option_id', o.get('id', ''))}. {o.get('option_text', o.get('text', o))}"
                        )
                    else:
                        opt_lines.append(str(o))
                opt_block = "\n".join(opt_lines)
            if opt_block:
                question = question + "\n" + opt_block

            cat_l = cat.lower()
            if any(k in cat_l for k in ("chat", "summary", "hallu")):
                question += (
                    "\n\n请只输出最终选项字母（A/B/C/D），不要解释。"
                )
            elif "math" in cat_l:
                question += "\n\n请只输出最终数字答案，不要解释。"
            elif "price" in cat_l:
                question += "\n\n请只输出最终判定数字（0或1），不要解释。"
            elif "relevance" in cat_l:
                question += (
                    "\n\n请只输出最终相关度标签（如 0/1/2 或同格式参考答案），不要解释。"
                )

            org_ctx = context
            sub_ctx = (
                f"You must follow these domain guidelines:\n{rules_block}\n\n"
                f"Task context:\n{context}"
            )
            # Binary / MC: opposite as parametric proxy; open: keep ref as both with note
            org_ans: list[str]
            if ref_s in {"0", "1"}:
                org_ans = ["1" if ref_s == "0" else "0"]
            elif ref_s.upper() in {"A", "B", "C", "D", "E"}:
                # weak parametric proxy — prefer non-ref letter
                org_ans = ["B" if ref_s.upper() == "A" else "A"]
            else:
                # open-ended: no reliable parametric gold; use a sentinel that Stage1 rarely hits
                org_ans = ["[parametric-default]"]
            rows.append(
                {
                    "question": question,
                    "org_context": org_ctx[:2500],
                    "sub_context": sub_ctx[:3500],
                    "org_answer": org_ans,
                    "sub_answer": [ref_s],
                    "options_text": opt_block[:2000] if opt_block else "",
                    "source_id": f"{cat}_{i}",
                    "taxonomy": "rule_guideline",
                    "dataset": "GuideBench",
                    "category": cat,
                }
            )
        out = OUT / "GuideBench" / f"{cat}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[GuideBench/{cat}] {n} -> {out}")
        outs.append(out)
    return outs


def _contract_nli_hypothesis(task: str) -> str:
    """Prefer README / base_prompt from LegalBench github tasks."""
    base = RAW / "rule_legal" / "LegalBench" / "tasks_github" / task
    for name in ("README.md", "base_prompt.txt"):
        p = base / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) > 40 and ("Identify" in line or "clause" in line.lower()):
                return line.rstrip(".") + "? Answer Yes or No."
        # fallback: first long non-empty line
        for line in text.splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) > 40:
                return line.rstrip(".") + "? Answer Yes or No."
    return (
        f"Does the contract clause entail the hypothesis for {task}? Answer Yes or No."
    )


def adapt_contract_nli() -> list[Path]:
    """LegalBench ContractNLI tasks: contract text as conflict context; Yes/No labels."""
    src_dir = RAW / "rule_legal" / "LegalBench"
    outs: list[Path] = []
    for src in sorted(src_dir.glob("contract_nli_*__test.jsonl")):
        task = src.name.split("__")[0]  # contract_nli_limited_use
        hyp = _contract_nli_hypothesis(task)
        rows: list[dict[str, Any]] = []
        with src.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                text = str(obj.get("text") or obj.get("document") or "").strip()
                ans = str(obj.get("answer") or "").strip()
                if not text or ans.lower() not in {"yes", "no"}:
                    continue
                ans_norm = "Yes" if ans.lower() == "yes" else "No"
                opp = "No" if ans_norm == "Yes" else "Yes"
                rows.append(
                    {
                        "question": hyp,
                        "org_context": "",  # clean = question only (parametric legal prior)
                        "sub_context": text[:3500],
                        "org_answer": [opp],  # without clause: opposite as weak prior proxy
                        "sub_answer": [ans_norm],  # with clause → labeled answer
                        "source_id": str(obj.get("index", i)),
                        "taxonomy": "rule_legal",
                        "dataset": "ContractNLI",
                        "category": task,
                    }
                )
        out = OUT / "ContractNLI" / f"{task}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[ContractNLI/{task}] {n} -> {out}")
        outs.append(out)
    return outs


def adapt_nqswap_copy() -> Path:
    """Copy canonical NQ-Swap split into processed/ for a single data root."""
    src = RAW / "content_factual" / "NQ-Swap" / "dev_dedup_p_only.jsonl"
    out = OUT / "NQ-Swap" / "dev_dedup_p_only.jsonl"
    rows = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n = _write_jsonl(out, rows)
    print(f"[NQ-Swap] {n} -> {out}")
    return out


def _extract_api_names_from_instruction(instruction: str) -> list[str]:
    names: list[str] = []
    for m in re.finditer(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"', instruction or ""):
        n = m.group(1)
        if n not in names:
            names.append(n)
    return names


def _last_user_utterance(dialogue: str) -> str:
    last = ""
    for line in (dialogue or "").splitlines():
        s = line.strip()
        if s.lower().startswith("user:"):
            last = s.split(":", 1)[-1].strip()
    return last or (dialogue or "").strip()[:300]


def adapt_apibank() -> list[Path]:
    """API-Bank level-1/2 api-call rows: dialogue + API docs vs expected API-Request."""
    outs: list[Path] = []
    base = RAW / "policy_api" / "API-Bank" / "hf" / "test-data"
    for fname, cat in (("level-1-api.json", "level-1-api"), ("level-2-api.json", "level-2-api")):
        src = base / fname
        if not src.is_file():
            print(f"[API-Bank/{cat}] missing {src}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        rows: list[dict[str, Any]] = []
        for i, obj in enumerate(data):
            exp = str(obj.get("expected_output") or "").strip()
            inp = str(obj.get("input") or "").strip()
            instr = str(obj.get("instruction") or "").strip()
            if not exp or not inp:
                continue
            q = _last_user_utterance(inp)
            if not q:
                q = "Generate the next API request."
            names = _extract_api_names_from_instruction(instr)
            m = re.search(r"\[([A-Za-z_][A-Za-z0-9_]*)\s*\(", exp)
            gold_name = m.group(1) if m else ""
            decoy = next((n for n in names if n != gold_name), "ToolNone")
            org_ans = [f"API-Request: [{decoy}()]"]
            # Conflict side: full API policy docs + dialogue
            sub_ctx = (instr.strip() + "\n\nDialogue:\n" + inp).strip()
            rows.append(
                {
                    "question": q,
                    "org_context": "",
                    "sub_context": sub_ctx[:4000],
                    "org_answer": org_ans,
                    "sub_answer": [exp],
                    "expected_output": exp,
                    "source_id": f"{cat}_{obj.get('id', i)}",
                    "taxonomy": "policy_api",
                    "dataset": "API-Bank",
                    "category": cat,
                }
            )
        out = OUT / "API-Bank" / f"{cat}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[API-Bank/{cat}] {n} -> {out}")
        outs.append(out)
    return outs


def _longmemeval_session_blocks(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (session_date, session_nl) pairs in official order."""
    blocks: list[tuple[str, str]] = []
    dates = entry.get("haystack_dates") or []
    sessions = entry.get("haystack_sessions") or []
    for session_date, session_entry in zip(dates, sessions):
        sess_string = ""
        if isinstance(session_entry, list):
            for turn in session_entry:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role") or "").strip()
                content = str(turn.get("content") or "").strip()
                if not role or not content:
                    continue
                sess_string += f"\n\n{role}: {content}"
        elif isinstance(session_entry, dict):
            role = str(session_entry.get("role") or "").strip()
            content = str(session_entry.get("content") or "").strip()
            if role and content:
                sess_string += f"\n\n{role}: {content}"
        blocks.append((str(session_date), sess_string))
    return blocks


def _longmemeval_oracle_history(
    entry: dict[str, Any], *, max_chars: int = 12000
) -> str:
    """Official ``oracle-session`` + ``nl`` history.

    Keep the **latest** sessions that fit ``max_chars`` (same spirit as
    LongMemEval ``orig-session`` top-k keep-latest). Full oracle histories are
    often 20k–100k chars and OOM even at batch size 1 on 8B.
    """
    blocks = _longmemeval_session_blocks(entry)
    if not blocks:
        return ""
    # Prefer recent sessions first, then restore chronological order for the prompt.
    kept_rev: list[tuple[str, str]] = []
    used = 0
    for session_date, sess_string in reversed(blocks):
        piece = f"\n### Session:\nSession Date: {session_date}\nSession Content:\n{sess_string}\n"
        # provisional; renumber after selection
        if kept_rev and used + len(piece) > max_chars:
            break
        if not kept_rev and len(piece) > max_chars:
            # Single oversized session: hard-truncate content tail.
            budget = max(500, max_chars - 120)
            sess_string = sess_string[:budget]
            piece = f"\n### Session:\nSession Date: {session_date}\nSession Content:\n{sess_string}\n"
            kept_rev.append((session_date, sess_string))
            used = len(piece)
            break
        kept_rev.append((session_date, sess_string))
        used += len(piece)
    kept = list(reversed(kept_rev))
    parts: list[str] = []
    for i, (session_date, sess_string) in enumerate(kept, start=1):
        parts.append(
            f"\n### Session {i}:\nSession Date: {session_date}\nSession Content:\n{sess_string}\n"
        )
    return "".join(parts)


def adapt_longmemeval_oracle() -> Path:
    """Official LongMemEval oracle split: evidence sessions as context; LLM Yes/No judge via question_type."""
    src = RAW / "policy_personalization" / "LongMemEval" / "original" / "longmemeval_oracle"
    out = OUT / "LongMemEval" / "oracle.jsonl"
    if not src.is_file():
        print(f"[LongMemEval/oracle] missing {src}")
        return out

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"expected list in {src}")

    # ~3k tokens of history; leave room for chat template + generation under 4k–8k ctx.
    max_hist_chars = int((__import__("os").environ.get("LME_MAX_HISTORY_CHARS") or "12000"))

    rows: list[dict[str, Any]] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        q = str(obj.get("question") or "").strip()
        ans = str(obj.get("answer") or "").strip()
        history = _longmemeval_oracle_history(obj, max_chars=max_hist_chars)
        if not (q and ans and history.strip()):
            continue
        qdate = str(obj.get("question_date") or "").strip()
        # Official answer prompt (non-CoT) with (possibly truncated) oracle sessions.
        sub_ctx = (
            "I will give you several history chats between you and a user. "
            "Please answer the question based on the relevant chat history.\n\n\n"
            f"History Chats:\n\n{history}\n\n"
            f"Current Date: {qdate}\n"
            f"Question: {q}"
        )
        rows.append(
            {
                "question": q,
                "org_context": "",
                "sub_context": sub_ctx,
                "org_answer": ["[unknown]"],
                "sub_answer": [ans],
                "question_type": str(obj.get("question_type") or "multi-session"),
                "question_date": qdate,
                "source_id": str(obj.get("question_id") or i),
                "taxonomy": "policy_personalization",
                "dataset": "LongMemEval",
                "category": "oracle",
                "history_max_chars": max_hist_chars,
            }
        )
    n = _write_jsonl(out, rows)
    print(f"[LongMemEval/oracle] {n} -> {out} (history_max_chars={max_hist_chars})")
    # Remove stale broken v2_small if present.
    stale = OUT / "LongMemEval" / "v2_small.jsonl"
    if stale.is_file():
        stale.unlink()
        print(f"[LongMemEval] removed stale {stale}")
    return out


def _mem2act_turn(turn: dict[str, Any]) -> str:
    role = str(turn.get("role") or "unknown")
    content = str(turn.get("content") or "").strip()
    lines: list[str] = []
    if role == "tool":
        name = str(turn.get("name") or "tool")
        lines.append(f"Tool ({name}): {content}" if content else f"Tool ({name}):")
    else:
        who = {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, role)
        if content:
            lines.append(f"{who}: {content}")
        for call in turn.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(fn.get("name") or call.get("name") or "").strip()
            args = fn.get("arguments") if fn else call.get("arguments")
            if isinstance(args, str):
                args_s = args
            else:
                args_s = json.dumps(args or {}, ensure_ascii=False)
            if name or args_s:
                lines.append(f"Assistant tool call: {name}({args_s})")
    return "\n".join(lines)


def adapt_mem2act() -> Path:
    """Mem2ActBench official context: full session dialogue + tool schema + query.

    Gold is the official tool_call (name + arguments). QA rows whose
    source_conversation_ids do not appear in any session are dropped.
    """
    base = RAW / "policy_personalization" / "Mem2ActBench" / "Mem2ActBench"
    qa_path = base / "qa_dataset.jsonl"
    conv_path = base / "toolmem_conversation.jsonl"
    out = OUT / "Mem2ActBench" / "qa.jsonl"
    if not qa_path.is_file() or not conv_path.is_file():
        small = RAW / "policy_personalization" / "Mem2ActBench" / "toolmembench_small"
        qa_path = small / "qa_dataset.jsonl"
        conv_path = small / "toolmem_conversation.jsonl"
    by_src: dict[str, list[str]] = {}
    sessions: dict[str, dict[str, Any]] = {}
    with conv_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj.get("session_id") or "")
            if not sid:
                continue
            sessions[sid] = obj
            for src in obj.get("original_conversation_ids") or []:
                by_src.setdefault(str(src), []).append(sid)

    rows: list[dict[str, Any]] = []
    n_drop = 0
    n_multi = 0
    with qa_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            q = str(obj.get("query") or "").strip()
            tc = obj.get("tool_call") or {}
            if not q or not isinstance(tc, dict) or not tc.get("name"):
                n_drop += 1
                continue
            src_ids = [str(s) for s in (obj.get("source_conversation_ids") or [])]
            chosen: list[str] = []
            seen: set[str] = set()
            linked = bool(src_ids)
            for src in src_ids:
                hits = by_src.get(src) or []
                if not hits:
                    linked = False
                    break
                for sid in hits:
                    if sid not in seen:
                        seen.add(sid)
                        chosen.append(sid)
            if not linked or not chosen:
                n_drop += 1
                continue
            chosen.sort()
            if len(chosen) > 1:
                n_multi += 1
            # Keep the tail of the dialogue. Tokenizer max length is 4096 and
            # truncates the right side, which would otherwise drop the request.
            max_hist = int(os.environ.get("M2A_MAX_HISTORY_CHARS", "12000"))
            turn_blocks: list[str] = []
            for sid in chosen:
                sess = sessions[sid]
                header = f"### {sid}"
                body_turns = [
                    _mem2act_turn(t)
                    for t in (sess.get("turns") or [])
                    if isinstance(t, dict)
                ]
                body_turns = [b for b in body_turns if b]
                turn_blocks.append(header)
                turn_blocks.extend(body_turns)
            schema = obj.get("target_tool_schema") or {}
            schema_s = json.dumps(schema, ensure_ascii=False) if schema else "{}"
            tail = (
                "Tool schema:\n"
                f"{schema_s}\n\n"
                f"Request: {q}\n"
                "JSON tool call:"
            )

            def _pack(blocks: list[str]) -> str:
                history = "\n".join(blocks)
                return f"Conversation history:\n\n{history}\n\n{tail}"

            while len(turn_blocks) > 1 and len(_pack(turn_blocks)) > max_hist:
                turn_blocks.pop(0)
            sub_ctx = _pack(turn_blocks)
            gold_s = json.dumps(
                {"name": tc.get("name"), "arguments": tc.get("arguments") or {}},
                ensure_ascii=False,
            )
            meta = obj.get("complexity_metadata") or {}
            rows.append(
                {
                    "question": q,
                    "org_context": "",
                    "sub_context": sub_ctx,
                    "org_answer": [json.dumps({"name": "[no-tool]", "arguments": {}}, ensure_ascii=False)],
                    "sub_answer": [gold_s],
                    "tool_call": {
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments") or {},
                    },
                    "source_id": str(obj.get("qa_id", i)),
                    "session_ids": chosen,
                    "complexity_level": str(meta.get("level") or ""),
                    "complexity_name": str(meta.get("level_name") or ""),
                    "taxonomy": "policy_personalization",
                    "dataset": "Mem2ActBench",
                    "category": "qa",
                }
            )
    n = _write_jsonl(out, rows)
    print(
        f"[Mem2ActBench/qa] {n} -> {out} (dropped_unlinked={n_drop}, multi_session={n_multi})"
    )
    return out


_FOLLOWBENCH_RULE_SOURCES = {
    "E2E",
    "WIKIEVENTS",
    "CONLL2003",
    "text_editing",
    "cnn_dailymail",
    "xsum",
    "samsum",
    "gigaword",
    "arxiv",
    "BBH_logical",
    "BBH_time",
    "self_made_space",
    "gsm_8k",
}


def adapt_legalbench() -> list[Path]:
    """LegalBench tasks other than ContractNLI. Prompt = base_prompt + text. Gold = answer.

    Use only the released test.tsv. Train examples are not evaluation examples
    and must never be silently substituted for test data.
    """
    import csv

    root = RAW / "rule_legal" / "LegalBench" / "tasks_github"
    outs: list[Path] = []
    n_tasks = n_rows = 0
    if not root.is_dir():
        print(f"[LegalBench] missing {root}")
        return outs
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if task_dir.name.startswith("contract_nli"):
            continue
        src = task_dir / "test.tsv"
        split = "test"
        if not src.is_file():
            print(f"[LegalBench/{task_dir.name}] no test.tsv; skipped (train.tsv is not evaluation data)")
            continue
        prompt = ""
        bp = task_dir / "base_prompt.txt"
        if bp.is_file():
            prompt = bp.read_text(encoding="utf-8").strip()
        rows: list[dict[str, Any]] = []
        with src.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, obj in enumerate(reader):
                text = str(obj.get("text") or obj.get("question") or "").strip()
                ans = str(obj.get("answer") or "").strip()
                if not text or not ans:
                    continue
                if prompt and "{{text}}" in prompt:
                    body = prompt.replace("{{text}}", text)
                elif prompt:
                    body = f"{prompt}\n\n{text}".strip()
                else:
                    body = text
                rows.append(
                    {
                        "question": text[:500],
                        "org_context": "",
                        "sub_context": body,
                        "org_answer": ["[unknown]"],
                        "sub_answer": [ans],
                        "source_id": str(obj.get("index", i)),
                        "taxonomy": "rule_legal",
                        "dataset": "LegalBench",
                        "category": task_dir.name,
                        "question_type": task_dir.name,
                        "split": split,
                    }
                )
        if not rows:
            continue
        out = OUT / "LegalBench" / f"{task_dir.name}.jsonl"
        n = _write_jsonl(out, rows)
        n_tasks += 1
        n_rows += n
        outs.append(out)
    print(f"[LegalBench] {n_tasks} tasks, {n_rows} rows")
    return outs


def adapt_followbench() -> list[Path]:
    """FollowBench level-1..5 constraint rows for the native/API judge.

    The released benchmark keeps level 0 as the unconstrained parent prompt.
    It is retained as metadata for each subsequent level, but it is not itself
    an evaluation item.  Rule-checkable rows use the upstream checker; all
    other rows are judged with the configured OpenAI-compatible judge.
    """
    src_dir = RAW / "rule_instruction" / "FollowBench" / "data"
    outs: list[Path] = []
    if not src_dir.is_dir():
        print(f"[FollowBench] missing {src_dir}")
        return outs
    kept = skipped_base = 0
    for path in sorted(src_dir.glob("*_constraints.json")):
        rows: list[dict[str, Any]] = []
        data = json.loads(path.read_text(encoding="utf-8"))
        histories: dict[str, list[str]] = {}
        for obj in data:
            level = int(obj.get("level") or 0)
            example_id = str(obj.get("example_id") or "")
            instruction = str(obj.get("instruction") or "").strip()
            if level <= 0:
                if instruction:
                    histories[example_id] = [instruction]
                skipped_base += 1
                continue
            source = str(obj.get("source") or "")
            rule_based = source in _FOLLOWBENCH_RULE_SOURCES or (
                str(obj.get("category") or "") == "format" and example_id in {"22", "30"}
            )
            target = str(obj.get("target") or "")
            if not instruction:
                continue
            cat = str(obj.get("category") or path.stem)
            history = histories.setdefault(example_id, [])
            history.append(instruction)
            # Reference targets are meaningful only to FollowBench's local
            # rule checker.  API-judged rows score constraint satisfaction
            # from the full instruction progression and model response.
            if rule_based and target.strip():
                gold_s = target
            else:
                gold_s = "[followbench-constraint]"
            rows.append(
                {
                    "question": instruction[:400],
                    "org_context": "",
                    "sub_context": f"{instruction}\nAnswer:",
                    "org_answer": ["[unknown]"],
                    "sub_answer": [gold_s],
                    "source_id": f"{cat}-{example_id}-L{level}",
                    "source": source,
                    "example_id": example_id,
                    "level": str(level),
                    "target": target,
                    "followbench_evolve_instructions": list(history),
                    "followbench_rule_based": rule_based,
                    "taxonomy": "rule_instruction",
                    "dataset": "FollowBench",
                    "category": cat,
                    "question_type": cat,
                    "complexity_level": str(level),
                }
            )
        if not rows:
            continue
        out = OUT / "FollowBench" / f"{path.stem}.jsonl"
        n = _write_jsonl(out, rows)
        kept += n
        outs.append(out)
        print(f"[FollowBench/{path.stem}] {n}")
    print(f"[FollowBench] kept={kept} skipped_level0={skipped_base}")
    return outs


def adapt_iheval() -> list[Path]:
    """IHEval conflict split. ``iheval_answer`` keeps the official answer object."""
    root = RAW / "rule_instruction" / "IHEval" / "benchmark"
    outs: list[Path] = []
    if not root.is_dir():
        print(f"[IHEval] missing {root}")
        return outs
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("input_data.json")):
        if "/conflict/" not in path.as_posix():
            continue
        rel = path.relative_to(root)
        cat = "__".join(rel.parts[:-1]).replace("/", "__")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            ans_raw = obj.get("answer")
            instruction = str(obj.get("instruction") or "").strip()
            if ans_raw is None or ans_raw == "" or not instruction:
                continue
            task = ""
            for part in rel.parts:
                if part in {
                    "verb-extract",
                    "translation",
                    "lang-detect",
                    "user-prompt-hijack",
                    "system-prompt-extract",
                    "single-turn",
                    "multi-turn",
                    "get-webpage",
                    "slack-user",
                }:
                    task = part
                    break
            if not task:
                continue
            system = str(obj.get("system") or "").strip()
            tool = obj.get("tool")
            tool_s = ""
            if isinstance(tool, dict):
                ret = tool.get("return")
                if ret is not None:
                    tool_s = json.dumps(ret, ensure_ascii=False)
            parts = []
            if system and system.lower() != "none":
                parts.append(f"System: {system}")
            history = obj.get("conversation_history") or []
            if isinstance(history, list):
                for turn_i, msg in enumerate(history):
                    role = "User" if turn_i % 2 == 0 else "Assistant"
                    parts.append(f"{role}: {msg}")
            parts.append(f"User: {instruction}")
            if tool_s:
                parts.append(f"Tool output: {tool_s}")
            parts.append("Answer:")
            gold_text = ans_raw if isinstance(ans_raw, str) else json.dumps(ans_raw, ensure_ascii=False)
            grouped.setdefault(cat, []).append(
                {
                    "question": instruction,
                    "org_context": "",
                    "sub_context": "\n\n".join(parts),
                    "org_answer": ["[unknown]"],
                    "sub_answer": [gold_text],
                    "iheval_task": task,
                    "iheval_answer": ans_raw,
                    "iheval_prompt": instruction,
                    "source_id": str(obj.get("id", i)),
                    "taxonomy": "rule_instruction",
                    "dataset": "IHEval",
                    "category": cat,
                    "question_type": cat,
                }
            )
    n_all = 0
    for cat, rows in sorted(grouped.items()):
        out = OUT / "IHEval" / f"{cat}.jsonl"
        n = _write_jsonl(out, rows)
        n_all += n
        outs.append(out)
    print(f"[IHEval/conflict] {len(outs)} files, {n_all} rows")
    return outs


def _locomo_history(conv: dict[str, Any], max_chars: int = 12000) -> str:
    blocks: list[str] = []
    i = 1
    while f"session_{i}" in conv:
        turns = conv.get(f"session_{i}") or []
        date = str(conv.get(f"session_{i}_date_time") or "").strip()
        lines = [f"### Session {i}" + (f" ({date})" if date else "")]
        if isinstance(turns, list):
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or "Speaker").strip()
                text = str(turn.get("text") or turn.get("clean_text") or "").strip()
                if text:
                    lines.append(f"{speaker}: {text}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))
        i += 1
    kept: list[str] = []
    used = 0
    for block in reversed(blocks):
        if kept and used + len(block) > max_chars:
            break
        kept.append(block)
        used += len(block)
    kept.reverse()
    return "\n\n".join(kept)


def adapt_locomo() -> Path:
    """LoCoMo QA over the 10 long conversations. Gold is the official answer string."""
    src = RAW / "policy_personalization" / "LoCoMo" / "data" / "locomo10.json"
    out = OUT / "LoCoMo" / "qa.jsonl"
    if not src.is_file():
        print(f"[LoCoMo] missing {src}")
        return out
    data = json.loads(src.read_text(encoding="utf-8"))
    names = {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }
    rows: list[dict[str, Any]] = []
    for sample in data:
        conv = sample.get("conversation") or {}
        history = _locomo_history(conv)
        if not history:
            continue
        sid = str(sample.get("sample_id") or "")
        for i, qa in enumerate(sample.get("qa") or []):
            if not isinstance(qa, dict):
                continue
            q = str(qa.get("question") or "").strip()
            ans = str(qa.get("answer") or "").strip()
            if not q or not ans:
                continue
            cat = int(qa.get("category") or 0)
            q_eval = q
            cat5_a = cat5_b = None
            if cat == 2:
                q_eval = q + " Use DATE of CONVERSATION to answer with an approximate date."
            elif cat == 5:
                rng = random.Random(f"{sid}-{i}")
                not_mentioned = "Not mentioned in the conversation"
                if rng.random() < 0.5:
                    cat5_a, cat5_b = not_mentioned, ans
                else:
                    cat5_a, cat5_b = ans, not_mentioned
                q_eval = q + f" Select the correct answer: (a) {cat5_a} (b) {cat5_b}. "
            rows.append(
                {
                    "question": q_eval,
                    "org_context": "",
                    "sub_context": f"Conversation:\n\n{history}\n\nQuestion: {q_eval}\nAnswer:",
                    "org_answer": ["[unknown]"],
                    "sub_answer": [ans],
                    "source_id": f"{sid}-{i}",
                    "taxonomy": "policy_personalization",
                    "dataset": "LoCoMo",
                    "category": names.get(cat, str(cat)),
                    "question_type": names.get(cat, str(cat)),
                    "locomo_category": cat,
                    "locomo_cat5_a": cat5_a,
                    "locomo_cat5_b": cat5_b,
                }
            )
    n = _write_jsonl(out, rows)
    print(f"[LoCoMo/qa] {n} -> {out}")
    return out


def adapt_ifeval() -> Path:
    """Google IFEval prompts with their released strict-evaluator metadata."""
    src = (
        RAW
        / "rule_instruction"
        / "IFEval"
        / "instruction_following_eval"
        / "data"
        / "input_data.jsonl"
    )
    out = OUT / "IFEval" / "official.jsonl"
    if not src.is_file():
        print(f"[IFEval] missing official input data: {src}")
        return out
    rows: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or "").strip()
            ids = obj.get("instruction_id_list")
            kwargs = obj.get("kwargs")
            if not prompt or not isinstance(ids, list) or not isinstance(kwargs, list):
                continue
            target = {
                "key": obj.get("key"),
                "instruction_id_list": ids,
                "kwargs": kwargs,
            }
            rows.append(
                {
                    "question": prompt,
                    "org_context": "",
                    "sub_context": f"{prompt}\n\nResponse:",
                    "org_answer": ["[unknown]"],
                    "sub_answer": [json.dumps(target, ensure_ascii=False)],
                    "source_id": str(obj.get("key")),
                    "taxonomy": "rule_instruction",
                    "dataset": "IFEval",
                    "category": "strict",
                    "question_type": "strict",
                    "official_metric": "google_ifeval_strict",
                    "ifeval_target": target,
                    "ifeval_prompt": prompt,
                }
            )
    n = _write_jsonl(out, rows)
    print(f"[IFEval/official] {n} -> {out}")
    return out


def adapt_freshqa() -> Path:
    """FreshQA official April 21, 2026 CSV snapshot.

    FreshQA intentionally has metadata rows before its header.  The published
    TEST split is the 500-question benchmark; DEV is retained as a separately
    runnable category.
    """
    candidates = (
        RAW / "content_factual" / "FreshQA" / "freshqa.csv",
        Path("<DATA_DIR>/FreshQA_v042126 - freshqa.csv"),
    )
    src = next((path for path in candidates if path.is_file()), None)
    out_dir = OUT / "FreshQA"
    if src is None:
        print("[FreshQA] missing official CSV; expected one of: " + ", ".join(map(str, candidates)))
        return out_dir / "test.jsonl"
    with src.open(encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.reader(f))
    try:
        header_at = next(i for i, row in enumerate(raw_rows) if row and row[0].strip().lower() == "id")
    except StopIteration:
        raise RuntimeError(f"FreshQA CSV has no id header: {src}")
    header = raw_rows[header_at]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_rows[header_at + 1 :]:
        if not any(str(value).strip() for value in raw):
            continue
        row = dict(zip(header, raw + [""] * max(0, len(header) - len(raw))))
        question = str(row.get("question") or "").strip()
        answers = [str(row.get(f"answer_{i}") or "").strip() for i in range(10)]
        answers = [answer for answer in answers if answer]
        split = str(row.get("split") or "").strip().lower()
        if not question or not answers or split not in {"test", "dev"}:
            continue
        by_split.setdefault(split, []).append(
            {
                "question": question,
                "org_context": "",
                "sub_context": question,
                "org_answer": ["[unknown]"],
                "sub_answer": answers,
                "source_id": str(row.get("id") or ""),
                "taxonomy": "content_factual",
                "dataset": "FreshQA",
                "category": split,
                "question_type": str(row.get("fact_type") or ""),
                "official_metric": "fresheval_strict",
                "freshqa_question": question,
                "freshqa_answers": answers,
                "freshqa_effective_year": str(row.get("effective_year") or ""),
                "freshqa_false_premise": str(row.get("false_premise") or ""),
                "freshqa_num_hops": str(row.get("num_hops") or ""),
                "freshqa_fact_type": str(row.get("fact_type") or ""),
            }
        )
    for split, rows in sorted(by_split.items()):
        out = out_dir / f"{split}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[FreshQA/{split}] {n} -> {out} (source={src})")
    return out_dir / "test.jsonl"


def adapt_locomo_plus() -> Path:
    """LoCoMo-Plus cognitive items: cue is the implicit constraint, trigger is the later query.

    Official scoring is constraint consistency, not a short string. ``sub_answer`` stores the cue.
    """
    src = RAW / "policy_personalization" / "LoCoMo-Plus" / "data" / "locomo_plus.json"
    out = OUT / "LoCoMo-Plus" / "cognitive.jsonl"
    if not src.is_file():
        print(f"[LoCoMo-Plus] missing {src}")
        return out
    data = json.loads(src.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        cue = str(obj.get("cue_dialogue") or "").strip()
        trigger = str(obj.get("trigger_query") or "").strip()
        if not cue or not trigger:
            continue
        rel = str(obj.get("relation_type") or "cognitive")
        rows.append(
            {
                "question": trigger,
                "org_context": "",
                "sub_context": (
                    "Earlier conversation:\n"
                    f"{cue}\n\n"
                    "Later message:\n"
                    f"{trigger}\n"
                    "Answer:"
                ),
                "org_answer": ["[unknown]"],
                "sub_answer": [cue],
                "constraint": cue,
                "relation_type": rel,
                "source_id": str(i),
                "taxonomy": "policy_personalization",
                "dataset": "LoCoMo-Plus",
                "category": rel,
                "question_type": rel,
            }
        )
    n = _write_jsonl(out, rows)
    print(f"[LoCoMo-Plus/cognitive] {n} -> {out}")
    return out


def adapt_bfcl() -> list[Path]:
    """BFCL single-turn AST categories scored by official ``ast_checker``."""
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from official_judges.bfcl_scorer import BFCL_AST_CATEGORIES, load_bfcl_category_rows

    outs: list[Path] = []
    for cat in BFCL_AST_CATEGORIES:
        rows = load_bfcl_category_rows(cat)
        if not rows:
            print(f"[BFCL/{cat}] missing data or no rows")
            continue
        out = OUT / "BFCL" / f"{cat}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[BFCL/{cat}] {n} -> {out}")
        outs.append(out)
    return outs


def adapt_toolsandbox() -> list[Path]:
    """One row per official named scenario, including distraction and scramble variants.

    The runner plays the scenario and scores ``EvaluationResult.similarity``.
    ``sub_context`` is only the user utterance; it is not the agent prompt.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from official_judges.toolsandbox_play import scenario_rows

    outs: list[Path] = []
    grouped = scenario_rows()
    if not grouped:
        print("[ToolSandbox] no scenarios")
        return outs
    for cat in (
        "single_tool_call",
        "multiple_tool_call",
        "multiple_user_turn",
        "insufficient_information",
        "other",
    ):
        rows = grouped.get(cat) or []
        if not rows:
            continue
        out = OUT / "ToolSandbox" / f"{cat}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[ToolSandbox/{cat}] {n} -> {out}")
        outs.append(out)
    return outs


def adapt_external_runner_dataset(name: str, taxonomy: str) -> list[Path]:
    """Register agent benchmarks without silently flattening their official rollout.

    ToolBench/tau-bench/MemoryAgentBench require their released data and, for the
    latter two, an executable official environment.  Their runner adapters are
    intentionally kept separate from string-answer JSONL evaluation.
    """
    root = RAW / taxonomy / name
    if not root.is_dir():
        print(f"[{name}] missing official data at {root}; no JSONL emitted")
        return []
    print(f"[{name}] official runner required; use its environment adapter, not generic JSONL scoring")
    return []


def adapt_taubench() -> list[Path]:
    print(
        "[tau-bench] official rollout supported via run/run_taubench_steer.sh; "
        "no generic JSONL is emitted"
    )
    return adapt_external_runner_dataset("tau-bench", "policy_api")


def adapt_memoryagentbench() -> list[Path]:
    """MemoryAgentBench FactConsolidation using the released external answers.

    Each released row contains one fact haystack and 100 questions.  Flatten
    these into independent fixed-context QA examples, preserving the official
    source name as the category.  By default only the 6k variants are emitted:
    they fit the current 8k-token Llama evaluation without truncating away the
    benchmark context.  Set ``MAB_CONTEXT_LENGTHS=all`` (or a comma-separated
    list such as ``6k,32k``) to opt into longer variants.
    """
    import pyarrow.parquet as pq

    src = (
        RAW
        / "policy_personalization"
        / "MemoryAgentBench"
        / "data"
        / "Conflict_Resolution-00000-of-00001.parquet"
    )
    if not src.is_file():
        print(f"[MemoryAgentBench] missing official Conflict Resolution data: {src}")
        return []

    requested_raw = str(os.environ.get("MAB_CONTEXT_LENGTHS") or "6k").strip().lower()
    requested = None if requested_raw == "all" else {
        x.strip() for x in requested_raw.split(",") if x.strip()
    }
    records = pq.read_table(
        src, columns=["context", "questions", "answers", "metadata"]
    ).to_pylist()
    outs: list[Path] = []
    for record in records:
        meta = record.get("metadata") or {}
        source = str(meta.get("source") or "").strip()
        if not source.startswith("factconsolidation_"):
            continue
        length_tag = source.rsplit("_", 1)[-1].lower()
        if requested is not None and length_tag not in requested:
            continue
        context = str(record.get("context") or "").strip()
        questions = list(record.get("questions") or [])
        answers = list(record.get("answers") or [])
        pair_ids = list(meta.get("qa_pair_ids") or [])
        rows: list[dict[str, Any]] = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            q = str(question or "").strip()
            gold = [str(x).strip() for x in (answer if isinstance(answer, list) else [answer]) if str(x).strip()]
            if not (q and gold and context):
                continue
            rows.append(
                {
                    "question": q,
                    "org_context": "",
                    "sub_context": context,
                    "org_answer": ["[unknown]"],
                    "sub_answer": gold,
                    "source_id": str(pair_ids[i] if i < len(pair_ids) else f"{source}_{i}"),
                    "source": source,
                    "taxonomy": "policy_personalization",
                    "dataset": "MemoryAgentBench",
                    "category": source,
                    "question_type": "fact_consolidation",
                    "official_metric": "substring_exact_match",
                    "context_length_tag": length_tag,
                }
            )
        out = OUT / "MemoryAgentBench" / f"{source}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"[MemoryAgentBench/{source}] {n} -> {out}")
        outs.append(out)
    if not outs:
        print(f"[MemoryAgentBench] no categories selected by MAB_CONTEXT_LENGTHS={requested_raw!r}")
    return outs


def adapt_personachat() -> list[Path]:
    return adapt_external_runner_dataset("PersonaChat", "policy_personalization")


ADAPTERS = {
    "NQ-Swap": adapt_nqswap_copy,
    "RGB": adapt_rgb_en_fact,
    "CRAG": adapt_crag_with_alt,
    "GuideBench": adapt_guidebench,
    "ContractNLI": adapt_contract_nli,
    "LegalBench": adapt_legalbench,
    "FollowBench": adapt_followbench,
    "IHEval": adapt_iheval,
    "API-Bank": adapt_apibank,
    "LongMemEval": adapt_longmemeval_oracle,
    "Mem2ActBench": adapt_mem2act,
    "IFEval": adapt_ifeval,
    "FreshQA": adapt_freshqa,
    "LoCoMo": adapt_locomo,
    "LoCoMo-Plus": adapt_locomo_plus,
    "ToolSandbox": adapt_toolsandbox,
    "BFCL": adapt_bfcl,
    "ToolBench": lambda: adapt_external_runner_dataset("ToolBench", "policy_api"),
    "tau-bench": adapt_taubench,
    "MemoryAgentBench": adapt_memoryagentbench,
    "PersonaChat": adapt_personachat,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        default="NQ-Swap,RGB,CRAG,GuideBench,ContractNLI,LegalBench,FollowBench,IHEval,BFCL,API-Bank,LongMemEval,Mem2ActBench,LoCoMo,LoCoMo-Plus",
        help="Comma-separated adapter names",
    )
    args = ap.parse_args()
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        fn = ADAPTERS.get(name)
        if fn is None:
            raise SystemExit(f"unknown dataset adapter: {name}. choose from {list(ADAPTERS)}")
        fn()
    print(f"[done] processed root: {OUT}")


if __name__ == "__main__":
    main()
