#!/usr/bin/env python3
"""Convert the WebArena expert next-action benchmark into compact JSONL."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_call(value: str) -> dict[str, Any]:
    raw = value.strip()
    try:
        node = ast.parse(raw, mode="eval").body
    except SyntaxError:
        # One released human answer has SQL-like doubled quotes inside
        # send_msg_to_user(...). Preserve its intended text as one argument.
        match = re.fullmatch(r"([A-Za-z_]\w*)\((.*)\)", raw, re.S)
        if not match:
            raise
        payload = match.group(2).strip()
        if len(payload) >= 2 and payload[0] == payload[-1] == "'":
            payload = payload[1:-1]
        return {
            "function": match.group(1),
            "args": [payload.replace("''", "'")],
            "kwargs": {},
            "parse_repaired": True,
        }
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported action: {value!r}")
    args = [ast.literal_eval(arg) for arg in node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    return {"function": node.func.id, "args": args, "kwargs": kwargs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = pq.read_table(
        args.source,
        columns=[
            "intent", "start_url", "thought_history", "action_history",
            "current_url", "text_observation", "chosen", "task_id",
            "step_id", "annotation_id", "source_name",
        ],
    ).to_pylist()
    output: list[dict[str, Any]] = []
    non_human = 0
    for row in rows:
        chosen = list(row.get("chosen") or [])
        human = next((item for item in chosen if item.get("response_source") == "human"), None)
        if human is None:
            non_human += 1
            continue
        output.append(
            {
                "sample_id": f"task{row['task_id']}:step{row['step_id']}:ann{row['annotation_id']}",
                "task_id": row["task_id"],
                "step": row["step_id"],
                "task": row["intent"],
                "start_url": row.get("start_url") or "",
                "url": row.get("current_url") or "",
                "observation": row["text_observation"],
                "action_history": list(row.get("action_history") or []),
                "thought_history": list(row.get("thought_history") or []),
                "reference_action_text": human["action"],
                "reference_action": parse_call(human["action"]),
                "reference_thought": human.get("thought") or "",
                "reference_source": "human",
                "source_name": row.get("source_name") or "webarena",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[webarena_expert] input={len(rows)} human={len(output)} "
        f"skipped_non_human={non_human} output={args.output}"
    )


if __name__ == "__main__":
    main()
