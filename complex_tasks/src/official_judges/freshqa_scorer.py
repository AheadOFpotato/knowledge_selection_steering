"""FreshQA's released FreshEval Strict protocol.

The upstream repository publishes the evaluator as a notebook rather than a
Python package.  This adapter loads its released prompt, demonstrations, and
rating parser directly from that notebook.  The judge transport is this
project's configured OpenAI-compatible client; the protocol itself is
unchanged.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .llm_client import get_chat_llm


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("FRESHQA_ROOT", str(ROOT / "data" / "content_factual" / "FreshQA"))).expanduser()
_LOCK = threading.Lock()
_PROTOCOL: tuple[str, list[str]] | None = None


def _upstream_notebook() -> Path:
    local = ROOT / "data" / "content_factual" / "FreshQA" / "fresheval_strict.ipynb"
    for path in (local, UPSTREAM / "fresheval_strict.ipynb"):
        if path.is_file():
            return path
    raise RuntimeError(
        "FreshQA official repository is missing fresheval_strict.ipynb; "
        "clone freshllms/freshqa under data/content_factual/FreshQA or <DATA_DIR>/conflict_taxonomy_steer_data/content_factual/FreshQA"
    )


def _protocol() -> tuple[str, list[str]]:
    """Load upstream Strict prefix and all its few-shot demonstrations once."""
    global _PROTOCOL
    with _LOCK:
        if _PROTOCOL is not None:
            return _PROTOCOL
        notebook = json.loads(_upstream_notebook().read_text(encoding="utf-8"))
        source = ""
        for cell in notebook.get("cells", []):
            code = "".join(cell.get("source", []))
            if "demo_examples =" in code and "evaluation_template" in code:
                source = code
                break
        if not source:
            raise RuntimeError("FreshQA official notebook has no FreshEval Strict examples")
        namespace: dict[str, Any] = {"current_date": "April 21, 2026"}
        # This is the released local notebook cell: assignments plus a loop that
        # formats its static demonstrations.  No imported code is executed.
        exec(compile(source, str(_upstream_notebook()), "exec"), namespace, namespace)
        prefix = str(namespace["prefix"])
        demos = [
            f"\n\n\nquestion: {question}{evaluation}"
            for question, evaluation in zip(
                namespace["demo_questions"], namespace["demo_evaluations"]
            )
        ]
        _PROTOCOL = (prefix, demos)
        return _PROTOCOL


def _parse_rating(text: str) -> bool | None:
    """Exact decision order from upstream ``extract_ratings``."""
    rating: str | None = None
    for line in str(text or "").split("\n"):
        if "evaluation: " in line:
            value = line.split(" ")[-1].strip().lower()
            if value not in {"correct", "incorrect"}:
                return None
            rating = value
    if rating is not None:
        return rating == "correct"
    if "Thus, the response is credited." in str(text or ""):
        return True
    if "Thus, the response is not credited." in str(text or ""):
        return False
    return None


def freshqa_strict_score(
    prediction: str,
    correct_answers: Any,
    *,
    question: str,
    evaluation_date: str | None = None,
) -> float:
    """Score one response with the official FreshEval Strict prompt (0 or 1)."""
    answers = correct_answers if isinstance(correct_answers, list) else [correct_answers]
    answers = [str(answer).strip() for answer in answers if str(answer).strip()]
    if not str(question).strip() or not answers:
        return 0.0
    prefix, demos = _protocol()
    evaluation = (
        "\ncorrect answer(s): {correct_answers}"
        "\nresponse: {response}"
        "\ncomment: "
    ).format(correct_answers=" | ".join(answers), response=str(prediction or ""))
    prompt = prefix + "\n\n\n" + "".join(demos).strip()
    prompt += f"\n\n\nquestion: {question}{evaluation}"
    date = evaluation_date or os.getenv("FRESHQA_EVAL_DATE", "April 21, 2026")
    response = get_chat_llm().chat(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond as concisely as possible. "
                f"Knowledge cutoff: {date}.",
            },
            {"role": "user", "content": "What's today's date?"},
            {
                "role": "assistant",
                "content": f"Today is {date} in Pacific Standard Time.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return 1.0 if _parse_rating(response) is True else 0.0
