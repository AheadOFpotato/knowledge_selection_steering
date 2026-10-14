#!/usr/bin/env python3
"""Shared baseline metrics: first-sentence bidirectional containment.

- external (sub / context / counter answer): hit sub first
- internal (org / parametric answer): hit org only when sub missed
"""
from __future__ import annotations

import json
import re
import string
import unicodedata
from typing import Any


def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    parts = re.split(r"[.!?。！？]+", line, maxsplit=1)
    return (parts[0] if parts else line).strip()


def bidirectional_contains(text_a: str, text_b: str) -> bool:
    a = normalize_answer(text_a)
    b = normalize_answer(text_b)
    if not a or not b:
        return False
    return a in b or b in a


def as_answer_list(xs: Any) -> list[str]:
    if isinstance(xs, str):
        t = xs.strip()
        return [t] if t else []
    if isinstance(xs, list):
        return [str(x).strip() for x in xs if str(x).strip()]
    return []


def score_prediction(
    output: str,
    sub_answers: list[str],
    org_answers: list[str],
) -> dict[str, Any]:
    """Judge by first sentence; prefer external (sub) when both match."""
    first_sent = extract_first_sentence(output)
    hit_sub = any(bidirectional_contains(first_sent, a) for a in sub_answers)
    hit_org = any(bidirectional_contains(first_sent, a) for a in org_answers)

    if hit_sub:
        choice = "external"
    elif hit_org:
        choice = "internal"
    else:
        choice = "other"

    return {
        "first_sentence": first_sent,
        "knowledge_choice": choice,
        "correct_external": int(hit_sub),
        "correct_internal": int(hit_org and not hit_sub),
    }


def aggregate_knowledge_rates(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {
            "n": 0,
            "external": 0,
            "internal": 0,
            "other": 0,
            "external_rate": 0.0,
            "internal_rate": 0.0,
            "other_rate": 0.0,
            "metric": "first_sentence_bidirectional_contain",
        }
    external_n = sum(1 for s in scores if s["knowledge_choice"] == "external")
    internal_n = sum(1 for s in scores if s["knowledge_choice"] == "internal")
    other_n = n - external_n - internal_n
    return {
        "n": n,
        "external": external_n,
        "internal": internal_n,
        "other": other_n,
        "external_rate": external_n / n,
        "internal_rate": internal_n / n,
        "other_rate": other_n / n,
        "metric": "first_sentence_bidirectional_contain",
    }


def metrics_file_valid(path: str) -> bool:
    """True if JSON metrics file exists and has knowledge rates."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "external_rate" in data:
            return True
        for key in ("knowledge_metrics", "metrics", "summary_metrics"):
            km = data.get(key)
            if isinstance(km, dict) and "external_rate" in km:
                return True
            if isinstance(km, dict) and ("UseC_C" in km or "external" in km):
                return True
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return False
