#!/usr/bin/env python3
"""OpenAI-compatible chat client for official LLM judges.

Reads settings from ``official_judges.llm_settings``.
Tracks token usage / CNY cost and prints after every call.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from . import llm_settings


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    input_yuan_per_1m: float = 0.71
    output_yuan_per_1m: float = 2.86

    @classmethod
    def from_settings(cls) -> "LLMConfig":
        base = str(getattr(llm_settings, "LLM_BASE_URL", "") or "").strip()
        key = str(getattr(llm_settings, "LLM_API_KEY", "") or "").strip()
        model = str(getattr(llm_settings, "LLM_MODEL", "") or "").strip()
        if not base:
            raise RuntimeError(
                "Missing LLM_BASE_URL in official_judges/llm_settings.py"
            )
        if not key:
            raise RuntimeError(
                "Missing LLM_API_KEY in official_judges/llm_settings.py — fill in your key."
            )
        if not model:
            raise RuntimeError(
                "Missing LLM_MODEL in official_judges/llm_settings.py"
            )
        return cls(
            base_url=base.rstrip("/"),
            api_key=key,
            model=model,
            input_yuan_per_1m=float(getattr(llm_settings, "LLM_INPUT_YUAN_PER_1M", 0.71)),
            output_yuan_per_1m=float(getattr(llm_settings, "LLM_OUTPUT_YUAN_PER_1M", 2.86)),
        )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Backward-compatible alias → file settings."""
        return cls.from_settings()


@dataclass
class UsageTotals:
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls_without_usage: int = 0
    cost_yuan: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_yuan: float,
        has_usage: bool,
    ) -> None:
        with self.lock:
            self.api_calls += 1
            if not has_usage:
                self.calls_without_usage += 1
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self.total_tokens += int(total_tokens)
            self.cost_yuan += float(cost_yuan)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "api_calls": self.api_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "calls_without_usage": self.calls_without_usage,
                "cost_yuan": round(self.cost_yuan, 6),
            }


# Process-wide accumulator (shared across ChatLLM instances + OpenAI patch).
USAGE = UsageTotals()
_PATCH_INSTALLED = False
_PATCH_LOCK = threading.Lock()


def _pricing() -> tuple[str, float, float]:
    cfg = LLMConfig.from_settings()
    return cfg.model, cfg.input_yuan_per_1m, cfg.output_yuan_per_1m


def _cost_yuan(prompt_tokens: int, completion_tokens: int, inp_rate: float, out_rate: float) -> float:
    return (prompt_tokens / 1_000_000.0) * inp_rate + (completion_tokens / 1_000_000.0) * out_rate


def _extract_usage(resp: Any) -> tuple[int, int, int, bool]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0, 0, False
    pt = int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0)
    ct = int(
        getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
    )
    tt = int(getattr(usage, "total_tokens", None) or (pt + ct))
    if pt == 0 and ct == 0 and tt == 0:
        return 0, 0, 0, False
    if pt == 0 and ct == 0 and tt > 0:
        pt, ct = tt // 2, tt - tt // 2
    return pt, ct, tt, True


def record_and_print_usage(
    resp: Any,
    *,
    model: Optional[str] = None,
    tag: str = "chat",
) -> dict[str, Any]:
    """Record usage from an OpenAI response and print a cost line."""
    model_name, inp_rate, out_rate = _pricing()
    if model:
        model_name = model
    pt, ct, tt, has = _extract_usage(resp)
    call_cost = _cost_yuan(pt, ct, inp_rate, out_rate)
    USAGE.add(
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        cost_yuan=call_cost,
        has_usage=has,
    )
    snap = USAGE.snapshot()
    print_every = bool(getattr(llm_settings, "LLM_PRINT_COST_EVERY_CALL", True))
    if print_every:
        print(
            f"[llm_cost] model={model_name} tag={tag} "
            f"call=#{snap['api_calls']} "
            f"this=+{pt}/{ct}/{tt} tok (¥{call_cost:.6f}) | "
            f"cum={snap['prompt_tokens']}/{snap['completion_tokens']}/{snap['total_tokens']} tok "
            f"(¥{snap['cost_yuan']:.6f}) | "
            f"rate=in¥{inp_rate}/M out¥{out_rate}/M",
            flush=True,
        )
    return {
        "model": model_name,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "call_cost_yuan": round(call_cost, 6),
        "cumulative": snap,
        "input_yuan_per_1m": inp_rate,
        "output_yuan_per_1m": out_rate,
    }


def print_cost_summary(prefix: str = "[llm_cost_summary]") -> dict[str, Any]:
    model_name, inp_rate, out_rate = _pricing()
    snap = USAGE.snapshot()
    print(
        f"{prefix} model={model_name} "
        f"calls={snap['api_calls']} "
        f"tokens_in/out/total={snap['prompt_tokens']}/{snap['completion_tokens']}/{snap['total_tokens']} "
        f"cost_yuan=¥{snap['cost_yuan']:.6f} "
        f"(in¥{inp_rate}/M out¥{out_rate}/M) "
        f"no_usage_calls={snap['calls_without_usage']}",
        flush=True,
    )
    return {"model": model_name, **snap, "input_yuan_per_1m": inp_rate, "output_yuan_per_1m": out_rate}


def install_openai_usage_patch() -> None:
    """Patch OpenAI chat.completions.create so LongMemEval-native clients also count."""
    global _PATCH_INSTALLED
    with _PATCH_LOCK:
        if _PATCH_INSTALLED:
            return
        try:
            from openai.resources.chat.completions import Completions
        except Exception:
            return
        if getattr(Completions.create, "_llm_cost_patched", False):
            _PATCH_INSTALLED = True
            return
        _orig = Completions.create

        def _wrapped(self, *args: Any, **kwargs: Any):  # noqa: ANN001
            resp = _orig(self, *args, **kwargs)
            try:
                model = kwargs.get("model") or getattr(self, "_default_model", None)
                record_and_print_usage(resp, model=str(model) if model else None, tag="openai_patch")
            except Exception as e:
                print(f"[llm_cost] warn: failed to record usage ({e})", flush=True)
            return resp

        _wrapped._llm_cost_patched = True  # type: ignore[attr-defined]
        Completions.create = _wrapped  # type: ignore[method-assign]
        _PATCH_INSTALLED = True


class ChatLLM:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig.from_settings()
        from openai import OpenAI

        install_openai_usage_patch()
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        print(
            f"[llm_cost] ready model={self.cfg.model} base={self.cfg.base_url} "
            f"price=in¥{self.cfg.input_yuan_per_1m}/M out¥{self.cfg.output_yuan_per_1m}/M",
            flush=True,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 64,
        **kwargs: Any,
    ) -> str:
        # Patch already records usage; still call create here.
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()


_thread_llm = threading.local()


def get_chat_llm(cfg: Optional[LLMConfig] = None) -> ChatLLM:
    """One ChatLLM per thread (safe for parallel judges)."""
    llm = getattr(_thread_llm, "client", None)
    if llm is None:
        llm = ChatLLM(cfg)
        _thread_llm.client = llm
    return llm


# Eagerly patch so LongMemEval-native OpenAI clients also report cost.
try:
    install_openai_usage_patch()
except Exception:
    pass
