"""Apply per-model chat templates before tokenization (extraction + evaluation)."""
from __future__ import annotations

import json
from typing import Any

# Chat/instruct model keys: wrap raw task prompts with tokenizer.apply_chat_template.
CHAT_MODEL_KEYS = frozenset(
    {
        "llama3-8b-it",
        "llama3.1-8b-it",
        "gemma2-9b-it",
        "qwen3-8b",
        "mistral-7b-instruct-v0.3",
        "yi-6b-chat",
    }
)

# Extra kwargs passed to tokenizer.apply_chat_template for specific models.
_CHAT_TEMPLATE_KWARGS: dict[str, dict[str, Any]] = {
    "qwen3-8b": {"enable_thinking": False},
}


def should_use_chat_template(
    model_key: str,
    tokenizer,
    *,
    use_chat_template: bool = True,
) -> bool:
    if not use_chat_template:
        return False
    key = str(model_key or "").strip()
    if key not in CHAT_MODEL_KEYS:
        return False
    return bool(getattr(tokenizer, "chat_template", None))


def _apply_chat_template(tokenizer, raw_prompt: str, model_key: str) -> str:
    messages = [{"role": "user", "content": str(raw_prompt)}]
    return format_chat_messages_for_inference(
        tokenizer, messages, model_key, tools=None, use_chat_template=True
    )


def format_chat_messages_for_inference(
    tokenizer,
    messages: list[dict[str, Any]],
    model_key: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    use_chat_template: bool = True,
) -> str:
    """Format OpenAI-style chat messages (optional tools) for local generation."""
    if not should_use_chat_template(model_key, tokenizer, use_chat_template=use_chat_template):
        # Fallback: concatenate plain text turns.
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            if content is None:
                content = ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant:"
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer has no apply_chat_template")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    extra = _CHAT_TEMPLATE_KWARGS.get(str(model_key or "").strip(), {})
    kwargs.update(extra)
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        if tools:
            # Older templates may not accept tools=; fold a compact schema into system.
            msgs = list(messages)
            schema = json.dumps(tools, ensure_ascii=False)
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {
                    **msgs[0],
                    "content": str(msgs[0].get("content") or "")
                    + "\n\nAvailable tools (JSON schema):\n"
                    + schema,
                }
            else:
                msgs.insert(
                    0,
                    {
                        "role": "system",
                        "content": "Available tools (JSON schema):\n" + schema,
                    },
                )
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(msgs, **kwargs)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        # Some instruct tokenizers (notably Gemma 2) reject the OpenAI-style
        # system role even though the benchmark prompt uses one. Preserve the
        # system instructions by folding them into the first user turn.
        if "System role not supported" not in str(exc):
            raise
        msgs = list(messages)
        if msgs and str(msgs[0].get("role") or "") == "system":
            system_text = str(msgs[0].get("content") or "")
            msgs = msgs[1:]
            if msgs:
                msgs[0] = {
                    **msgs[0],
                    "content": system_text + "\n\n" + str(msgs[0].get("content") or ""),
                }
            else:
                msgs = [{"role": "user", "content": system_text}]
        return tokenizer.apply_chat_template(msgs, **kwargs)


def format_prompt_for_inference(
    tokenizer,
    raw_prompt: str,
    model_key: str,
    *,
    use_chat_template: bool = True,
) -> str:
    raw = str(raw_prompt)
    if not should_use_chat_template(model_key, tokenizer, use_chat_template=use_chat_template):
        return raw
    if not hasattr(tokenizer, "apply_chat_template"):
        return raw
    return _apply_chat_template(tokenizer, raw, model_key)


def format_prompts_for_inference(
    tokenizer,
    prompts: list[str],
    model_key: str,
    *,
    use_chat_template: bool = True,
) -> list[str]:
    return [
        format_prompt_for_inference(tokenizer, p, model_key, use_chat_template=use_chat_template)
        for p in prompts
    ]
