"""Play a ToolSandbox scenario and return official ``EvaluationResult.similarity``.

The score is ``int(minefield_similarity == 0) * milestone_similarity`` from
``scenario.evaluation.evaluate`` after ``scenario.play``. Tool calls are executed
by the sandbox ``ExecutionEnvironment``. Steering, when used, applies only to the
agent turn; the user simulator turn is the caller's responsibility.
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(os.environ.get("COMPLEX_RAW_ROOT", str(ROOT / "data"))).expanduser() / "policy_api" / "ToolSandbox"

CompleteFn = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], str]

_PRIMARY = (
    ("INSUFFICIENT_INFORMATION", "insufficient_information"),
    ("MULTIPLE_USER_TURN", "multiple_user_turn"),
    ("MULTIPLE_TOOL_CALL", "multiple_tool_call"),
    ("SINGLE_TOOL_CALL", "single_tool_call"),
)

_FN = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.S)
_TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_PY_ARGS = re.compile(
    r"^(?P<tool_id>.+)_parameters = (?P<arguments>[^\n]+)\n",
    re.M,
)

_NS: dict[str, Any] = {}
_SCENARIOS: dict[str, Any] | None = None
_ROLES: tuple[Any, Any, Any] | None = None


def _ensure_import() -> dict[str, Any]:
    if _NS:
        return _NS
    if not (SANDBOX_ROOT / "tool_sandbox" / "common" / "evaluation.py").is_file():
        raise FileNotFoundError(f"ToolSandbox is not at {SANDBOX_ROOT}")
    root = str(SANDBOX_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        RoleType,
        get_current_context,
    )
    from tool_sandbox.common.message_conversion import Message
    from tool_sandbox.common.tool_conversion import convert_to_openai_tools
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.roles.base_role import BaseRole
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment
    from tool_sandbox.scenarios import named_scenarios

    _NS.update(
        {
            "DatabaseNamespace": DatabaseNamespace,
            "RoleType": RoleType,
            "get_current_context": get_current_context,
            "Message": Message,
            "convert_to_openai_tools": convert_to_openai_tools,
            "ToolBackend": ToolBackend,
            "BaseRole": BaseRole,
            "ExecutionEnvironment": ExecutionEnvironment,
            "named_scenarios": named_scenarios,
        }
    )
    return _NS


def load_scenarios() -> dict[str, Any]:
    """Official ``named_scenarios``, with the CLI seed (``random.seed(42)``)."""
    global _SCENARIOS
    if _SCENARIOS is not None:
        return _SCENARIOS
    ns = _ensure_import()
    state = random.getstate()
    try:
        random.seed(42)
        _SCENARIOS = ns["named_scenarios"](
            preferred_tool_backend=ns["ToolBackend"].DEFAULT
        )
    finally:
        random.setstate(state)
    return _SCENARIOS


def _load_args(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("tool arguments are not an object")
    if ("name" in obj or "tool_name" in obj) and isinstance(obj.get("arguments"), dict):
        return obj["arguments"]
    if ("name" in obj or "tool_name" in obj) and isinstance(obj.get("parameters"), dict):
        return obj["parameters"]
    return obj


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Tool calls from a local chat completion (Llama ``<function=...>`` or ``<tool_call>``)."""
    raw = str(text or "")
    calls: list[tuple[str, dict[str, Any]]] = []
    for match in _FN.finditer(raw):
        calls.append((match.group(1).strip(), _load_args(match.group(2))))
    if calls:
        return calls
    for match in _TC.finditer(raw):
        obj = json.loads(match.group(1))
        if not isinstance(obj, dict):
            raise ValueError("tool_call is not an object")
        name = str(obj.get("name") or obj.get("tool_name") or "").strip()
        if not name:
            raise ValueError("tool_call has no name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if not isinstance(args, dict):
            raise ValueError("tool_call arguments are not an object")
        calls.append((name, args))
    if calls:
        return calls
    # Llama-3.1 python_tag / bare JSON tool call object(s).
    work = raw
    for token in ("<|python_tag|>", "<|tool_call|>", "<|eot_id|>", "<|end_of_text|>"):
        work = work.replace(token, "\n")
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", work):
        chunk = match.group(0)
        if not any(key in chunk for key in ('"name"', '"tool_name"', "'name'")):
            continue
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or obj.get("tool_name") or "").strip()
        if not name:
            continue
        if "arguments" in obj and isinstance(obj["arguments"], dict):
            calls.append((name, obj["arguments"]))
        elif "parameters" in obj and isinstance(obj["parameters"], dict):
            calls.append((name, obj["parameters"]))
    return calls


def _python_call(tool_id: str, fn_name: str, arguments: dict[str, Any]) -> str:
    if not tool_id.isidentifier() or not fn_name.isidentifier():
        raise KeyError(f"tool name is not a python identifier: {fn_name}")
    return (
        f"{tool_id}_parameters = {arguments!r}\n"
        f"{tool_id}_response = {fn_name}(**{tool_id}_parameters)\n"
        f"print(repr({tool_id}_response))"
    )


def _args_from_python(content: str) -> dict[str, Any]:
    match = _PY_ARGS.match(str(content or ""))
    if match is None:
        return {}
    try:
        value = ast.literal_eval(match.group("arguments"))
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _plain_text(text: str) -> str:
    cleaned = _FN.sub("", str(text or ""))
    cleaned = _TC.sub("", cleaned)
    for token in ("<|python_tag|>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _agent_chat(messages: list[Any]) -> list[dict[str, Any]]:
    ns = _ensure_import()
    role = ns["RoleType"]
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.sender == role.SYSTEM and message.recipient == role.AGENT:
            out.append({"role": "system", "content": message.content})
        elif message.sender == role.USER and message.recipient == role.AGENT:
            out.append({"role": "user", "content": message.content})
        elif message.sender == role.EXECUTION_ENVIRONMENT and message.recipient == role.AGENT:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.openai_tool_call_id or "",
                    "name": message.openai_function_name or "",
                    "content": message.content,
                }
            )
        elif message.sender == role.AGENT and message.recipient == role.EXECUTION_ENVIRONMENT:
            if not out or "tool_calls" not in out[-1]:
                out.append({"role": "assistant", "content": "", "tool_calls": []})
            name = message.openai_function_name or ""
            out[-1]["tool_calls"].append(
                {
                    "id": message.openai_tool_call_id or f"call_{len(out[-1]['tool_calls'])}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(_args_from_python(message.content), ensure_ascii=False),
                    },
                }
            )
        elif message.sender == role.AGENT and message.recipient == role.USER:
            out.append({"role": "assistant", "content": message.content})
        else:
            raise ValueError(
                f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
            )
    return out


def _user_chat(messages: list[Any]) -> list[dict[str, Any]]:
    """Same pairing as ``OpenAIAPIUser.to_openai_messages``."""
    ns = _ensure_import()
    role = ns["RoleType"]
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.sender == role.SYSTEM and message.recipient == role.USER:
            out.append({"role": "system", "content": message.content})
        elif message.sender == role.AGENT and message.recipient == role.USER:
            out.append({"role": "user", "content": message.content})
        elif message.sender == role.USER and message.recipient == role.AGENT:
            out.append({"role": "assistant", "content": message.content})
        elif message.recipient == role.EXECUTION_ENVIRONMENT or message.sender == role.EXECUTION_ENVIRONMENT:
            continue
        else:
            raise ValueError(
                f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
            )
    return out


def _agent_respond(self: Any, ending_index: int | None = None) -> None:
    ns = _ensure_import()
    role = ns["RoleType"]
    message_cls = ns["Message"]
    messages = self.get_messages(ending_index=ending_index)
    self.messages_validation(messages=messages)
    messages = self.filter_messages(messages=messages)
    if messages[-1].sender == role.SYSTEM:
        return
    tools = None
    if messages[-1].sender in {role.USER, role.EXECUTION_ENVIRONMENT}:
        available = self.get_available_tools()
        tools = ns["convert_to_openai_tools"](available)
        names = set(available)
    text = self._complete(_agent_chat(messages), tools)
    if tools is None:
        calls: list[tuple[str, dict[str, Any]]] = []
    else:
        calls = parse_tool_calls(text)
    if not calls:
        self.add_messages(
            [
                message_cls(
                    sender=self.role_type,
                    recipient=role.USER,
                    content=_plain_text(text) or text.strip(),
                )
            ]
        )
        return
    ctx = ns["get_current_context"]()
    wrote: list[Any] = []
    for idx, (name, arguments) in enumerate(calls):
        if name not in names:
            raise KeyError(f"Agent tool call {name} is not a known allowed tool. Options are {names}")
        execution_name = ctx.get_execution_facing_tool_name(name)
        tool_id = f"call_{idx}"
        wrote.append(
            message_cls(
                sender=self.role_type,
                recipient=role.EXECUTION_ENVIRONMENT,
                content=_python_call(tool_id, execution_name, arguments),
                openai_tool_call_id=tool_id,
                openai_function_name=name,
            )
        )
    self.add_messages(wrote)


def _user_respond(self: Any, ending_index: int | None = None) -> None:
    ns = _ensure_import()
    role = ns["RoleType"]
    message_cls = ns["Message"]
    messages = self.get_messages(ending_index=ending_index)
    self.messages_validation(messages=messages)
    messages = self.filter_messages(messages=messages)
    if messages[-1].sender == role.SYSTEM:
        return
    tools = None
    names: set[str] = set()
    if messages[-1].sender == role.AGENT:
        available = self.get_available_tools()
        tools = ns["convert_to_openai_tools"](available)
        names = set(available)
    text = self._complete(_user_chat(messages), tools)
    calls = parse_tool_calls(text) if tools is not None else []
    if not calls:
        self.add_messages(
            [
                message_cls(
                    sender=self.role_type,
                    recipient=role.AGENT,
                    content=_plain_text(text) or text.strip(),
                )
            ]
        )
        return
    wrote = []
    for idx, (name, arguments) in enumerate(calls):
        if name not in names:
            raise KeyError(f"User tool call {name} is not a known allowed tool. Options are {names}")
        tool_id = f"user_call_{idx}"
        wrote.append(
            message_cls(
                sender=self.role_type,
                recipient=role.EXECUTION_ENVIRONMENT,
                content=_python_call(tool_id, name, arguments),
                openai_tool_call_id=tool_id,
                openai_function_name=name,
            )
        )
    self.add_messages(wrote)


def _roles() -> tuple[Any, Any, Any]:
    global _ROLES
    if _ROLES is not None:
        return _ROLES
    ns = _ensure_import()
    base = ns["BaseRole"]
    role = ns["RoleType"]

    class SteeredAgent(base):
        role_type = role.AGENT

        def __init__(self, complete: CompleteFn) -> None:
            self._complete = complete

        respond = _agent_respond

    class SteeredUser(base):
        role_type = role.USER

        def __init__(self, complete: CompleteFn) -> None:
            self._complete = complete

        respond = _user_respond

    _ROLES = (SteeredAgent, SteeredUser, ns["ExecutionEnvironment"])
    return _ROLES


def play_similarity(
    scenario_name: str,
    agent_complete: CompleteFn,
    user_complete: CompleteFn,
) -> tuple[float, str]:
    """Run one scenario. The returned float is ``EvaluationResult.similarity``."""
    name = str(scenario_name or "").strip()
    scenarios = load_scenarios()
    scenario = scenarios.get(name)
    if scenario is None:
        return 0.0, f"unknown scenario {name}"
    agent_cls, user_cls, env_cls = _roles()
    ns = _ensure_import()
    role = ns["RoleType"]
    roles = {
        role.USER: user_cls(user_complete),
        role.AGENT: agent_cls(agent_complete),
        role.EXECUTION_ENVIRONMENT: env_cls(),
    }
    try:
        # Scenario.play draws a tqdm bar; keep runner logs readable.
        import os

        os.environ.setdefault("TQDM_DISABLE", "1")
        context = scenario.play(roles=roles, scenario_name=name)
        result = scenario.evaluation.evaluate(
            execution_context=context,
            max_turn_count=scenario.max_messages,
        )
        summary = (
            f"similarity={float(result.similarity):.4f} "
            f"milestone={float(result.milestone_similarity):.4f} "
            f"minefield={float(result.minefield_similarity):.4f} "
            f"turns={int(result.turn_count)}"
        )
        return float(result.similarity), summary
    except Exception:
        return 0.0, traceback.format_exc(limit=12)
    finally:
        for item in roles.values():
            item.teardown()


def _first_user_utterance(scenario: Any) -> str:
    ns = _ensure_import()
    role = ns["RoleType"]
    database = scenario.starting_context.get_database(ns["DatabaseNamespace"].SANDBOX)
    chosen = ""
    for row in database.to_dicts():
        if row.get("sender") != role.USER or row.get("recipient") != role.AGENT:
            continue
        visible = row.get("visible_to")
        if visible is not None:
            shown = {str(item) for item in list(visible)}
            if "AGENT" not in shown and role.AGENT not in list(visible):
                continue
        text = str(row.get("content") or "").strip()
        if text:
            chosen = text
    return chosen


def scenario_rows() -> dict[str, list[dict[str, Any]]]:
    """One JSONL group per task family. Each scenario is stored once.

    Augmentations from ``named_scenarios`` (distraction tools, scrambling) stay
    in the same family as their base scenario. The item score is still the
    official similarity of that full scenario.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, scenario in load_scenarios().items():
        labels = [str(category) for category in scenario.categories]
        family = "other"
        for label, filename in _PRIMARY:
            if label in labels:
                family = filename
                break
        utterance = _first_user_utterance(scenario) or name
        grouped.setdefault(family, []).append(
            {
                "question": utterance,
                "org_context": "",
                "sub_context": utterance,
                "org_answer": ["[unknown]"],
                "sub_answer": [name],
                "toolsandbox_scenario": name,
                "toolsandbox_categories": labels,
                "source_id": name,
                "taxonomy": "policy_api",
                "dataset": "ToolSandbox",
                "category": family,
                "question_type": family,
            }
        )
    return grouped
