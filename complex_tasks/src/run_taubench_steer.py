#!/usr/bin/env python3
"""Run official tau-bench environment rollouts with residual steering.

The local causal LM is the agent.  The official tau-bench environment executes
tools and computes reward from final database state / required outputs.  The
user simulator uses this project's configured OpenAI-compatible LLM and is
never steered.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAU_ROOT = Path(
    "<DATA_DIR>/conflict_taxonomy_steer_data/policy_api/tau-bench"
)
DEFAULT_DIR_NPZ = Path(
    "<DATA_DIR>/conflict_taxonomy_steer_500/llama3.1-8b-it/NQ-Swap/"
    "dev_dedup_p_only/resid_hidden_state__unfiltered_all_pairs.npz"
)
DEFAULT_OUT_ROOT = Path(
    "<DATA_DIR>/conflict_taxonomy_steer_result_500/tau-bench"
)

from official_judges.llm_client import ChatLLM, print_cost_summary
from run_resid_projection_ablation import (
    MODEL_PATHS,
    _build_projection_dirs,
    _iter_alpha_grid,
    _layer_hook_with_vector,
    generate_chat_completion,
    get_model_and_tokenizer,
)


class ConfiguredLLMUser:
    """tau-bench-compatible user simulator backed by local LLM settings."""

    def __init__(self) -> None:
        self.client = ChatLLM()
        self.messages: list[dict[str, str]] = []
        self.total_cost = 0.0

    @staticmethod
    def _system(instruction: str) -> str:
        return f"""You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate one short, natural user message at a time.
- Reveal only information necessary for the current step.
- Never invent information absent from the instruction.
- If the goal is satisfied, output ###STOP### alone.
- Do not quote the full instruction verbatim."""

    def _next(self) -> str:
        text = self.client.chat(self.messages, temperature=0.0, max_tokens=160)
        text = str(text or "").strip()
        self.messages.append({"role": "assistant", "content": text})
        return text

    def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self._system(str(instruction or ""))},
            {"role": "user", "content": "Begin the conversation."},
        ]
        return self._next()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": str(content or "")})
        return self._next()

    def get_total_cost(self) -> float:
        return float(self.total_cost)


def _json_blob(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    candidates = [raw]
    # Extract balanced objects so nested tool arguments survive surrounding prose.
    for start, char in enumerate(raw):
        if char != "{":
            continue
        depth = 0
        quoted = False
        escaped = False
        for end in range(start, len(raw)):
            cur = raw[end]
            if quoted:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    quoted = False
                continue
            if cur == '"':
                quoted = True
            elif cur == "{":
                depth += 1
            elif cur == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : end + 1])
                    break
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    kind = str(schema.get("type") or "")
    if kind in {"array", "object"} and isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    if kind == "array" and isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [_coerce_value(x, item_schema) for x in value]
    if kind == "object" and isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        return {k: _coerce_value(v, props.get(k, {})) for k, v in value.items()}
    return value


def _function_style_call(
    text: str, valid_tools: set[str]
) -> tuple[str, dict[str, Any]] | None:
    """Extract calls such as ``get_order_details(order_id="#W1")`` from prose."""
    raw = str(text or "")
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", raw):
        name = match.group(1)
        if name not in valid_tools:
            continue
        start = match.start(1)
        open_pos = raw.find("(", match.start(1))
        depth = 0
        quoted: str | None = None
        escaped = False
        for end in range(open_pos, len(raw)):
            cur = raw[end]
            if quoted is not None:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == quoted:
                    quoted = None
                continue
            if cur in {"'", '"'}:
                quoted = cur
            elif cur == "(":
                depth += 1
            elif cur == ")":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : end + 1]
                    try:
                        node = ast.parse(candidate, mode="eval").body
                    except (SyntaxError, ValueError):
                        break
                    if not isinstance(node, ast.Call) or node.args:
                        break
                    if not isinstance(node.func, ast.Name) or node.func.id != name:
                        break
                    kwargs: dict[str, Any] = {}
                    try:
                        for kw in node.keywords:
                            if kw.arg is None:
                                raise ValueError("**kwargs are unsupported")
                            kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError, SyntaxError):
                        break
                    return name, kwargs
    return None


def _prediction_to_action(
    text: str,
    valid_tools: set[str],
    tool_schemas: dict[str, dict[str, Any]],
    Action,
    respond_name: str,
):
    """Parse Llama chat-template JSON tool calls, else return a response."""
    obj = _json_blob(text)
    if obj is not None:
        name = str(obj.get("name") or obj.get("tool") or obj.get("function") or "")
        args = obj.get("parameters")
        if args is None:
            args = obj.get("arguments")
        if args is None:
            args = {}
        if name in valid_tools and isinstance(args, dict):
            schema = tool_schemas.get(name, {})
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            args = {k: _coerce_value(v, props.get(k, {})) for k, v in args.items()}
            return Action(name=name, kwargs=args), {
                "role": "assistant",
                "content": text,
                "tool_call": {"name": name, "arguments": args},
            }
    function_call = _function_style_call(text, valid_tools)
    if function_call is not None:
        name, args = function_call
        schema = tool_schemas.get(name, {})
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        args = {k: _coerce_value(v, props.get(k, {})) for k, v in args.items()}
        return Action(name=name, kwargs=args), {
            "role": "assistant",
            "content": text,
            "tool_call": {"name": name, "arguments": args, "syntax": "function_style"},
        }
    return Action(name=respond_name, kwargs={"content": str(text or "")}), {
        "role": "assistant",
        "content": str(text or ""),
    }


def _chat_messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tau-bench turns for portable Hugging Face chat templates.

    tau-bench represents an action result as an OpenAI ``tool`` message.  Some
    local templates (notably Mistral's) only accept that role when it is paired
    with a template-specific tool-call id.  The runner deliberately emits
    model-agnostic JSON actions, so it has no such id.  Preserve the result as
    the next user observation instead; this is also the form expected by the
    agent loop (the model must decide its next action from the observation).
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if content is None:
            content = ""
        if role == "tool":
            name = str(message.get("name") or "tool")
            out.append(
                {
                    "role": "user",
                    "content": f"Tool result from {name}:\n{content}",
                }
            )
        else:
            out.append({"role": role, "content": str(content)})
    return out


def _compact_tools(tools_info: list[dict[str, Any]]) -> str:
    """Compact tool instructions; avoids Llama echoing the long native schema."""
    lines = [
        "Available tools. To call one, output JSON only as "
        '{"name":"tool_name","parameters":{...}}. '
        "To talk to the user, output normal text. Never repeat these schemas."
    ]
    for item in tools_info:
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        desc = re.sub(r"\s+", " ", str(fn.get("description") or "")).strip()[:240]
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        required = set(params.get("required") or [])
        arg_parts = []
        for arg, schema in props.items():
            schema = schema if isinstance(schema, dict) else {}
            typ = str(schema.get("type") or "any")
            req = "required" if arg in required else "optional"
            note = re.sub(r"\s+", " ", str(schema.get("description") or "")).strip()[:120]
            arg_parts.append(f"{arg}:{typ},{req}" + (f" ({note})" if note else ""))
        lines.append(f"- {name}({'; '.join(arg_parts)}): {desc}")
    return "\n".join(lines)


def _run_task(
    *,
    env_name: str,
    task_index: int,
    model,
    tokenizer,
    model_key: str,
    tools_info: list[dict[str, Any]],
    wiki: str,
    max_steps: int,
    context_length: int,
    get_env,
    Action,
    respond_name: str,
) -> dict[str, Any]:
    env = get_env(
        env_name,
        user_strategy="human",
        user_model="unused",
        user_provider=None,
        task_split="test",
        task_index=task_index,
    )
    env.user = ConfiguredLLMUser()
    reset = env.reset(task_index=task_index)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": wiki},
        {"role": "user", "content": reset.observation},
    ]
    valid_tools = {str(t.get("function", {}).get("name") or "") for t in tools_info}
    tool_schemas = {
        str(t.get("function", {}).get("name") or ""): (
            t.get("function", {}).get("parameters") or {}
        )
        for t in tools_info
    }
    compact_tools = _compact_tools(tools_info)
    messages[0]["content"] = str(messages[0]["content"]) + "\n\n" + compact_tools
    reward = 0.0
    last_info: dict[str, Any] = reset.info.model_dump()
    done = False
    for step in range(max_steps):
        prediction = generate_chat_completion(
            model,
            tokenizer,
            _chat_messages_for_model(messages),
            tools=None,
            model_key=model_key,
            max_length=int(context_length),
            max_new_tokens=300,
        )
        action, assistant_message = _prediction_to_action(
            prediction, valid_tools, tool_schemas, Action, respond_name
        )
        response = env.step(action)
        reward = float(response.reward)
        last_info = response.info.model_dump()
        assistant_message["parsed_action"] = action.model_dump()
        messages.append(assistant_message)
        if action.name == respond_name:
            messages.append({"role": "user", "content": response.observation})
        else:
            messages.append(
                {
                    "role": "tool",
                    "content": response.observation,
                    "name": action.name,
                }
            )
        if response.done:
            done = True
            break
    return {
        "task_id": int(task_index),
        "reward": reward,
        "done": done,
        "num_steps": sum(1 for m in messages if m.get("role") == "assistant"),
        "info": last_info,
        "trajectory": messages,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Official tau-bench rollout with NQ-Swap residual steering."
    )
    ap.add_argument("--env", choices=("airline", "retail"), default="retail")
    ap.add_argument("--model", default="llama3.1-8b-it")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--layers", default="10")
    ap.add_argument("--npz", type=Path, default=DEFAULT_DIR_NPZ)
    ap.add_argument("--tau-root", type=Path, default=DEFAULT_TAU_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--alpha-start", type=float, default=0.0)
    ap.add_argument("--alpha-end", type=float, default=2.0)
    ap.add_argument("--alpha-step", type=float, default=0.5)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=-1)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all selected tasks")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument(
        "--context-length",
        type=int,
        default=8192,
        help="Agent prompt token limit; tau-bench schemas exceed 4096 tokens.",
    )
    ap.add_argument(
        "--direction-mode",
        choices=("true", "shuffled", "random_split", "gaussian"),
        default="true",
        help="How to construct the steering direction from the NPZ activations.",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tau_root = args.tau_root.resolve()
    if not tau_root.is_dir():
        raise SystemExit(f"tau-bench repository missing: {tau_root}")
    if not args.npz.is_file():
        raise SystemExit(f"direction NPZ missing: {args.npz}")
    sys.path.insert(0, str(tau_root))
    from tau_bench.envs import get_env
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    probe = get_env(
        args.env,
        user_strategy="human",
        user_model="unused",
        user_provider=None,
        task_split="test",
        task_index=0,
    )
    n_tasks = len(probe.tasks)
    start = max(0, int(args.start_index))
    end = n_tasks if int(args.end_index) < 0 else min(n_tasks, int(args.end_index))
    task_ids = list(range(start, end))
    if int(args.max_samples) > 0:
        task_ids = task_ids[: int(args.max_samples)]
    alphas = _iter_alpha_grid(args.alpha_start, args.alpha_end, args.alpha_step)
    print(
        f"[tau] env={args.env} tasks={len(task_ids)}/{n_tasks} "
        f"ids={start}:{end} alphas={alphas} out={args.out_root}",
        flush=True,
    )
    if args.dry_run:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model, tokenizer = get_model_and_tokenizer(args.model)
    layers = [int(x) for x in str(args.layers).split(",") if x.strip()]
    with np.load(args.npz, allow_pickle=True) as z:
        n_rows = len(z["resid_act_clean"])
    proj, _scales, direction_meta = _build_projection_dirs(
        args.npz,
        layers,
        list(range(n_rows)),
        steer_coeff_mode="alpha",
        direction_mode=args.direction_mode,
        steer_toward="context",
    )
    if not proj:
        raise SystemExit("No valid residual direction for requested layers")

    model_layers = model.model.layers  # type: ignore[attr-defined]
    gate = {"enabled": True}
    hooks = []
    current_alpha = {"value": 0.0}
    for layer_idx, vector in proj.items():
        if 0 <= layer_idx < len(model_layers):
            # Hook reads mutable alpha so the model is loaded only once.
            base_hook = _layer_hook_with_vector(vector, 1.0, gate=gate)

            def hook(module, inputs, output, *, _base=base_hook):
                a = float(current_alpha["value"])
                if a == 0.0:
                    return output
                # The base hook adds v; temporarily scale the returned delta.
                steered = _base(module, inputs, output)
                hs = output[0] if isinstance(output, tuple) else output
                shs = steered[0] if isinstance(steered, tuple) else steered
                mixed = hs + a * (shs - hs)
                return (mixed,) + output[1:] if isinstance(output, tuple) else mixed

            hooks.append(model_layers[layer_idx].register_forward_hook(hook))

    try:
        for alpha in alphas:
            current_alpha["value"] = float(alpha)
            out_path = (
                args.out_root
                / args.model
                / args.env
                / f"L{str(args.layers).replace(',', '_')}_alpha{float(alpha):g}.json"
            )
            existing: list[dict[str, Any]] = []
            if args.skip_existing and out_path.is_file():
                try:
                    doc = json.loads(out_path.read_text(encoding="utf-8"))
                    existing = list(doc.get("results") or [])
                except Exception:
                    existing = []
            # A task that raised before receiving an official reward is not
            # complete.  Keeping it in ``done_ids`` made --skip-existing
            # permanently preserve template failures and prevented recovery.
            done_ids = {
                int(x.get("task_id"))
                for x in existing
                if "task_id" in x and x.get("reward") is not None
            }
            if args.skip_existing and done_ids.issuperset(task_ids):
                print(f"[tau alpha={alpha:g}] skip complete {out_path}", flush=True)
                continue
            results = [x for x in existing if int(x.get("task_id", -1)) in task_ids]
            for task_id in task_ids:
                if args.skip_existing and task_id in done_ids:
                    continue
                print(f"[tau alpha={alpha:g}] task {task_id}", flush=True)
                try:
                    item = _run_task(
                        env_name=args.env,
                        task_index=task_id,
                        model=model,
                        tokenizer=tokenizer,
                        model_key=args.model,
                        tools_info=probe.tools_info,
                        wiki=probe.wiki,
                        max_steps=int(args.max_steps),
                        context_length=int(args.context_length),
                        get_env=get_env,
                        Action=Action,
                        respond_name=RESPOND_ACTION_NAME,
                    )
                except Exception as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    item = {
                        "task_id": task_id,
                        "reward": None,
                        "done": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "trajectory": [],
                    }
                results = [x for x in results if int(x.get("task_id", -1)) != task_id]
                results.append(item)
                results.sort(key=lambda x: int(x.get("task_id", -1)))
                evaluated = [x for x in results if x.get("reward") is not None]
                reward = sum(float(x.get("reward", 0.0)) for x in evaluated)
                doc = {
                    "benchmark": "tau-bench",
                    "official_reward": True,
                    "env": args.env,
                    "model": args.model,
                    "gpu": str(args.gpu),
                    "layers": layers,
                    "alpha": float(alpha),
                    "direction_npz": str(args.npz),
                    "direction": direction_meta,
                    "task_ids": task_ids,
                    "num_completed": len(results),
                    "num_evaluated": len(evaluated),
                    "num_errors": len(results) - len(evaluated),
                    "pass_rate": reward / max(1, len(evaluated)),
                    "results": results,
                }
                _write_json(out_path, doc)
                print(
                    f"[tau alpha={alpha:g}] task={task_id} reward={item.get('reward')} "
                    f"running_pass={doc['pass_rate']:.4f}",
                    flush=True,
                )
            print(out_path, flush=True)
    finally:
        for handle in hooks:
            handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_cost_summary(prefix="[tau user_llm_cost]")


if __name__ == "__main__":
    main()
