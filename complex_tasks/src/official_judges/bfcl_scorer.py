"""Official BFCL AST scoring for single-turn function-calling categories."""
from __future__ import annotations

import copy
import ast
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BFCL_ROOT = Path(os.environ.get("COMPLEX_RAW_ROOT", str(ROOT / "data"))).expanduser() / "policy_api" / "BFCL"

# Single-turn AST categories with possible_answer (no multi-turn loop).
# Keep this list explicit: the remaining BFCL v4 categories require execution,
# multi-turn state, or an agentic evaluator and cannot be reduced to one text
# completion plus ast_checker.
BFCL_AST_CATEGORIES: tuple[str, ...] = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)

DEFAULT_BFCL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def _bfcl_load_file(path: Path) -> list[dict[str, Any]]:
    """Load BFCL data files, which are JSONL despite their .json suffix."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda x: str(x.get("id", "")))


def _model_config_stub_module() -> types.ModuleType:
    class _DefaultConfig:
        underscore_to_dot = False

    class _DefaultModelMap(dict):
        def __missing__(self, key):
            value = _DefaultConfig()
            self[key] = value
            return value

    mod = types.ModuleType("bfcl_eval.constants.model_config")
    mod.MODEL_CONFIG_MAPPING = _DefaultModelMap(
        {
            DEFAULT_BFCL_MODEL_NAME: _DefaultConfig(),
            f"{DEFAULT_BFCL_MODEL_NAME}-FC": _DefaultConfig(),
        }
    )
    return mod


def _import_ast_checker():
    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        return ast_checker
    except ModuleNotFoundError as e:
        if e.name not in {"tree_sitter", "tree_sitter_java", "tree_sitter_javascript"}:
            raise
        sys.modules.pop("bfcl_eval.constants.model_config", None)
        sys.modules["bfcl_eval.constants.model_config"] = _model_config_stub_module()
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        return ast_checker


def _ensure_bfcl() -> None:
    if not (BFCL_ROOT / "bfcl_eval").is_dir():
        raise FileNotFoundError(f"BFCL is not at {BFCL_ROOT}")
    root = str(BFCL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def bfcl_language_and_format(test_category: str) -> tuple[Any, Any]:
    _ensure_bfcl()
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.utils import is_java, is_js

    if is_java(test_category):
        return Language.JAVA, ReturnFormat.JAVA
    if is_js(test_category):
        return Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
    return Language.PYTHON, ReturnFormat.PYTHON


def _is_function_calling_format_output(decoded_output: Any) -> bool:
    if not isinstance(decoded_output, list):
        return False
    for item in decoded_output:
        if not isinstance(item, dict) or len(item) != 1:
            return False
        if not isinstance(list(item.values())[0], dict):
            return False
    return True


def _resolve_ast_call(elem: ast.Call) -> dict[str, Any]:
    func_parts: list[str] = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {arg.arg: _resolve_ast_value(arg.value) for arg in elem.keywords}
    return {func_name: args_dict}


def _resolve_ast_value(value: ast.AST) -> Any:
    if isinstance(value, ast.Constant):
        return "..." if value.value is Ellipsis else value.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        inner = _resolve_ast_value(value.operand)
        return -inner if isinstance(inner, (int, float)) else inner
    if isinstance(value, ast.List):
        return [_resolve_ast_value(v) for v in value.elts]
    if isinstance(value, ast.Tuple):
        return tuple(_resolve_ast_value(v) for v in value.elts)
    if isinstance(value, ast.Dict):
        return {
            _resolve_ast_value(k): _resolve_ast_value(v)
            for k, v in zip(value.keys, value.values)
            if k is not None
        }
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        return ast.unparse(value) if not value.keywords else _resolve_ast_call(value)
    if isinstance(value, ast.Subscript):
        return ast.unparse(value)
    raise ValueError(f"unsupported AST value type: {type(value)}")


def _decode_python_prompting(result: str) -> list[dict[str, Any]]:
    text = str(result or "").strip("`\n ").strip("'")
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"
    parsed = ast.parse(text, mode="eval")
    body = parsed.body
    if isinstance(body, ast.Call):
        return [_resolve_ast_call(body)]
    if not isinstance(body, (ast.List, ast.Tuple)):
        raise ValueError("decoded BFCL output is not a function-call list")
    out: list[dict[str, Any]] = []
    for elem in body.elts:
        if not isinstance(elem, ast.Call):
            raise ValueError("decoded BFCL output item is not a call")
        out.append(_resolve_ast_call(elem))
    return out


def _decode_ast_prompting(prediction: str, return_format: Any) -> list[dict[str, Any]]:
    # Avoid importing bfcl_eval.model_handler.utils in the common Python path;
    # that module imports Java/JS tree-sitter parsers even when only Python AST
    # scoring is needed.
    if str(getattr(return_format, "value", return_format)) == "python":
        return _decode_python_prompting(prediction)
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting

    return default_decode_ast_prompting(str(prediction or ""), return_format, has_tool_call_tag=False)


def _convert_to_tool_schema(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI

    def cast_props(props: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(props)
        for value in out.values():
            if not isinstance(value, dict):
                continue
            typ = value.get("type", "string")
            value["type"] = GORILLA_TO_OPENAPI.get(typ, "string")
            if value["type"] == "array" and isinstance(value.get("items"), dict):
                item_type = value["items"].get("type", "string")
                value["items"]["type"] = GORILLA_TO_OPENAPI.get(item_type, "string")
            if value["type"] == "object" and isinstance(value.get("properties"), dict):
                value["properties"] = cast_props(value["properties"])
        return out

    tools: list[dict[str, Any]] = []
    for func in functions:
        item = copy.deepcopy(func)
        params = item.setdefault("parameters", {})
        params["type"] = "object"
        params["properties"] = cast_props(params.get("properties") or {})
        tools.append(item)
    return tools


def _bfcl_system_prompt(functions: list[dict[str, Any]]) -> str:
    # DEFAULT_SYSTEM_PROMPT_FORMAT in BFCL v4:
    # ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic
    return (
        "You are an expert in composing functions. You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the format of "
        "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  "
        "You SHOULD NOT include any other text in the response.\n\n"
        "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
        "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
        "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\n"
        "Here is a list of functions in json format that you can invoke.\n"
        f"{json.dumps(functions, indent=4, ensure_ascii=False)}\n"
    )


def _bfcl_messages(first_turn: list[dict[str, Any]], functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(first_turn)
    system_prompt = _bfcl_system_prompt(functions)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt + "\n\n" + str(messages[0].get("content") or "")
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def bfcl_official_score(prediction: str, sample: dict[str, Any]) -> float:
    """Return 1.0 when ``prediction`` passes official ``ast_checker``, else 0.0."""
    _ensure_bfcl()
    ast_checker = _import_ast_checker()

    test_category = str(sample.get("bfcl_test_category") or sample.get("category") or "")
    functions = sample.get("bfcl_functions") or []
    ground_truth = sample.get("bfcl_ground_truth") or []
    model_name = str(sample.get("bfcl_model_name") or DEFAULT_BFCL_MODEL_NAME)
    if not functions or not ground_truth or not test_category:
        return 0.0

    language, return_format = bfcl_language_and_format(test_category)
    if str(getattr(return_format, "value", return_format)) in {"java", "javascript"}:
        try:
            import tree_sitter_java  # noqa: F401
            import tree_sitter_javascript  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "BFCL Java/JavaScript scoring requires tree-sitter dependencies from requirements.txt"
            ) from exc
    try:
        decoded = _decode_ast_prompting(str(prediction or ""), return_format)
    except Exception:
        return 0.0

    if not _is_function_calling_format_output(decoded):
        return 0.0

    try:
        checker_result = ast_checker(
            functions,
            decoded,
            ground_truth,
            language,
            test_category,
            model_name,
        )
    except Exception:
        # A malformed model call, unknown model config, or an upstream checker
        # exception is an incorrect BFCL item, not a failed whole evaluation.
        return 0.0
    return 1.0 if checker_result.get("valid") else 0.0


def bfcl_row_from_entry(
    entry: dict[str, Any],
    gt_entry: dict[str, Any],
    *,
    category: str,
) -> dict[str, Any] | None:
    """Build one processed JSONL row from official BFCL prompt + possible_answer."""
    _ensure_bfcl()

    entry_id = str(entry.get("id") or "").strip()
    functions = entry.get("function") or []
    question_turns = entry.get("question") or []
    if not entry_id or not functions or not question_turns:
        return None
    first_turn = question_turns[0]
    if not isinstance(first_turn, list) or not first_turn:
        return None

    ground_truth = gt_entry.get("ground_truth")
    if not ground_truth:
        return None

    messages = _bfcl_messages(first_turn, functions)
    tools = _convert_to_tool_schema(functions)

    user_q = ""
    for message in first_turn:
        if message.get("role") == "user":
            user_q = str(message.get("content") or "").strip()
            break
    if not user_q:
        return None

    sub_ctx = ""
    for message in messages:
        if message.get("role") == "system":
            sub_ctx = str(message.get("content") or "").strip()
            break
    if not sub_ctx:
        sub_ctx = user_q

    gold_name = ""
    if isinstance(ground_truth, list) and ground_truth and isinstance(ground_truth[0], dict):
        gold_name = next(iter(ground_truth[0].keys()), "")
    decoy = next((f["name"] for f in functions if f.get("name") != gold_name), "ToolNone")

    return {
        "question": user_q,
        "org_context": "",
        "sub_context": sub_ctx[:8000],
        "org_answer": [f"[{decoy}()]"],
        "sub_answer": [entry_id],
        "bfcl_id": entry_id,
        "bfcl_test_category": category,
        "bfcl_functions": functions,
        "bfcl_ground_truth": ground_truth,
        "bfcl_messages": messages,
        # Keep the converted schema for callers that need it, but inference in
        # this project uses bfcl_messages. The BFCL system prompt already
        # contains the official function documentation; passing tools again to
        # a tokenizer chat template can duplicate or alter that prompt.
        "bfcl_tools": tools,
        "bfcl_model_name": DEFAULT_BFCL_MODEL_NAME,
        "source_id": entry_id,
        "taxonomy": "policy_api",
        "dataset": "BFCL",
        "category": category,
        "question_type": category,
    }


def load_bfcl_category_rows(category: str) -> list[dict[str, Any]]:
    _ensure_bfcl()
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH

    prompt_path = PROMPT_PATH / f"{VERSION_PREFIX}_{category}.json"
    answer_path = POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{category}.json"
    if not prompt_path.is_file() or not answer_path.is_file():
        return []

    prompts = _bfcl_load_file(prompt_path)
    answers = _bfcl_load_file(answer_path)
    gt_by_id = {str(item.get("id")): item for item in answers}

    rows: list[dict[str, Any]] = []
    for entry in prompts:
        entry_id = str(entry.get("id") or "")
        gt_entry = gt_by_id.get(entry_id)
        if not gt_entry:
            continue
        row = bfcl_row_from_entry(entry, gt_entry, category=category)
        if row:
            rows.append(row)
    return rows
