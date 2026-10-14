"""Google IFEval strict evaluator adapter.

This invokes the released ``instruction_following_eval.evaluation_lib`` rather
than reimplementing individual instruction checks.  IFEval's official headline
metric is prompt-level strict accuracy: every instruction attached to a prompt
must be followed.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
IFEVAL_PARENT = Path(os.environ.get("COMPLEX_RAW_ROOT", str(ROOT / "data"))).expanduser() / "rule_instruction" / "IFEval"
_LOCK = threading.Lock()
_EVAL_LIB: Any = None


def _official_lib() -> Any:
    global _EVAL_LIB
    with _LOCK:
        if _EVAL_LIB is not None:
            return _EVAL_LIB
        package = IFEVAL_PARENT / "instruction_following_eval"
        if not (package / "evaluation_lib.py").is_file():
            raise RuntimeError(
                "IFEval official package missing at "
                f"{package}; restore data/rule_instruction/IFEval/instruction_following_eval"
            )
        if str(IFEVAL_PARENT) not in sys.path:
            sys.path.insert(0, str(IFEVAL_PARENT))
        _EVAL_LIB = importlib.import_module(
            "instruction_following_eval.evaluation_lib"
        )
        return _EVAL_LIB


def ifeval_strict_score(prediction: str, target: Any, *, prompt: str) -> float:
    """Return the official IFEval strict prompt-level score (0 or 1)."""
    if not isinstance(target, dict) or not str(prompt).strip():
        return 0.0
    instruction_ids = target.get("instruction_id_list")
    kwargs = target.get("kwargs")
    if not isinstance(instruction_ids, list) or not isinstance(kwargs, list):
        return 0.0
    lib = _official_lib()
    inp = lib.InputExample(
        key=target.get("key", ""),
        instruction_id_list=instruction_ids,
        prompt=str(prompt),
        kwargs=kwargs,
    )
    result = lib.test_instruction_following_strict(inp, {str(prompt): str(prediction)})
    return 1.0 if result.follow_all_instructions else 0.0
