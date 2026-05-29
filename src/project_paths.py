"""Repository-relative paths (override via env: DATA_ROOT, DUMP_ROOT, MODEL_ROOT, …)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else default


DATA_ROOT = _env_path("DATA_ROOT", ROOT / "data")
DUMP_ROOT = _env_path("DUMP_ROOT", ROOT / "output" / "dump")
MODEL_ROOT = _env_path("MODEL_ROOT", ROOT / "models")

STEER_OUT_ROOT = _env_path("STEER_OUT_ROOT", ROOT / "output" / "steer_result")
CROSS_FORMAT_DUMP_ROOT = _env_path("CROSS_FORMAT_DUMP_ROOT", DUMP_ROOT / "cross_format")
CROSS_FORMAT_STEER_ROOT = _env_path("CROSS_FORMAT_STEER_ROOT", ROOT / "output" / "cross_format")
KNOWLEDGE_TYPE_STEER_ROOT = _env_path(
    "KNOWLEDGE_TYPE_STEER_ROOT", ROOT / "output" / "knowledge_type"
)
CROSS_DATASET_STEER_ROOT = _env_path(
    "CROSS_DATASET_STEER_ROOT", ROOT / "output" / "cross_dataset"
)

# Subdir names under MODEL_ROOT (not absolute paths).
MODEL_DIR_NAMES: dict[str, str] = {
    "llama3-8b-it": "Llama3-8b-it",
    "gemma2-9b-it": "Gemma2-9b-it",
    "qwen3-8b": "Qwen3-8B",
    "mistral-7b-v0.1": "Mistral-7B-v0.1",
    "yi-6b": "Yi-6B",
    "yi-6b-chat": "Yi-6B-Chat",
}


def resolve_model_path(model_key: str) -> str:
    """Local path under MODEL_ROOT, or pass through HF hub id / explicit path."""
    key = str(model_key).strip()
    sub = MODEL_DIR_NAMES.get(key)
    if sub:
        return str((MODEL_ROOT / sub).resolve())
    return key


def display_path(p: Path | str, base: Path | None = None) -> str:
    """Prefer repo-relative paths in logs and JSON artifacts."""
    base = base or ROOT
    rp = Path(p).expanduser().resolve()
    try:
        return str(rp.relative_to(base.resolve()))
    except ValueError:
        return str(rp)
