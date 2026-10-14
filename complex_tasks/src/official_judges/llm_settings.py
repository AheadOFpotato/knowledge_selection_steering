"""OpenAI-compatible judge settings.

All values are intentionally read from the environment so the same evaluator
works with a local vLLM/Ollama server or a hosted endpoint.  For a local
server, set ``LLM_BASE_URL=http://127.0.0.1:8000/v1`` and ``LLM_API_KEY=local``.
"""
import os

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "local")
LLM_MODEL = os.environ.get("LLM_MODEL", "local-model")
LLM_INPUT_YUAN_PER_1M = float(os.environ.get("LLM_INPUT_YUAN_PER_1M", "0"))
LLM_OUTPUT_YUAN_PER_1M = float(os.environ.get("LLM_OUTPUT_YUAN_PER_1M", "0"))
LLM_PRINT_COST_EVERY_CALL = os.environ.get("LLM_PRINT_COST_EVERY_CALL", "1") != "0"
