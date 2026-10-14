# Shared helpers for ifeval run scripts.

# Block Hugging Face hub access for model weights (local checkpoints only).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

resolve_model_path() {
  local model_key="$1"
  python3 - "${BUNDLE_ROOT}" "${model_key}" <<'PY'
import json, sys
from pathlib import Path

bundle_root = Path(sys.argv[1])
model_key = sys.argv[2]
cfg = bundle_root / "config" / "models.json"
raw = json.loads(cfg.read_text(encoding="utf-8"))
path = raw.get(model_key)
if not path:
    raise SystemExit(f"[error] Model key not found in config/models.json: {model_key!r}")
p = Path(path)
if not p.is_absolute():
    p = (bundle_root / p).resolve()
print(p)
PY
}

check_model_weights() {
  local model_key="$1"
  local model_path
  model_path="$(resolve_model_path "${model_key}")" || return 1
  if [[ ! -d "${model_path}" ]]; then
    echo "[error] Model weights not found: ${model_path}" >&2
    echo "  Copy checkpoints into models/<ModelDir>/ at the repo root (see ifeval/config/models.json)." >&2
    echo "  This repo does not download model weights automatically." >&2
    return 1
  fi
  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "[error] Invalid model directory (missing config.json): ${model_path}" >&2
    echo "  Expected a local Hugging Face checkpoint folder under models/." >&2
    return 1
  fi
}

resolve_npz_path() {
  local model="$1"
  local npz_rel="$2"
  if [[ "${npz_rel}" = /* ]]; then
    printf '%s' "${npz_rel}"
  else
    printf '%s' "${BUNDLE_ROOT}/${npz_rel}"
  fi
}
