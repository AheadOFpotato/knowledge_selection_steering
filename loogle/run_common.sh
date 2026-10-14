# Shared helpers for loogle run scripts.

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
    echo "  Copy checkpoints into models/<ModelDir>/ at the repo root (see loogle/config/models.json)." >&2
    echo "  This repo does not download model weights automatically." >&2
    return 1
  fi
}
