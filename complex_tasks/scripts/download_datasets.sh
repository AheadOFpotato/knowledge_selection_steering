#!/usr/bin/env bash
# Download all taxonomy datasets into conflict_taxonomy_steer/data/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Keep large raw benchmark assets off the repository volume.  Existing local
# layouts remain supported when the variable is not set.
DATA="${CONFLICT_TAXONOMY_DATA_ROOT:-$ROOT/data}"
export CONFLICT_TAXONOMY_DATA_ROOT="$DATA"
mkdir -p "$DATA"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# This host often needs the HF mirror; override with HF_ENDPOINT= if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GH_MIRROR_PREFIX="${GH_MIRROR_PREFIX:-https://ghfast.top/}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------- Content / Factual: NQ-Swap ----------
download_nqswap() {
  local out="$DATA/content_factual/NQ-Swap"
  mkdir -p "$out"
  if [[ -f "$out/dev.jsonl" ]]; then
    log "NQ-Swap already present ($out)"
    return 0
  fi
  log "Downloading NQ-Swap from Hugging Face (pminervini/NQ-Swap)..."
  python3 - <<'PY'
from pathlib import Path
from datasets import load_dataset
import json
import os
out = Path(os.environ["CONFLICT_TAXONOMY_DATA_ROOT"]) / "content_factual" / "NQ-Swap"
out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("pminervini/NQ-Swap")
for split, subset in ds.items():
    path = out / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    print(f"wrote {path} n={len(subset)}")
PY
}

# ---------- Content / RAG: RGB ----------
download_rgb() {
  local out="$DATA/content_rag/RGB"
  mkdir -p "$out"
  if [[ -f "$out/data/en_fact.json" ]] || [[ -f "$out/en_fact.json" ]]; then
    log "RGB already present ($out)"
    return 0
  fi
  log "Cloning RGB (chen700564/RGB)..."
  rm -rf "$out/.tmp_rgb"
  git clone --depth 1 https://github.com/chen700564/RGB.git "$out/.tmp_rgb"
  if [[ -d "$out/.tmp_rgb/data" ]]; then
    cp -a "$out/.tmp_rgb/data/." "$out/"
  else
    cp -a "$out/.tmp_rgb/." "$out/"
  fi
  rm -rf "$out/.tmp_rgb"
  # keep a marker README
  [[ -f "$out/readme.md" ]] || [[ -f "$out/README.md" ]] || true
  log "RGB done"
}

# ---------- Content / RAG: CRAG ----------
download_crag() {
  local out="$DATA/content_rag/CRAG"
  mkdir -p "$out"
  if [[ -f "$out/crag_task_1_and_2_dev_v4.jsonl" ]]; then
    log "CRAG already present ($out)"
    return 0
  fi
  log "Downloading CRAG task1/2 (facebookresearch/CRAG)..."
  local raw="https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"
  local url="${GH_MIRROR_PREFIX}${raw}"
  wget -c --show-progress -O "$out/crag_task_1_and_2_dev_v4.jsonl.bz2" "$url" \
    || wget -c --show-progress -O "$out/crag_task_1_and_2_dev_v4.jsonl.bz2" "$raw"
  bunzip2 -fk "$out/crag_task_1_and_2_dev_v4.jsonl.bz2"
  # also clone repo docs (lightweight) for schema
  if [[ ! -d "$out/repo_docs" ]]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/CRAG.git "$out/.tmp_crag" || true
    if [[ -d "$out/.tmp_crag" ]]; then
      (cd "$out/.tmp_crag" && git sparse-checkout set docs README.md) || true
      mkdir -p "$out/repo_docs"
      cp -a "$out/.tmp_crag/docs/." "$out/repo_docs/" 2>/dev/null || true
      cp -a "$out/.tmp_crag/README.md" "$out/repo_docs/" 2>/dev/null || true
      rm -rf "$out/.tmp_crag"
    fi
  fi
  log "CRAG done"
}

# ---------- Rule / Guideline: GuideBench ----------
download_guidebench() {
  local out="$DATA/rule_guideline/GuideBench"
  mkdir -p "$out"
  if [[ -d "$out/data" ]] && [[ -n "$(ls -A "$out/data" 2>/dev/null || true)" ]]; then
    log "GuideBench already present ($out)"
    return 0
  fi
  log "Cloning GuideBench (Dlxxx/GuideBench)..."
  rm -rf "$out/.tmp"
  git clone --depth 1 https://github.com/Dlxxx/GuideBench.git "$out/.tmp"
  cp -a "$out/.tmp/." "$out/"
  rm -rf "$out/.tmp/.git" "$out/.tmp"
  log "GuideBench done"
}

# ---------- Rule / Legal: LegalBench (ContractNLI tasks) + ContractNLI ----------
download_legal() {
  local lb="$DATA/rule_legal/LegalBench"
  local cn="$DATA/rule_legal/ContractNLI"
  mkdir -p "$lb" "$cn"
  if [[ -f "$lb/contract_nli_limited_use_test.jsonl" ]]; then
    log "LegalBench ContractNLI tasks already present"
  else
    log "Downloading LegalBench ContractNLI task subset from HF..."
    python3 - <<'PY'
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names
import json
import os

out = Path(os.environ["CONFLICT_TAXONOMY_DATA_ROOT"]) / "rule_legal" / "LegalBench"
out.mkdir(parents=True, exist_ok=True)
configs = get_dataset_config_names("nguha/legalbench")
cn_cfgs = [c for c in configs if c.startswith("contract_nli_")]
print(f"found {len(cn_cfgs)} contract_nli configs")
# download all contract_nli_* ; also keep a few other high-signal legal rule tasks
extra = [c for c in configs if c in {
    "cua_v1_contracts", "definition_classification", "international_citizenship_questions",
    "privacy_policy_qa", "proa", "ssla_company_persons",
}]
todo = sorted(set(cn_cfgs + extra))
manifest = []
for cfg in todo:
    try:
        ds = load_dataset("nguha/legalbench", cfg)
    except Exception as e:
        print(f"skip {cfg}: {e}")
        continue
    for split, subset in ds.items():
        path = out / f"{cfg}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        manifest.append({"config": cfg, "split": split, "n": len(subset), "path": str(path.name)})
        print(f"wrote {path.name} n={len(subset)}")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"manifest n={len(manifest)}")
PY
  fi

  # Original ContractNLI release (document-level NLI JSON)
  if [[ -f "$cn/train.json" ]] || [[ -f "$cn/dev.json" ]] || [[ -f "$cn/test.json" ]]; then
    log "ContractNLI already present"
  else
    log "Downloading ContractNLI from Stanford NLP release..."
    # Prefer HF mirror if available; else Stanford GitHub pages asset
    python3 - <<'PY'
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import shutil
import os
out = Path(os.environ["CONFLICT_TAXONOMY_DATA_ROOT"]) / "rule_legal" / "ContractNLI"
out.mkdir(parents=True, exist_ok=True)
# Try common HF repos / fallbacks
candidates = [
    ("kiddothe2b/contract-nli", None),
]
ok = False
for repo, _ in candidates:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print(repo, "files", files[:20])
        for fn in files:
            if fn.endswith((".json", ".jsonl", ".zip", ".tar.gz")):
                p = hf_hub_download(repo, fn, repo_type="dataset", local_dir=str(out))
                print("got", p)
                ok = True
    except Exception as e:
        print("fail", repo, e)
if not ok:
    # Stanford zip via known URL pattern used by papers
    import urllib.request
    urls = [
        "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
        "https://github.com/stanfordnlp/contract-nli/releases/download/v1.0/contract-nli.zip",
    ]
    for u in urls:
        try:
            dest = out / "contract-nli.zip"
            print("trying", u)
            urllib.request.urlretrieve(u, dest)
            print("downloaded", dest, dest.stat().st_size)
            ok = True
            break
        except Exception as e:
            print("url fail", u, e)
print("contractnli_ok", ok)
PY
    if [[ -f "$cn/contract-nli.zip" ]]; then
      (cd "$cn" && unzip -qo contract-nli.zip)
    fi
  fi
}

# ---------- Policy / API: ToolSandbox ----------
download_toolsandbox() {
  local out="$DATA/policy_api/ToolSandbox"
  mkdir -p "$out"
  if [[ -d "$out/tool_sandbox" ]] || [[ -f "$out/README.md" && -d "$out/scenarios" ]]; then
    log "ToolSandbox already present"
    return 0
  fi
  log "Cloning ToolSandbox (apple/ToolSandbox)..."
  rm -rf "$out/.tmp"
  git clone --depth 1 https://github.com/apple/ToolSandbox.git "$out/.tmp"
  cp -a "$out/.tmp/." "$out/"
  rm -rf "$out/.tmp/.git" "$out/.tmp"
  log "ToolSandbox done"
}

# ---------- Policy / API: API-Bank ----------
download_apibank() {
  local out="$DATA/policy_api/API-Bank"
  mkdir -p "$out"
  if [[ -d "$out/lv1-lv2-samples" ]] || [[ -f "$out/test.json" ]] || [[ -n "$(ls "$out"/*.jsonl 2>/dev/null || true)" ]]; then
    log "API-Bank already present"
    return 0
  fi
  log "Downloading API-Bank from HF (liminghao1630/API-Bank) + GitHub subtree..."
  python3 - <<'PY'
from pathlib import Path
from datasets import load_dataset
import json
import os
out = Path(os.environ["CONFLICT_TAXONOMY_DATA_ROOT"]) / "policy_api" / "API-Bank"
out.mkdir(parents=True, exist_ok=True)
try:
    ds = load_dataset("liminghao1630/API-Bank")
    for split, subset in ds.items():
        path = out / f"hf_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"wrote {path} n={len(subset)}")
except Exception as e:
    print("HF API-Bank failed:", e)
PY
  # Also pull runnable evaluation samples from DAMO-ConvAI
  if [[ ! -d "$out/lv1-lv2-samples" ]]; then
    rm -rf "$out/.tmp_repo"
    git clone --depth 1 --filter=blob:none --sparse https://github.com/AlibabaResearch/DAMO-ConvAI.git "$out/.tmp_repo" || true
    if [[ -d "$out/.tmp_repo" ]]; then
      (cd "$out/.tmp_repo" && git sparse-checkout set api-bank) || true
      if [[ -d "$out/.tmp_repo/api-bank" ]]; then
        cp -a "$out/.tmp_repo/api-bank/." "$out/"
      fi
      rm -rf "$out/.tmp_repo"
    fi
  fi
  log "API-Bank done"
}

# ---------- Policy / Personalization: LongMemEval ----------
download_longmemeval() {
  local out="$DATA/policy_personalization/LongMemEval"
  mkdir -p "$out"
  if [[ -f "$out/questions.jsonl" ]] || [[ -f "$out/longmemeval_s_cleaned.json" ]] || [[ -f "$out/data/longmemeval_s.json" ]]; then
    log "LongMemEval already present"
  else
    log "Downloading LongMemEval (original) + LongMemEval-V2..."
    # Original LongMemEval data often on HF / Google Drive; try common HF ids
    python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
import os
out = Path(os.environ["CONFLICT_TAXONOMY_DATA_ROOT"]) / "policy_personalization" / "LongMemEval"
out.mkdir(parents=True, exist_ok=True)
# Try original LongMemEval releases
for repo in ["xiaowu0162/longmemeval", "xiaowu0162/LongMemEval"]:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print("repo", repo, "nfiles", len(files))
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(out / "original"))
        print("downloaded", repo)
        break
    except Exception as e:
        print("skip", repo, e)

# LongMemEval-V2 (text-only core files; skip huge screenshot tarballs if possible)
v2 = out / "v2"
v2.mkdir(parents=True, exist_ok=True)
try:
    from huggingface_hub import hf_hub_download
    for fn in ["questions.jsonl", "trajectories.jsonl", "SCHEMA.md", "DATA_CARD.md", "LICENSE"]:
        try:
            p = hf_hub_download("xiaowu0162/longmemeval-v2", fn, repo_type="dataset", local_dir=str(v2))
            print("got", p)
        except Exception as e:
            print("missing", fn, e)
    # haystacks if present
    for fn in ["haystacks/lme_v2_small.json", "haystacks/lme_v2_medium.json"]:
        try:
            p = hf_hub_download("xiaowu0162/longmemeval-v2", fn, repo_type="dataset", local_dir=str(v2))
            print("got", p)
        except Exception as e:
            print("missing", fn, e)
except Exception as e:
    print("v2 fail", e)
PY
  fi
  # Also clone harness repos (shallow) for loaders
  if [[ ! -d "$out/LongMemEval-code" ]]; then
    git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git "$out/LongMemEval-code" || true
  fi
  if [[ ! -d "$out/LongMemEval-V2-code" ]]; then
    git clone --depth 1 https://github.com/xiaowu0162/LongMemEval-V2.git "$out/LongMemEval-V2-code" || true
  fi
  log "LongMemEval done"
}

# ---------- Policy / Personalization: Mem2ActBench ----------
download_mem2act() {
  local out="$DATA/policy_personalization/Mem2ActBench"
  mkdir -p "$out"
  if [[ -d "$out/Mem2ActBench" ]] || [[ -f "$out/qa_dataset.jsonl" ]] || [[ -f "$out/README.md" && -d "$out/prompts" ]]; then
    log "Mem2ActBench already present"
    return 0
  fi
  log "Cloning Mem2ActBench (Cantaloupe-M/Mem2ActBench)..."
  rm -rf "$out/.tmp"
  git clone --depth 1 https://github.com/Cantaloupe-M/Mem2ActBench.git "$out/.tmp"
  cp -a "$out/.tmp/." "$out/"
  rm -rf "$out/.tmp/.git" "$out/.tmp"
  log "Mem2ActBench done"
}

main() {
  local target="${1:-all}"
  case "$target" in
    nqswap) download_nqswap ;;
    rgb) download_rgb ;;
    crag) download_crag ;;
    guidebench) download_guidebench ;;
    legal) download_legal ;;
    toolsandbox) download_toolsandbox ;;
    apibank) download_apibank ;;
    longmemeval) download_longmemeval ;;
    mem2act) download_mem2act ;;
    all)
      download_nqswap
      download_rgb
      download_crag
      download_guidebench
      download_legal
      download_toolsandbox
      download_apibank
      download_longmemeval
      download_mem2act
      ;;
    *)
      echo "Usage: $0 [all|nqswap|rgb|crag|guidebench|legal|toolsandbox|apibank|longmemeval|mem2act]"
      exit 1
      ;;
  esac
  log "Finished target=$target"
  du -sh "$DATA"/*/* 2>/dev/null || true
}

main "$@"
