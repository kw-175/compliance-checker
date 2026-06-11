#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
ENV_DIR="${ENV_DIR:-$PROJECT/.venvs/qwen-hardcase}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
HARD_CASE_MODEL_DIR="${HARD_CASE_MODEL_DIR:-$PROJECT/models/Qwen/Qwen3.5-9B}"
TRANSFORMERS_SOURCE="${TRANSFORMERS_SOURCE:-transformers[serving] @ git+https://github.com/huggingface/transformers.git@main}"
BACKUP_SUFFIX="${BACKUP_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_path() {
  if [[ ! -e "$1" ]]; then
    echo "Required path not found: $1" >&2
    exit 1
  fi
}

echo "[qwen-hardcase] Validating prerequisites"
need_cmd uv
need_path "$PROJECT"
need_path "$HARD_CASE_MODEL_DIR"

if [[ -d "$ENV_DIR" ]]; then
  BACKUP_DIR="${ENV_DIR}.bak-${BACKUP_SUFFIX}"
  echo "[qwen-hardcase] Backing up existing env to $BACKUP_DIR"
  mv "$ENV_DIR" "$BACKUP_DIR"
fi

echo "[qwen-hardcase] Creating virtualenv at $ENV_DIR"
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"

echo "[qwen-hardcase] Installing CUDA 12.8 PyTorch stack"
uv pip install \
  --python "$ENV_DIR/bin/python" \
  --extra-index-url "$PYTORCH_INDEX_URL" \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  "torchaudio==2.10.0"

echo "[qwen-hardcase] Installing Qwen3.5 runtime packages"
uv pip install \
  --python "$ENV_DIR/bin/python" \
  "$TRANSFORMERS_SOURCE" \
  "accelerate>=1.12,<2.0" \
  "pydantic>=2.10,<3.0" \
  "pydantic-settings>=2.6,<3.0" \
  "fastapi>=0.115,<1.0" \
  "uvicorn[standard]>=0.30,<1.0" \
  "httpx>=0.28,<1.0" \
  "pillow>=10,<12" \
  "sentencepiece>=0.2,<1.0" \
  "safetensors>=0.5,<1.0" \
  "numpy<2.5"

echo "[qwen-hardcase] Running smoke checks"
cd "$PROJECT"
"$ENV_DIR/bin/python" - <<'PY'
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig

project = Path("/data/kw/compliance-checker")
model_dir = project / "models/Qwen/Qwen3.5-9B"
assert model_dir.is_dir(), f"Missing hard-case model dir: {model_dir}"

cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("hard_case_model_type", cfg.model_type)
print("hard_case_architectures", getattr(cfg, "architectures", []))
PY

cat <<EOF
[qwen-hardcase] Rebuild complete

Use with audio start script:
  export HARD_CASE_ENV_ACTIVATE="$ENV_DIR/bin/activate"
  bash "$PROJECT/scripts/start_audio_a100.sh" restart

Optional alternate transformers source:
  export TRANSFORMERS_SOURCE='transformers[serving] @ git+https://<reachable-mirror-or-fork>.git@main'
  export TRANSFORMERS_SOURCE='/absolute/path/to/local/transformers'
  export TRANSFORMERS_SOURCE='/absolute/path/to/transformers_whl_or_sdist'
EOF
