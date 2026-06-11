#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
ENV_DIR="${ENV_DIR:-$PROJECT/qwen-serving/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-$PROJECT/models/Qwen/Qwen3-ASR-0.6B}"
GUARD_MODEL_DIR="${GUARD_MODEL_DIR:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"
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

echo "[qwen-serving] Validating prerequisites"
need_cmd uv
need_path "$PROJECT"
need_path "$ASR_MODEL_DIR"
need_path "$GUARD_MODEL_DIR"

if [[ -d "$ENV_DIR" ]]; then
  BACKUP_DIR="${ENV_DIR}.bak-${BACKUP_SUFFIX}"
  echo "[qwen-serving] Backing up existing env to $BACKUP_DIR"
  mv "$ENV_DIR" "$BACKUP_DIR"
fi

echo "[qwen-serving] Creating virtualenv at $ENV_DIR"
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"

echo "[qwen-serving] Installing CUDA 12.8 PyTorch stack"
uv pip install \
  --python "$ENV_DIR/bin/python" \
  --extra-index-url "$PYTORCH_INDEX_URL" \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  "torchaudio==2.10.0"

echo "[qwen-serving] Installing shared Qwen runtime packages"
uv pip install \
  --python "$ENV_DIR/bin/python" \
  "qwen-asr==0.0.6" \
  "pydantic>=2.10,<3.0" \
  "pydantic-settings>=2.6,<3.0" \
  "transformers==4.57.6" \
  "accelerate==1.12.0" \
  "fastapi>=0.115,<1.0" \
  "uvicorn[standard]>=0.30,<1.0" \
  "httpx>=0.28,<1.0" \
  "librosa>=0.11,<1.0" \
  "soundfile>=0.13,<1.0" \
  "sox>=1.5,<2.0" \
  "sentencepiece>=0.2,<1.0" \
  "safetensors>=0.5,<1.0" \
  "pillow>=10,<12" \
  "numpy<2.5"

echo "[qwen-serving] Running import smoke checks"
cd "$PROJECT"
"$ENV_DIR/bin/python" - <<'PY'
from pathlib import Path

import torch
import torchaudio
import torchvision
import transformers
import qwen_asr

from audio.adapters import qwen_asr_adapter, qwen_guard_adapter
from audio.config.settings import get_settings
from transformers import AutoConfig, AutoTokenizer

project = Path("/data/kw/compliance-checker")
asr_model_dir = Path(project / "models/Qwen/Qwen3-ASR-0.6B")
guard_model_dir = Path(project / "models/Qwen/Qwen3Guard-Gen-0.6B")

assert asr_model_dir.is_dir(), f"Missing ASR model dir: {asr_model_dir}"
assert guard_model_dir.is_dir(), f"Missing guard model dir: {guard_model_dir}"

settings = get_settings()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("transformers", transformers.__version__)
print("qwen_asr", getattr(qwen_asr, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
print("qwen_asr_device", qwen_asr_adapter.resolve_device(settings))
print("qwen_guard_device", qwen_guard_adapter.resolve_device(settings))

# Guard is a standard text model; tokenizer/config loading is enough for smoke validation.
guard_config = AutoConfig.from_pretrained(guard_model_dir, trust_remote_code=True)
guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_dir, trust_remote_code=True)
print("guard_model_type", guard_config.model_type)
print("guard_vocab_size", getattr(guard_tokenizer, "vocab_size", "unknown"))
PY

cat <<EOF
[qwen-serving] Rebuild complete

Activate:
  source "$ENV_DIR/bin/activate"

ASR service:
  source "$ENV_DIR/bin/activate"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export COMPLIANCE_QWEN_ASR_MODEL="$ASR_MODEL_DIR"
  export COMPLIANCE_QWEN_ASR_DEVICE=cuda
  python -m uvicorn audio.adapters.qwen_asr_service:app --host 127.0.0.1 --port 8011

Guard service:
  source "$ENV_DIR/bin/activate"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export COMPLIANCE_QWEN_GUARD_MODEL="$GUARD_MODEL_DIR"
  export COMPLIANCE_QWEN_GUARD_DEVICE=cuda
  python -m uvicorn audio.adapters.qwen_guard_service:app --host 127.0.0.1 --port 8012
EOF
