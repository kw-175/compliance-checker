#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
PICTURE_HOST="${PICTURE_HOST:-127.0.0.1}"
PICTURE_PORT="${PICTURE_PORT:-19012}"
QWEN35_BASE_URL="${QWEN35_BASE_URL:-http://127.0.0.1:8200/openai/v1}"
QWEN35_HEALTH_URL="${QWEN35_HEALTH_URL:-http://127.0.0.1:8200/health}"
QWEN35_MODEL="${QWEN35_MODEL:-Qwen/Qwen3.5-9B}"
TEXT_API_BASE_URL="${TEXT_API_BASE_URL:-http://127.0.0.1:19002}"
START_TEXT_STACK="${START_TEXT_STACK:-false}"
TEXT_STACK_SCRIPT="${TEXT_STACK_SCRIPT:-$PROJECT/scripts/start_text_8200_stack.sh}"
TEXT_SESSION="${TEXT_SESSION:-text-8200}"
PICTURE_ENV_ACTIVATE="${PICTURE_ENV_ACTIVATE:-$PROJECT/.venvs/picture/bin/activate}"
PICTURE_PADDLEOCR_MODEL_DIR="${PICTURE_PADDLEOCR_MODEL_DIR:-$PROJECT/models/paddleocr_vl/PaddleOCR-VL-1.5}"
PICTURE_OCR_PROVIDER="${PICTURE_OCR_PROVIDER:-paddleocr_vl_api}"
PICTURE_PADDLEOCR_VL_API_URL="${PICTURE_PADDLEOCR_VL_API_URL:-http://127.0.0.1:8217}"
PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS="${PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS:-300}"
PICTURE_SAM3_MODEL_DIR="${PICTURE_SAM3_MODEL_DIR:-$PROJECT/models/facebook/sam3}"
PICTURE_SAM3_API_URL="${PICTURE_SAM3_API_URL:-http://127.0.0.1:8218}"
PICTURE_SAM3_API_TIMEOUT_SECONDS="${PICTURE_SAM3_API_TIMEOUT_SECONDS:-180}"
PICTURE_GPU="${PICTURE_GPU:-2}"
PICTURE_PADDLEOCR_USE_GPU="${PICTURE_PADDLEOCR_USE_GPU:-true}"
PICTURE_PADDLEOCR_DEVICE="${PICTURE_PADDLEOCR_DEVICE:-gpu:0}"
PICTURE_PADDLEOCR_VL_TASK="${PICTURE_PADDLEOCR_VL_TASK:-spotting}"
PICTURE_PADDLEOCR_VL_BACKEND="${PICTURE_PADDLEOCR_VL_BACKEND:-transformers}"
PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS="${PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS:-768}"
PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS="${PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS:-90}"
PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED="${PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED:-true}"
PICTURE_QWEN_OCR_TIMEOUT_SECONDS="${PICTURE_QWEN_OCR_TIMEOUT_SECONDS:-180}"
PICTURE_QWEN_OCR_MAX_TOKENS="${PICTURE_QWEN_OCR_MAX_TOKENS:-4096}"
PICTURE_SAM3_DEVICE="${PICTURE_SAM3_DEVICE:-cuda:0}"
PICTURE_YOLO_DEVICE="${PICTURE_YOLO_DEVICE:-cuda:0}"
PICTURE_SAFETY_PROVIDER="${PICTURE_SAFETY_PROVIDER:-qwen_sam3_safety_fusion}"
PICTURE_VISION_PROVIDER="${PICTURE_VISION_PROVIDER:-qwen_sam3_api_fusion}"
PICTURE_SEGMENTATION_PROVIDER="${PICTURE_SEGMENTATION_PROVIDER:-sam3_api}"
PICTURE_TEXT_API_POLL_INTERVAL_SECONDS="${PICTURE_TEXT_API_POLL_INTERVAL_SECONDS:-0.5}"
PICTURE_QWEN35_VL_MAX_TOKENS="${PICTURE_QWEN35_VL_MAX_TOKENS:-384}"
PICTURE_QWEN35_VL_IMAGE_MAX_SIDE="${PICTURE_QWEN35_VL_IMAGE_MAX_SIDE:-1280}"
PICTURE_QWEN35_VL_IMAGE_JPEG_QUALITY="${PICTURE_QWEN35_VL_IMAGE_JPEG_QUALITY:-85}"

usage() {
  cat <<EOF
Usage: $0 [start|check]

This script starts only the picture service by default. It expects the text
8200 stack to be running already, usually via:
  ATTACH=false bash scripts/start_text_8200_stack.sh start

Set START_TEXT_STACK=true if this script should start the text 8200 stack
when TEXT_API_BASE_URL is not healthy.

Environment overrides:
  PROJECT=$PROJECT
  PICTURE_HOST=$PICTURE_HOST
  PICTURE_PORT=$PICTURE_PORT
  QWEN35_BASE_URL=$QWEN35_BASE_URL
  QWEN35_HEALTH_URL=$QWEN35_HEALTH_URL
  QWEN35_MODEL=$QWEN35_MODEL
  TEXT_API_BASE_URL=$TEXT_API_BASE_URL
  START_TEXT_STACK=$START_TEXT_STACK
  TEXT_STACK_SCRIPT=$TEXT_STACK_SCRIPT
  TEXT_SESSION=$TEXT_SESSION
  PICTURE_ENV_ACTIVATE=$PICTURE_ENV_ACTIVATE
  PICTURE_OCR_PROVIDER=$PICTURE_OCR_PROVIDER
  PICTURE_PADDLEOCR_VL_API_URL=$PICTURE_PADDLEOCR_VL_API_URL
  PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS=$PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS
  PICTURE_PADDLEOCR_MODEL_DIR=$PICTURE_PADDLEOCR_MODEL_DIR
  PICTURE_SAM3_MODEL_DIR=$PICTURE_SAM3_MODEL_DIR
  PICTURE_SAM3_API_URL=$PICTURE_SAM3_API_URL
  PICTURE_SAM3_API_TIMEOUT_SECONDS=$PICTURE_SAM3_API_TIMEOUT_SECONDS
  PICTURE_SAFETY_PROVIDER=$PICTURE_SAFETY_PROVIDER
  PICTURE_VISION_PROVIDER=$PICTURE_VISION_PROVIDER
  PICTURE_SEGMENTATION_PROVIDER=$PICTURE_SEGMENTATION_PROVIDER
  PICTURE_GPU=$PICTURE_GPU
  PICTURE_PADDLEOCR_USE_GPU=$PICTURE_PADDLEOCR_USE_GPU
  PICTURE_PADDLEOCR_DEVICE=$PICTURE_PADDLEOCR_DEVICE
  PICTURE_PADDLEOCR_VL_TASK=$PICTURE_PADDLEOCR_VL_TASK
  PICTURE_PADDLEOCR_VL_BACKEND=$PICTURE_PADDLEOCR_VL_BACKEND
  PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS=$PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS
  PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS=$PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS
  PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED=$PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED
  PICTURE_QWEN_OCR_TIMEOUT_SECONDS=$PICTURE_QWEN_OCR_TIMEOUT_SECONDS
  PICTURE_QWEN_OCR_MAX_TOKENS=$PICTURE_QWEN_OCR_MAX_TOKENS
  PICTURE_SAM3_DEVICE=$PICTURE_SAM3_DEVICE
  PICTURE_YOLO_DEVICE=$PICTURE_YOLO_DEVICE
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

python_bin_for() {
  local activate="$1"
  printf '%s\n' "${activate%/activate}/python"
}

require_python_module() {
  local activate="$1"
  local module_name="$2"
  local python_bin
  python_bin="$(python_bin_for "$activate")"
  [[ -x "$python_bin" ]] || {
    echo "Python executable not found for env: $activate" >&2
    exit 1
  }
  "$python_bin" - <<PY >/dev/null
import importlib.util
import sys
module_name = ${module_name@Q}
if importlib.util.find_spec(module_name) is None:
    sys.exit(1)
PY
}

probe() {
  local url="$1"
  curl -sS --connect-timeout 5 --max-time 15 "$url" >/dev/null
}

check_qwen35_service() {
  local response
  local chat_url="${QWEN35_BASE_URL%/}/chat/completions"
  probe "$QWEN35_HEALTH_URL" || {
    echo "Qwen3.5 health endpoint is not reachable: $QWEN35_HEALTH_URL" >&2
    exit 1
  }
  response="$(
    curl -sS --connect-timeout 5 --max-time 60 \
      "$chat_url" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$QWEN35_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ok\"}],\"temperature\":0,\"max_tokens\":8}"
  )" || {
    echo "Qwen3.5 chat endpoint is not reachable: $chat_url" >&2
    exit 1
  }
  python - "$response" "$QWEN35_MODEL" <<'PY'
import json
import sys

raw = sys.argv[1]
expected_model = sys.argv[2]
try:
    payload = json.loads(raw)
except Exception as exc:
    raise SystemExit(f"Qwen3.5 response is not JSON: {exc}; raw={raw[:200]!r}")

if payload.get("code") and payload.get("code") != 0:
    raise SystemExit(f"Qwen3.5 wrapper returned error: {payload}")

model = str(payload.get("model") or "")
choices = payload.get("choices") or []
if model != expected_model:
    raise SystemExit(f"Qwen3.5 model mismatch: got {model!r}, expected {expected_model!r}")
if not choices:
    raise SystemExit(f"Qwen3.5 response has no choices: {payload}")
print(f"Qwen3.5 check passed: model={model}")
PY
}

ensure_text_api() {
  local health_url="${TEXT_API_BASE_URL%/}/api/v1/health"
  if probe "$health_url"; then
    return
  fi
  if [[ "$START_TEXT_STACK" != "true" ]]; then
    echo "text.api_server is not reachable: $health_url" >&2
    echo "Start it first: ATTACH=false bash scripts/start_text_8200_stack.sh start" >&2
    echo "Or set START_TEXT_STACK=true when starting picture service." >&2
    exit 1
  fi
  [[ -x "$TEXT_STACK_SCRIPT" || -f "$TEXT_STACK_SCRIPT" ]] || {
    echo "TEXT_STACK_SCRIPT not found: $TEXT_STACK_SCRIPT" >&2
    exit 1
  }
  ATTACH=false \
    SESSION="$TEXT_SESSION" \
    QWEN35_BASE_URL="$QWEN35_BASE_URL" \
    QWEN35_HEALTH_URL="$QWEN35_HEALTH_URL" \
    QWEN35_MODEL="$QWEN35_MODEL" \
    bash "$TEXT_STACK_SCRIPT" start || true
  probe "$health_url" || {
    echo "text.api_server did not become ready: $health_url" >&2
    exit 1
  }
}

configure_runtime_env() {
  mkdir -p \
    "$PROJECT/caches/hf/hub" \
    "$PROJECT/caches/hf/transformers" \
    "$PROJECT/caches/modelscope" \
    "$PROJECT/caches/torch" \
    "$PROJECT/caches/matplotlib" || {
      echo "Cannot create picture cache directories under: $PROJECT/caches" >&2
      echo "Check directory permissions or run the script from a context that can write to PROJECT." >&2
      exit 1
    }

  export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
  export CUDA_VISIBLE_DEVICES="$PICTURE_GPU"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_HOME="$PROJECT/caches/hf"
  export HUGGINGFACE_HUB_CACHE="$PROJECT/caches/hf/hub"
  export TRANSFORMERS_CACHE="$PROJECT/caches/hf/transformers"
  export MODELSCOPE_CACHE="$PROJECT/caches/modelscope"
  export TORCH_HOME="$PROJECT/caches/torch"
  export MPLCONFIGDIR="$PROJECT/caches/matplotlib"
  export TOKENIZERS_PARALLELISM=false
}

check_text_api_provider() {
  local health_url="${TEXT_API_BASE_URL%/}/api/v1/health"
  python - "$health_url" "$QWEN35_MODEL" <<'PY'
import json
import sys
import urllib.request

health_url = sys.argv[1]
expected_model = sys.argv[2]
try:
    with urllib.request.urlopen(health_url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"Unable to read text.api_server health payload: {exc}")

provider_mode = str(payload.get("provider_mode") or "")
provider_model = str(payload.get("provider_model") or "")
if provider_mode and provider_mode != "local_model":
    raise SystemExit(
        f"text.api_server provider_mode is {provider_mode!r}, expected 'local_model'."
    )
if provider_model and provider_model != expected_model:
    raise SystemExit(
        f"text.api_server provider_model is {provider_model!r}, expected {expected_model!r}."
    )
print(f"text.api_server provider check passed: mode={provider_mode or 'unknown'}, model={provider_model or 'unknown'}")
PY
}

check_ready() {
  require_cmd curl
  [[ -d "$PROJECT" ]] || { echo "PROJECT does not exist: $PROJECT" >&2; exit 1; }
  [[ -f "$PICTURE_ENV_ACTIVATE" ]] || { echo "PICTURE_ENV_ACTIVATE does not exist: $PICTURE_ENV_ACTIVATE" >&2; exit 1; }
  require_python_module "$PICTURE_ENV_ACTIVATE" "fastapi"
  require_python_module "$PICTURE_ENV_ACTIVATE" "multipart"
  if [[ "$PICTURE_OCR_PROVIDER" != "paddleocr_vl_api" ]]; then
    [[ -d "$PICTURE_PADDLEOCR_MODEL_DIR" ]] || { echo "PICTURE_PADDLEOCR_MODEL_DIR does not exist: $PICTURE_PADDLEOCR_MODEL_DIR" >&2; exit 1; }
  fi
  [[ -d "$PICTURE_SAM3_MODEL_DIR" ]] || { echo "PICTURE_SAM3_MODEL_DIR does not exist: $PICTURE_SAM3_MODEL_DIR" >&2; exit 1; }

  cd "$PROJECT"
  configure_runtime_env
  export COMPLIANCE_COMPLIANCE_PROVIDER_MODE=local
  export COMPLIANCE_LOCAL_COMPLIANCE_BASE_URL="$QWEN35_BASE_URL"
  export COMPLIANCE_LOCAL_COMPLIANCE_MODEL="$QWEN35_MODEL"
  export PICTURE_PADDLEOCR_MODEL_DIR="$PICTURE_PADDLEOCR_MODEL_DIR"
  export PICTURE_OCR_PROVIDER="$PICTURE_OCR_PROVIDER"
  export PICTURE_PADDLEOCR_VL_API_URL="$PICTURE_PADDLEOCR_VL_API_URL"
  export PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS="$PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS"
  export PICTURE_SAM3_MODEL_DIR="$PICTURE_SAM3_MODEL_DIR"
  export PICTURE_GPU="$PICTURE_GPU"
  export PICTURE_PADDLEOCR_USE_GPU="$PICTURE_PADDLEOCR_USE_GPU"
  export PICTURE_PADDLEOCR_DEVICE="$PICTURE_PADDLEOCR_DEVICE"
  export PICTURE_PADDLEOCR_VL_TASK="$PICTURE_PADDLEOCR_VL_TASK"
  export PICTURE_PADDLEOCR_VL_BACKEND="$PICTURE_PADDLEOCR_VL_BACKEND"
  export PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS="$PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS"
  export PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS="$PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS"
  export PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED="$PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED"
  export PICTURE_QWEN_OCR_TIMEOUT_SECONDS="$PICTURE_QWEN_OCR_TIMEOUT_SECONDS"
  export PICTURE_QWEN_OCR_MAX_TOKENS="$PICTURE_QWEN_OCR_MAX_TOKENS"
  export PICTURE_SAM3_DEVICE="$PICTURE_SAM3_DEVICE"
  export PICTURE_SAM3_API_URL="$PICTURE_SAM3_API_URL"
  export PICTURE_SAM3_API_TIMEOUT_SECONDS="$PICTURE_SAM3_API_TIMEOUT_SECONDS"
  export PICTURE_VISION_PROVIDER="$PICTURE_VISION_PROVIDER"
  export PICTURE_SEGMENTATION_PROVIDER="$PICTURE_SEGMENTATION_PROVIDER"
  export PICTURE_YOLO_DEVICE="$PICTURE_YOLO_DEVICE"
  source "$PICTURE_ENV_ACTIVATE"

  check_qwen35_service
  ensure_text_api
  check_text_api_provider

  if [[ "$PICTURE_OCR_PROVIDER" == "paddleocr_vl_api" ]]; then
    probe "${PICTURE_PADDLEOCR_VL_API_URL%/}/docs" || {
      echo "PaddleOCR-VL PaddleX Serving is not reachable: ${PICTURE_PADDLEOCR_VL_API_URL%/}/docs" >&2
      echo "Start it first: bash scripts/start_paddleocr_vl_serving.sh start" >&2
      exit 1
    }
  fi
  if [[ "$PICTURE_VISION_PROVIDER" == "qwen_sam3_api_fusion" || "$PICTURE_VISION_PROVIDER" == "sam3_api" || "$PICTURE_SEGMENTATION_PROVIDER" == "sam3_api" ]]; then
    probe "${PICTURE_SAM3_API_URL%/}/health" || {
      echo "SAM3 API is not reachable: ${PICTURE_SAM3_API_URL%/}/health" >&2
      echo "Start it first: bash scripts/start_sam3_api.sh start" >&2
      exit 1
    }
  fi

  python - <<'PY'
from picture.infra.model_readiness import check_picture_model_readiness
report = check_picture_model_readiness()
missing = []
if report["paddleocr_vl_api"]["provider_configured"]:
    if not report["paddleocr_vl_api"]["endpoint_reachable"]:
        missing.append("PaddleOCR-VL PaddleX Serving API")
else:
    if not report["ocr"]["ready"]:
        missing.append("PaddleOCR-VL files")
    if not report["dependencies"].get("paddleocr_vl_pipeline"):
        missing.append("paddleocr.PaddleOCRVL runtime")
    if not report["dependencies"].get("paddlex_ocr_extra"):
        missing.append("paddlex[ocr] extra dependencies")
if not report["sam3"]["ready"]:
    missing.append("SAM3 files")
if report["sam3_api"]["provider_configured"]:
    if not report["sam3_api"]["endpoint_reachable"]:
        missing.append("SAM3 API")
else:
    if not report["dependencies"].get("transformers_sam3") and not report["dependencies"].get("official_sam3"):
        missing.append("SAM3 Python runtime")
if missing:
    raise SystemExit("Picture model readiness failed: " + ", ".join(missing))
print("Picture local model files and Python runtime are ready.")
PY
}

start_service() {
  check_ready
  cd "$PROJECT"
  configure_runtime_env
  export COMPLIANCE_COMPLIANCE_PROVIDER_MODE=local
  export COMPLIANCE_LOCAL_COMPLIANCE_BASE_URL="$QWEN35_BASE_URL"
  export COMPLIANCE_LOCAL_COMPLIANCE_MODEL="$QWEN35_MODEL"
  export PICTURE_SERVER_HOST="$PICTURE_HOST"
  export PICTURE_SERVER_PORT="$PICTURE_PORT"
  export PICTURE_GPU="$PICTURE_GPU"
  export PICTURE_OCR_PROVIDER="$PICTURE_OCR_PROVIDER"
  export PICTURE_PADDLEOCR_VL_API_URL="$PICTURE_PADDLEOCR_VL_API_URL"
  export PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS="$PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS"
  export PICTURE_PADDLEOCR_MODEL_DIR="$PICTURE_PADDLEOCR_MODEL_DIR"
  export PICTURE_PADDLEOCR_USE_GPU="$PICTURE_PADDLEOCR_USE_GPU"
  export PICTURE_PADDLEOCR_DEVICE="$PICTURE_PADDLEOCR_DEVICE"
  export PICTURE_PADDLEOCR_VL_TASK="$PICTURE_PADDLEOCR_VL_TASK"
  export PICTURE_PADDLEOCR_VL_BACKEND="$PICTURE_PADDLEOCR_VL_BACKEND"
  export PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS="$PICTURE_PADDLEOCR_VL_MAX_NEW_TOKENS"
  export PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS="$PICTURE_PADDLEOCR_VL_GENERATION_TIMEOUT_SECONDS"
  export PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED="$PICTURE_PADDLEOCR_VL_QWEN_FALLBACK_ENABLED"
  export PICTURE_QWEN_OCR_TIMEOUT_SECONDS="$PICTURE_QWEN_OCR_TIMEOUT_SECONDS"
  export PICTURE_QWEN_OCR_MAX_TOKENS="$PICTURE_QWEN_OCR_MAX_TOKENS"
  export PICTURE_PII_PROVIDER=text_compliance
  export PICTURE_TEXT_COMPLIANCE_PROVIDER=text_api
  export PICTURE_TEXT_API_BASE_URL="$TEXT_API_BASE_URL"
  export PICTURE_TEXT_API_POLL_INTERVAL_SECONDS="$PICTURE_TEXT_API_POLL_INTERVAL_SECONDS"
  export PICTURE_SAFETY_PROVIDER="$PICTURE_SAFETY_PROVIDER"
  export PICTURE_QWEN35_VL_MAX_TOKENS="$PICTURE_QWEN35_VL_MAX_TOKENS"
  export PICTURE_QWEN35_VL_IMAGE_MAX_SIDE="$PICTURE_QWEN35_VL_IMAGE_MAX_SIDE"
  export PICTURE_QWEN35_VL_IMAGE_JPEG_QUALITY="$PICTURE_QWEN35_VL_IMAGE_JPEG_QUALITY"
  export PICTURE_VISION_PROVIDER="$PICTURE_VISION_PROVIDER"
  export PICTURE_SAM3_API_URL="$PICTURE_SAM3_API_URL"
  export PICTURE_SAM3_API_TIMEOUT_SECONDS="$PICTURE_SAM3_API_TIMEOUT_SECONDS"
  export PICTURE_YOLO_DEVICE="$PICTURE_YOLO_DEVICE"
  export PICTURE_SEGMENTATION_PROVIDER="$PICTURE_SEGMENTATION_PROVIDER"
  export PICTURE_SAM3_MODEL_DIR="$PICTURE_SAM3_MODEL_DIR"
  export PICTURE_SAM3_DEVICE="$PICTURE_SAM3_DEVICE"
  echo "[picture-local] Starting picture service on ${PICTURE_HOST}:${PICTURE_PORT}"
  echo "[picture-local] Physical GPU=${PICTURE_GPU}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; in-process CUDA/Paddle index is 0"
  echo "[picture-local] OCR provider=${PICTURE_OCR_PROVIDER} paddleocr_vl_api=${PICTURE_PADDLEOCR_VL_API_URL} timeout=${PICTURE_PADDLEOCR_VL_API_TIMEOUT_SECONDS}s"
  echo "[picture-local] Safety provider=${PICTURE_SAFETY_PROVIDER} qwen=${QWEN35_BASE_URL} sam3_api=${PICTURE_SAM3_API_URL}"
  echo "[picture-local] Vision provider=${PICTURE_VISION_PROVIDER} sam3_api=${PICTURE_SAM3_API_URL} sam3_device=${PICTURE_SAM3_DEVICE} model_dir=${PICTURE_SAM3_MODEL_DIR}"
  echo "[picture-local] YOLO device=${PICTURE_YOLO_DEVICE}"
  echo "[picture-local] Text API=${TEXT_API_BASE_URL} poll_interval=${PICTURE_TEXT_API_POLL_INTERVAL_SECONDS}s"
  echo "[picture-local] Qwen endpoint=${QWEN35_BASE_URL} model=${QWEN35_MODEL}"
  python -m uvicorn picture.api.app:app --host "$PICTURE_HOST" --port "$PICTURE_PORT"
}

case "${1:-start}" in
  start)
    start_service
    ;;
  check)
    check_ready
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
