#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
PADDLEOCR_VL_PIPELINE_ENV="${PADDLEOCR_VL_PIPELINE_ENV:-$PROJECT/.venvs/paddleocr-vl}"
PADDLEOCR_VL_VLLM_ENV="${PADDLEOCR_VL_VLLM_ENV:-$PROJECT/.venvs/paddleocr-vl-vllm}"
PADDLEOCR_VL_ENV_ACTIVATE="${PADDLEOCR_VL_ENV_ACTIVATE:-$PADDLEOCR_VL_PIPELINE_ENV/bin/activate}"
PADDLEOCR_VL_MODEL_DIR="${PADDLEOCR_VL_MODEL_DIR:-$PROJECT/models/paddleocr_vl/PaddleOCR-VL-1.5}"
PADDLEOCR_VL_VLLM_HOST="${PADDLEOCR_VL_VLLM_HOST:-0.0.0.0}"
PADDLEOCR_VL_VLLM_PORT="${PADDLEOCR_VL_VLLM_PORT:-8216}"
PADDLEOCR_VL_API_HOST="${PADDLEOCR_VL_API_HOST:-0.0.0.0}"
PADDLEOCR_VL_API_PORT="${PADDLEOCR_VL_API_PORT:-8217}"
PADDLEOCR_VL_GPU="${PADDLEOCR_VL_GPU:-2}"
PADDLEOCR_VL_VLLM_CONFIG="${PADDLEOCR_VL_VLLM_CONFIG:-$PROJECT/ops/paddleocr_vl_vllm_config.yaml}"
PADDLEOCR_VL_PIPELINE_CONFIG="${PADDLEOCR_VL_PIPELINE_CONFIG:-$PROJECT/ops/PaddleOCR-VL-serving.yaml}"
PID_DIR="${PID_DIR:-$PROJECT/tmp/pids}"
LOG_DIR="${LOG_DIR:-$PROJECT/tmp/logs}"

usage() {
  cat <<EOF
Usage: $0 [start|stop|check]

Starts the official PaddleOCR-VL API route:
  vLLM env        : ${PADDLEOCR_VL_VLLM_ENV}
  PaddleX env     : ${PADDLEOCR_VL_PIPELINE_ENV}
  PaddleX Serving :${PADDLEOCR_VL_API_PORT} /layout-parsing
    -> PaddleOCR-VL pipeline
    -> VLM recognition accelerated by vLLM server :${PADDLEOCR_VL_VLLM_PORT}

Before first start, generate the PaddleX pipeline config and set:
  VLRecognition.genai_config.backend: vllm-server
  VLRecognition.genai_config.server_url: http://127.0.0.1:${PADDLEOCR_VL_VLLM_PORT}/v1
  VLRecognition.genai_config.max_concurrency: 128
  Serving.visualize: False

Config path:
  PADDLEOCR_VL_PIPELINE_CONFIG=${PADDLEOCR_VL_PIPELINE_CONFIG}
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

configure_env() {
  mkdir -p "$PID_DIR" "$LOG_DIR" "$PROJECT/caches/paddle" "$PROJECT/caches/hf" "$PROJECT/caches/modelscope"
  export CUDA_VISIBLE_DEVICES="$PADDLEOCR_VL_GPU"
  export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
  export HF_HOME="$PROJECT/caches/hf"
  export MODELSCOPE_CACHE="$PROJECT/caches/modelscope"
}

check_ready() {
  require_cmd curl
  [[ -d "$PROJECT" ]] || { echo "PROJECT does not exist: $PROJECT" >&2; exit 1; }
  [[ -d "$PADDLEOCR_VL_MODEL_DIR" ]] || { echo "PADDLEOCR_VL_MODEL_DIR does not exist: $PADDLEOCR_VL_MODEL_DIR" >&2; exit 1; }
  [[ -x "$PADDLEOCR_VL_VLLM_ENV/bin/paddleocr" ]] || {
    echo "PaddleOCR vLLM CLI does not exist: $PADDLEOCR_VL_VLLM_ENV/bin/paddleocr" >&2
    exit 1
  }
  [[ -x "$PADDLEOCR_VL_VLLM_ENV/bin/python" ]] || {
    echo "PaddleOCR vLLM Python does not exist: $PADDLEOCR_VL_VLLM_ENV/bin/python" >&2
    exit 1
  }
  [[ -x "$PADDLEOCR_VL_PIPELINE_ENV/bin/paddlex" ]] || {
    echo "PaddleX Serving CLI does not exist: $PADDLEOCR_VL_PIPELINE_ENV/bin/paddlex" >&2
    exit 1
  }
  [[ -x "$PADDLEOCR_VL_PIPELINE_ENV/bin/python" ]] || {
    echo "PaddleOCR pipeline Python does not exist: $PADDLEOCR_VL_PIPELINE_ENV/bin/python" >&2
    exit 1
  }
  [[ -f "$PADDLEOCR_VL_VLLM_CONFIG" ]] || { echo "PADDLEOCR_VL_VLLM_CONFIG does not exist: $PADDLEOCR_VL_VLLM_CONFIG" >&2; exit 1; }
  [[ -f "$PADDLEOCR_VL_PIPELINE_CONFIG" ]] || {
    echo "PADDLEOCR_VL_PIPELINE_CONFIG does not exist: $PADDLEOCR_VL_PIPELINE_CONFIG" >&2
    echo "Generate it with: paddlex --get_pipeline_config PaddleOCR-VL" >&2
    echo "Then set VLRecognition.genai_config to the vLLM server at http://127.0.0.1:${PADDLEOCR_VL_VLLM_PORT}/v1." >&2
    exit 1
  }
  configure_env
  "$PADDLEOCR_VL_VLLM_ENV/bin/python" - <<'PY'
from importlib.metadata import version
from paddlex.utils import deps
required = ["torch", "vllm", "xformers", "flash-attn", "paddlex", "paddleocr"]
for package in required:
    version(package)
if not deps.is_genai_engine_plugin_available("vllm-server"):
    raise SystemExit("PaddleX vllm-server plugin is not available in the vLLM env.")
print("PaddleOCR-VL vLLM env is ready.")
PY
  "$PADDLEOCR_VL_PIPELINE_ENV/bin/python" - <<'PY'
import paddle
from paddleocr import PaddleOCRVL
import paddlex
print("PaddleOCR-VL pipeline env is ready.")
PY
}

start_service() {
  check_ready

  local vllm_pid="$PID_DIR/paddleocr-vl-vllm.pid"
  local api_pid="$PID_DIR/paddleocr-vl-api.pid"
  if [[ -f "$vllm_pid" ]] && kill -0 "$(cat "$vllm_pid")" 2>/dev/null; then
    echo "PaddleOCR-VL vLLM server already running with pid $(cat "$vllm_pid")"
  else
    echo "[paddleocr-vl] Starting VLM vLLM server on ${PADDLEOCR_VL_VLLM_HOST}:${PADDLEOCR_VL_VLLM_PORT}"
    nohup "$PADDLEOCR_VL_VLLM_ENV/bin/paddleocr" genai_server \
      --model_name PaddleOCR-VL-1.5-0.9B \
      --model_dir "$PADDLEOCR_VL_MODEL_DIR" \
      --host "$PADDLEOCR_VL_VLLM_HOST" \
      --port "$PADDLEOCR_VL_VLLM_PORT" \
      --backend vllm \
      --backend_config "$PADDLEOCR_VL_VLLM_CONFIG" \
      >"$LOG_DIR/paddleocr-vl-vllm.log" 2>&1 &
    echo "$!" > "$vllm_pid"
  fi

  echo "[paddleocr-vl] Waiting for VLM server"
  for _ in $(seq 1 120); do
    if curl -sS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${PADDLEOCR_VL_VLLM_PORT}/v1/models" >/dev/null; then
      break
    fi
    sleep 2
  done
  curl -sS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${PADDLEOCR_VL_VLLM_PORT}/v1/models" >/dev/null || {
    echo "PaddleOCR-VL vLLM server did not become ready. See $LOG_DIR/paddleocr-vl-vllm.log" >&2
    exit 1
  }

  if [[ -f "$api_pid" ]] && kill -0 "$(cat "$api_pid")" 2>/dev/null; then
    echo "PaddleOCR-VL API server already running with pid $(cat "$api_pid")"
  else
    echo "[paddleocr-vl] Starting PaddleX Serving API on ${PADDLEOCR_VL_API_HOST}:${PADDLEOCR_VL_API_PORT}"
    nohup "$PADDLEOCR_VL_PIPELINE_ENV/bin/paddlex" --serve \
      --pipeline "$PADDLEOCR_VL_PIPELINE_CONFIG" \
      --host "$PADDLEOCR_VL_API_HOST" \
      --port "$PADDLEOCR_VL_API_PORT" \
      >"$LOG_DIR/paddleocr-vl-api.log" 2>&1 &
    echo "$!" > "$api_pid"
  fi

  echo "[paddleocr-vl] Waiting for API server"
  for _ in $(seq 1 120); do
    if curl -sS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${PADDLEOCR_VL_API_PORT}/docs" >/dev/null; then
      echo "[paddleocr-vl] Ready: http://127.0.0.1:${PADDLEOCR_VL_API_PORT}/layout-parsing"
      return 0
    fi
    sleep 2
  done
  echo "PaddleOCR-VL API server did not become ready. See $LOG_DIR/paddleocr-vl-api.log" >&2
  exit 1
}

stop_service() {
  for name in paddleocr-vl-api paddleocr-vl-vllm; do
    local pid_file="$PID_DIR/${name}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      kill "$(cat "$pid_file")"
      echo "Stopped $name pid $(cat "$pid_file")"
    fi
    rm -f "$pid_file"
  done
}

case "${1:-start}" in
  start)
    start_service
    ;;
  stop)
    stop_service
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
