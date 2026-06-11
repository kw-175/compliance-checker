#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
SAM3_ENV_ACTIVATE="${SAM3_ENV_ACTIVATE:-$PROJECT/.venvs/sam3/bin/activate}"
SAM3_MODEL_DIR="${SAM3_MODEL_DIR:-$PROJECT/models/facebook/sam3}"
SAM3_HOST="${SAM3_HOST:-0.0.0.0}"
SAM3_PORT="${SAM3_PORT:-8218}"
SAM3_GPU="${SAM3_GPU:-2}"
SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
SAM3_CONFIDENCE="${SAM3_CONFIDENCE:-0.35}"
SAM3_USE_BF16_AUTOCAST="${SAM3_USE_BF16_AUTOCAST:-1}"
PID_DIR="${PID_DIR:-$PROJECT/tmp/pids}"
LOG_DIR="${LOG_DIR:-$PROJECT/tmp/logs}"

usage() {
  cat <<EOF
Usage: $0 [start|stop|check]

Starts the official SAM3 image API service:
  http://127.0.0.1:${SAM3_PORT}/v1/sam3/detect
  http://127.0.0.1:${SAM3_PORT}/v1/sam3/refine
  http://127.0.0.1:${SAM3_PORT}/v1/sam3/video-track
EOF
}

check_ready() {
  [[ -d "$PROJECT" ]] || { echo "PROJECT does not exist: $PROJECT" >&2; exit 1; }
  [[ -f "$SAM3_ENV_ACTIVATE" ]] || { echo "SAM3_ENV_ACTIVATE does not exist: $SAM3_ENV_ACTIVATE" >&2; exit 1; }
  [[ -d "$SAM3_MODEL_DIR" ]] || { echo "SAM3_MODEL_DIR does not exist: $SAM3_MODEL_DIR" >&2; exit 1; }
  source "$SAM3_ENV_ACTIVATE"
  export CUDA_VISIBLE_DEVICES="$SAM3_GPU"
  python - <<'PY'
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
print("SAM3 Python runtime is ready.")
PY
  python -c "import fastapi, uvicorn, pydantic; print('SAM3 API dependencies are ready.')"
}

start_service() {
  check_ready
  mkdir -p "$PID_DIR" "$LOG_DIR"
  local pid_file="$PID_DIR/sam3-api.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "SAM3 API already running with pid $(cat "$pid_file")"
    return 0
  fi
  cd "$PROJECT"
  source "$SAM3_ENV_ACTIVATE"
  export SAM3_MODEL_DIR="$SAM3_MODEL_DIR"
  export CUDA_VISIBLE_DEVICES="$SAM3_GPU"
  export SAM3_DEVICE="$SAM3_DEVICE"
  export SAM3_CONFIDENCE="$SAM3_CONFIDENCE"
  export SAM3_USE_BF16_AUTOCAST="$SAM3_USE_BF16_AUTOCAST"
  echo "[sam3-api] Starting on ${SAM3_HOST}:${SAM3_PORT} physical_gpu=${SAM3_GPU} device=${SAM3_DEVICE} bf16_autocast=${SAM3_USE_BF16_AUTOCAST} model_dir=${SAM3_MODEL_DIR}"
  nohup python -m uvicorn ops.sam3_api:app --host "$SAM3_HOST" --port "$SAM3_PORT" \
    >"$LOG_DIR/sam3-api.log" 2>&1 &
  echo "$!" > "$pid_file"
  for _ in $(seq 1 60); do
    if curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${SAM3_PORT}/health" >/dev/null 2>&1; then
      echo "[sam3-api] Ready: http://127.0.0.1:${SAM3_PORT}"
      return 0
    fi
    sleep 2
  done
  echo "SAM3 API did not become ready. See $LOG_DIR/sam3-api.log" >&2
  exit 1
}

stop_service() {
  local pid_file="$PID_DIR/sam3-api.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    kill "$(cat "$pid_file")"
    echo "Stopped SAM3 API pid $(cat "$pid_file")"
  fi
  rm -f "$pid_file"
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
