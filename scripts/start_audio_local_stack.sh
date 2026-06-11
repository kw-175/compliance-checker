#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
SESSION="${SESSION:-audio-local}"
ATTACH="${ATTACH:-false}"

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/audio_local_stack_tmux}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_local_output}"

TEXT_API_HOST="${TEXT_API_HOST:-127.0.0.1}"
TEXT_API_PORT="${TEXT_API_PORT:-19002}"
TEXT_API_BASE_URL="${TEXT_API_BASE_URL:-http://${TEXT_API_HOST}:${TEXT_API_PORT}}"
START_TEXT_STACK="${START_TEXT_STACK:-true}"
STOP_TEXT_STACK="${STOP_TEXT_STACK:-false}"
TEXT_STACK_SCRIPT="${TEXT_STACK_SCRIPT:-$PROJECT/scripts/start_text_8200_stack.sh}"
TEXT_SESSION="${TEXT_SESSION:-text-8200}"
TEXT_QWEN35_BASE_URL="${TEXT_QWEN35_BASE_URL:-http://127.0.0.1:8200/openai/v1}"
TEXT_QWEN35_HEALTH_URL="${TEXT_QWEN35_HEALTH_URL:-http://127.0.0.1:8200/health}"
TEXT_QWEN35_MODEL="${TEXT_QWEN35_MODEL:-Qwen/Qwen3.5-9B}"
TEXT_QWEN35_GPU_MEMORY_UTILIZATION="${TEXT_QWEN35_GPU_MEMORY_UTILIZATION:-0.68}"
TEXT_QWEN3GUARD_GPU_MEMORY_UTILIZATION="${TEXT_QWEN3GUARD_GPU_MEMORY_UTILIZATION:-0.06}"
TEXT_QWEN35_GPU="${TEXT_QWEN35_GPU:-0}"
TEXT_QWEN3GUARD_GPU="${TEXT_QWEN3GUARD_GPU:-0}"

ASR_HOST="${ASR_HOST:-127.0.0.1}"
ASR_PORT="${ASR_PORT:-19011}"
ASR_GPU="${ASR_GPU:-0}"
ASR_BACKEND="${ASR_BACKEND:-vllm}"
START_ASR_ADAPTER="${START_ASR_ADAPTER:-true}"

AUDIO_HOST="${AUDIO_HOST:-0.0.0.0}"
AUDIO_PORT="${AUDIO_PORT:-19001}"
AUDIO_PROBE_HOST="${AUDIO_PROBE_HOST:-127.0.0.1}"

QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$PROJECT/models/Qwen/Qwen3-ASR-0.6B}"
FFMPEG_BIN="${FFMPEG_BIN:-$PROJECT/models/ffmpeg-7.0.2-amd64-static/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-$PROJECT/models/ffmpeg-7.0.2-amd64-static/ffprobe}"

ASR_ENV_ACTIVATE="${ASR_ENV_ACTIVATE:-$PROJECT/qwen-serving/asr-vllm/.venv/bin/activate}"
AUDIO_ENV_ACTIVATE="${AUDIO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"

QWEN_ASR_DEVICE="${QWEN_ASR_DEVICE:-cuda}"
QWEN3ASR_GPU_MEMORY_UTILIZATION="${QWEN3ASR_GPU_MEMORY_UTILIZATION:-0.12}"
QWEN3ASR_MAX_INFERENCE_BATCH_SIZE="${QWEN3ASR_MAX_INFERENCE_BATCH_SIZE:-1}"
QWEN3ASR_MAX_NEW_TOKENS="${QWEN3ASR_MAX_NEW_TOKENS:-2048}"
QWEN3ASR_MAX_MODEL_LEN="${QWEN3ASR_MAX_MODEL_LEN:-4096}"
QWEN3ASR_TENSOR_PARALLEL_SIZE="${QWEN3ASR_TENSOR_PARALLEL_SIZE:-1}"
AUDIO_REDACTION_ENABLED="${AUDIO_REDACTION_ENABLED:-false}"
WAIT_SECONDS="${WAIT_SECONDS:-900}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-3}"
TEXT_API_TASK_TIMEOUT_SECONDS="${TEXT_API_TASK_TIMEOUT_SECONDS:-1800}"

usage() {
  cat <<EOF
Usage: $0 [start|restart|stop|status|attach]

Starts the transcript-text audio compliance stack:
  optional text local stack -> Qwen3-ASR adapter -> audio.server

Important overrides:
  PROJECT=$PROJECT
  SESSION=$SESSION
  START_TEXT_STACK=$START_TEXT_STACK
  TEXT_STACK_SCRIPT=$TEXT_STACK_SCRIPT
  TEXT_SESSION=$TEXT_SESSION
  TEXT_API_BASE_URL=$TEXT_API_BASE_URL
  TEXT_QWEN35_BASE_URL=$TEXT_QWEN35_BASE_URL
  TEXT_QWEN35_HEALTH_URL=$TEXT_QWEN35_HEALTH_URL
  TEXT_QWEN35_MODEL=$TEXT_QWEN35_MODEL
  TEXT_QWEN35_GPU_MEMORY_UTILIZATION=$TEXT_QWEN35_GPU_MEMORY_UTILIZATION
  TEXT_QWEN3GUARD_GPU_MEMORY_UTILIZATION=$TEXT_QWEN3GUARD_GPU_MEMORY_UTILIZATION
  START_ASR_ADAPTER=$START_ASR_ADAPTER
  ASR_BACKEND=$ASR_BACKEND
  ASR_ENV_ACTIVATE=$ASR_ENV_ACTIVATE
  AUDIO_ENV_ACTIVATE=$AUDIO_ENV_ACTIVATE
  ASR_PORT=$ASR_PORT
  AUDIO_PORT=$AUDIO_PORT
  ASR_GPU=$ASR_GPU
  QWEN3ASR_GPU_MEMORY_UTILIZATION=$QWEN3ASR_GPU_MEMORY_UTILIZATION
  QWEN3ASR_MAX_INFERENCE_BATCH_SIZE=$QWEN3ASR_MAX_INFERENCE_BATCH_SIZE
  QWEN3ASR_MAX_NEW_TOKENS=$QWEN3ASR_MAX_NEW_TOKENS
  QWEN3ASR_MAX_MODEL_LEN=$QWEN3ASR_MAX_MODEL_LEN
  QWEN3ASR_TENSOR_PARALLEL_SIZE=$QWEN3ASR_TENSOR_PARALLEL_SIZE
  TEXT_API_TASK_TIMEOUT_SECONDS=$TEXT_API_TASK_TIMEOUT_SECONDS
  AUDIO_WORK_DIR=$AUDIO_WORK_DIR
  ATTACH=$ATTACH
EOF
}

log() {
  printf '[audio-local] %s\n' "$*"
}

fail() {
  printf '[audio-local] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

q() {
  printf "%q" "$1"
}

activation_line() {
  local command="$1"
  if [[ -n "$command" ]]; then
    printf 'source %q\n' "$command"
  fi
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
  [[ -x "$python_bin" ]] || fail "Python executable not found for env: $activate"
  "$python_bin" - <<PY >/dev/null
import importlib.util
import sys
module_name = ${module_name@Q}
if importlib.util.find_spec(module_name) is None:
    sys.exit(1)
PY
}

session_exists() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
}

probe_http() {
  local url="$1"
  curl -sS --connect-timeout 5 --max-time 20 "$url" >/dev/null
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local deadline=$((SECONDS + WAIT_SECONDS))
  log "Waiting for $label: $url"
  while (( SECONDS < deadline )); do
    if probe_http "$url"; then
      log "$label is ready"
      return
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
  fail "Timed out waiting for $label at $url"
}

write_runner() {
  local path="$1"
  local label="$2"
  local body="$3"
  mkdir -p "$RUN_DIR"
  cat > "$path" <<EOF
#!/usr/bin/env bash
set -u

$body

status=\$?
echo
echo "[$label] Process exited with status \$status. Press Ctrl+D or close this tmux pane to exit."
exec bash
EOF
  chmod +x "$path"
}

validate_paths() {
  [[ -d "$PROJECT" ]] || fail "PROJECT does not exist: $PROJECT"
  [[ -f "$AUDIO_ENV_ACTIVATE" ]] || fail "AUDIO_ENV_ACTIVATE does not exist: $AUDIO_ENV_ACTIVATE"
  [[ -d "$QWEN_ASR_MODEL" ]] || fail "QWEN_ASR_MODEL does not exist: $QWEN_ASR_MODEL"
  [[ -f "$FFMPEG_BIN" ]] || fail "FFMPEG_BIN does not exist: $FFMPEG_BIN"
  [[ -f "$FFPROBE_BIN" ]] || fail "FFPROBE_BIN does not exist: $FFPROBE_BIN"
  if [[ "$START_ASR_ADAPTER" == "true" ]]; then
    [[ "$ASR_BACKEND" == "vllm" || "$ASR_BACKEND" == "transformers" ]] || fail "ASR_BACKEND must be 'vllm' or 'transformers', got: $ASR_BACKEND"
    [[ -f "$ASR_ENV_ACTIVATE" ]] || fail "ASR_ENV_ACTIVATE does not exist: $ASR_ENV_ACTIVATE"
    require_python_module "$ASR_ENV_ACTIVATE" "fastapi"
    if [[ "$ASR_BACKEND" == "vllm" ]]; then
      [[ "${QWEN_ASR_DEVICE,,}" != "cpu" ]] || fail "ASR_BACKEND=vllm requires a CUDA device. Use ASR_BACKEND=transformers for CPU ASR."
      require_python_module "$ASR_ENV_ACTIVATE" "qwen_asr"
      require_python_module "$ASR_ENV_ACTIVATE" "vllm"
    else
      require_python_module "$ASR_ENV_ACTIVATE" "pydantic_settings"
      require_python_module "$ASR_ENV_ACTIVATE" "transformers"
    fi
  fi
  require_python_module "$AUDIO_ENV_ACTIVATE" "fastapi"
  require_python_module "$AUDIO_ENV_ACTIVATE" "httpx"
  require_python_module "$AUDIO_ENV_ACTIVATE" "multipart"
  mkdir -p "$AUDIO_WORK_DIR" "$RUN_DIR"
}

ensure_text_stack() {
  if probe_http "${TEXT_API_BASE_URL}/api/v1/health"; then
    log "Text API is already healthy: ${TEXT_API_BASE_URL}"
    return
  fi
  if [[ "$START_TEXT_STACK" != "true" ]]; then
    fail "Text API is not healthy at ${TEXT_API_BASE_URL}. Start it or set START_TEXT_STACK=true."
  fi
  [[ -x "$TEXT_STACK_SCRIPT" || -f "$TEXT_STACK_SCRIPT" ]] || fail "TEXT_STACK_SCRIPT not found: $TEXT_STACK_SCRIPT"
  log "Starting text local stack via $TEXT_STACK_SCRIPT"
  ATTACH=false \
    SESSION="$TEXT_SESSION" \
    QWEN35_BASE_URL="$TEXT_QWEN35_BASE_URL" \
    QWEN35_HEALTH_URL="$TEXT_QWEN35_HEALTH_URL" \
    QWEN35_MODEL="$TEXT_QWEN35_MODEL" \
    QWEN35_GPU="$TEXT_QWEN35_GPU" \
    QWEN3GUARD_GPU="$TEXT_QWEN3GUARD_GPU" \
    QWEN35_GPU_MEMORY_UTILIZATION="$TEXT_QWEN35_GPU_MEMORY_UTILIZATION" \
    QWEN3GUARD_GPU_MEMORY_UTILIZATION="$TEXT_QWEN3GUARD_GPU_MEMORY_UTILIZATION" \
    bash "$TEXT_STACK_SCRIPT" start || true
  wait_for_http "text-api" "${TEXT_API_BASE_URL}/api/v1/health"
}

write_runners() {
  local asr_module="audio.adapters.qwen_asr_service:app"
  local asr_extra_exports="
export COMPLIANCE_QWEN_ASR_MODEL=$(q "$QWEN_ASR_MODEL")
export COMPLIANCE_QWEN_ASR_DEVICE=$(q "$QWEN_ASR_DEVICE")
export COMPLIANCE_QWEN_ASR_ENDPOINT=
"
  if [[ "$ASR_BACKEND" == "vllm" ]]; then
    asr_module="ops.qwen3asr_vllm_adapter:app"
    asr_extra_exports="
export QWEN3ASR_MODEL=$(q "$QWEN_ASR_MODEL")
export QWEN3ASR_DEVICE=$(q "$QWEN_ASR_DEVICE")
export QWEN3ASR_GPU_MEMORY_UTILIZATION=$(q "$QWEN3ASR_GPU_MEMORY_UTILIZATION")
export QWEN3ASR_MAX_INFERENCE_BATCH_SIZE=$(q "$QWEN3ASR_MAX_INFERENCE_BATCH_SIZE")
export QWEN3ASR_MAX_NEW_TOKENS=$(q "$QWEN3ASR_MAX_NEW_TOKENS")
export QWEN3ASR_MAX_MODEL_LEN=$(q "$QWEN3ASR_MAX_MODEL_LEN")
export QWEN3ASR_TENSOR_PARALLEL_SIZE=$(q "$QWEN3ASR_TENSOR_PARALLEL_SIZE")
"
  fi

  write_runner "$RUN_DIR/qwen_asr_adapter.sh" "audio-asr" "
cd $(q "$PROJECT")
$(activation_line "$ASR_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$ASR_GPU")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
$asr_extra_exports
echo '[audio-asr] Starting Qwen3-ASR ${ASR_BACKEND} adapter on ${ASR_HOST}:${ASR_PORT}'
python -m uvicorn $asr_module --host $(q "$ASR_HOST") --port $(q "$ASR_PORT")
"

  local asr_endpoint=""
  if [[ "$START_ASR_ADAPTER" == "true" ]]; then
    asr_endpoint="http://${ASR_HOST}:${ASR_PORT}/transcribe"
  fi

  write_runner "$RUN_DIR/audio_server.sh" "audio-server" "
cd $(q "$PROJECT")
$(activation_line "$AUDIO_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=\"\"
export COMPLIANCE_WORK_DIR=$(q "$AUDIO_WORK_DIR")
export COMPLIANCE_SERVER_HOST=$(q "$AUDIO_HOST")
export COMPLIANCE_SERVER_PORT=$(q "$AUDIO_PORT")
export COMPLIANCE_AUDIO_EXECUTION_ROUTE=api
export COMPLIANCE_TEXT_API_BASE_URL=$(q "$TEXT_API_BASE_URL")
export COMPLIANCE_TEXT_API_TASK_TIMEOUT_SECONDS=$(q "$TEXT_API_TASK_TIMEOUT_SECONDS")
export COMPLIANCE_FFMPEG_BIN=$(q "$FFMPEG_BIN")
export COMPLIANCE_FFPROBE_BIN=$(q "$FFPROBE_BIN")
export COMPLIANCE_QWEN_ASR_ENABLED=true
export COMPLIANCE_QWEN_ASR_MODEL=$(q "$QWEN_ASR_MODEL")
export COMPLIANCE_QWEN_ASR_ENDPOINT=$(q "$asr_endpoint")
export COMPLIANCE_QWEN_ASR_TIMEOUT_SECONDS=600
export COMPLIANCE_ASR_REQUIRED=true
export COMPLIANCE_ASR_UNAVAILABLE_FALLBACK_ENABLED=false
export COMPLIANCE_FASTER_WHISPER_ENABLED=false
export COMPLIANCE_PYANNOTE_ENABLED=false
export COMPLIANCE_AUDIO_REDACTION_ENABLED=$(q "$AUDIO_REDACTION_ENABLED")
echo '[audio-server] Starting audio compliance API on ${AUDIO_HOST}:${AUDIO_PORT}'
echo '[audio-server] Text API: ${TEXT_API_BASE_URL}'
echo '[audio-server] ASR endpoint: ${asr_endpoint:-local in-process Qwen3-ASR} (${ASR_BACKEND})'
python -m uvicorn audio.server:app --host $(q "$AUDIO_HOST") --port $(q "$AUDIO_PORT")
"
}

start_session() {
  require_cmd tmux
  require_cmd curl
  validate_paths
  ensure_text_stack

  if session_exists; then
    fail "tmux session already exists: $SESSION. Use '$0 attach', '$0 stop', or '$0 restart'."
  fi

  write_runners
  if [[ "$START_ASR_ADAPTER" == "true" ]]; then
    tmux new-session -d -s "$SESSION" -n asr "$RUN_DIR/qwen_asr_adapter.sh"
    tmux new-window -t "$SESSION" -n audio-server "$RUN_DIR/audio_server.sh"
  else
    tmux new-session -d -s "$SESSION" -n audio-server "$RUN_DIR/audio_server.sh"
  fi
  tmux select-window -t "$SESSION:audio-server"

  if [[ "$START_ASR_ADAPTER" == "true" ]]; then
    local asr_probe_path="/health"
    if [[ "$ASR_BACKEND" == "vllm" ]]; then
      asr_probe_path="/ready"
    fi
    wait_for_http "qwen3-asr-adapter" "http://${ASR_HOST}:${ASR_PORT}${asr_probe_path}"
  fi
  wait_for_http "audio-api" "http://${AUDIO_PROBE_HOST}:${AUDIO_PORT}/api/v1/health"

  cat <<EOF
Started tmux session: $SESSION

Services:
  text-api   -> ${TEXT_API_BASE_URL}
  asr        -> http://${ASR_HOST}:${ASR_PORT}  (START_ASR_ADAPTER=$START_ASR_ADAPTER, ASR_BACKEND=$ASR_BACKEND)
  audio-api  -> http://${AUDIO_PROBE_HOST}:${AUDIO_PORT}

Text stack defaults used when this script starts the text stack:
  Qwen3.5     ${TEXT_QWEN35_BASE_URL} model=${TEXT_QWEN35_MODEL}
  Qwen3Guard  GPU ${TEXT_QWEN3GUARD_GPU}, gpu_memory_utilization=${TEXT_QWEN3GUARD_GPU_MEMORY_UTILIZATION}
  Qwen3-ASR   GPU ${ASR_GPU}, gpu_memory_utilization=${QWEN3ASR_GPU_MEMORY_UTILIZATION} (vLLM)

Useful commands:
  tmux attach -t $SESSION
  $0 status
  $0 stop
EOF

  if [[ "$ATTACH" == "true" ]]; then
    tmux attach -t "$SESSION"
  fi
}

stop_session() {
  require_cmd tmux
  if session_exists; then
    tmux kill-session -t "$SESSION"
    log "Stopped tmux session: $SESSION"
  else
    log "No tmux session found: $SESSION"
  fi
  if [[ "$STOP_TEXT_STACK" == "true" ]]; then
    ATTACH=false SESSION="$TEXT_SESSION" bash "$TEXT_STACK_SCRIPT" stop || true
  fi
}

status_session() {
  require_cmd tmux
  if session_exists; then
    tmux list-windows -t "$SESSION"
  else
    log "No tmux session found: $SESSION"
    exit 1
  fi
}

case "$ACTION" in
  start)
    start_session
    ;;
  restart)
    stop_session
    start_session
    ;;
  stop)
    stop_session
    ;;
  status)
    status_session
    ;;
  attach)
    require_cmd tmux
    tmux attach -t "$SESSION"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
