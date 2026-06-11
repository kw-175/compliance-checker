#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
PROJECT="${PROJECT:-/data/kw/compliance-checker}"
SESSION="${SESSION:-video-compliance}"
RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/video_compliance_stack_tmux}"

ATTACH="${ATTACH:-false}"
START_TEXT_STACK="${START_TEXT_STACK:-true}"
START_PADDLEOCR_VL="${START_PADDLEOCR_VL:-true}"
START_SAM3_API="${START_SAM3_API:-true}"
START_AUDIO_STACK="${START_AUDIO_STACK:-true}"
START_PICTURE_API="${START_PICTURE_API:-true}"
STOP_DEPENDENCIES="${STOP_DEPENDENCIES:-false}"

TEXT_STACK_SCRIPT="${TEXT_STACK_SCRIPT:-$PROJECT/scripts/start_text_8200_stack.sh}"
AUDIO_STACK_SCRIPT="${AUDIO_STACK_SCRIPT:-$PROJECT/scripts/start_audio_local_stack.sh}"
PADDLEOCR_SCRIPT="${PADDLEOCR_SCRIPT:-$PROJECT/scripts/start_paddleocr_vl_serving.sh}"
SAM3_SCRIPT="${SAM3_SCRIPT:-$PROJECT/scripts/start_sam3_api.sh}"
PICTURE_SCRIPT="${PICTURE_SCRIPT:-$PROJECT/scripts/start_picture_local_stack.sh}"

TEXT_SESSION="${TEXT_SESSION:-text-8200}"
AUDIO_SESSION="${AUDIO_SESSION:-audio-local}"

QWEN35_BASE_URL="${QWEN35_BASE_URL:-http://127.0.0.1:8200/openai/v1}"
QWEN35_HEALTH_URL="${QWEN35_HEALTH_URL:-http://127.0.0.1:8200/health}"
QWEN35_MODEL="${QWEN35_MODEL:-Qwen/Qwen3.5-9B}"

TEXT_API_BASE_URL="${TEXT_API_BASE_URL:-http://127.0.0.1:19002}"
PADDLEOCR_VL_API_URL="${PADDLEOCR_VL_API_URL:-http://127.0.0.1:8217}"
SAM3_API_URL="${SAM3_API_URL:-http://127.0.0.1:8218}"
PICTURE_API_BASE_URL="${PICTURE_API_BASE_URL:-http://127.0.0.1:19012}"
ASR_ENDPOINT="${ASR_ENDPOINT:-http://127.0.0.1:19011/transcribe}"
ASR_HEALTH_URL="${ASR_HEALTH_URL:-http://127.0.0.1:19011/health}"
ASR_READY_URL="${ASR_READY_URL:-http://127.0.0.1:19011/ready}"

PICTURE_HOST="${PICTURE_HOST:-127.0.0.1}"
PICTURE_PORT="${PICTURE_PORT:-19012}"
VIDEO_HOST="${VIDEO_HOST:-0.0.0.0}"
VIDEO_PORT="${VIDEO_PORT:-19003}"
VIDEO_PROBE_HOST="${VIDEO_PROBE_HOST:-127.0.0.1}"

VIDEO_ENV_ACTIVATE="${VIDEO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
VIDEO_WORK_DIR="${VIDEO_WORK_DIR:-$PROJECT/temp/video_output}"
VIDEO_STORAGE_BASE_PATH="${VIDEO_STORAGE_BASE_PATH:-$PROJECT/temp/video_storage}"
FFMPEG_BIN="${FFMPEG_BIN:-$PROJECT/models/ffmpeg-7.0.2-amd64-static/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-$PROJECT/models/ffmpeg-7.0.2-amd64-static/ffprobe}"

PICTURE_API_TIMEOUT_SECONDS="${PICTURE_API_TIMEOUT_SECONDS:-30}"
PICTURE_API_TASK_TIMEOUT_SECONDS="${PICTURE_API_TASK_TIMEOUT_SECONDS:-1800}"
PICTURE_API_POLL_INTERVAL_SECONDS="${PICTURE_API_POLL_INTERVAL_SECONDS:-1.0}"

VIDEO_MAX_WORKERS="${VIDEO_MAX_WORKERS:-2}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-1}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-0}"
VIDEO_ENABLE_AUDIO_SIDECAR="${VIDEO_ENABLE_AUDIO_SIDECAR:-true}"
VIDEO_EXTRACT_NATIVE_AUDIO="${VIDEO_EXTRACT_NATIVE_AUDIO:-true}"
VIDEO_SCENE_DETECTION_ENABLED="${VIDEO_SCENE_DETECTION_ENABLED:-true}"
VIDEO_SCENE_CHANGE_THRESHOLD="${VIDEO_SCENE_CHANGE_THRESHOLD:-0.18}"
VIDEO_SCENE_MIN_DURATION_MS="${VIDEO_SCENE_MIN_DURATION_MS:-1000}"
VIDEO_CLIP_WINDOW_MS="${VIDEO_CLIP_WINDOW_MS:-5000}"
VIDEO_CLIP_WINDOW_OVERLAP_MS="${VIDEO_CLIP_WINDOW_OVERLAP_MS:-1000}"
VIDEO_CLIP_MODERATION_ENABLED="${VIDEO_CLIP_MODERATION_ENABLED:-true}"
VIDEO_CLIP_MODERATION_BASE_URL="${VIDEO_CLIP_MODERATION_BASE_URL:-http://127.0.0.1:8200}"
VIDEO_CLIP_MODERATION_ENDPOINT="${VIDEO_CLIP_MODERATION_ENDPOINT:-/video/action-recognition}"
VIDEO_CLIP_MODERATION_TIMEOUT_SECONDS="${VIDEO_CLIP_MODERATION_TIMEOUT_SECONDS:-120}"
VIDEO_CLIP_MODERATION_MAX_FRAMES="${VIDEO_CLIP_MODERATION_MAX_FRAMES:-8}"
VIDEO_CLIP_MODERATION_CONFIDENCE_THRESHOLD="${VIDEO_CLIP_MODERATION_CONFIDENCE_THRESHOLD:-0.55}"
VIDEO_TRACK_GAP_TOLERANCE_MS="${VIDEO_TRACK_GAP_TOLERANCE_MS:-1000}"
VIDEO_TRACK_IOU_THRESHOLD="${VIDEO_TRACK_IOU_THRESHOLD:-0.35}"
VIDEO_SAM3_VIDEO_TRACKING_ENABLED="${VIDEO_SAM3_VIDEO_TRACKING_ENABLED:-true}"
VIDEO_SAM3_VIDEO_TRACKER_BASE_URL="${VIDEO_SAM3_VIDEO_TRACKER_BASE_URL:-$SAM3_API_URL}"
VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT="${VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT:-/v1/sam3/video-track}"
VIDEO_SAM3_VIDEO_TRACKER_TIMEOUT_SECONDS="${VIDEO_SAM3_VIDEO_TRACKER_TIMEOUT_SECONDS:-300}"
VIDEO_SAM3_VIDEO_TRACKER_FAIL_ON_ERROR="${VIDEO_SAM3_VIDEO_TRACKER_FAIL_ON_ERROR:-false}"
VIDEO_SAM3_VIDEO_TRACKER_RETURN_MASKS="${VIDEO_SAM3_VIDEO_TRACKER_RETURN_MASKS:-true}"

usage() {
  cat <<EOF
Usage: $0 [start|restart|stop|status|check|attach]

Starts the full video compliance runtime:
  external Qwen3.5 on 8200 -> text API -> PaddleOCR-VL -> SAM3 -> audio ASR stack
  -> picture compliance API -> video compliance API

The current video engine requires picture API; it does not fall back to in-process
picture pipeline.

Important overrides:
  PROJECT=$PROJECT
  SESSION=$SESSION
  ATTACH=$ATTACH
  START_TEXT_STACK=$START_TEXT_STACK
  START_PADDLEOCR_VL=$START_PADDLEOCR_VL
  START_SAM3_API=$START_SAM3_API
  START_AUDIO_STACK=$START_AUDIO_STACK
  START_PICTURE_API=$START_PICTURE_API
  STOP_DEPENDENCIES=$STOP_DEPENDENCIES

Service URLs:
  QWEN35_HEALTH_URL=$QWEN35_HEALTH_URL
  QWEN35_BASE_URL=$QWEN35_BASE_URL
  TEXT_API_BASE_URL=$TEXT_API_BASE_URL
  PADDLEOCR_VL_API_URL=$PADDLEOCR_VL_API_URL
  SAM3_API_URL=$SAM3_API_URL
  VIDEO_SAM3_VIDEO_TRACKING_ENABLED=$VIDEO_SAM3_VIDEO_TRACKING_ENABLED
  VIDEO_SAM3_VIDEO_TRACKER_BASE_URL=$VIDEO_SAM3_VIDEO_TRACKER_BASE_URL
  VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT=$VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT
  PICTURE_API_BASE_URL=$PICTURE_API_BASE_URL
  ASR_ENDPOINT=$ASR_ENDPOINT
  ASR_HEALTH_URL=$ASR_HEALTH_URL
  ASR_READY_URL=$ASR_READY_URL
  VIDEO_PORT=$VIDEO_PORT

GPU-related dependency overrides are forwarded to the underlying scripts, e.g.:
  QWEN3GUARD_GPU=3 QWEN3GUARD_GPU_MEMORY_UTILIZATION=0.08
  ASR_GPU=0 QWEN3ASR_GPU_MEMORY_UTILIZATION=0.12
  PADDLEOCR_VL_GPU=2
  SAM3_GPU=2
EOF
}

log() {
  printf '[video-stack] %s\n' "$*"
}

fail() {
  printf '[video-stack] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

q() {
  printf "%q" "$1"
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

probe() {
  curl -fsS --connect-timeout 2 --max-time 8 "$1" >/dev/null 2>&1
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local timeout="${3:-300}"
  local started
  started="$(date +%s)"
  while true; do
    if probe "$url"; then
      log "$name ready: $url"
      return 0
    fi
    if (( $(date +%s) - started >= timeout )); then
      fail "$name did not become ready: $url"
    fi
    sleep 2
  done
}

session_exists() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
}

check_required_files() {
  [[ -d "$PROJECT" ]] || fail "PROJECT does not exist: $PROJECT"
  [[ -f "$VIDEO_ENV_ACTIVATE" ]] || fail "VIDEO_ENV_ACTIVATE does not exist: $VIDEO_ENV_ACTIVATE"
  require_python_module "$VIDEO_ENV_ACTIVATE" "fastapi"
  require_python_module "$VIDEO_ENV_ACTIVATE" "multipart"
  [[ -x "$FFMPEG_BIN" ]] || fail "FFMPEG_BIN does not exist or is not executable: $FFMPEG_BIN"
  [[ -x "$FFPROBE_BIN" ]] || fail "FFPROBE_BIN does not exist or is not executable: $FFPROBE_BIN"
  [[ -f "$PICTURE_SCRIPT" ]] || fail "PICTURE_SCRIPT not found: $PICTURE_SCRIPT"
}

start_text_stack() {
  if probe "${TEXT_API_BASE_URL%/}/api/v1/health"; then
    log "Text API already ready: ${TEXT_API_BASE_URL}"
    return
  fi
  [[ "$START_TEXT_STACK" == "true" ]] || fail "Text API is not ready and START_TEXT_STACK=false"
  [[ -f "$TEXT_STACK_SCRIPT" ]] || fail "TEXT_STACK_SCRIPT not found: $TEXT_STACK_SCRIPT"
  log "Starting text stack"
  ATTACH=false \
    SESSION="$TEXT_SESSION" \
    QWEN35_BASE_URL="$QWEN35_BASE_URL" \
    QWEN35_HEALTH_URL="$QWEN35_HEALTH_URL" \
    QWEN35_MODEL="$QWEN35_MODEL" \
    bash "$TEXT_STACK_SCRIPT" start
  wait_for_http "text-api" "${TEXT_API_BASE_URL%/}/api/v1/health" 300
}

start_paddleocr_vl() {
  if probe "${PADDLEOCR_VL_API_URL%/}/docs"; then
    log "PaddleOCR-VL API already ready: ${PADDLEOCR_VL_API_URL}"
    return
  fi
  [[ "$START_PADDLEOCR_VL" == "true" ]] || fail "PaddleOCR-VL API is not ready and START_PADDLEOCR_VL=false"
  [[ -f "$PADDLEOCR_SCRIPT" ]] || fail "PADDLEOCR_SCRIPT not found: $PADDLEOCR_SCRIPT"
  log "Starting PaddleOCR-VL serving"
  bash "$PADDLEOCR_SCRIPT" start
  wait_for_http "paddleocr-vl-api" "${PADDLEOCR_VL_API_URL%/}/docs" 300
}

start_sam3_api() {
  if probe "${SAM3_API_URL%/}/health"; then
    log "SAM3 API already ready: ${SAM3_API_URL}"
    return
  fi
  [[ "$START_SAM3_API" == "true" ]] || fail "SAM3 API is not ready and START_SAM3_API=false"
  [[ -f "$SAM3_SCRIPT" ]] || fail "SAM3_SCRIPT not found: $SAM3_SCRIPT"
  log "Starting SAM3 API"
  bash "$SAM3_SCRIPT" start
  wait_for_http "sam3-api" "${SAM3_API_URL%/}/health" 180
}

start_audio_stack() {
  if probe "$ASR_READY_URL" || probe "$ASR_HEALTH_URL"; then
    log "ASR adapter already ready: ${ASR_ENDPOINT}"
    return
  fi
  [[ "$START_AUDIO_STACK" == "true" ]] || fail "ASR adapter is not ready and START_AUDIO_STACK=false"
  [[ -f "$AUDIO_STACK_SCRIPT" ]] || fail "AUDIO_STACK_SCRIPT not found: $AUDIO_STACK_SCRIPT"
  log "Starting audio stack for ASR adapter"
  ATTACH=false \
    SESSION="$AUDIO_SESSION" \
    START_TEXT_STACK=false \
    TEXT_API_BASE_URL="$TEXT_API_BASE_URL" \
    bash "$AUDIO_STACK_SCRIPT" start
  if ! probe "$ASR_READY_URL"; then
    wait_for_http "asr-adapter" "$ASR_HEALTH_URL" 900
  fi
}

write_runners() {
  mkdir -p "$RUN_DIR"
  cat > "$RUN_DIR/picture_api.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd $(q "$PROJECT")
export START_TEXT_STACK=false
export QWEN35_BASE_URL=$(q "$QWEN35_BASE_URL")
export QWEN35_HEALTH_URL=$(q "$QWEN35_HEALTH_URL")
export QWEN35_MODEL=$(q "$QWEN35_MODEL")
export TEXT_API_BASE_URL=$(q "$TEXT_API_BASE_URL")
export PICTURE_HOST=$(q "$PICTURE_HOST")
export PICTURE_PORT=$(q "$PICTURE_PORT")
export PICTURE_PADDLEOCR_VL_API_URL=$(q "$PADDLEOCR_VL_API_URL")
export PICTURE_SAM3_API_URL=$(q "$SAM3_API_URL")
exec bash $(q "$PICTURE_SCRIPT") start
EOF
  chmod +x "$RUN_DIR/picture_api.sh"

  cat > "$RUN_DIR/video_server.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd $(q "$PROJECT")
source $(q "$VIDEO_ENV_ACTIVATE")
export VIDEO_SERVER_HOST=$(q "$VIDEO_HOST")
export VIDEO_SERVER_PORT=$(q "$VIDEO_PORT")
export VIDEO_WORK_DIR=$(q "$VIDEO_WORK_DIR")
export VIDEO_STORAGE_BASE_PATH=$(q "$VIDEO_STORAGE_BASE_PATH")
export VIDEO_FFMPEG_BIN=$(q "$FFMPEG_BIN")
export VIDEO_FFPROBE_BIN=$(q "$FFPROBE_BIN")
export VIDEO_MAX_WORKERS=$(q "$VIDEO_MAX_WORKERS")
export VIDEO_FRAME_STRIDE=$(q "$VIDEO_FRAME_STRIDE")
export VIDEO_MAX_FRAMES=$(q "$VIDEO_MAX_FRAMES")
export VIDEO_ENABLE_AUDIO_SIDECAR=$(q "$VIDEO_ENABLE_AUDIO_SIDECAR")
export VIDEO_EXTRACT_NATIVE_AUDIO=$(q "$VIDEO_EXTRACT_NATIVE_AUDIO")
export VIDEO_SCENE_DETECTION_ENABLED=$(q "$VIDEO_SCENE_DETECTION_ENABLED")
export VIDEO_SCENE_CHANGE_THRESHOLD=$(q "$VIDEO_SCENE_CHANGE_THRESHOLD")
export VIDEO_SCENE_MIN_DURATION_MS=$(q "$VIDEO_SCENE_MIN_DURATION_MS")
export VIDEO_CLIP_WINDOW_MS=$(q "$VIDEO_CLIP_WINDOW_MS")
export VIDEO_CLIP_WINDOW_OVERLAP_MS=$(q "$VIDEO_CLIP_WINDOW_OVERLAP_MS")
export VIDEO_CLIP_MODERATION_ENABLED=$(q "$VIDEO_CLIP_MODERATION_ENABLED")
export VIDEO_CLIP_MODERATION_BASE_URL=$(q "$VIDEO_CLIP_MODERATION_BASE_URL")
export VIDEO_CLIP_MODERATION_ENDPOINT=$(q "$VIDEO_CLIP_MODERATION_ENDPOINT")
export VIDEO_CLIP_MODERATION_TIMEOUT_SECONDS=$(q "$VIDEO_CLIP_MODERATION_TIMEOUT_SECONDS")
export VIDEO_CLIP_MODERATION_MAX_FRAMES=$(q "$VIDEO_CLIP_MODERATION_MAX_FRAMES")
export VIDEO_CLIP_MODERATION_CONFIDENCE_THRESHOLD=$(q "$VIDEO_CLIP_MODERATION_CONFIDENCE_THRESHOLD")
export VIDEO_TRACK_GAP_TOLERANCE_MS=$(q "$VIDEO_TRACK_GAP_TOLERANCE_MS")
export VIDEO_TRACK_IOU_THRESHOLD=$(q "$VIDEO_TRACK_IOU_THRESHOLD")
export VIDEO_SAM3_VIDEO_TRACKING_ENABLED=$(q "$VIDEO_SAM3_VIDEO_TRACKING_ENABLED")
export VIDEO_SAM3_VIDEO_TRACKER_BASE_URL=$(q "$VIDEO_SAM3_VIDEO_TRACKER_BASE_URL")
export VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT=$(q "$VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT")
export VIDEO_SAM3_VIDEO_TRACKER_TIMEOUT_SECONDS=$(q "$VIDEO_SAM3_VIDEO_TRACKER_TIMEOUT_SECONDS")
export VIDEO_SAM3_VIDEO_TRACKER_FAIL_ON_ERROR=$(q "$VIDEO_SAM3_VIDEO_TRACKER_FAIL_ON_ERROR")
export VIDEO_SAM3_VIDEO_TRACKER_RETURN_MASKS=$(q "$VIDEO_SAM3_VIDEO_TRACKER_RETURN_MASKS")
export VIDEO_PICTURE_API_BASE_URL=$(q "$PICTURE_API_BASE_URL")
export VIDEO_PICTURE_API_TIMEOUT_SECONDS=$(q "$PICTURE_API_TIMEOUT_SECONDS")
export VIDEO_PICTURE_API_TASK_TIMEOUT_SECONDS=$(q "$PICTURE_API_TASK_TIMEOUT_SECONDS")
export VIDEO_PICTURE_API_POLL_INTERVAL_SECONDS=$(q "$PICTURE_API_POLL_INTERVAL_SECONDS")
export COMPLIANCE_TEXT_API_BASE_URL=$(q "$TEXT_API_BASE_URL")
export COMPLIANCE_QWEN_ASR_ENDPOINT=$(q "$ASR_ENDPOINT")
export COMPLIANCE_QWEN_ASR_ENABLED=true
export COMPLIANCE_FASTER_WHISPER_ENABLED=false
export COMPLIANCE_ASR_REQUIRED=true
export COMPLIANCE_ASR_UNAVAILABLE_FALLBACK_ENABLED=false
export COMPLIANCE_ENABLE_HARD_CASE_ADJUDICATION=false
export COMPLIANCE_OPA_ENABLED=false
export COMPLIANCE_FFMPEG_BIN=$(q "$FFMPEG_BIN")
export COMPLIANCE_FFPROBE_BIN=$(q "$FFPROBE_BIN")
echo "[video-server] Starting video compliance API on \${VIDEO_SERVER_HOST}:\${VIDEO_SERVER_PORT}"
echo "[video-server] Picture API: \${VIDEO_PICTURE_API_BASE_URL}"
echo "[video-server] Text API: \${COMPLIANCE_TEXT_API_BASE_URL}"
echo "[video-server] ASR endpoint: \${COMPLIANCE_QWEN_ASR_ENDPOINT}"
echo "[video-server] Clip moderation: \${VIDEO_CLIP_MODERATION_ENABLED} via \${VIDEO_CLIP_MODERATION_BASE_URL}\${VIDEO_CLIP_MODERATION_ENDPOINT}"
echo "[video-server] SAM3 video tracking: \${VIDEO_SAM3_VIDEO_TRACKING_ENABLED} via \${VIDEO_SAM3_VIDEO_TRACKER_BASE_URL}\${VIDEO_SAM3_VIDEO_TRACKER_ENDPOINT}"
exec python -m uvicorn video.server:app --host "\$VIDEO_SERVER_HOST" --port "\$VIDEO_SERVER_PORT"
EOF
  chmod +x "$RUN_DIR/video_server.sh"
}

start_picture_and_video() {
  require_cmd tmux
  if session_exists; then
    fail "tmux session already exists: $SESSION. Use '$0 attach', '$0 stop', or '$0 restart'."
  fi
  write_runners

  if probe "${PICTURE_API_BASE_URL%/}/api/v1/health"; then
    log "Picture API already ready: ${PICTURE_API_BASE_URL}"
    tmux new-session -d -s "$SESSION" -n video-server "$RUN_DIR/video_server.sh"
    wait_for_http "video-api" "http://${VIDEO_PROBE_HOST}:${VIDEO_PORT}/api/v1/health" 180
    return
  fi

  if [[ "$START_PICTURE_API" == "true" ]]; then
    tmux new-session -d -s "$SESSION" -n picture-api "$RUN_DIR/picture_api.sh"
    wait_for_http "picture-api" "${PICTURE_API_BASE_URL%/}/api/v1/health" 420
  else
    fail "Picture API is not ready and START_PICTURE_API=false"
  fi

  tmux new-window -t "$SESSION" -n video-server "$RUN_DIR/video_server.sh"
  tmux select-window -t "$SESSION:video-server"
  wait_for_http "video-api" "http://${VIDEO_PROBE_HOST}:${VIDEO_PORT}/api/v1/health" 180
}

start_stack() {
  require_cmd curl
  check_required_files
  wait_for_http "external-qwen35" "$QWEN35_HEALTH_URL" 10
  start_text_stack
  start_paddleocr_vl
  start_sam3_api
  start_audio_stack
  start_picture_and_video
  cat <<EOF
Started video compliance stack.

tmux session:
  $SESSION

Services:
  text-api       ${TEXT_API_BASE_URL}
  paddleocr-vl   ${PADDLEOCR_VL_API_URL}
  sam3-api       ${SAM3_API_URL}
  asr-adapter    ${ASR_ENDPOINT}
  picture-api    ${PICTURE_API_BASE_URL}
  video-api      http://${VIDEO_PROBE_HOST}:${VIDEO_PORT}

Useful commands:
  tmux attach -t $SESSION
  $0 status
  $0 stop
EOF
  if [[ "$ATTACH" == "true" ]]; then
    tmux attach -t "$SESSION"
  fi
}

stop_stack() {
  require_cmd tmux
  if session_exists; then
    tmux kill-session -t "$SESSION"
    log "Stopped tmux session: $SESSION"
  else
    log "No tmux session found: $SESSION"
  fi
  if [[ "$STOP_DEPENDENCIES" == "true" ]]; then
    ATTACH=false SESSION="$AUDIO_SESSION" bash "$AUDIO_STACK_SCRIPT" stop || true
    bash "$SAM3_SCRIPT" stop || true
    bash "$PADDLEOCR_SCRIPT" stop || true
    ATTACH=false SESSION="$TEXT_SESSION" bash "$TEXT_STACK_SCRIPT" stop || true
  fi
}

status_stack() {
  require_cmd tmux
  require_cmd curl
  if session_exists; then
    tmux list-windows -t "$SESSION"
  else
    log "No tmux session found: $SESSION"
  fi
  for item in \
    "text-api ${TEXT_API_BASE_URL%/}/api/v1/health" \
    "paddleocr-vl ${PADDLEOCR_VL_API_URL%/}/docs" \
    "sam3-api ${SAM3_API_URL%/}/health" \
    "asr-adapter ${ASR_HEALTH_URL}" \
    "picture-api ${PICTURE_API_BASE_URL%/}/api/v1/health" \
    "video-api http://${VIDEO_PROBE_HOST}:${VIDEO_PORT}/api/v1/health"; do
    local name="${item%% *}"
    local url="${item#* }"
    if probe "$url"; then
      log "$name OK"
    else
      log "$name unavailable"
    fi
  done
}

case "$ACTION" in
  start)
    start_stack
    ;;
  restart)
    stop_stack
    start_stack
    ;;
  stop)
    stop_stack
    ;;
  status)
    status_stack
    ;;
  check)
    status_stack
    ;;
  attach)
    require_cmd tmux
    tmux attach -t "$SESSION"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
