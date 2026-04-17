#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
SESSION="${SESSION:-audio-a100}"
ATTACH="${ATTACH:-true}"

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
AUDIO_HOST="${AUDIO_HOST:-0.0.0.0}"
AUDIO_PORT="${AUDIO_PORT:-8010}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_a100_output}"

PII_ROOT="${PII_ROOT:-$PROJECT/models/compliance-pii}"
PII_STANZA_RESOURCES_DIR="${PII_STANZA_RESOURCES_DIR:-$PII_ROOT/stanza_resources}"
GLINER_MODEL_DIR="${GLINER_MODEL_DIR:-$PII_ROOT/gliner-pii-large-v1.0}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$PROJECT/models/Qwen/Qwen3-ASR-0.6B}"
QWEN_GUARD_MODEL="${QWEN_GUARD_MODEL:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"
HARD_CASE_MODEL="${HARD_CASE_MODEL:-$PROJECT/models/Qwen/Qwen3.5-9B}"

FFMPEG_BIN="${FFMPEG_BIN:-/data/kw/.local/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/data/kw/.local/bin/ffprobe}"

AUDIO_GPU="${AUDIO_GPU:-0}"
QWEN_ASR_DEVICE="${QWEN_ASR_DEVICE:-cuda}"
QWEN_GUARD_DEVICE="${QWEN_GUARD_DEVICE:-cuda}"
HARD_CASE_DEVICE="${HARD_CASE_DEVICE:-cuda}"
HARD_CASE_MAX_NEW_TOKENS="${HARD_CASE_MAX_NEW_TOKENS:-512}"

QWEN_ASR_ENABLED="${QWEN_ASR_ENABLED:-true}"
FASTER_WHISPER_ENABLED="${FASTER_WHISPER_ENABLED:-false}"
PYANNOTE_ENABLED="${PYANNOTE_ENABLED:-false}"
QWEN_GUARD_ENABLED="${QWEN_GUARD_ENABLED:-true}"
HARD_CASE_ENABLED="${HARD_CASE_ENABLED:-true}"
OPA_ENABLED="${OPA_ENABLED:-false}"

PII_ENABLE_PRESIDIO="${PII_ENABLE_PRESIDIO:-true}"
PII_ENABLE_GLINER="${PII_ENABLE_GLINER:-true}"
PII_ENABLE_REGEX_RULES="${PII_ENABLE_REGEX_RULES:-true}"
PII_STANZA_EN_MODEL="${PII_STANZA_EN_MODEL:-en}"
PII_STANZA_ZH_MODEL="${PII_STANZA_ZH_MODEL:-zh}"
PII_STANZA_DOWNLOAD_IF_MISSING="${PII_STANZA_DOWNLOAD_IF_MISSING:-false}"
PII_GLINER_THRESHOLD="${PII_GLINER_THRESHOLD:-0.50}"
PII_SCORE_THRESHOLD="${PII_SCORE_THRESHOLD:-0.45}"

AUDIO_ENV_ACTIVATE="${AUDIO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
AUDIO_RUNNER="${AUDIO_RUNNER:-}"
RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/audio_a100_tmux}"

usage() {
  cat <<EOF
Usage: $0 [start|restart|stop|status|attach]

Environment overrides:
  SESSION=$SESSION
  PROJECT=$PROJECT
  AUDIO_HOST=$AUDIO_HOST
  AUDIO_PORT=$AUDIO_PORT
  AUDIO_WORK_DIR=$AUDIO_WORK_DIR
  AUDIO_GPU=$AUDIO_GPU
  AUDIO_ENV_ACTIVATE=$AUDIO_ENV_ACTIVATE
  AUDIO_RUNNER=$AUDIO_RUNNER
  QWEN_ASR_MODEL=$QWEN_ASR_MODEL
  QWEN_GUARD_MODEL=$QWEN_GUARD_MODEL
  HARD_CASE_MODEL=$HARD_CASE_MODEL
  PII_ROOT=$PII_ROOT
  FFMPEG_BIN=$FFMPEG_BIN
  FFPROBE_BIN=$FFPROBE_BIN
  ATTACH=$ATTACH
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

activation_line() {
  local command="$1"
  if [[ -z "$command" ]]; then
    return
  fi
  printf 'source %q\n' "$command"
}

session_exists() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
}

q() {
  printf "%q" "$1"
}

write_runner() {
  local path="$1"
  local body="$2"
  mkdir -p "$RUN_DIR"
  cat > "$path" <<EOF
#!/usr/bin/env bash
set -u

$body

status=\$?
echo
echo "[audio-a100] Process exited with status \$status. Press Ctrl+D or close this tmux pane to exit."
exec bash
EOF
  chmod +x "$path"
}

validate_paths() {
  if [[ ! -d "$PROJECT" ]]; then
    echo "PROJECT does not exist: $PROJECT" >&2
    exit 1
  fi
  if [[ -n "$AUDIO_ENV_ACTIVATE" && ! -f "$AUDIO_ENV_ACTIVATE" ]]; then
    echo "Virtual environment activation script does not exist: $AUDIO_ENV_ACTIVATE" >&2
    exit 1
  fi
  if [[ ! -f "$FFMPEG_BIN" ]]; then
    echo "FFMPEG_BIN does not exist: $FFMPEG_BIN" >&2
    exit 1
  fi
  if [[ ! -f "$FFPROBE_BIN" ]]; then
    echo "FFPROBE_BIN does not exist: $FFPROBE_BIN" >&2
    exit 1
  fi
  if [[ "$QWEN_ASR_ENABLED" == "true" && ! -d "$QWEN_ASR_MODEL" ]]; then
    echo "QWEN_ASR_MODEL does not exist: $QWEN_ASR_MODEL" >&2
    exit 1
  fi
  if [[ "$QWEN_GUARD_ENABLED" == "true" && ! -d "$QWEN_GUARD_MODEL" ]]; then
    echo "QWEN_GUARD_MODEL does not exist: $QWEN_GUARD_MODEL" >&2
    exit 1
  fi
  if [[ "$HARD_CASE_ENABLED" == "true" && ! -d "$HARD_CASE_MODEL" ]]; then
    echo "HARD_CASE_MODEL does not exist: $HARD_CASE_MODEL" >&2
    exit 1
  fi
  if [[ "$PII_ENABLE_PRESIDIO" == "true" && ! -d "$PII_STANZA_RESOURCES_DIR" ]]; then
    echo "PII_STANZA_RESOURCES_DIR does not exist: $PII_STANZA_RESOURCES_DIR" >&2
    exit 1
  fi
  if [[ "$PII_ENABLE_GLINER" == "true" && ! -d "$GLINER_MODEL_DIR" ]]; then
    echo "GLINER_MODEL_DIR does not exist: $GLINER_MODEL_DIR" >&2
    exit 1
  fi
  mkdir -p "$AUDIO_WORK_DIR" "$RUN_DIR"
}

write_runners() {
  local audio_script="$RUN_DIR/audio_server.sh"

  write_runner "$audio_script" "
cd $(q "$PROJECT")
$(activation_line "$AUDIO_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export STANZA_RESOURCES_DIR=$(q "$PII_STANZA_RESOURCES_DIR")

export COMPLIANCE_WORK_DIR=$(q "$AUDIO_WORK_DIR")
export COMPLIANCE_SERVER_HOST=$(q "$AUDIO_HOST")
export COMPLIANCE_SERVER_PORT=$(q "$AUDIO_PORT")
export COMPLIANCE_FFMPEG_BIN=$(q "$FFMPEG_BIN")
export COMPLIANCE_FFPROBE_BIN=$(q "$FFPROBE_BIN")

export COMPLIANCE_QWEN_ASR_ENABLED=$(q "$QWEN_ASR_ENABLED")
export COMPLIANCE_QWEN_ASR_MODEL=$(q "$QWEN_ASR_MODEL")
export COMPLIANCE_QWEN_ASR_DEVICE=$(q "$QWEN_ASR_DEVICE")
export COMPLIANCE_FASTER_WHISPER_ENABLED=$(q "$FASTER_WHISPER_ENABLED")
export COMPLIANCE_PYANNOTE_ENABLED=$(q "$PYANNOTE_ENABLED")

export COMPLIANCE_PII_MODEL_ROOT=$(q "$PII_ROOT")
export COMPLIANCE_PII_STANZA_RESOURCES_DIR=$(q "$PII_STANZA_RESOURCES_DIR")
export COMPLIANCE_PII_STANZA_EN_MODEL=$(q "$PII_STANZA_EN_MODEL")
export COMPLIANCE_PII_STANZA_ZH_MODEL=$(q "$PII_STANZA_ZH_MODEL")
export COMPLIANCE_PII_STANZA_DOWNLOAD_IF_MISSING=$(q "$PII_STANZA_DOWNLOAD_IF_MISSING")
export COMPLIANCE_PII_ENABLE_PRESIDIO=$(q "$PII_ENABLE_PRESIDIO")
export COMPLIANCE_PII_ENABLE_GLINER=$(q "$PII_ENABLE_GLINER")
export COMPLIANCE_PII_ENABLE_REGEX_RULES=$(q "$PII_ENABLE_REGEX_RULES")
export COMPLIANCE_PII_GLINER_MODEL=$(q "$GLINER_MODEL_DIR")
export COMPLIANCE_PII_GLINER_THRESHOLD=$(q "$PII_GLINER_THRESHOLD")
export COMPLIANCE_PII_SCORE_THRESHOLD=$(q "$PII_SCORE_THRESHOLD")

export COMPLIANCE_QWEN_GUARD_ENABLED=$(q "$QWEN_GUARD_ENABLED")
export COMPLIANCE_QWEN_GUARD_MODEL=$(q "$QWEN_GUARD_MODEL")
export COMPLIANCE_QWEN_GUARD_DEVICE=$(q "$QWEN_GUARD_DEVICE")

export COMPLIANCE_ENABLE_HARD_CASE_ADJUDICATION=$(q "$HARD_CASE_ENABLED")
export COMPLIANCE_HARD_CASE_MODEL_NAME=Qwen3.5-9B
export COMPLIANCE_HARD_CASE_LOCAL_MODEL_PATH=$(q "$HARD_CASE_MODEL")
export COMPLIANCE_HARD_CASE_ENDPOINT=
export COMPLIANCE_HARD_CASE_DEVICE=$(q "$HARD_CASE_DEVICE")
export COMPLIANCE_HARD_CASE_MAX_NEW_TOKENS=$(q "$HARD_CASE_MAX_NEW_TOKENS")
export COMPLIANCE_OPA_ENABLED=$(q "$OPA_ENABLED")

echo \"[audio-a100] Starting audio server on $(q "$AUDIO_HOST"):$(q "$AUDIO_PORT")\"
echo \"[audio-a100] Work dir: $(q "$AUDIO_WORK_DIR")\"
${AUDIO_RUNNER} python -m uvicorn audio.server:app --host $(q "$AUDIO_HOST") --port $(q "$AUDIO_PORT")
"
}

start_session() {
  require_cmd tmux
  validate_paths

  if session_exists; then
    echo "tmux session already exists: $SESSION" >&2
    echo "Use '$0 attach', '$0 stop', or '$0 restart'." >&2
    exit 1
  fi

  write_runners
  tmux new-session -d -s "$SESSION" -n audio-server "$RUN_DIR/audio_server.sh"

  cat <<EOF
Started tmux session: $SESSION

Windows:
  1. audio-server -> http://${AUDIO_HOST}:${AUDIO_PORT}

Useful commands:
  tmux attach -t $SESSION
  tmux ls
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
    echo "Stopped tmux session: $SESSION"
  else
    echo "No tmux session found: $SESSION"
  fi
}

status_session() {
  require_cmd tmux
  if session_exists; then
    tmux list-windows -t "$SESSION"
  else
    echo "No tmux session found: $SESSION"
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
    usage
    exit 1
    ;;
esac
