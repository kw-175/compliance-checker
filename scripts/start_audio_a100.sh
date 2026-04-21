#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
SESSION="${SESSION:-audio-a100}"
ATTACH="${ATTACH:-true}"

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/audio_a100_tmux}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_a100_output}"

PII_HOST="${PII_HOST:-127.0.0.1}"
PII_PORT="${PII_PORT:-5012}"
ASR_HOST="${ASR_HOST:-127.0.0.1}"
ASR_PORT="${ASR_PORT:-8011}"
GUARD_HOST="${GUARD_HOST:-127.0.0.1}"
GUARD_PORT="${GUARD_PORT:-8012}"
HARD_CASE_HOST="${HARD_CASE_HOST:-127.0.0.1}"
HARD_CASE_PORT="${HARD_CASE_PORT:-8013}"
AUDIO_HOST="${AUDIO_HOST:-0.0.0.0}"
AUDIO_PORT="${AUDIO_PORT:-8010}"

PII_ROOT="${PII_ROOT:-$PROJECT/models/compliance-pii}"
PII_STANZA_RESOURCES_DIR="${PII_STANZA_RESOURCES_DIR:-$PII_ROOT/stanza_resources}"
GLINER_MODEL_DIR="${GLINER_MODEL_DIR:-$PII_ROOT/gliner-pii-large-v1.0}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$PROJECT/models/Qwen/Qwen3-ASR-0.6B}"
QWEN_GUARD_MODEL="${QWEN_GUARD_MODEL:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"
HARD_CASE_MODEL="${HARD_CASE_MODEL:-$PROJECT/models/Qwen/Qwen3.5-9B}"

FFMPEG_BIN="${FFMPEG_BIN:-/data/kw/.local/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/data/kw/.local/bin/ffprobe}"

PII_ENV_ACTIVATE="${PII_ENV_ACTIVATE:-$PROJECT/.venvs/compliance-pii/bin/activate}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-$PROJECT/qwen-serving/.venv/bin/activate}"
AUDIO_ENV_ACTIVATE="${AUDIO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PII_RUNNER="${PII_RUNNER:-}"
QWEN_RUNNER="${QWEN_RUNNER:-}"
AUDIO_RUNNER="${AUDIO_RUNNER:-}"

AUDIO_GPU="${AUDIO_GPU:-0}"
QWEN_ASR_DEVICE="${QWEN_ASR_DEVICE:-cuda}"
QWEN_GUARD_DEVICE="${QWEN_GUARD_DEVICE:-cuda}"
HARD_CASE_DEVICE="${HARD_CASE_DEVICE:-cuda}"
HARD_CASE_MAX_NEW_TOKENS="${HARD_CASE_MAX_NEW_TOKENS:-512}"

PII_STANZA_EN_MODEL="${PII_STANZA_EN_MODEL:-en}"
PII_STANZA_ZH_MODEL="${PII_STANZA_ZH_MODEL:-zh}"
PII_STANZA_DOWNLOAD_IF_MISSING="${PII_STANZA_DOWNLOAD_IF_MISSING:-false}"
PII_GLINER_THRESHOLD="${PII_GLINER_THRESHOLD:-0.50}"
PII_SCORE_THRESHOLD="${PII_SCORE_THRESHOLD:-0.45}"

usage() {
  cat <<EOF
Usage: $0 [start|restart|stop|status|attach]

Environment overrides:
  SESSION=$SESSION
  PROJECT=$PROJECT
  PII_ENV_ACTIVATE=$PII_ENV_ACTIVATE
  QWEN_ENV_ACTIVATE=$QWEN_ENV_ACTIVATE
  AUDIO_ENV_ACTIVATE=$AUDIO_ENV_ACTIVATE
  PII_PORT=$PII_PORT ASR_PORT=$ASR_PORT GUARD_PORT=$GUARD_PORT HARD_CASE_PORT=$HARD_CASE_PORT AUDIO_PORT=$AUDIO_PORT
  QWEN_ASR_MODEL=$QWEN_ASR_MODEL
  QWEN_GUARD_MODEL=$QWEN_GUARD_MODEL
  HARD_CASE_MODEL=$HARD_CASE_MODEL
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
  [[ -d "$PROJECT" ]] || { echo "PROJECT does not exist: $PROJECT" >&2; exit 1; }
  [[ -f "$PII_ENV_ACTIVATE" ]] || { echo "PII_ENV_ACTIVATE does not exist: $PII_ENV_ACTIVATE" >&2; exit 1; }
  [[ -f "$QWEN_ENV_ACTIVATE" ]] || { echo "QWEN_ENV_ACTIVATE does not exist: $QWEN_ENV_ACTIVATE" >&2; exit 1; }
  [[ -f "$AUDIO_ENV_ACTIVATE" ]] || { echo "AUDIO_ENV_ACTIVATE does not exist: $AUDIO_ENV_ACTIVATE" >&2; exit 1; }
  [[ -f "$FFMPEG_BIN" ]] || { echo "FFMPEG_BIN does not exist: $FFMPEG_BIN" >&2; exit 1; }
  [[ -f "$FFPROBE_BIN" ]] || { echo "FFPROBE_BIN does not exist: $FFPROBE_BIN" >&2; exit 1; }
  [[ -d "$PII_STANZA_RESOURCES_DIR" ]] || { echo "PII_STANZA_RESOURCES_DIR does not exist: $PII_STANZA_RESOURCES_DIR" >&2; exit 1; }
  [[ -d "$GLINER_MODEL_DIR" ]] || { echo "GLINER_MODEL_DIR does not exist: $GLINER_MODEL_DIR" >&2; exit 1; }
  [[ -d "$QWEN_ASR_MODEL" ]] || { echo "QWEN_ASR_MODEL does not exist: $QWEN_ASR_MODEL" >&2; exit 1; }
  [[ -d "$QWEN_GUARD_MODEL" ]] || { echo "QWEN_GUARD_MODEL does not exist: $QWEN_GUARD_MODEL" >&2; exit 1; }
  [[ -d "$HARD_CASE_MODEL" ]] || { echo "HARD_CASE_MODEL does not exist: $HARD_CASE_MODEL" >&2; exit 1; }
  mkdir -p "$AUDIO_WORK_DIR" "$RUN_DIR"
}

write_runners() {
  write_runner "$RUN_DIR/pii_gateway.sh" "audio-pii" "
cd $(q "$PROJECT")
$(activation_line "$PII_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=\"\"
export STANZA_RESOURCES_DIR=$(q "$PII_STANZA_RESOURCES_DIR")
export PII_STANZA_EN_MODEL=$(q "$PII_STANZA_EN_MODEL")
export PII_STANZA_ZH_MODEL=$(q "$PII_STANZA_ZH_MODEL")
export PII_STANZA_DOWNLOAD_IF_MISSING=$(q "$PII_STANZA_DOWNLOAD_IF_MISSING")
export PII_ENABLE_REGEX_RULES=true
export PII_ENABLE_PRESIDIO=true
export PII_ENABLE_GLINER=true
export PII_GLINER_MODEL=$(q "$GLINER_MODEL_DIR")
export PII_GLINER_THRESHOLD=$(q "$PII_GLINER_THRESHOLD")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo \"[audio-pii] Starting PII Gateway on $(q "$PII_HOST"):$(q "$PII_PORT")\"
${PII_RUNNER} python -m uvicorn ops.presidio_bilingual.app:app --host $(q "$PII_HOST") --port $(q "$PII_PORT")
"

  write_runner "$RUN_DIR/qwen_asr.sh" "audio-asr" "
cd $(q "$PROJECT")
$(activation_line "$QWEN_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COMPLIANCE_QWEN_ASR_MODEL=$(q "$QWEN_ASR_MODEL")
export COMPLIANCE_QWEN_ASR_DEVICE=$(q "$QWEN_ASR_DEVICE")
export COMPLIANCE_QWEN_ASR_ENDPOINT=
echo \"[audio-asr] Starting Qwen ASR adapter on $(q "$ASR_HOST"):$(q "$ASR_PORT")\"
${QWEN_RUNNER} python -m uvicorn audio.adapters.qwen_asr_service:app --host $(q "$ASR_HOST") --port $(q "$ASR_PORT")
"

  write_runner "$RUN_DIR/qwen_guard.sh" "audio-guard" "
cd $(q "$PROJECT")
$(activation_line "$QWEN_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COMPLIANCE_QWEN_GUARD_MODEL=$(q "$QWEN_GUARD_MODEL")
export COMPLIANCE_QWEN_GUARD_DEVICE=$(q "$QWEN_GUARD_DEVICE")
export COMPLIANCE_QWEN_GUARD_ENDPOINT=
echo \"[audio-guard] Starting Qwen Guard adapter on $(q "$GUARD_HOST"):$(q "$GUARD_PORT")\"
${QWEN_RUNNER} python -m uvicorn audio.adapters.qwen_guard_service:app --host $(q "$GUARD_HOST") --port $(q "$GUARD_PORT")
"

  write_runner "$RUN_DIR/qwen_hard_case.sh" "audio-hardcase" "
cd $(q "$PROJECT")
$(activation_line "$QWEN_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COMPLIANCE_HARD_CASE_MODEL_NAME=Qwen3.5-9B
export COMPLIANCE_HARD_CASE_LOCAL_MODEL_PATH=$(q "$HARD_CASE_MODEL")
export COMPLIANCE_HARD_CASE_DEVICE=$(q "$HARD_CASE_DEVICE")
export COMPLIANCE_HARD_CASE_MAX_NEW_TOKENS=$(q "$HARD_CASE_MAX_NEW_TOKENS")
export COMPLIANCE_HARD_CASE_ENDPOINT=
echo \"[audio-hardcase] Starting Qwen hard-case adapter on $(q "$HARD_CASE_HOST"):$(q "$HARD_CASE_PORT")\"
${QWEN_RUNNER} python -m uvicorn audio.adapters.qwen_hard_case_service:app --host $(q "$HARD_CASE_HOST") --port $(q "$HARD_CASE_PORT")
"

  write_runner "$RUN_DIR/audio_server.sh" "audio-server" "
cd $(q "$PROJECT")
$(activation_line "$AUDIO_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=\"\"
export COMPLIANCE_WORK_DIR=$(q "$AUDIO_WORK_DIR")
export COMPLIANCE_SERVER_HOST=$(q "$AUDIO_HOST")
export COMPLIANCE_SERVER_PORT=$(q "$AUDIO_PORT")
export COMPLIANCE_FFMPEG_BIN=$(q "$FFMPEG_BIN")
export COMPLIANCE_FFPROBE_BIN=$(q "$FFPROBE_BIN")
export COMPLIANCE_PII_ENDPOINT=http://${PII_HOST}:${PII_PORT}/analyze
export COMPLIANCE_PII_TIMEOUT_SECONDS=90
export COMPLIANCE_PII_SCORE_THRESHOLD=$(q "$PII_SCORE_THRESHOLD")
export COMPLIANCE_QWEN_ASR_ENABLED=true
export COMPLIANCE_QWEN_ASR_ENDPOINT=http://${ASR_HOST}:${ASR_PORT}/transcribe
export COMPLIANCE_QWEN_ASR_TIMEOUT_SECONDS=300
export COMPLIANCE_FASTER_WHISPER_ENABLED=false
export COMPLIANCE_PYANNOTE_ENABLED=false
export COMPLIANCE_QWEN_GUARD_ENABLED=true
export COMPLIANCE_QWEN_GUARD_ENDPOINT=http://${GUARD_HOST}:${GUARD_PORT}/moderate
export COMPLIANCE_QWEN_GUARD_TIMEOUT_SECONDS=120
export COMPLIANCE_ENABLE_HARD_CASE_ADJUDICATION=true
export COMPLIANCE_HARD_CASE_ENDPOINT=http://${HARD_CASE_HOST}:${HARD_CASE_PORT}/adjudicate
export COMPLIANCE_HARD_CASE_LOCAL_MODEL_PATH=
export COMPLIANCE_HARD_CASE_TIMEOUT_SECONDS=180
export COMPLIANCE_OPA_ENABLED=false
echo \"[audio-server] Starting audio server on $(q "$AUDIO_HOST"):$(q "$AUDIO_PORT")\"
echo \"[audio-server] Work dir: $(q "$AUDIO_WORK_DIR")\"
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
  tmux new-session -d -s "$SESSION" -n pii "$RUN_DIR/pii_gateway.sh"
  tmux new-window -t "$SESSION" -n asr "$RUN_DIR/qwen_asr.sh"
  tmux new-window -t "$SESSION" -n guard "$RUN_DIR/qwen_guard.sh"
  tmux new-window -t "$SESSION" -n hardcase "$RUN_DIR/qwen_hard_case.sh"
  tmux new-window -t "$SESSION" -n audio-server "$RUN_DIR/audio_server.sh"
  tmux select-window -t "$SESSION:audio-server"

  cat <<EOF
Started tmux session: $SESSION

Windows:
  1. pii          -> http://${PII_HOST}:${PII_PORT}/analyze
  2. asr          -> http://${ASR_HOST}:${ASR_PORT}/transcribe
  3. guard        -> http://${GUARD_HOST}:${GUARD_PORT}/moderate
  4. hardcase     -> http://${HARD_CASE_HOST}:${HARD_CASE_PORT}/adjudicate
  5. audio-server -> http://${AUDIO_HOST}:${AUDIO_PORT}

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

