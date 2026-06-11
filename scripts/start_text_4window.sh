#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
SESSION="${SESSION:-text4}"
ATTACH="${ATTACH:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/data/kw/compliance-checker}"

PII_ROOT="${PII_ROOT:-$PROJECT/models/compliance-pii}"
PII_STANZA_RESOURCES_DIR="${PII_STANZA_RESOURCES_DIR:-$PII_ROOT/stanza_resources}"
STANZA_RESOURCES_DIR="$PII_STANZA_RESOURCES_DIR"
GLINER_MODEL_DIR="${GLINER_MODEL_DIR:-$PII_ROOT/gliner-pii-large-v1.0}"
QWEN3GUARD_MODEL="${QWEN3GUARD_MODEL:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"
TEXT_WORK_DIR="${TEXT_WORK_DIR:-$PROJECT/temp/text_4window_output}"

PII_HOST="${PII_HOST:-127.0.0.1}"
PII_PORT="${PII_PORT:-5002}"
QWEN3GUARD_VLLM_HOST="${QWEN3GUARD_VLLM_HOST:-127.0.0.1}"
QWEN3GUARD_VLLM_PORT="${QWEN3GUARD_VLLM_PORT:-8155}"
QWEN3GUARD_ADAPTER_HOST="${QWEN3GUARD_ADAPTER_HOST:-127.0.0.1}"
QWEN3GUARD_ADAPTER_PORT="${QWEN3GUARD_ADAPTER_PORT:-8001}"
TEXT_HOST="${TEXT_HOST:-127.0.0.1}"
TEXT_PORT="${TEXT_PORT:-8000}"

QWEN3GUARD_GPU="${QWEN3GUARD_GPU:-0}"
QWEN3GUARD_GPU_MEMORY_UTILIZATION="${QWEN3GUARD_GPU_MEMORY_UTILIZATION:-0.12}"
QWEN3GUARD_MAX_MODEL_LEN="${QWEN3GUARD_MAX_MODEL_LEN:-4096}"
QWEN3GUARD_MAX_NUM_SEQS="${QWEN3GUARD_MAX_NUM_SEQS:-4}"
VLLM_CMD="${VLLM_CMD:-vllm}"
PII_ENV_ACTIVATE="${PII_ENV_ACTIVATE:-$PROJECT/.venvs/compliance-pii/bin/activate}"
QWEN3GUARD_ENV_ACTIVATE="${QWEN3GUARD_ENV_ACTIVATE:-$PROJECT/qwen-serving/.venv/bin/activate}"
GUARD_ADAPTER_ENV_ACTIVATE="${GUARD_ADAPTER_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
TEXT_ENV_ACTIVATE="${TEXT_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PII_RUNNER="${PII_RUNNER:-}"
GUARD_ADAPTER_RUNNER="${GUARD_ADAPTER_RUNNER:-}"
TEXT_RUNNER="${TEXT_RUNNER:-}"

PII_STANZA_EN_MODEL="${PII_STANZA_EN_MODEL:-en}"
PII_STANZA_ZH_MODEL="${PII_STANZA_ZH_MODEL:-zh}"
PII_STANZA_DOWNLOAD_IF_MISSING="${PII_STANZA_DOWNLOAD_IF_MISSING:-false}"
PII_GLINER_THRESHOLD="${PII_GLINER_THRESHOLD:-0.50}"

RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/text_4window_tmux}"

usage() {
  cat <<EOF
Usage: $0 [start|restart|stop|status|attach]

Legacy note:
  This script starts the older text.server-based four-window stack.
  For the preferred local-model-first text pipeline, use:
    bash scripts/start_text_local_stack.sh start

Environment overrides:
  SESSION=$SESSION
  PROJECT=$PROJECT
  PII_ROOT=$PII_ROOT
  PII_STANZA_RESOURCES_DIR=$PII_STANZA_RESOURCES_DIR
  STANZA_RESOURCES_DIR=$STANZA_RESOURCES_DIR
  GLINER_MODEL_DIR=$GLINER_MODEL_DIR
  QWEN3GUARD_MODEL=$QWEN3GUARD_MODEL
  TEXT_WORK_DIR=$TEXT_WORK_DIR
  QWEN3GUARD_GPU=$QWEN3GUARD_GPU
  QWEN3GUARD_GPU_MEMORY_UTILIZATION=$QWEN3GUARD_GPU_MEMORY_UTILIZATION
  VLLM_CMD=$VLLM_CMD
  PII_ENV_ACTIVATE=$PII_ENV_ACTIVATE
  QWEN3GUARD_ENV_ACTIVATE=$QWEN3GUARD_ENV_ACTIVATE
  GUARD_ADAPTER_ENV_ACTIVATE=$GUARD_ADAPTER_ENV_ACTIVATE
  TEXT_ENV_ACTIVATE=$TEXT_ENV_ACTIVATE
  PII_RUNNER=$PII_RUNNER
  GUARD_ADAPTER_RUNNER=$GUARD_ADAPTER_RUNNER
  TEXT_RUNNER=$TEXT_RUNNER
  PII_STANZA_DOWNLOAD_IF_MISSING=$PII_STANZA_DOWNLOAD_IF_MISSING
  ATTACH=$ATTACH
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_vllm_cmd() {
  if [[ -n "$QWEN3GUARD_ENV_ACTIVATE" ]]; then
    return
  fi
  if [[ "$VLLM_CMD" == *" "* ]]; then
    return
  fi
  require_cmd "$VLLM_CMD"
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
echo "[text4] Process exited with status \$status. Press Ctrl+D or close this tmux pane to exit."
exec bash
EOF
  chmod +x "$path"
}

validate_paths() {
  if [[ ! -d "$PROJECT" ]]; then
    echo "PROJECT does not exist: $PROJECT" >&2
    exit 1
  fi
  if [[ ! -d "$STANZA_RESOURCES_DIR" ]]; then
    echo "STANZA_RESOURCES_DIR does not exist: $STANZA_RESOURCES_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$GLINER_MODEL_DIR" ]]; then
    echo "GLINER_MODEL_DIR does not exist: $GLINER_MODEL_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$QWEN3GUARD_MODEL" ]]; then
    echo "QWEN3GUARD_MODEL does not exist: $QWEN3GUARD_MODEL" >&2
    exit 1
  fi
  for activation in "$PII_ENV_ACTIVATE" "$QWEN3GUARD_ENV_ACTIVATE" "$GUARD_ADAPTER_ENV_ACTIVATE" "$TEXT_ENV_ACTIVATE"; do
    if [[ -n "$activation" && ! -f "$activation" ]]; then
      echo "Virtual environment activation script does not exist: $activation" >&2
      exit 1
    fi
  done
  mkdir -p "$TEXT_WORK_DIR"
}

write_runners() {
  local pii_script="$RUN_DIR/pii_gateway.sh"
  local guard_vllm_script="$RUN_DIR/qwen3guard_vllm.sh"
  local guard_adapter_script="$RUN_DIR/qwen3guard_adapter.sh"
  local text_script="$RUN_DIR/text_server.sh"

  write_runner "$pii_script" "
cd $(q "$PROJECT")
$(activation_line "$PII_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=\"\"
export STANZA_RESOURCES_DIR=$(q "$STANZA_RESOURCES_DIR")
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
echo '[text4] Starting PII Gateway on ${PII_HOST}:${PII_PORT}'
${PII_RUNNER} python -m uvicorn ops.presidio_bilingual.app:app --host $(q "$PII_HOST") --port $(q "$PII_PORT")
"

  write_runner "$guard_vllm_script" "
cd $(q "$PROJECT")
$(activation_line "$QWEN3GUARD_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$QWEN3GUARD_GPU")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo '[text4] Starting Qwen3Guard vLLM on ${QWEN3GUARD_VLLM_HOST}:${QWEN3GUARD_VLLM_PORT}'
${VLLM_CMD} serve $(q "$QWEN3GUARD_MODEL") \\
  --served-model-name Qwen3Guard-Gen-0.6B \\
  --host $(q "$QWEN3GUARD_VLLM_HOST") \\
  --port $(q "$QWEN3GUARD_VLLM_PORT") \\
  --dtype auto \\
  --max-model-len $(q "$QWEN3GUARD_MAX_MODEL_LEN") \\
  --max-num-seqs $(q "$QWEN3GUARD_MAX_NUM_SEQS") \\
  --gpu-memory-utilization $(q "$QWEN3GUARD_GPU_MEMORY_UTILIZATION") \\
  --trust-remote-code
"

  write_runner "$guard_adapter_script" "
cd $(q "$PROJECT")
$(activation_line "$GUARD_ADAPTER_ENV_ACTIVATE")
export QWEN3GUARD_VLLM_URL=http://${QWEN3GUARD_VLLM_HOST}:${QWEN3GUARD_VLLM_PORT}/v1/chat/completions
export QWEN3GUARD_API_KEY=guard-token
export QWEN3GUARD_MODEL=Qwen3Guard-Gen-0.6B
echo '[text4] Starting Qwen3Guard Adapter on ${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}'
${GUARD_ADAPTER_RUNNER} python -m uvicorn ops.qwen3guard_adapter:app --host $(q "$QWEN3GUARD_ADAPTER_HOST") --port $(q "$QWEN3GUARD_ADAPTER_PORT")
"

  write_runner "$text_script" "
cd $(q "$PROJECT")
$(activation_line "$TEXT_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=\"\"
export COMPLIANCE_WORK_DIR=$(q "$TEXT_WORK_DIR")
export COMPLIANCE_ENABLE_PRESIDIO=true
export COMPLIANCE_PRESIDIO_ANALYZER_ENDPOINT=http://${PII_HOST}:${PII_PORT}/analyze
export COMPLIANCE_PRESIDIO_LANGUAGE=auto
export COMPLIANCE_PRESIDIO_SUPPORTED_LANGUAGES=en,zh
export COMPLIANCE_PRESIDIO_LANGUAGE_FALLBACK=en
export COMPLIANCE_PRESIDIO_SCORE_THRESHOLD=0.45
export COMPLIANCE_PRESIDIO_TIMEOUT_SECONDS=60
export COMPLIANCE_ENABLE_QWEN3GUARD=true
export COMPLIANCE_QWEN3GUARD_ENDPOINT=http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate
export COMPLIANCE_QWEN3GUARD_MODEL_NAME=Qwen3Guard-Gen-0.6B
export COMPLIANCE_QWEN3GUARD_TIMEOUT_SECONDS=90
export COMPLIANCE_QWEN3GUARD_MAX_CHARS=8000
export COMPLIANCE_ENABLE_HARD_CASE_ADJUDICATION=true
export COMPLIANCE_HARD_CASE_ENDPOINT=\"\"
export COMPLIANCE_HARD_CASE_LOCAL_MODEL_PATH=\"\"
echo '[text4] Starting text server on ${TEXT_HOST}:${TEXT_PORT}'
${TEXT_RUNNER} python -m uvicorn text.server:app --host $(q "$TEXT_HOST") --port $(q "$TEXT_PORT")
"
}

start_session() {
  require_cmd tmux
  require_vllm_cmd
  validate_paths

  if session_exists; then
    echo "tmux session already exists: $SESSION" >&2
    echo "Use '$0 attach', '$0 stop', or '$0 restart'." >&2
    exit 1
  fi

  write_runners

  tmux new-session -d -s "$SESSION" -n pii "$RUN_DIR/pii_gateway.sh"
  tmux new-window -t "$SESSION" -n guard-vllm "$RUN_DIR/qwen3guard_vllm.sh"
  tmux new-window -t "$SESSION" -n guard-adapter "$RUN_DIR/qwen3guard_adapter.sh"
  tmux new-window -t "$SESSION" -n text-server "$RUN_DIR/text_server.sh"
  tmux select-window -t "$SESSION:pii"

  cat <<EOF
Started tmux session: $SESSION

Legacy note:
  This is the older text.server four-window stack.
  Preferred local-model-first stack:
    bash scripts/start_text_local_stack.sh start

Windows:
  1. pii            -> http://${PII_HOST}:${PII_PORT}/analyze
  2. guard-vllm     -> http://${QWEN3GUARD_VLLM_HOST}:${QWEN3GUARD_VLLM_PORT}/v1/chat/completions
  3. guard-adapter  -> http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate
  4. text-server    -> http://${TEXT_HOST}:${TEXT_PORT}

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
