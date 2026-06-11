#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
SESSION="${SESSION:-text-local}"
ATTACH="${ATTACH:-true}"

PROJECT="${PROJECT:-/data/kw/compliance-checker}"

PII_ROOT="${PII_ROOT:-$PROJECT/models/compliance-pii}"
PII_STANZA_RESOURCES_DIR="${PII_STANZA_RESOURCES_DIR:-$PII_ROOT/stanza_resources}"
STANZA_RESOURCES_DIR="$PII_STANZA_RESOURCES_DIR"
GLINER_MODEL_DIR="${GLINER_MODEL_DIR:-$PII_ROOT/gliner-pii-large-v1.0}"
QWEN3GUARD_MODEL="${QWEN3GUARD_MODEL:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"
QWEN35_MODEL="${QWEN35_MODEL:-$PROJECT/models/Qwen/Qwen3.5-9B}"
TEXT_WORK_DIR="${TEXT_WORK_DIR:-$PROJECT/temp/text_local_output}"

PII_HOST="${PII_HOST:-127.0.0.1}"
PII_PORT="${PII_PORT:-5002}"
QWEN3GUARD_VLLM_HOST="${QWEN3GUARD_VLLM_HOST:-127.0.0.1}"
QWEN3GUARD_VLLM_PORT="${QWEN3GUARD_VLLM_PORT:-8212}"
QWEN3GUARD_ADAPTER_HOST="${QWEN3GUARD_ADAPTER_HOST:-127.0.0.1}"
QWEN3GUARD_ADAPTER_PORT="${QWEN3GUARD_ADAPTER_PORT:-8215}"
QWEN35_HOST="${QWEN35_HOST:-127.0.0.1}"
QWEN35_PORT="${QWEN35_PORT:-8301}"
TEXT_API_HOST="${TEXT_API_HOST:-127.0.0.1}"
TEXT_API_PORT="${TEXT_API_PORT:-19002}"

QWEN3GUARD_GPU="${QWEN3GUARD_GPU:-3}"
QWEN35_GPU="${QWEN35_GPU:-3}"
QWEN3GUARD_GPU_MEMORY_UTILIZATION="${QWEN3GUARD_GPU_MEMORY_UTILIZATION:-0.08}"
QWEN35_GPU_MEMORY_UTILIZATION="${QWEN35_GPU_MEMORY_UTILIZATION:-0.5}"
QWEN3GUARD_MAX_MODEL_LEN="${QWEN3GUARD_MAX_MODEL_LEN:-1536}"
QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-6144}"
QWEN3GUARD_MAX_NUM_SEQS="${QWEN3GUARD_MAX_NUM_SEQS:-1}"
QWEN35_MAX_NUM_SEQS="${QWEN35_MAX_NUM_SEQS:-1}"
VLLM_CMD="${VLLM_CMD:-vllm}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-300}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-3}"

PII_ENV_ACTIVATE="${PII_ENV_ACTIVATE:-$PROJECT/.venvs/compliance-pii/bin/activate}"
QWEN3GUARD_ENV_ACTIVATE="${QWEN3GUARD_ENV_ACTIVATE:-$PROJECT/qwen-serving/asr-vllm/.venv/bin/activate}"
QWEN35_ENV_ACTIVATE="${QWEN35_ENV_ACTIVATE:-$PROJECT/qwen-serving/text-vllm/.venv/bin/activate}"
GUARD_ADAPTER_ENV_ACTIVATE="${GUARD_ADAPTER_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
TEXT_ENV_ACTIVATE="${TEXT_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"

QWEN3GUARD_SERVED_MODEL="${QWEN3GUARD_SERVED_MODEL:-Qwen3Guard-Gen-0.6B}"
QWEN35_SERVED_MODEL="${QWEN35_SERVED_MODEL:-Qwen3.5-9B}"

PII_STANZA_EN_MODEL="${PII_STANZA_EN_MODEL:-en}"
PII_STANZA_ZH_MODEL="${PII_STANZA_ZH_MODEL:-zh}"
PII_STANZA_DOWNLOAD_IF_MISSING="${PII_STANZA_DOWNLOAD_IF_MISSING:-false}"
PII_GLINER_THRESHOLD="${PII_GLINER_THRESHOLD:-0.50}"

RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/text_local_stack_tmux}"

usage() {
  cat <<EOF
Usage: $0 [start|restart|stop|status|attach]

This is the preferred local-model-first text compliance stack:
  pii-gateway -> qwen3guard-vllm -> qwen3guard-adapter -> qwen35-vllm -> text-api-server

Environment overrides:
  SESSION=$SESSION
  PROJECT=$PROJECT
  PII_ROOT=$PII_ROOT
  GLINER_MODEL_DIR=$GLINER_MODEL_DIR
  QWEN3GUARD_MODEL=$QWEN3GUARD_MODEL
  QWEN35_MODEL=$QWEN35_MODEL
  TEXT_WORK_DIR=$TEXT_WORK_DIR
  PII_ENV_ACTIVATE=$PII_ENV_ACTIVATE
  QWEN3GUARD_ENV_ACTIVATE=$QWEN3GUARD_ENV_ACTIVATE
  QWEN35_ENV_ACTIVATE=$QWEN35_ENV_ACTIVATE
  GUARD_ADAPTER_ENV_ACTIVATE=$GUARD_ADAPTER_ENV_ACTIVATE
  TEXT_ENV_ACTIVATE=$TEXT_ENV_ACTIVATE
  QWEN3GUARD_ADAPTER_PORT=$QWEN3GUARD_ADAPTER_PORT
  QWEN35_PORT=$QWEN35_PORT
  TEXT_API_PORT=$TEXT_API_PORT
  QWEN3GUARD_GPU=$QWEN3GUARD_GPU
  QWEN35_GPU=$QWEN35_GPU
  VLLM_CMD=$VLLM_CMD
  ATTACH=$ATTACH
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

probe_http() {
  local url="$1"
  curl -sS --connect-timeout 5 --max-time 15 "$url" >/dev/null
}

q() {
  printf "%q" "$1"
}

activation_line() {
  local command="$1"
  if [[ -z "$command" ]]; then
    return
  fi
  printf 'source %q\n' "$command"
}

python_bin_for() {
  local activate="$1"
  printf '%s\n' "${activate%/activate}/python"
}

session_exists() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
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

require_vllm_in_env() {
  local activate="$1"
  require_python_module "$activate" "vllm" || {
    echo "vLLM is not installed in env: $activate" >&2
    exit 1
  }
  bash -lc "source $(q "$activate") && command -v $(q "$VLLM_CMD") >/dev/null 2>&1" || {
    echo "vLLM command '$VLLM_CMD' is not available after activating env: $activate" >&2
    exit 1
  }
}

require_transformers_symbol() {
  local activate="$1"
  local symbol_name="$2"
  local python_bin
  python_bin="$(python_bin_for "$activate")"
  [[ -x "$python_bin" ]] || {
    echo "Python executable not found for env: $activate" >&2
    exit 1
  }
  "$python_bin" - <<PY >/dev/null
from transformers import ${symbol_name}
print(${symbol_name})
PY
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
echo "[text-local] Process exited with status \$status. Press Ctrl+D or close this tmux pane to exit."
exec bash
EOF
  chmod +x "$path"
}

validate_paths() {
  [[ -d "$PROJECT" ]] || { echo "PROJECT does not exist: $PROJECT" >&2; exit 1; }
  [[ -d "$STANZA_RESOURCES_DIR" ]] || { echo "STANZA_RESOURCES_DIR does not exist: $STANZA_RESOURCES_DIR" >&2; exit 1; }
  [[ -d "$GLINER_MODEL_DIR" ]] || { echo "GLINER_MODEL_DIR does not exist: $GLINER_MODEL_DIR" >&2; exit 1; }
  [[ -d "$QWEN3GUARD_MODEL" ]] || { echo "QWEN3GUARD_MODEL does not exist: $QWEN3GUARD_MODEL" >&2; exit 1; }
  [[ -d "$QWEN35_MODEL" ]] || { echo "QWEN35_MODEL does not exist: $QWEN35_MODEL" >&2; exit 1; }

  for activation in "$PII_ENV_ACTIVATE" "$QWEN3GUARD_ENV_ACTIVATE" "$QWEN35_ENV_ACTIVATE" "$GUARD_ADAPTER_ENV_ACTIVATE" "$TEXT_ENV_ACTIVATE"; do
    [[ -f "$activation" ]] || { echo "Virtual environment activation script does not exist: $activation" >&2; exit 1; }
  done

  require_python_module "$PII_ENV_ACTIVATE" "fastapi"
  require_python_module "$PII_ENV_ACTIVATE" "presidio_analyzer"
  require_python_module "$PII_ENV_ACTIVATE" "gliner"
  require_python_module "$QWEN3GUARD_ENV_ACTIVATE" "transformers"
  require_transformers_symbol "$QWEN3GUARD_ENV_ACTIVATE" "GenerationConfig"
  require_vllm_in_env "$QWEN3GUARD_ENV_ACTIVATE"
  require_python_module "$GUARD_ADAPTER_ENV_ACTIVATE" "fastapi"
  require_python_module "$GUARD_ADAPTER_ENV_ACTIVATE" "httpx"
  require_python_module "$TEXT_ENV_ACTIVATE" "fastapi"
  require_python_module "$TEXT_ENV_ACTIVATE" "httpx"
  require_python_module "$TEXT_ENV_ACTIVATE" "multipart"
  require_transformers_symbol "$QWEN35_ENV_ACTIVATE" "GenerationConfig"
  require_vllm_in_env "$QWEN35_ENV_ACTIVATE"

  mkdir -p "$TEXT_WORK_DIR"
}

write_runners() {
  local pii_script="$RUN_DIR/pii_gateway.sh"
  local guard_vllm_script="$RUN_DIR/qwen3guard_vllm.sh"
  local guard_adapter_script="$RUN_DIR/qwen3guard_adapter.sh"
  local qwen35_vllm_script="$RUN_DIR/qwen35_vllm.sh"
  local text_api_script="$RUN_DIR/text_api_server.sh"

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
export TOKENIZERS_PARALLELISM=false
echo '[text-local] Starting PII Gateway on ${PII_HOST}:${PII_PORT}'
python -m uvicorn ops.presidio_bilingual.app:app --host $(q "$PII_HOST") --port $(q "$PII_PORT")
"

  write_runner "$guard_vllm_script" "
cd $(q "$PROJECT")
$(activation_line "$QWEN3GUARD_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$QWEN3GUARD_GPU")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
unset PYTHONPATH
echo '[text-local] Starting Qwen3Guard vLLM on ${QWEN3GUARD_VLLM_HOST}:${QWEN3GUARD_VLLM_PORT}'
${VLLM_CMD} serve $(q "$QWEN3GUARD_MODEL") \\
  --served-model-name $(q "$QWEN3GUARD_SERVED_MODEL") \\
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
export QWEN3GUARD_API_KEY=
export QWEN3GUARD_MODEL=$(q "$QWEN3GUARD_SERVED_MODEL")
echo '[text-local] Starting Qwen3Guard Adapter on ${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}'
python -m uvicorn ops.qwen3guard_adapter:app --host $(q "$QWEN3GUARD_ADAPTER_HOST") --port $(q "$QWEN3GUARD_ADAPTER_PORT")
"

  write_runner "$qwen35_vllm_script" "
cd $(q "$PROJECT")
$(activation_line "$QWEN35_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$QWEN35_GPU")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
unset PYTHONPATH
echo '[text-local] Starting Qwen3.5 vLLM on ${QWEN35_HOST}:${QWEN35_PORT}'
${VLLM_CMD} serve $(q "$QWEN35_MODEL") \\
  --served-model-name $(q "$QWEN35_SERVED_MODEL") \\
  --host $(q "$QWEN35_HOST") \\
  --port $(q "$QWEN35_PORT") \\
  --dtype auto \\
  --max-model-len $(q "$QWEN35_MAX_MODEL_LEN") \\
  --max-num-seqs $(q "$QWEN35_MAX_NUM_SEQS") \\
  --gpu-memory-utilization $(q "$QWEN35_GPU_MEMORY_UTILIZATION") \\
  --trust-remote-code
"

  write_runner "$text_api_script" "
cd $(q "$PROJECT")
$(activation_line "$TEXT_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=\"\"
export COMPLIANCE_WORK_DIR=$(q "$TEXT_WORK_DIR")
export COMPLIANCE_SERVER_HOST=$(q "$TEXT_API_HOST")
export COMPLIANCE_API_SERVER_PORT=$(q "$TEXT_API_PORT")
export COMPLIANCE_COMPLIANCE_PROVIDER_MODE=local
export COMPLIANCE_LOCAL_COMPLIANCE_BASE_URL=http://${QWEN35_HOST}:${QWEN35_PORT}/v1
export COMPLIANCE_LOCAL_COMPLIANCE_API_KEY=
export COMPLIANCE_LOCAL_COMPLIANCE_MODEL=$(q "$QWEN35_SERVED_MODEL")
export COMPLIANCE_LOCAL_COMPLIANCE_TIMEOUT_SECONDS=180
export COMPLIANCE_LOCAL_COMPLIANCE_MAX_CHARS=4800
export COMPLIANCE_LOCAL_COMPLIANCE_MAX_TOKENS=1024
export COMPLIANCE_ENABLE_PRESIDIO=true
export COMPLIANCE_PRESIDIO_ANALYZER_ENDPOINT=http://${PII_HOST}:${PII_PORT}/analyze
export COMPLIANCE_PRESIDIO_LANGUAGE=auto
export COMPLIANCE_PRESIDIO_SUPPORTED_LANGUAGES=en,zh
export COMPLIANCE_PRESIDIO_LANGUAGE_FALLBACK=en
export COMPLIANCE_PRESIDIO_SCORE_THRESHOLD=0.45
export COMPLIANCE_PRESIDIO_TIMEOUT_SECONDS=60
export COMPLIANCE_ENABLE_QWEN3GUARD=true
export COMPLIANCE_QWEN3GUARD_ENDPOINT=http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate
export COMPLIANCE_QWEN3GUARD_MODEL_NAME=$(q "$QWEN3GUARD_SERVED_MODEL")
export COMPLIANCE_QWEN3GUARD_TIMEOUT_SECONDS=90
export COMPLIANCE_QWEN3GUARD_MAX_CHARS=8000
export COMPLIANCE_ENABLE_HARD_CASE_ADJUDICATION=true
export COMPLIANCE_HARD_CASE_ENDPOINT=
export COMPLIANCE_HARD_CASE_LOCAL_MODEL_PATH=
echo '[text-local] Starting text.api_server on ${TEXT_API_HOST}:${TEXT_API_PORT}'
python -m uvicorn text.api_server:app --host $(q "$TEXT_API_HOST") --port $(q "$TEXT_API_PORT")
"
}

start_session() {
  require_cmd tmux
  require_cmd curl
  validate_paths

  if session_exists; then
    echo "tmux session already exists: $SESSION" >&2
    echo "Use '$0 attach', '$0 stop', or '$0 restart'." >&2
    exit 1
  fi

  write_runners

  tmux new-session -d -s "$SESSION" -n pii-gateway "$RUN_DIR/pii_gateway.sh"
  tmux new-window -t "$SESSION" -n qwen35-vllm "$RUN_DIR/qwen35_vllm.sh"
  _wait_for_endpoint "Qwen3.5 vLLM" "http://${QWEN35_HOST}:${QWEN35_PORT}/v1/models"

  tmux new-window -t "$SESSION" -n qwen3guard-vllm "$RUN_DIR/qwen3guard_vllm.sh"
  _wait_for_endpoint "Qwen3Guard vLLM" "http://${QWEN3GUARD_VLLM_HOST}:${QWEN3GUARD_VLLM_PORT}/v1/models"

  tmux new-window -t "$SESSION" -n qwen3guard-adapter "$RUN_DIR/qwen3guard_adapter.sh"
  _wait_for_endpoint "Qwen3Guard adapter" "http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/health"

  tmux new-window -t "$SESSION" -n text-api-server "$RUN_DIR/text_api_server.sh"
  _wait_for_endpoint "text.api_server" "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/health"
  tmux select-window -t "$SESSION:pii-gateway"

  cat <<EOF
Started tmux session: $SESSION

Windows:
  1. pii-gateway        -> http://${PII_HOST}:${PII_PORT}/analyze
  2. qwen35-vllm        -> http://${QWEN35_HOST}:${QWEN35_PORT}/v1/chat/completions
  3. qwen3guard-vllm    -> http://${QWEN3GUARD_VLLM_HOST}:${QWEN3GUARD_VLLM_PORT}/v1/chat/completions
  4. qwen3guard-adapter -> http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate
  5. text-api-server    -> http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/health

Useful commands:
  tmux attach -t $SESSION
  tmux ls
  $0 stop
EOF

  if [[ "$ATTACH" == "true" ]]; then
    tmux attach -t "$SESSION"
  fi
}

_wait_for_endpoint() {
  local label="$1"
  local url="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  echo "[text-local] Waiting for ${label} at ${url}"
  while (( SECONDS < deadline )); do
    if probe_http "$url"; then
      echo "[text-local] ${label} is ready"
      return 0
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
  echo "[text-local] ${label} did not become ready within ${STARTUP_TIMEOUT_SECONDS}s" >&2
  echo "[text-local] Inspect the tmux session with: tmux attach -t ${SESSION}" >&2
  exit 1
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
