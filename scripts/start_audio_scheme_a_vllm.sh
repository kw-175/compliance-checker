#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-start}"
SESSION="${SESSION:-audio-scheme-a-vllm}"
ATTACH="${ATTACH:-true}"

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
QWEN_ROOT="${QWEN_ROOT:-$PROJECT/qwen-serving}"
RUN_DIR="${RUN_DIR:-$PROJECT/.tmp/audio_scheme_a_vllm_tmux}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_a100_output}"

TEXT_ENV_DIR="${TEXT_ENV_DIR:-$QWEN_ROOT/text-vllm/.venv}"
ASR_ENV_DIR="${ASR_ENV_DIR:-$QWEN_ROOT/asr-vllm/.venv}"
PII_ENV_ACTIVATE="${PII_ENV_ACTIVATE:-$PROJECT/.venvs/compliance-pii/bin/activate}"
AUDIO_ENV_ACTIVATE="${AUDIO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"

TEXT_ENV_ACTIVATE="${TEXT_ENV_ACTIVATE:-$TEXT_ENV_DIR/bin/activate}"
ASR_ENV_ACTIVATE="${ASR_ENV_ACTIVATE:-$ASR_ENV_DIR/bin/activate}"

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

GUARD_VLLM_HOST="${GUARD_VLLM_HOST:-127.0.0.1}"
GUARD_VLLM_PORT="${GUARD_VLLM_PORT:-8212}"
HARD_CASE_VLLM_HOST="${HARD_CASE_VLLM_HOST:-127.0.0.1}"
HARD_CASE_VLLM_PORT="${HARD_CASE_VLLM_PORT:-8213}"

PII_ROOT="${PII_ROOT:-$PROJECT/models/compliance-pii}"
PII_STANZA_RESOURCES_DIR="${PII_STANZA_RESOURCES_DIR:-$PII_ROOT/stanza_resources}"
GLINER_MODEL_DIR="${GLINER_MODEL_DIR:-$PII_ROOT/gliner-pii-large-v1.0}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$PROJECT/models/Qwen/Qwen3-ASR-0.6B}"
QWEN_GUARD_MODEL="${QWEN_GUARD_MODEL:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"
QWEN35_MODEL="${QWEN35_MODEL:-$PROJECT/models/Qwen/Qwen3.5-9B}"

GUARD_VLLM_SERVED_MODEL="${GUARD_VLLM_SERVED_MODEL:-Qwen3Guard-Gen-0.6B}"
HARD_CASE_VLLM_SERVED_MODEL="${HARD_CASE_VLLM_SERVED_MODEL:-Qwen3.5-9B}"

FFMPEG_BIN="${FFMPEG_BIN:-/data/kw/.local/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/data/kw/.local/bin/ffprobe}"

AUDIO_GPU="${AUDIO_GPU:-0}"
QWEN_ASR_DEVICE="${QWEN_ASR_DEVICE:-cuda}"
PII_STANZA_EN_MODEL="${PII_STANZA_EN_MODEL:-en}"
PII_STANZA_ZH_MODEL="${PII_STANZA_ZH_MODEL:-zh}"
PII_STANZA_DOWNLOAD_IF_MISSING="${PII_STANZA_DOWNLOAD_IF_MISSING:-false}"
PII_GLINER_THRESHOLD="${PII_GLINER_THRESHOLD:-0.50}"
PII_SCORE_THRESHOLD="${PII_SCORE_THRESHOLD:-0.45}"

TEXT_VLLM_SPEC="${TEXT_VLLM_SPEC:-vllm}"
TEXT_VLLM_EXTRA_INDEX_URL="${TEXT_VLLM_EXTRA_INDEX_URL:-https://wheels.vllm.ai/nightly}"
TEXT_TORCH_BACKEND="${TEXT_TORCH_BACKEND:-cu128}"
ASR_QWEN_ASR_SPEC="${ASR_QWEN_ASR_SPEC:-qwen-asr[vllm]==0.0.6}"
ASR_PYTORCH_INDEX_URL="${ASR_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

GUARD_GPU_MEMORY_UTILIZATION="${GUARD_GPU_MEMORY_UTILIZATION:-0.35}"
HARD_CASE_GPU_MEMORY_UTILIZATION="${HARD_CASE_GPU_MEMORY_UTILIZATION:-0.45}"
HARD_CASE_MAX_MODEL_LEN="${HARD_CASE_MAX_MODEL_LEN:-32768}"
QWEN3ASR_GPU_MEMORY_UTILIZATION="${QWEN3ASR_GPU_MEMORY_UTILIZATION:-0.7}"
QWEN3ASR_MAX_INFERENCE_BATCH_SIZE="${QWEN3ASR_MAX_INFERENCE_BATCH_SIZE:-32}"
QWEN3ASR_MAX_NEW_TOKENS="${QWEN3ASR_MAX_NEW_TOKENS:-4096}"
HARD_CASE_MAX_NEW_TOKENS="${HARD_CASE_MAX_NEW_TOKENS:-1024}"

usage() {
  cat <<EOF
Usage: $0 [setup|start|restart|stop|status|attach]

Scheme A:
  - text-vllm env: Qwen3Guard + Qwen3.5-9B
  - asr-vllm env:  Qwen3-ASR

Environment directories:
  TEXT_ENV_DIR=$TEXT_ENV_DIR
  ASR_ENV_DIR=$ASR_ENV_DIR
EOF
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
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

session_exists() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
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
  [[ -f "$AUDIO_ENV_ACTIVATE" ]] || { echo "AUDIO_ENV_ACTIVATE does not exist: $AUDIO_ENV_ACTIVATE" >&2; exit 1; }
  [[ -f "$TEXT_ENV_ACTIVATE" ]] || { echo "TEXT_ENV_ACTIVATE does not exist: $TEXT_ENV_ACTIVATE" >&2; exit 1; }
  [[ -f "$ASR_ENV_ACTIVATE" ]] || { echo "ASR_ENV_ACTIVATE does not exist: $ASR_ENV_ACTIVATE" >&2; exit 1; }
  [[ -f "$FFMPEG_BIN" ]] || { echo "FFMPEG_BIN does not exist: $FFMPEG_BIN" >&2; exit 1; }
  [[ -f "$FFPROBE_BIN" ]] || { echo "FFPROBE_BIN does not exist: $FFPROBE_BIN" >&2; exit 1; }
  [[ -d "$PII_STANZA_RESOURCES_DIR" ]] || { echo "PII_STANZA_RESOURCES_DIR does not exist: $PII_STANZA_RESOURCES_DIR" >&2; exit 1; }
  [[ -d "$GLINER_MODEL_DIR" ]] || { echo "GLINER_MODEL_DIR does not exist: $GLINER_MODEL_DIR" >&2; exit 1; }
  [[ -d "$QWEN_ASR_MODEL" ]] || { echo "QWEN_ASR_MODEL does not exist: $QWEN_ASR_MODEL" >&2; exit 1; }
  [[ -d "$QWEN_GUARD_MODEL" ]] || { echo "QWEN_GUARD_MODEL does not exist: $QWEN_GUARD_MODEL" >&2; exit 1; }
  [[ -d "$QWEN35_MODEL" ]] || { echo "QWEN35_MODEL does not exist: $QWEN35_MODEL" >&2; exit 1; }
  mkdir -p "$AUDIO_WORK_DIR" "$RUN_DIR"
}

setup_text_env() {
  mkdir -p "$(dirname "$TEXT_ENV_DIR")"
  if [[ ! -x "$TEXT_ENV_DIR/bin/python" ]]; then
    uv venv "$TEXT_ENV_DIR" --python 3.12
  fi
  if [[ -n "$TEXT_VLLM_EXTRA_INDEX_URL" ]]; then
    uv pip install \
      --python "$TEXT_ENV_DIR/bin/python" \
      --torch-backend "$TEXT_TORCH_BACKEND" \
      --extra-index-url "$TEXT_VLLM_EXTRA_INDEX_URL" \
      "$TEXT_VLLM_SPEC"
  else
    uv pip install \
      --python "$TEXT_ENV_DIR/bin/python" \
      --torch-backend "$TEXT_TORCH_BACKEND" \
      "$TEXT_VLLM_SPEC"
  fi
  uv pip install \
    --python "$TEXT_ENV_DIR/bin/python" \
    "fastapi>=0.115,<1.0" \
    "uvicorn[standard]>=0.30,<1.0" \
    "httpx>=0.28,<1.0" \
    "pydantic>=2.10,<3.0" \
    "pydantic-settings>=2.6,<3.0" \
    "pillow>=10,<12" \
    "numpy<2.5"
}

setup_asr_env() {
  mkdir -p "$(dirname "$ASR_ENV_DIR")"
  if [[ ! -x "$ASR_ENV_DIR/bin/python" ]]; then
    uv venv "$ASR_ENV_DIR" --python 3.12
  fi
  uv pip install \
    --python "$ASR_ENV_DIR/bin/python" \
    --extra-index-url "$ASR_PYTORCH_INDEX_URL" \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "torchaudio==2.10.0"
  uv pip install \
    --python "$ASR_ENV_DIR/bin/python" \
    "$ASR_QWEN_ASR_SPEC" \
    "fastapi>=0.115,<1.0" \
    "uvicorn[standard]>=0.30,<1.0" \
    "httpx>=0.28,<1.0" \
    "pydantic>=2.10,<3.0" \
    "pydantic-settings>=2.6,<3.0" \
    "numpy<2.5"
}

setup_envs() {
  need_cmd uv
  validate_model_paths_only
  setup_text_env
  setup_asr_env
  cat <<EOF
Setup complete.

Text env:
  $TEXT_ENV_DIR
ASR env:
  $ASR_ENV_DIR

Next:
  $0 start
EOF
}

validate_model_paths_only() {
  [[ -d "$PROJECT" ]] || { echo "PROJECT does not exist: $PROJECT" >&2; exit 1; }
  [[ -d "$QWEN_ASR_MODEL" ]] || { echo "QWEN_ASR_MODEL does not exist: $QWEN_ASR_MODEL" >&2; exit 1; }
  [[ -d "$QWEN_GUARD_MODEL" ]] || { echo "QWEN_GUARD_MODEL does not exist: $QWEN_GUARD_MODEL" >&2; exit 1; }
  [[ -d "$QWEN35_MODEL" ]] || { echo "QWEN35_MODEL does not exist: $QWEN35_MODEL" >&2; exit 1; }
}

write_runners() {
  write_runner "$RUN_DIR/pii_gateway.sh" "pii" "
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
python -m uvicorn ops.presidio_bilingual.app:app --host $(q "$PII_HOST") --port $(q "$PII_PORT")
"

  write_runner "$RUN_DIR/guard_vllm.sh" "guard-vllm" "
cd $(q "$PROJECT")
$(activation_line "$TEXT_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
vllm serve $(q "$QWEN_GUARD_MODEL") \\
  --served-model-name $(q "$GUARD_VLLM_SERVED_MODEL") \\
  --host $(q "$GUARD_VLLM_HOST") \\
  --port $(q "$GUARD_VLLM_PORT") \\
  --dtype auto \\
  --gpu-memory-utilization $(q "$GUARD_GPU_MEMORY_UTILIZATION") \\
  --max-model-len 32768 \\
  --trust-remote-code
"

  write_runner "$RUN_DIR/guard_adapter.sh" "guard-adapter" "
cd $(q "$PROJECT")
$(activation_line "$TEXT_ENV_ACTIVATE")
export QWEN3GUARD_VLLM_URL=http://${GUARD_VLLM_HOST}:${GUARD_VLLM_PORT}/v1/chat/completions
export QWEN3GUARD_VLLM_MODEL=$(q "$GUARD_VLLM_SERVED_MODEL")
export QWEN3GUARD_MODEL=$(q "$QWEN_GUARD_MODEL")
python -m uvicorn ops.qwen3guard_vllm_adapter:app --host $(q "$GUARD_HOST") --port $(q "$GUARD_PORT")
"

  write_runner "$RUN_DIR/hardcase_vllm.sh" "hardcase-vllm" "
cd $(q "$PROJECT")
$(activation_line "$TEXT_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
vllm serve $(q "$QWEN35_MODEL") \\
  --served-model-name $(q "$HARD_CASE_VLLM_SERVED_MODEL") \\
  --host $(q "$HARD_CASE_VLLM_HOST") \\
  --port $(q "$HARD_CASE_VLLM_PORT") \\
  --tensor-parallel-size 1 \\
  --max-model-len $(q "$HARD_CASE_MAX_MODEL_LEN") \\
  --gpu-memory-utilization $(q "$HARD_CASE_GPU_MEMORY_UTILIZATION") \\
  --reasoning-parser qwen3 \\
  --language-model-only \\
  --trust-remote-code
"

  write_runner "$RUN_DIR/hardcase_adapter.sh" "hardcase-adapter" "
cd $(q "$PROJECT")
$(activation_line "$TEXT_ENV_ACTIVATE")
export QWEN35_HARDCASE_VLLM_URL=http://${HARD_CASE_VLLM_HOST}:${HARD_CASE_VLLM_PORT}/v1/chat/completions
export QWEN35_HARDCASE_VLLM_MODEL=$(q "$HARD_CASE_VLLM_SERVED_MODEL")
export QWEN35_HARDCASE_MODEL=$(q "$QWEN35_MODEL")
export QWEN35_HARDCASE_MAX_TOKENS=$(q "$HARD_CASE_MAX_NEW_TOKENS")
python -m uvicorn ops.qwen35_hardcase_vllm_adapter:app --host $(q "$HARD_CASE_HOST") --port $(q "$HARD_CASE_PORT")
"

  write_runner "$RUN_DIR/asr_adapter.sh" "asr-adapter" "
cd $(q "$PROJECT")
$(activation_line "$ASR_ENV_ACTIVATE")
export CUDA_VISIBLE_DEVICES=$(q "$AUDIO_GPU")
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN3ASR_MODEL=$(q "$QWEN_ASR_MODEL")
export QWEN3ASR_DEVICE=$(q "$QWEN_ASR_DEVICE")
export QWEN3ASR_GPU_MEMORY_UTILIZATION=$(q "$QWEN3ASR_GPU_MEMORY_UTILIZATION")
export QWEN3ASR_MAX_INFERENCE_BATCH_SIZE=$(q "$QWEN3ASR_MAX_INFERENCE_BATCH_SIZE")
export QWEN3ASR_MAX_NEW_TOKENS=$(q "$QWEN3ASR_MAX_NEW_TOKENS")
python -m uvicorn ops.qwen3asr_vllm_adapter:app --host $(q "$ASR_HOST") --port $(q "$ASR_PORT")
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
python -m uvicorn audio.server:app --host $(q "$AUDIO_HOST") --port $(q "$AUDIO_PORT")
"
}

start_session() {
  need_cmd tmux
  validate_paths
  if session_exists; then
    echo "tmux session already exists: $SESSION" >&2
    echo "Use '$0 attach', '$0 stop', or '$0 restart'." >&2
    exit 1
  fi
  write_runners
  tmux new-session -d -s "$SESSION" -n pii "$RUN_DIR/pii_gateway.sh"
  tmux new-window -t "$SESSION" -n guard-vllm "$RUN_DIR/guard_vllm.sh"
  tmux new-window -t "$SESSION" -n guard-adapter "$RUN_DIR/guard_adapter.sh"
  tmux new-window -t "$SESSION" -n hardcase-vllm "$RUN_DIR/hardcase_vllm.sh"
  tmux new-window -t "$SESSION" -n hardcase-adapter "$RUN_DIR/hardcase_adapter.sh"
  tmux new-window -t "$SESSION" -n asr-adapter "$RUN_DIR/asr_adapter.sh"
  tmux new-window -t "$SESSION" -n audio-server "$RUN_DIR/audio_server.sh"
  tmux select-window -t "$SESSION:audio-server"

  cat <<EOF
Started tmux session: $SESSION

Endpoints:
  PII          -> http://${PII_HOST}:${PII_PORT}/analyze
  ASR          -> http://${ASR_HOST}:${ASR_PORT}/transcribe
  Guard        -> http://${GUARD_HOST}:${GUARD_PORT}/moderate
  Hardcase     -> http://${HARD_CASE_HOST}:${HARD_CASE_PORT}/adjudicate
  Audio server -> http://${AUDIO_HOST}:${AUDIO_PORT}

Use existing smoke test:
  bash $PROJECT/scripts/test_audio_a100.sh
EOF

  if [[ "$ATTACH" == "true" ]]; then
    tmux attach -t "$SESSION"
  fi
}

stop_session() {
  need_cmd tmux
  if session_exists; then
    tmux kill-session -t "$SESSION"
    echo "Stopped tmux session: $SESSION"
  else
    echo "No tmux session found: $SESSION"
  fi
}

status_session() {
  need_cmd tmux
  if session_exists; then
    tmux list-windows -t "$SESSION"
  else
    echo "No tmux session found: $SESSION"
    exit 1
  fi
}

case "$ACTION" in
  setup)
    setup_envs
    ;;
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
    need_cmd tmux
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
