#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
TEMP_DIR="${TEMP_DIR:-$PROJECT/temp/audio_real_wav_test}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_real_wav_output}"
AUDIO_BASE_URL="${AUDIO_BASE_URL:-http://127.0.0.1:19001}"
TEXT_API_BASE_URL="${TEXT_API_BASE_URL:-http://127.0.0.1:19002}"

CONTENT_AUDIO="${CONTENT_AUDIO:-$PROJECT/test_data/Content_safety.wav}"
PRIVACY_AUDIO="${PRIVACY_AUDIO:-$PROJECT/test_data/Privacy detection.wav}"

START_STACK="${START_STACK:-true}"
START_STACK_SCRIPT="${START_STACK_SCRIPT:-$PROJECT/scripts/start_audio_local_stack.sh}"
ASR_BACKEND="${ASR_BACKEND:-vllm}"
ASR_GPU="${ASR_GPU:-0}"
QWEN3ASR_GPU_MEMORY_UTILIZATION="${QWEN3ASR_GPU_MEMORY_UTILIZATION:-0.12}"
QWEN3ASR_MAX_INFERENCE_BATCH_SIZE="${QWEN3ASR_MAX_INFERENCE_BATCH_SIZE:-1}"
QWEN3ASR_MAX_NEW_TOKENS="${QWEN3ASR_MAX_NEW_TOKENS:-2048}"
QWEN3ASR_MAX_MODEL_LEN="${QWEN3ASR_MAX_MODEL_LEN:-4096}"
QWEN3ASR_TENSOR_PARALLEL_SIZE="${QWEN3ASR_TENSOR_PARALLEL_SIZE:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-5}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"

MIN_CONTENT_RISK_RECORDS="${MIN_CONTENT_RISK_RECORDS:-1}"
MIN_PRIVACY_RISK_RECORDS="${MIN_PRIVACY_RISK_RECORDS:-1}"
MIN_ASR_SEGMENTS="${MIN_ASR_SEGMENTS:-1}"

CONTENT_SUBMIT_JSON="$TEMP_DIR/content_submit.json"
CONTENT_STATUS_JSON="$TEMP_DIR/content_status.json"
CONTENT_RESULT_JSON="$TEMP_DIR/content_result.json"
PRIVACY_SUBMIT_JSON="$TEMP_DIR/privacy_submit.json"
PRIVACY_STATUS_JSON="$TEMP_DIR/privacy_status.json"
PRIVACY_RESULT_JSON="$TEMP_DIR/privacy_result.json"

log() {
  printf '[audio-real-test] %s\n' "$*"
}

fail() {
  printf '[audio-real-test] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

get_json() {
  local url="$1"
  curl -sS "$url" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME"
}

post_json() {
  local url="$1"
  local payload="$2"
  curl -sS -X POST "$url" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME" \
    -H "Content-Type: application/json" \
    -d "$payload"
}

json_pretty() {
  python -m json.tool
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

validate_inputs() {
  [[ -d "$PROJECT" ]] || fail "PROJECT does not exist: $PROJECT"
  [[ -f "$CONTENT_AUDIO" ]] || fail "CONTENT_AUDIO does not exist: $CONTENT_AUDIO"
  [[ -f "$PRIVACY_AUDIO" ]] || fail "PRIVACY_AUDIO does not exist: $PRIVACY_AUDIO"
  mkdir -p "$TEMP_DIR" "$AUDIO_WORK_DIR"
}

ensure_stack() {
  if probe_http "$AUDIO_BASE_URL/api/v1/health" && probe_http "$TEXT_API_BASE_URL/api/v1/health"; then
    log "Audio and text APIs are already healthy"
    return
  fi
  if [[ "$START_STACK" != "true" ]]; then
    fail "Required services are not healthy. Start them first or set START_STACK=true."
  fi
  [[ -f "$START_STACK_SCRIPT" ]] || fail "START_STACK_SCRIPT not found: $START_STACK_SCRIPT"
  log "Starting audio local stack via $START_STACK_SCRIPT"
  ATTACH=false \
    AUDIO_WORK_DIR="$AUDIO_WORK_DIR" \
    TEXT_API_BASE_URL="$TEXT_API_BASE_URL" \
    ASR_BACKEND="$ASR_BACKEND" \
    ASR_GPU="$ASR_GPU" \
    QWEN3ASR_GPU_MEMORY_UTILIZATION="$QWEN3ASR_GPU_MEMORY_UTILIZATION" \
    QWEN3ASR_MAX_INFERENCE_BATCH_SIZE="$QWEN3ASR_MAX_INFERENCE_BATCH_SIZE" \
    QWEN3ASR_MAX_NEW_TOKENS="$QWEN3ASR_MAX_NEW_TOKENS" \
    QWEN3ASR_MAX_MODEL_LEN="$QWEN3ASR_MAX_MODEL_LEN" \
    QWEN3ASR_TENSOR_PARALLEL_SIZE="$QWEN3ASR_TENSOR_PARALLEL_SIZE" \
    bash "$START_STACK_SCRIPT" start || true
  wait_for_http "text-api" "$TEXT_API_BASE_URL/api/v1/health"
  wait_for_http "audio-api" "$AUDIO_BASE_URL/api/v1/health"
}

submit_job() {
  local label="$1"
  local audio_path="$2"
  local operator_id="$3"
  local pipeline_profile="$4"
  local submit_path="$5"

  log "Submitting $label audio: $audio_path"
  local payload
  payload=$(python - <<PY
import json
print(json.dumps({
    "input_paths": [${audio_path@Q}],
    "config_overrides": {
        "operator_id": ${operator_id@Q},
        "dataset_name": ${label@Q},
        "pipeline_profile": ${pipeline_profile@Q},
        "work_dir": ${AUDIO_WORK_DIR@Q},
        "text_api_base_url": ${TEXT_API_BASE_URL@Q},
        "audio_execution_route": "api",
        "qwen_asr_enabled": True,
        "faster_whisper_enabled": False,
        "pyannote_enabled": False
    }
}, ensure_ascii=False))
PY
)
  post_json "$AUDIO_BASE_URL/api/v1/check" "$payload" | tee "$submit_path" >/dev/null
  json_pretty < "$submit_path" >/dev/null
}

extract_task_id() {
  local submit_path="$1"
  python - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${submit_path@Q}).read_text(encoding="utf-8"))
task_id = payload.get("task_id")
if not task_id:
    raise SystemExit(f"No task_id in submit response: {payload}")
print(task_id)
PY
}

wait_for_result() {
  local label="$1"
  local task_id="$2"
  local status_path="$3"
  local result_path="$4"
  local deadline=$((SECONDS + WAIT_SECONDS))
  local status="unknown"

  log "Waiting for $label task: $task_id"
  while (( SECONDS < deadline )); do
    get_json "$AUDIO_BASE_URL/api/v1/status/$task_id" > "$status_path" || true
    status="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${status_path@Q}).read_text(encoding="utf-8"))
print(str(payload.get("status", "")).lower())
PY
)"
    log "$label status: $status"
    if [[ "$status" == "completed" ]]; then
      break
    fi
    if [[ "$status" == "failed" ]]; then
      cat "$status_path" >&2
      fail "$label task failed"
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
  [[ "$status" == "completed" ]] || fail "Timed out waiting for $label task after ${WAIT_SECONDS}s"

  get_json "$AUDIO_BASE_URL/api/v1/result/$task_id" | tee "$result_path" >/dev/null
  json_pretty < "$result_path" >/dev/null
}

validate_result() {
  local label="$1"
  local result_path="$2"
  local expected_chain="$3"
  local min_risk_records="$4"

  log "Validating $label result"
  python - <<PY
import json
from pathlib import Path

result_path = Path(${result_path@Q})
expected_chain = ${expected_chain@Q}
min_risk_records = int(${min_risk_records@Q})
min_asr_segments = int(${MIN_ASR_SEGMENTS@Q})

payload = json.loads(result_path.read_text(encoding="utf-8"))
report = payload.get("result")
if not isinstance(report, dict):
    raise SystemExit(f"Missing result report in {result_path}: {payload}")

audio_paths = ((report.get("artifact_paths") or {}).get("audio") or {})
text_paths = ((report.get("artifact_paths") or {}).get("text_api") or {})
required_audio = [
    "intake",
    "normalized_audio",
    "asr",
    "transcript",
    "alignment_index",
    "text_api_input",
    "text_api_result",
    "audio_text_risk_records",
    "audio_document_assessments",
    "audio_policy_decisions",
    "audio_annotation",
    "audio_audit",
    "audio_summary",
    "audio_report",
]
missing = [name for name in required_audio if not audio_paths.get(name) or not Path(audio_paths[name]).exists()]
if missing:
    raise SystemExit(f"Missing audio artifacts for {result_path}: {missing}")

required_text = ["intake", "document_context", "policy", "summary", "annotation", "audit"]
missing_text = [name for name in required_text if not text_paths.get(name) or not Path(text_paths[name]).exists()]
if missing_text:
    raise SystemExit(f"Text artifact paths were reported but missing: {missing_text}")

asr_rows = [
    json.loads(line)
    for line in Path(audio_paths["asr"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(asr_rows) < min_asr_segments:
    raise SystemExit(f"Expected at least {min_asr_segments} ASR segment(s), got {len(asr_rows)}")
if not any(str(row.get("text", "")).strip() and row.get("engine_name") != "fallback" for row in asr_rows):
    raise SystemExit("ASR did not produce a usable non-fallback transcript")

risk_rows = [
    json.loads(line)
    for line in Path(audio_paths["audio_text_risk_records"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(risk_rows) < min_risk_records:
    raise SystemExit(f"Expected at least {min_risk_records} audio text risk record(s), got {len(risk_rows)}")
if not any(row.get("chain") == expected_chain for row in risk_rows):
    raise SystemExit(f"No risk record for expected chain {expected_chain}: {risk_rows[:3]}")
if not any((row.get("audio_span") or {}).get("mapping_status") == "mapped" for row in risk_rows):
    raise SystemExit("No risk record was mapped back to the audio timeline")

audio_summary = json.loads(Path(audio_paths["audio_summary"]).read_text(encoding="utf-8"))
if audio_summary.get("processed_sources", 0) < 1:
    raise SystemExit(f"Invalid audio summary: {audio_summary}")
if report.get("decision") not in {"allow", "review", "quarantine", "reject"}:
    raise SystemExit(f"Unexpected report decision: {report.get('decision')}")

print(json.dumps({
    "label": ${label@Q},
    "task_id": payload.get("task_id"),
    "decision": report.get("decision"),
    "overall_disposition": report.get("overall_disposition"),
    "trust_level": report.get("trust_level"),
    "risk_records": len(risk_rows),
    "asr_segments": len(asr_rows),
    "run_dir": str(Path(audio_paths["audio_report"]).parent),
}, ensure_ascii=False, indent=2))
PY
}

main() {
  require_cmd curl
  validate_inputs
  ensure_stack

  submit_job "content-safety-real-audio" "$CONTENT_AUDIO" "CMP_002" "safety_only" "$CONTENT_SUBMIT_JSON"
  local content_task_id
  content_task_id="$(extract_task_id "$CONTENT_SUBMIT_JSON")"
  wait_for_result "content" "$content_task_id" "$CONTENT_STATUS_JSON" "$CONTENT_RESULT_JSON"
  validate_result "content" "$CONTENT_RESULT_JSON" "content_safety" "$MIN_CONTENT_RISK_RECORDS"

  submit_job "privacy-real-audio" "$PRIVACY_AUDIO" "CMP_001" "privacy_only" "$PRIVACY_SUBMIT_JSON"
  local privacy_task_id
  privacy_task_id="$(extract_task_id "$PRIVACY_SUBMIT_JSON")"
  wait_for_result "privacy" "$privacy_task_id" "$PRIVACY_STATUS_JSON" "$PRIVACY_RESULT_JSON"
  validate_result "privacy" "$PRIVACY_RESULT_JSON" "privacy" "$MIN_PRIVACY_RISK_RECORDS"

  log "Real audio compliance test passed."
  log "Content result: $CONTENT_RESULT_JSON"
  log "Privacy result: $PRIVACY_RESULT_JSON"
}

main "$@"
