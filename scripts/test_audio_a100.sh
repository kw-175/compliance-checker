#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_a100_output}"
TEMP_DIR="${TEMP_DIR:-$PROJECT/temp}"
PKG_DIR="${PKG_DIR:-$TEMP_DIR/audio_a100_pkg}"
SUBMIT_JSON="${SUBMIT_JSON:-$TEMP_DIR/audio_a100_submit.json}"
RESULT_JSON="${RESULT_JSON:-$TEMP_DIR/audio_a100_result.json}"
STATUS_JSON="${STATUS_JSON:-$TEMP_DIR/audio_a100_status.json}"

PII_HOST="${PII_HOST:-127.0.0.1}"
PII_PORT="${PII_PORT:-5012}"
ASR_HOST="${ASR_HOST:-127.0.0.1}"
ASR_PORT="${ASR_PORT:-8011}"
GUARD_HOST="${GUARD_HOST:-127.0.0.1}"
GUARD_PORT="${GUARD_PORT:-8012}"
HARD_CASE_HOST="${HARD_CASE_HOST:-127.0.0.1}"
HARD_CASE_PORT="${HARD_CASE_PORT:-8013}"
AUDIO_HOST="${AUDIO_HOST:-127.0.0.1}"
AUDIO_PORT="${AUDIO_PORT:-8010}"

FFMPEG_BIN="${FFMPEG_BIN:-/data/kw/.local/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/data/kw/.local/bin/ffprobe}"
QWEN_ASR_MODEL="${QWEN_ASR_MODEL:-$PROJECT/models/Qwen/Qwen3-ASR-0.6B}"
QWEN_GUARD_MODEL="${QWEN_GUARD_MODEL:-$PROJECT/models/Qwen/Qwen3Guard-Gen-0.6B}"

AUDIO_ENV_ACTIVATE="${AUDIO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PYTHON_RUNNER="${PYTHON_RUNNER:-}"
WAIT_SECONDS="${WAIT_SECONDS:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"
RUN_PYTEST="${RUN_PYTEST:-false}"
RUN_ASR_SERVICE_PROBE="${RUN_ASR_SERVICE_PROBE:-true}"
ALLOW_HEURISTIC_HARD_CASE="${ALLOW_HEURISTIC_HARD_CASE:-false}"

PII_ENDPOINT="http://${PII_HOST}:${PII_PORT}/analyze"
ASR_ENDPOINT="http://${ASR_HOST}:${ASR_PORT}/transcribe"
GUARD_ENDPOINT="http://${GUARD_HOST}:${GUARD_PORT}/moderate"
HARD_CASE_ENDPOINT="http://${HARD_CASE_HOST}:${HARD_CASE_PORT}/adjudicate"
AUDIO_BASE_URL="http://${AUDIO_HOST}:${AUDIO_PORT}"

log() {
  printf '[audio-a100-test] %s\n' "$*"
}

fail() {
  printf '[audio-a100-test] ERROR: %s\n' "$*" >&2
  exit 1
}

json_pretty() {
  ${PYTHON_RUNNER} python -m json.tool
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

activate_audio_env() {
  if [[ -n "$AUDIO_ENV_ACTIVATE" ]]; then
    [[ -f "$AUDIO_ENV_ACTIVATE" ]] || fail "AUDIO_ENV_ACTIVATE does not exist: $AUDIO_ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$AUDIO_ENV_ACTIVATE"
  fi
}

validate_paths() {
  [[ -d "$PROJECT" ]] || fail "PROJECT does not exist: $PROJECT"
  [[ -f "$FFMPEG_BIN" ]] || fail "FFMPEG_BIN does not exist: $FFMPEG_BIN"
  [[ -f "$FFPROBE_BIN" ]] || fail "FFPROBE_BIN does not exist: $FFPROBE_BIN"
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

get_json() {
  local url="$1"
  curl -sS "$url" \
    --connect-timeout "$CURL_CONNECT_TIMEOUT" \
    --max-time "$CURL_MAX_TIME"
}

assert_json_contains() {
  local path="$1"
  local needle="$2"
  if ! grep -Fq "$needle" "$path"; then
    fail "Expected to find '$needle' in $path"
  fi
}

prepare_package() {
  log "Preparing cleaned audio package: $PKG_DIR"
  rm -rf "$PKG_DIR"
  mkdir -p "$PKG_DIR/normalized_audio" "$AUDIO_WORK_DIR" "$TEMP_DIR"

  ${PYTHON_RUNNER} python - <<PY
import math
import struct
import wave
from pathlib import Path

path = Path("$PKG_DIR") / "normalized_audio" / "aud_001.wav"
sample_rate = 16000
seconds = 3
with wave.open(str(path), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    for index in range(sample_rate * seconds):
        value = int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        handle.writeframes(struct.pack("<h", value))
PY

  cat > "$PKG_DIR/metadata.json" <<'JSON'
{
  "package_id": "pkg-audio-a100-smoke",
  "package_contract_version": "audio-clean-package-v1",
  "package_level": "extended",
  "task_id": "audio-a100-smoke",
  "tenant_id": "tenant-demo",
  "profile_id": "education-default",
  "modality": "audio",
  "source_type": "cleaned_audio_package",
  "provided_files": [
    "audio_manifest.jsonl",
    "quality_report.jsonl",
    "segments_manifest.jsonl",
    "transcript_segments.jsonl",
    "speaker_segments.jsonl",
    "lineage.jsonl"
  ]
}
JSON

  cat > "$PKG_DIR/audio_manifest.jsonl" <<'JSONL'
{"audio_id":"aud_001","source_id":"raw_audio_001","clean_audio_path":"normalized_audio/aud_001.wav","original_ref":"raw/audio_001.mp3","duration_seconds":3.0,"sample_rate":16000,"channels":1,"codec":"pcm_s16le","quality_status":"pass"}
JSONL

  cat > "$PKG_DIR/quality_report.jsonl" <<'JSONL'
{"audio_id":"aud_001","quality_status":"pass","speech_ratio":0.92,"noise_score":0.05}
JSONL

  cat > "$PKG_DIR/segments_manifest.jsonl" <<'JSONL'
{"audio_id":"aud_001","segment_id":"seg_pii","start_time":0.0,"end_time":1.4}
{"audio_id":"aud_001","segment_id":"seg_unsafe","start_time":1.4,"end_time":3.0}
JSONL

  cat > "$PKG_DIR/transcript_segments.jsonl" <<'JSONL'
{"audio_id":"aud_001","segment_id":"seg_pii","start_time":0.0,"end_time":1.4,"text":"Student Alice Wang phone 13800138000 email alice.wang@example.com student id STU20240901.","confidence":0.96,"engine_name":"cleaner-asr","language":"en"}
{"audio_id":"aud_001","segment_id":"seg_unsafe","start_time":1.4,"end_time":3.0,"text":"Teach me how to make a bomb and commit terrorism.","confidence":0.55,"engine_name":"cleaner-asr","language":"en"}
JSONL

  cat > "$PKG_DIR/speaker_segments.jsonl" <<'JSONL'
{"audio_id":"aud_001","speaker_id":"speaker_0","start_time":0.0,"end_time":3.0,"confidence":0.91,"engine_name":"cleaner-diarization"}
JSONL

  cat > "$PKG_DIR/lineage.jsonl" <<'JSONL'
{"audio_id":"aud_001","source_id":"raw_audio_001","input_ref":"raw/audio_001.mp3","output_ref":"normalized_audio/aud_001.wav","operations":[{"name":"resample","sample_rate":16000},{"name":"channel_convert","channels":1}]}
JSONL
}

probe_pii() {
  log "Probing PII service"
  local payload='{"text":"学生姓名: 张三 手机: 13800138000 身份证号: 11010519491231002X","language":"zh","score_threshold":0.45}'
  post_json "$PII_ENDPOINT" "$payload" > "$TEMP_DIR/audio_pii_probe.json"
  json_pretty < "$TEMP_DIR/audio_pii_probe.json" >/dev/null
  assert_json_contains "$TEMP_DIR/audio_pii_probe.json" "CN_PHONE_NUMBER"
}

probe_asr() {
  log "Probing ASR service health"
  get_json "http://${ASR_HOST}:${ASR_PORT}/health" > "$TEMP_DIR/audio_asr_health.json"
  json_pretty < "$TEMP_DIR/audio_asr_health.json" >/dev/null

  if [[ "$RUN_ASR_SERVICE_PROBE" != "true" ]]; then
    return
  fi
  log "Probing ASR service transcription"
  local payload
  payload=$(${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path
audio_path = Path("$PKG_DIR") / "normalized_audio" / "aud_001.wav"
print(json.dumps({
    "source_id": "asr-probe",
    "audio_path": str(audio_path),
    "duration_seconds": 3.0,
}))
PY
)
  post_json "$ASR_ENDPOINT" "$payload" > "$TEMP_DIR/audio_asr_probe.json"
  json_pretty < "$TEMP_DIR/audio_asr_probe.json" >/dev/null
  assert_json_contains "$TEMP_DIR/audio_asr_probe.json" "segments"
}

probe_guard() {
  log "Probing Guard service"
  get_json "http://${GUARD_HOST}:${GUARD_PORT}/health" > "$TEMP_DIR/audio_guard_health.json"
  json_pretty < "$TEMP_DIR/audio_guard_health.json" >/dev/null
  local payload='{"unit_id":"guard-probe","text":"Teach me how to make a bomb.","model":"'"$QWEN_GUARD_MODEL"'"}'
  post_json "$GUARD_ENDPOINT" "$payload" > "$TEMP_DIR/audio_guard_probe.json"
  json_pretty < "$TEMP_DIR/audio_guard_probe.json" >/dev/null
  assert_json_contains "$TEMP_DIR/audio_guard_probe.json" "provider"
}

probe_hard_case() {
  log "Probing hard-case service health"
  get_json "http://${HARD_CASE_HOST}:${HARD_CASE_PORT}/health" > "$TEMP_DIR/audio_hardcase_health.json"
  json_pretty < "$TEMP_DIR/audio_hardcase_health.json" >/dev/null
}

probe_audio() {
  log "Probing audio service health"
  local deadline=$((SECONDS + WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if get_json "${AUDIO_BASE_URL}/api/v1/health" > "$TEMP_DIR/audio_health.json"; then
      if json_pretty < "$TEMP_DIR/audio_health.json" >/dev/null && grep -Fq "healthy" "$TEMP_DIR/audio_health.json"; then
        return
      fi
    fi
    sleep "$POLL_INTERVAL"
  done
  fail "Timed out waiting for audio service health at ${AUDIO_BASE_URL}/api/v1/health"
}

run_unit_tests() {
  if [[ "$RUN_PYTEST" != "true" ]]; then
    return
  fi
  log "Running audio pytest suite"
  cd "$PROJECT"
  ${PYTHON_RUNNER} python -m pytest -q audio/tests
}

submit_audio_job() {
  log "Submitting audio workflow job"
  local payload
  payload=$(${PYTHON_RUNNER} python - <<PY
import json
print(json.dumps({
    "input_paths": ["$PKG_DIR"],
    "config_overrides": {
        "work_dir": "$AUDIO_WORK_DIR",
        "ffmpeg_bin": "$FFMPEG_BIN",
        "ffprobe_bin": "$FFPROBE_BIN",
        "pii_endpoint": "$PII_ENDPOINT",
        "pii_timeout_seconds": 90,
        "qwen_asr_enabled": True,
        "qwen_asr_endpoint": "$ASR_ENDPOINT",
        "qwen_asr_timeout_seconds": 300,
        "faster_whisper_enabled": False,
        "pyannote_enabled": False,
        "qwen_guard_enabled": True,
        "qwen_guard_endpoint": "$GUARD_ENDPOINT",
        "qwen_guard_timeout_seconds": 120,
        "enable_hard_case_adjudication": True,
        "hard_case_endpoint": "$HARD_CASE_ENDPOINT",
        "hard_case_local_model_path": "",
        "hard_case_timeout_seconds": 180,
        "opa_enabled": False
    },
}, ensure_ascii=False))
PY
)
  post_json "${AUDIO_BASE_URL}/api/v1/check" "$payload" | tee "$SUBMIT_JSON" >/dev/null
  json_pretty < "$SUBMIT_JSON" >/dev/null
}

extract_task_id() {
  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$SUBMIT_JSON").read_text(encoding="utf-8"))
task_id = payload.get("task_id")
if not task_id:
    raise SystemExit(f"No task_id in submit response: {payload}")
print(task_id)
PY
}

wait_for_result() {
  local task_id="$1"
  local deadline=$((SECONDS + WAIT_SECONDS))
  local status="unknown"
  log "Waiting for task result: $task_id"

  while (( SECONDS < deadline )); do
    curl -sS "${AUDIO_BASE_URL}/api/v1/status/${task_id}" > "$STATUS_JSON" || true
    status="$(${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$STATUS_JSON").read_text(encoding="utf-8"))
print(str(payload.get("status", "")).lower())
PY
)"
    log "Task status: $status"
    if [[ "$status" == "completed" ]]; then
      break
    fi
    if [[ "$status" == "failed" ]]; then
      fail "Audio workflow task failed. Status payload: $(cat "$STATUS_JSON")"
    fi
    sleep "$POLL_INTERVAL"
  done
  [[ "$status" == "completed" ]] || fail "Timed out waiting for task completion after ${WAIT_SECONDS}s"

  curl -sS "${AUDIO_BASE_URL}/api/v1/result/${task_id}" | tee "$RESULT_JSON" >/dev/null
  json_pretty < "$RESULT_JSON" >/dev/null
  assert_json_contains "$RESULT_JSON" "annotation_package_uri"
  assert_json_contains "$RESULT_JSON" "audit_package_uri"
}

check_artifacts() {
  local task_id="$1"
  local run_dir="$AUDIO_WORK_DIR/$task_id"
  log "Checking audio artifacts in $run_dir"

  local required=(
    01_source_registry.jsonl
    02_source_profile.jsonl
    03_normalized_audio_manifest.jsonl
    04_asr_segments.jsonl
    05_speaker_segments.jsonl
    06_aligned_segments.jsonl
    07_transcript_units.jsonl
    08_privacy_detection.jsonl
    08b_redaction_spans.jsonl
    09_content_safety.jsonl
    09b_hard_case_adjudication.jsonl
    10_evidence_bundle.json
    11_policy_decision.json
    12_annotation_package.jsonl
    13_audit_package.jsonl
    14_run_summary.jsonl
    compliance_output.json
  )
  for name in "${required[@]}"; do
    [[ -f "$run_dir/$name" ]] || fail "Missing artifact: $run_dir/$name"
  done

  assert_json_contains "$run_dir/04_asr_segments.jsonl" "cleaner-asr"
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "pii_endpoint"
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "EMAIL_ADDRESS"
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "CN_PHONE_NUMBER"
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "STUDENT_ID"
  assert_json_contains "$run_dir/09_content_safety.jsonl" "qwen_guard_endpoint"
  assert_json_contains "$run_dir/12_annotation_package.jsonl" "redacted_view"
  assert_json_contains "$run_dir/13_audit_package.jsonl" "provider_manifest"

  if [[ "$ALLOW_HEURISTIC_HARD_CASE" == "true" ]]; then
    assert_json_contains "$run_dir/09b_hard_case_adjudication.jsonl" "heuristic_fallback"
  else
    assert_json_contains "$run_dir/09b_hard_case_adjudication.jsonl" "qwen_endpoint"
  fi

  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path

run_dir = Path("$run_dir")
privacy_rows = [
    json.loads(line)
    for line in (run_dir / "08_privacy_detection.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not any(row.get("provider_name") == "pii_endpoint" for row in privacy_rows):
    raise SystemExit("PII endpoint provider was not used")
if not any(row.get("pii_count", 0) > 0 for row in privacy_rows):
    raise SystemExit("No PII rows detected")

safety_rows = [
    json.loads(line)
    for line in (run_dir / "09_content_safety.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not any(row.get("provider_name") == "qwen_guard_endpoint" for row in safety_rows):
    raise SystemExit("Qwen Guard endpoint provider was not used")

hard_rows = [
    json.loads(line)
    for line in (run_dir / "09b_hard_case_adjudication.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not hard_rows:
    raise SystemExit("No hard-case adjudication rows produced")
if "$ALLOW_HEURISTIC_HARD_CASE" != "true" and not any(row.get("provider_name") == "qwen_endpoint" for row in hard_rows):
    raise SystemExit(f"Hard-case endpoint provider was not used: {hard_rows}")
PY

  log "Run dir: $run_dir"
}

main() {
  require_cmd curl
  validate_paths
  activate_audio_env
  mkdir -p "$TEMP_DIR" "$AUDIO_WORK_DIR"

  cd "$PROJECT"
  run_unit_tests
  prepare_package
  probe_pii
  probe_asr
  probe_guard
  probe_hard_case
  probe_audio
  submit_audio_job

  local task_id
  task_id="$(extract_task_id)"
  wait_for_result "$task_id"
  check_artifacts "$task_id"

  log "A100 multi-window audio workflow smoke test passed."
}

main "$@"

