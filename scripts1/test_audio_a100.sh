#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
AUDIO_HOST="${AUDIO_HOST:-127.0.0.1}"
AUDIO_PORT="${AUDIO_PORT:-8010}"
AUDIO_WORK_DIR="${AUDIO_WORK_DIR:-$PROJECT/temp/audio_a100_output}"

TEMP_DIR="${TEMP_DIR:-$PROJECT/temp}"
PKG_DIR="${PKG_DIR:-$TEMP_DIR/audio_a100_pkg}"
SUBMIT_JSON="${SUBMIT_JSON:-$TEMP_DIR/audio_a100_submit.json}"
RESULT_JSON="${RESULT_JSON:-$TEMP_DIR/audio_a100_result.json}"
STATUS_JSON="${STATUS_JSON:-$TEMP_DIR/audio_a100_status.json}"
HEALTH_JSON="${HEALTH_JSON:-$TEMP_DIR/audio_a100_health.json}"

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

AUDIO_ENV_ACTIVATE="${AUDIO_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PYTHON_RUNNER="${PYTHON_RUNNER:-}"
WAIT_SECONDS="${WAIT_SECONDS:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"
RUN_PYTEST="${RUN_PYTEST:-false}"
RUN_ASR_MODEL_LOAD_PROBE="${RUN_ASR_MODEL_LOAD_PROBE:-true}"
ALLOW_HEURISTIC_HARD_CASE="${ALLOW_HEURISTIC_HARD_CASE:-false}"

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

validate_paths() {
  [[ -d "$PROJECT" ]] || fail "PROJECT does not exist: $PROJECT"
  [[ -f "$FFMPEG_BIN" ]] || fail "FFMPEG_BIN does not exist: $FFMPEG_BIN"
  [[ -f "$FFPROBE_BIN" ]] || fail "FFPROBE_BIN does not exist: $FFPROBE_BIN"
  [[ -d "$PII_STANZA_RESOURCES_DIR" ]] || fail "PII_STANZA_RESOURCES_DIR does not exist: $PII_STANZA_RESOURCES_DIR"
  [[ -d "$GLINER_MODEL_DIR" ]] || fail "GLINER_MODEL_DIR does not exist: $GLINER_MODEL_DIR"
  [[ -d "$QWEN_GUARD_MODEL" ]] || fail "QWEN_GUARD_MODEL does not exist: $QWEN_GUARD_MODEL"
  if [[ "$RUN_ASR_MODEL_LOAD_PROBE" == "true" ]]; then
    [[ -d "$QWEN_ASR_MODEL" ]] || fail "QWEN_ASR_MODEL does not exist: $QWEN_ASR_MODEL"
  fi
  if [[ "$ALLOW_HEURISTIC_HARD_CASE" != "true" ]]; then
    [[ -d "$HARD_CASE_MODEL" ]] || fail "HARD_CASE_MODEL does not exist: $HARD_CASE_MODEL"
  fi
}

activate_audio_env() {
  if [[ -n "$AUDIO_ENV_ACTIVATE" ]]; then
    [[ -f "$AUDIO_ENV_ACTIVATE" ]] || fail "AUDIO_ENV_ACTIVATE does not exist: $AUDIO_ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$AUDIO_ENV_ACTIVATE"
  fi
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

probe_health() {
  log "Probing audio service health"
  local deadline=$((SECONDS + WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -sS "http://${AUDIO_HOST}:${AUDIO_PORT}/api/v1/health" > "$HEALTH_JSON"; then
      if json_pretty < "$HEALTH_JSON" >/dev/null && grep -Fq "healthy" "$HEALTH_JSON"; then
        return
      fi
    fi
    sleep "$POLL_INTERVAL"
  done
  fail "Timed out waiting for audio service health at http://${AUDIO_HOST}:${AUDIO_PORT}/api/v1/health"
}

run_unit_tests() {
  if [[ "$RUN_PYTEST" != "true" ]]; then
    return
  fi
  log "Running audio pytest suite"
  cd "$PROJECT"
  ${PYTHON_RUNNER} python -m pytest -q audio/tests
}

probe_asr_model_load() {
  if [[ "$RUN_ASR_MODEL_LOAD_PROBE" != "true" ]]; then
    return
  fi
  log "Loading Qwen ASR model once in a short-lived probe process"
  cd "$PROJECT"
  CUDA_VISIBLE_DEVICES="$AUDIO_GPU" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ${PYTHON_RUNNER} python - <<PY
from audio.adapters import qwen_asr_adapter
from audio.config.settings import Settings

settings = Settings(
    qwen_asr_model="$QWEN_ASR_MODEL",
    qwen_asr_device="$QWEN_ASR_DEVICE",
    qwen_guard_enabled=False,
    enable_hard_case_adjudication=False,
)
pipeline = qwen_asr_adapter.load_pipeline(settings)
print(f"Qwen ASR pipeline loaded: {type(pipeline).__name__}")
qwen_asr_adapter.reset_cache()
PY
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
        "qwen_asr_enabled": True,
        "qwen_asr_model": "$QWEN_ASR_MODEL",
        "qwen_asr_device": "$QWEN_ASR_DEVICE",
        "faster_whisper_enabled": False,
        "pyannote_enabled": False,
        "qwen_guard_enabled": True,
        "qwen_guard_model": "$QWEN_GUARD_MODEL",
        "qwen_guard_device": "$QWEN_GUARD_DEVICE",
        "pii_model_root": "$PII_ROOT",
        "pii_stanza_resources_dir": "$PII_STANZA_RESOURCES_DIR",
        "pii_gliner_model": "$GLINER_MODEL_DIR",
        "pii_enable_presidio": True,
        "pii_enable_gliner": True,
        "pii_enable_regex_rules": True,
        "enable_hard_case_adjudication": True,
        "hard_case_local_model_path": "$HARD_CASE_MODEL",
        "hard_case_device": "$HARD_CASE_DEVICE",
        "opa_enabled": False
    },
}, ensure_ascii=False))
PY
)
  post_json "http://${AUDIO_HOST}:${AUDIO_PORT}/api/v1/check" "$payload" | tee "$SUBMIT_JSON" >/dev/null
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
    curl -sS "http://${AUDIO_HOST}:${AUDIO_PORT}/api/v1/status/${task_id}" > "$STATUS_JSON" || true
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

  curl -sS "http://${AUDIO_HOST}:${AUDIO_PORT}/api/v1/result/${task_id}" | tee "$RESULT_JSON" >/dev/null
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
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "EMAIL_ADDRESS"
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "CN_PHONE_NUMBER"
  assert_json_contains "$run_dir/08_privacy_detection.jsonl" "STUDENT_ID"
  assert_json_contains "$run_dir/09_content_safety.jsonl" "qwen_guard"
  assert_json_contains "$run_dir/12_annotation_package.jsonl" "redacted_view"
  assert_json_contains "$run_dir/13_audit_package.jsonl" "provider_manifest"

  if [[ "$ALLOW_HEURISTIC_HARD_CASE" == "true" ]]; then
    assert_json_contains "$run_dir/09b_hard_case_adjudication.jsonl" "heuristic_fallback"
  else
    assert_json_contains "$run_dir/09b_hard_case_adjudication.jsonl" "qwen_local"
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
if not any(row.get("pii_count", 0) > 0 for row in privacy_rows):
    raise SystemExit("No PII rows detected")

safety_rows = [
    json.loads(line)
    for line in (run_dir / "09_content_safety.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not any(row.get("provider_name") == "qwen_guard" for row in safety_rows):
    raise SystemExit("Qwen Guard provider was not used")

hard_rows = [
    json.loads(line)
    for line in (run_dir / "09b_hard_case_adjudication.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not hard_rows:
    raise SystemExit("No hard-case adjudication rows produced")
if "$ALLOW_HEURISTIC_HARD_CASE" != "true" and not any(row.get("provider_name") == "qwen_local" for row in hard_rows):
    raise SystemExit(f"Hard-case Qwen local provider was not used: {hard_rows}")
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
  probe_health
  probe_asr_model_load
  submit_audio_job

  local task_id
  task_id="$(extract_task_id)"
  wait_for_result "$task_id"
  check_artifacts "$task_id"

  log "A100 audio workflow smoke test passed."
}

main "$@"
