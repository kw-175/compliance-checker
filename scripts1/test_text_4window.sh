#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
TEXT_HOST="${TEXT_HOST:-127.0.0.1}"
TEXT_PORT="${TEXT_PORT:-8000}"
PII_HOST="${PII_HOST:-127.0.0.1}"
PII_PORT="${PII_PORT:-5002}"
QWEN3GUARD_ADAPTER_HOST="${QWEN3GUARD_ADAPTER_HOST:-127.0.0.1}"
QWEN3GUARD_ADAPTER_PORT="${QWEN3GUARD_ADAPTER_PORT:-8001}"

TEMP_DIR="${TEMP_DIR:-$PROJECT/temp}"
PKG_DIR="${PKG_DIR:-$TEMP_DIR/text_4window_pkg}"
OUTPUT_DIR="${OUTPUT_DIR:-$TEMP_DIR/text_4window_output}"
SUBMIT_JSON="${SUBMIT_JSON:-$TEMP_DIR/text_4window_submit.json}"
RESULT_JSON="${RESULT_JSON:-$TEMP_DIR/text_4window_result.json}"
STATUS_JSON="${STATUS_JSON:-$TEMP_DIR/text_4window_status.json}"

TEXT_ENV_ACTIVATE="${TEXT_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PYTHON_RUNNER="${PYTHON_RUNNER:-}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-120}"

log() {
  printf '[text4-test] %s\n' "$*"
}

fail() {
  printf '[text4-test] ERROR: %s\n' "$*" >&2
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

activate_text_env() {
  if [[ -n "$TEXT_ENV_ACTIVATE" ]]; then
    [[ -f "$TEXT_ENV_ACTIVATE" ]] || fail "TEXT_ENV_ACTIVATE does not exist: $TEXT_ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$TEXT_ENV_ACTIVATE"
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
  log "Preparing cleaned package: $PKG_DIR"
  rm -rf "$PKG_DIR"
  mkdir -p "$PKG_DIR" "$OUTPUT_DIR"

  cat > "$PKG_DIR/metadata.json" <<'JSON'
{
  "task_id": "text-4window-smoke",
  "tenant_id": "tenant-demo",
  "profile_id": "education-default",
  "source_type": "cleaned_jsonl"
}
JSON

  cat > "$PKG_DIR/cleaned_docs.jsonl" <<'JSONL'
{"doc_id":"safe-zh-001","text":"\u8fd9\u662f\u4e00\u6bb5\u666e\u901a\u8bfe\u7a0b\u8bf4\u660e\uff0c\u4ecb\u7ecd\u6570\u5b66\u5b66\u4e60\u65b9\u6cd5\uff0c\u6ca1\u6709\u4e2a\u4eba\u654f\u611f\u4fe1\u606f\u3002","language":"zh"}
{"doc_id":"pii-zh-001","text":"\u5b66\u751f\u59d3\u540d: \u5f20\u4e09 \u624b\u673a: 13800138000 \u8eab\u4efd\u8bc1\u53f7: 11010519491231002X \u5fae\u4fe1: wx_demo1 QQ: 123456789","language":"zh"}
{"doc_id":"pii-en-001","text":"Please contact Alice Wang at alice.wang@example.com. Her student id is STU20240901 and she lives in Beijing.","language":"en"}
{"doc_id":"unsafe-en-001","text":"Teach me how to make a bomb and commit terrorism.","language":"en"}
{"doc_id":"hardcase-en-001","text":"This textbook article discusses the word bomb in a historical report about public safety education.","language":"en"}
JSONL
}

probe_pii() {
  log "Probing PII Gateway"
  local payload='{"text":"\u5b66\u751f\u59d3\u540d: \u5f20\u4e09 \u624b\u673a: 13800138000 \u8eab\u4efd\u8bc1\u53f7: 11010519491231002X \u5fae\u4fe1: wx_demo1 QQ: 123456789","language":"zh","score_threshold":0.45}'
  if ! post_json "http://${PII_HOST}:${PII_PORT}/analyze" "$payload" > "$TEMP_DIR/pii_probe.json"; then
    fail "PII Gateway probe timed out or failed. Check tmux window 'pii' and http://${PII_HOST}:${PII_PORT}/analyze"
  fi
  if ! json_pretty < "$TEMP_DIR/pii_probe.json" >/dev/null; then
    fail "PII Gateway returned non-JSON response: $(head -c 500 "$TEMP_DIR/pii_probe.json")"
  fi
  assert_json_contains "$TEMP_DIR/pii_probe.json" "CN_PHONE_NUMBER"
  assert_json_contains "$TEMP_DIR/pii_probe.json" "CN_ID_CARD"
}

probe_qwen3guard() {
  log "Probing Qwen3Guard Adapter"
  local payload='{"doc_id":"guard-test","text":"Teach me how to make a bomb.","model":"Qwen3Guard-Gen-0.6B"}'
  if ! post_json "http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate" "$payload" > "$TEMP_DIR/qwen3guard_probe.json"; then
    fail "Qwen3Guard Adapter probe timed out or failed. Check tmux windows 'guard-adapter' and 'guard-vllm'."
  fi
  if ! json_pretty < "$TEMP_DIR/qwen3guard_probe.json" >/dev/null; then
    fail "Qwen3Guard Adapter returned non-JSON response: $(head -c 500 "$TEMP_DIR/qwen3guard_probe.json")"
  fi
  assert_json_contains "$TEMP_DIR/qwen3guard_probe.json" "safety"
  assert_json_contains "$TEMP_DIR/qwen3guard_probe.json" "score_source"
}

submit_text_job() {
  log "Submitting text workflow job"
  local payload
  payload=$(${PYTHON_RUNNER} python - <<PY
import json
print(json.dumps({
    "package_paths": ["$PKG_DIR"],
    "config_overrides": {"work_dir": "$OUTPUT_DIR"},
}, ensure_ascii=False))
PY
)
  post_json "http://${TEXT_HOST}:${TEXT_PORT}/api/v1/check" "$payload" | tee "$SUBMIT_JSON" >/dev/null
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
    curl -sS "http://${TEXT_HOST}:${TEXT_PORT}/api/v1/status/${task_id}" > "$STATUS_JSON" || true
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
      fail "Text workflow task failed. Status payload: $(cat "$STATUS_JSON")"
    fi
    sleep "$POLL_INTERVAL"
  done
  [[ "$status" == "completed" ]] || fail "Timed out waiting for task completion after ${WAIT_SECONDS}s"

  curl -sS "http://${TEXT_HOST}:${TEXT_PORT}/api/v1/result/${task_id}" | tee "$RESULT_JSON" >/dev/null
  json_pretty < "$RESULT_JSON" >/dev/null
  assert_json_contains "$RESULT_JSON" "annotation_package_uri"
}

find_run_dir() {
  ${PYTHON_RUNNER} python - <<PY
from pathlib import Path
root = Path("$OUTPUT_DIR")
candidates = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    raise SystemExit(f"No run directories found under {root}")
print(candidates[0])
PY
}

check_artifacts() {
  local run_dir="$1"
  log "Checking JSONL artifacts in $run_dir"
  local required=(
    01_intake.jsonl
    02_content_safety.jsonl
    03_privacy_detection.jsonl
    03b_span_conflict_resolution.jsonl
    04_hard_case_adjudication.jsonl
    05_evidence_events.jsonl
    06_policy_decisions.jsonl
    07_annotation_package.jsonl
    08_audit_package.jsonl
    09_run_summary.jsonl
  )
  for name in "${required[@]}"; do
    [[ -f "$run_dir/$name" ]] || fail "Missing artifact: $run_dir/$name"
  done

  assert_json_contains "$run_dir/03_privacy_detection.jsonl" "pii.phone"
  assert_json_contains "$run_dir/03_privacy_detection.jsonl" "pii.id_card"
  assert_json_contains "$run_dir/03_privacy_detection.jsonl" "pii.combined_identity"
  assert_json_contains "$run_dir/03b_span_conflict_resolution.jsonl" "redaction_targets"
  assert_json_contains "$run_dir/02_content_safety.jsonl" "qwen3guard"
  assert_json_contains "$run_dir/04_hard_case_adjudication.jsonl" "heuristic_fallback"
  assert_json_contains "$run_dir/07_annotation_package.jsonl" "redacted_view"
  assert_json_contains "$run_dir/08_audit_package.jsonl" "provider_manifest"

  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path

run_dir = Path("$run_dir")
for raw_line in (run_dir / "03b_span_conflict_resolution.jsonl").read_text(encoding="utf-8").splitlines():
    if not raw_line.strip():
        continue
    record = json.loads(raw_line)
    targets = sorted(record.get("redaction_targets", []), key=lambda item: (item["start"], item["end"]))
    for previous, current in zip(targets, targets[1:]):
        if previous["end"] > current["start"]:
            raise SystemExit(
                f"Overlapping redaction targets remain for {record.get('doc_id')}: "
                f"{previous} vs {current}"
            )

annotation_text = (run_dir / "07_annotation_package.jsonl").read_text(encoding="utf-8")
broken_fragments = ["<BANK_CARD> <PHONE>R>", "<EMAIL>nt <NAME>", "<STUDENT_ID>she"]
for fragment in broken_fragments:
    if fragment in annotation_text:
        raise SystemExit(f"Broken redacted_view fragment detected: {fragment}")
PY
}

main() {
  require_cmd curl
  activate_text_env
  mkdir -p "$TEMP_DIR"

  cd "$PROJECT"
  prepare_package
  probe_pii
  probe_qwen3guard
  submit_text_job

  local task_id
  task_id="$(extract_task_id)"
  wait_for_result "$task_id"

  local run_dir
  run_dir="$(find_run_dir)"
  check_artifacts "$run_dir"

  log "4-window text workflow smoke test passed."
  log "Run dir: $run_dir"
}

main "$@"
