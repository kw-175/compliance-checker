#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
TEXT_API_HOST="${TEXT_API_HOST:-127.0.0.1}"
TEXT_API_PORT="${TEXT_API_PORT:-19002}"
PII_HOST="${PII_HOST:-127.0.0.1}"
PII_PORT="${PII_PORT:-5002}"
QWEN3GUARD_ADAPTER_HOST="${QWEN3GUARD_ADAPTER_HOST:-127.0.0.1}"
QWEN3GUARD_ADAPTER_PORT="${QWEN3GUARD_ADAPTER_PORT:-8215}"
QWEN35_HOST="${QWEN35_HOST:-127.0.0.1}"
QWEN35_PORT="${QWEN35_PORT:-8301}"
QWEN35_MODEL="${QWEN35_MODEL:-Qwen3.5-9B}"

TEMP_DIR="${TEMP_DIR:-$PROJECT/temp}"
PKG_DIR="${PKG_DIR:-$TEMP_DIR/text_local_smoke_pkg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TEMP_DIR/text_local_smoke_output}"
TEXT_ENV_ACTIVATE="${TEXT_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PYTHON_RUNNER="${PYTHON_RUNNER:-}"

WAIT_SECONDS="${WAIT_SECONDS:-240}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-120}"

log() {
  printf '[text-local-test] %s\n' "$*" >&2
}

fail() {
  printf '[text-local-test] ERROR: %s\n' "$*" >&2
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
    --data-binary "$payload"
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

validate_privacy_details() {
  local run_dir="$1"
  local profile="$2"
  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path

run_dir = Path("$run_dir")
profile = "$profile"

def load_jsonl(name):
    path = run_dir / name
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

contexts = load_jsonl("01b_document_context.jsonl")
if not contexts:
    raise SystemExit("document context artifact is empty")
for item in contexts:
    if not item.get("doc_id") or not item.get("document_type") or "explanation" not in item:
        raise SystemExit(f"document context missing required fields: {item}")

privacy_rows = load_jsonl("03_privacy_detection.jsonl")
privacy_findings = []
for row in privacy_rows:
    for finding in row.get("findings") or []:
        if finding.get("risk_type") == "api_unavailable":
            continue
        attrs = finding.get("attributes") or {}
        privacy_context = attrs.get("privacy_context") or {}
        if not privacy_context.get("document_type"):
            raise SystemExit(f"privacy finding lacks privacy_context.document_type: {finding}")
        span = finding.get("span") or {}
        if span and (span.get("start") is None or span.get("end") is None):
            raise SystemExit(f"privacy finding span is incomplete: {finding}")
        privacy_findings.append(finding)

if not privacy_findings:
    raise SystemExit("no real privacy findings were produced")

fragment_rows = load_jsonl("03f_privacy_fragment_adjudications.jsonl")
adjudicated_ids = {str(item.get("finding_id") or "") for item in fragment_rows}
missing = [
    str(item.get("finding_id") or "")
    for item in privacy_findings
    if str(item.get("finding_id") or "") not in adjudicated_ids
]
if missing:
    raise SystemExit(f"privacy findings without Qwen3.5 fragment adjudication: {missing}")
for item in fragment_rows:
    if not item.get("explanation"):
        raise SystemExit(f"privacy fragment adjudication lacks natural-language explanation: {item}")
    if not item.get("governance_action"):
        raise SystemExit(f"privacy fragment adjudication lacks governance_action: {item}")

doc_rows = load_jsonl("03g_privacy_document_assessments.jsonl")
if not doc_rows:
    raise SystemExit("privacy document assessment artifact is empty")
for item in doc_rows:
    if not item.get("recommended_action") or not item.get("explanation"):
        raise SystemExit(f"privacy document assessment missing decision/explanation: {item}")

load_jsonl("03b_span_conflict_resolution.jsonl")
final_rows = load_jsonl("03i_privacy_final_decisions.jsonl")
if not final_rows:
    raise SystemExit("privacy final decisions artifact is empty")
for item in final_rows:
    if "fragment_adjudications" not in item or "document_assessment" not in item:
        raise SystemExit(f"privacy final decision lacks adjudication/assessment view: {item}")

print(f"privacy semantic checks passed for {profile}: {len(privacy_findings)} findings")
PY
}

validate_content_details() {
  local run_dir="$1"
  local profile="$2"
  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path

run_dir = Path("$run_dir")
profile = "$profile"

def load_jsonl(name):
    path = run_dir / name
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

windows = load_jsonl("02a_content_candidate_windows.jsonl")
if not windows:
    raise SystemExit("content candidate windows artifact is empty")
window_by_id = {}
for item in windows:
    window_id = str(item.get("window_id") or "")
    if not window_id:
        raise SystemExit(f"candidate window lacks window_id: {item}")
    if item.get("start") is None or item.get("end") is None or not item.get("text"):
        raise SystemExit(f"candidate window lacks span/text: {item}")
    if not item.get("recall_sources"):
        raise SystemExit(f"candidate window lacks recall_sources: {item}")
    if not item.get("candidate_labels"):
        raise SystemExit(f"candidate window lacks candidate_labels: {item}")
    window_by_id[window_id] = item

fragments = load_jsonl("02aa_content_fragment_localization.jsonl")
if not fragments:
    raise SystemExit("content localized fragments artifact is empty")
fragment_by_id = {}
for item in fragments:
    fragment_id = str(item.get("fragment_id") or "")
    window_id = str(item.get("window_id") or "")
    if not fragment_id:
        raise SystemExit(f"localized fragment lacks fragment_id: {item}")
    if window_id not in window_by_id:
        raise SystemExit(f"localized fragment references unknown window_id={window_id}: {item}")
    span = item.get("span") or {}
    if span.get("start") is None or span.get("end") is None or not span.get("text"):
        raise SystemExit(f"localized fragment lacks precise span/text: {item}")
    if not item.get("source_tool") or not item.get("explanation"):
        raise SystemExit(f"localized fragment lacks source/explanation: {item}")
    fragment_by_id[fragment_id] = item

safety_rows = load_jsonl("02_content_safety.jsonl")
content_findings = []
for row in safety_rows:
    for finding in row.get("findings") or []:
        attrs = finding.get("attributes") or {}
        localized = attrs.get("localized_fragment") or {}
        content_attrs = attrs.get("content_safety") or {}
        fragment_id = str(localized.get("fragment_id") or "")
        if fragment_id not in fragment_by_id:
            raise SystemExit(f"content finding does not reference localized fragment: {finding}")
        if str(localized.get("window_id") or content_attrs.get("candidate_window_id") or "") not in window_by_id:
            raise SystemExit(f"content finding does not reference candidate window: {finding}")
        content_findings.append(finding)

if not content_findings:
    raise SystemExit("no content safety findings were produced")

adjudications = load_jsonl("02g_content_fragment_adjudications.jsonl")
adjudicated_ids = {str(item.get("finding_id") or "") for item in adjudications}
missing = [
    str(item.get("finding_id") or "")
    for item in content_findings
    if str(item.get("finding_id") or "") not in adjudicated_ids
]
if missing:
    raise SystemExit(f"content findings without Qwen3.5 fragment adjudication: {missing}")
for item in adjudications:
    if not item.get("recommended_action") or not item.get("explanation"):
        raise SystemExit(f"content fragment adjudication lacks action/explanation: {item}")

assessments = load_jsonl("02h_content_document_assessments.jsonl")
if not assessments:
    raise SystemExit("content document assessments artifact is empty")
for item in assessments:
    if not item.get("overall_stance") or not item.get("recommended_action") or not item.get("explanation"):
        raise SystemExit(f"content document assessment missing stance/action/explanation: {item}")

decision_rows = load_jsonl("02b_content_safety_decisions.jsonl")
if not decision_rows:
    raise SystemExit("content safety decisions artifact is empty")
if not any((item.get("metadata") or {}).get("content_candidate_window_count", 0) for item in decision_rows):
    raise SystemExit("content safety decisions do not expose candidate window counts")
if not any((item.get("metadata") or {}).get("content_localized_fragment_count", 0) for item in decision_rows):
    raise SystemExit("content safety decisions do not expose localized fragment counts")

audit_rows = load_jsonl("02c_content_safety_audit.jsonl")
if not audit_rows:
    raise SystemExit("content safety audit artifact is empty")
if not any(item.get("candidate_window") and item.get("localized_fragment") for item in audit_rows):
    raise SystemExit("content safety audit does not expose candidate_window/localized_fragment")

review_rows = load_jsonl("02d_content_safety_review_tasks.jsonl")
if not review_rows:
    raise SystemExit("content safety review task artifact is empty")
if not any(item.get("candidate_window") and item.get("localized_fragment") for item in review_rows):
    raise SystemExit("content safety review tasks do not expose candidate_window/localized_fragment")

print(f"content semantic checks passed for {profile}: {len(windows)} windows, {len(fragments)} fragments")
PY
}

validate_full_details() {
  local run_dir="$1"
  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path

run_dir = Path("$run_dir")

def load_jsonl(name):
    path = run_dir / name
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

audit_rows = load_jsonl("08_audit_package.jsonl")
if not audit_rows:
    raise SystemExit("final audit package is empty")
if not any(item.get("content_candidate_windows") for item in audit_rows):
    raise SystemExit("final audit package lacks content_candidate_windows")
if not any(item.get("content_localized_fragments") for item in audit_rows):
    raise SystemExit("final audit package lacks content_localized_fragments")

policy_rows = load_jsonl("06_policy_decisions.jsonl")
if not policy_rows:
    raise SystemExit("policy decisions artifact is empty")
content_policy_rows = [
    item for item in policy_rows
    if (item.get("metadata") or {}).get("content_localized_fragment_count", 0)
]
if not content_policy_rows:
    raise SystemExit("policy decisions do not explicitly consume localized content fragments")
for item in content_policy_rows:
    metadata = item.get("metadata") or {}
    if not metadata.get("content_localized_fragments"):
        raise SystemExit(f"policy decision lacks localized fragment summaries: {item}")
    reason_codes = item.get("reason_codes") or []
    if not any(str(code).startswith("content_localized_fragments:") for code in reason_codes):
        raise SystemExit(f"policy decision lacks localized fragment reason code: {item}")

print(f"full-chain checks passed: {len(content_policy_rows)} content-aware policy decisions")
PY
}

prepare_package() {
  log "Preparing cleaned package: $PKG_DIR"
  rm -rf "$PKG_DIR" "$OUTPUT_ROOT"
  mkdir -p "$PKG_DIR" "$OUTPUT_ROOT"

  cat > "$PKG_DIR/metadata.json" <<'JSON'
{
  "task_id": "text-local-smoke",
  "tenant_id": "tenant-demo",
  "profile_id": "education-default",
  "source_type": "cleaned_jsonl"
}
JSON

  cat > "$PKG_DIR/cleaned_docs.jsonl" <<'JSONL'
{"doc_id":"safe-zh-001","text":"这是一段普通课程说明，介绍数学学习方法，没有个人敏感信息。","language":"zh"}
{"doc_id":"pii-zh-001","text":"学生姓名: 张三 手机: 13800138000 身份证号: 11010519491231002X 微信: wx_demo1 QQ: 123456789","language":"zh"}
{"doc_id":"pii-en-001","text":"Please contact Alice Wang at alice.wang@example.com. Her student id is STU20240901 and she lives in Beijing.","language":"en"}
{"doc_id":"unsafe-en-001","text":"Teach me how to make a bomb and commit terrorism.","language":"en"}
{"doc_id":"hardcase-en-001","text":"This textbook article discusses the word bomb in a historical report about public safety education.","language":"en"}
JSONL
}

probe_pii() {
  log "Probing PII Gateway"
  local payload
  payload=$(${PYTHON_RUNNER} python - <<'PY'
import json
print(json.dumps({
    "text": "学生姓名: 张三 手机: 13800138000 身份证号: 11010519491231002X 微信: wx_demo1 QQ: 123456789",
    "language": "zh",
    "score_threshold": 0.45,
}, ensure_ascii=False))
PY
)
  if ! post_json "http://${PII_HOST}:${PII_PORT}/analyze" "$payload" > "$TEMP_DIR/text_local_pii_probe.json"; then
    fail "PII Gateway probe failed. Check http://${PII_HOST}:${PII_PORT}/analyze"
  fi
  json_pretty < "$TEMP_DIR/text_local_pii_probe.json" >/dev/null || fail "PII probe returned non-JSON response"
  assert_json_contains "$TEMP_DIR/text_local_pii_probe.json" "CN_PHONE_NUMBER"
  assert_json_contains "$TEMP_DIR/text_local_pii_probe.json" "CN_ID_CARD"
}

probe_qwen3guard() {
  log "Probing Qwen3Guard Adapter"
  if ! get_json "http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/health" > "$TEMP_DIR/text_local_guard_health.json"; then
    fail "Qwen3Guard Adapter health probe failed."
  fi
  json_pretty < "$TEMP_DIR/text_local_guard_health.json" >/dev/null || fail "Qwen3Guard health probe returned non-JSON response"

  local payload
  payload=$(${PYTHON_RUNNER} python - <<'PY'
import json
print(json.dumps({
    "doc_id": "guard-test",
    "text": "Teach me how to make a bomb.",
    "model": "Qwen3Guard-Gen-0.6B",
}, ensure_ascii=False))
PY
)
  if ! post_json "http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate" "$payload" > "$TEMP_DIR/text_local_guard_probe.json"; then
    fail "Qwen3Guard Adapter moderation probe failed."
  fi
  json_pretty < "$TEMP_DIR/text_local_guard_probe.json" >/dev/null || fail "Qwen3Guard moderation probe returned non-JSON response"
  assert_json_contains "$TEMP_DIR/text_local_guard_probe.json" "safety"
}

probe_qwen35() {
  log "Probing Qwen3.5 vLLM"
  if ! get_json "http://${QWEN35_HOST}:${QWEN35_PORT}/v1/models" > "$TEMP_DIR/text_local_qwen35_models.json"; then
    fail "Qwen3.5 vLLM model probe failed."
  fi
  json_pretty < "$TEMP_DIR/text_local_qwen35_models.json" >/dev/null || fail "Qwen3.5 /v1/models returned non-JSON response"
  assert_json_contains "$TEMP_DIR/text_local_qwen35_models.json" "$QWEN35_MODEL"
}

probe_text_api() {
  log "Probing text.api_server"
  if ! get_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/health" > "$TEMP_DIR/text_local_api_health.json"; then
    fail "text.api_server health probe failed."
  fi
  json_pretty < "$TEMP_DIR/text_local_api_health.json" >/dev/null || fail "text.api_server health probe returned non-JSON response"
  assert_json_contains "$TEMP_DIR/text_local_api_health.json" "local_model"
}

submit_job() {
  local profile="$1"
  local submit_json="$TEMP_DIR/text_local_submit_${profile}.json"
  local output_dir="$OUTPUT_ROOT/$profile"
  mkdir -p "$output_dir"
  log "Submitting ${profile} job"

  local payload
  payload=$(${PYTHON_RUNNER} python - <<PY
import json
print(json.dumps({
    "package_paths": ["$PKG_DIR"],
    "config_overrides": {
        "work_dir": "$output_dir",
        "pipeline_profile": "$profile",
    },
}, ensure_ascii=False))
PY
)
  post_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/check" "$payload" | tee "$submit_json" >/dev/null
  json_pretty < "$submit_json" >/dev/null || fail "Submit response for ${profile} is not JSON"
  echo "$submit_json"
}

extract_task_id() {
  local submit_json="$1"
  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$submit_json").read_text(encoding="utf-8"))
task_id = payload.get("task_id")
if not task_id:
    raise SystemExit(f"No task_id in submit response: {payload}")
print(task_id)
PY
}

wait_for_result() {
  local task_id="$1"
  local profile="$2"
  local result_json="$TEMP_DIR/text_local_result_${profile}.json"
  local status_json="$TEMP_DIR/text_local_status_${profile}.json"
  local deadline=$((SECONDS + WAIT_SECONDS))
  local status="unknown"

  log "Waiting for ${profile} task result: $task_id"
  while (( SECONDS < deadline )); do
    get_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/status/${task_id}" > "$status_json" || true
    status="$(${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$status_json").read_text(encoding="utf-8"))
print(str(payload.get("status", "")).lower())
PY
)"
    log "${profile} task status: $status"
    if [[ "$status" == "completed" ]]; then
      break
    fi
    if [[ "$status" == "failed" ]]; then
      fail "${profile} task failed. Status payload: $(cat "$status_json")"
    fi
    sleep "$POLL_INTERVAL"
  done
  [[ "$status" == "completed" ]] || fail "Timed out waiting for ${profile} task completion after ${WAIT_SECONDS}s"

  get_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/result/${task_id}" | tee "$result_json" >/dev/null
  json_pretty < "$result_json" >/dev/null || fail "${profile} result payload is not JSON"
  echo "$result_json"
}

extract_run_dir() {
  local result_json="$1"
  ${PYTHON_RUNNER} python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$result_json").read_text(encoding="utf-8"))
paths = payload.get("metadata", {}).get("artifact_paths", {})
summary_path = paths.get("summary")
if not summary_path:
    raise SystemExit(f"No summary artifact path in result: {payload}")
print(Path(summary_path).parent)
PY
}

check_profile_artifacts() {
  local profile="$1"
  local run_dir="$2"
  log "Checking ${profile} artifacts in $run_dir"

  [[ -f "$run_dir/01_intake.jsonl" ]] || fail "Missing intake artifact for ${profile}"
  [[ -f "$run_dir/01b_document_context.jsonl" ]] || fail "Missing document context artifact for ${profile}"
  [[ -f "$run_dir/09_run_summary.jsonl" ]] || fail "Missing run summary artifact for ${profile}"

  assert_json_contains "$run_dir/01b_document_context.jsonl" "document_type"
  assert_json_contains "$run_dir/09_run_summary.jsonl" "local_model"

  case "$profile" in
    privacy_only)
      [[ -f "$run_dir/03_privacy_detection.jsonl" ]] || fail "Missing privacy artifact for ${profile}"
      assert_json_contains "$run_dir/03_privacy_detection.jsonl" "privacy_context"
      validate_privacy_details "$run_dir" "$profile"
      ;;
    safety_only)
      [[ -f "$run_dir/02a_content_candidate_windows.jsonl" ]] || fail "Missing content candidate window artifact for ${profile}"
      [[ -f "$run_dir/02aa_content_fragment_localization.jsonl" ]] || fail "Missing content fragment localization artifact for ${profile}"
      [[ -f "$run_dir/02_content_safety.jsonl" ]] || fail "Missing content safety artifact for ${profile}"
      assert_json_contains "$run_dir/02a_content_candidate_windows.jsonl" "window_id"
      assert_json_contains "$run_dir/02a_content_candidate_windows.jsonl" "recall_sources"
      assert_json_contains "$run_dir/02aa_content_fragment_localization.jsonl" "fragment_id"
      assert_json_contains "$run_dir/02aa_content_fragment_localization.jsonl" "window_id"
      assert_json_contains "$run_dir/02_content_safety.jsonl" "content_safety"
      validate_content_details "$run_dir" "$profile"
      ;;
    full)
      [[ -f "$run_dir/02a_content_candidate_windows.jsonl" ]] || fail "Missing content candidate window artifact for ${profile}"
      [[ -f "$run_dir/02aa_content_fragment_localization.jsonl" ]] || fail "Missing content fragment localization artifact for ${profile}"
      [[ -f "$run_dir/02_content_safety.jsonl" ]] || fail "Missing content safety artifact for ${profile}"
      [[ -f "$run_dir/03_privacy_detection.jsonl" ]] || fail "Missing privacy artifact for ${profile}"
      [[ -f "$run_dir/04_hard_case_adjudication.jsonl" ]] || fail "Missing hard-case artifact for ${profile}"
      assert_json_contains "$run_dir/02a_content_candidate_windows.jsonl" "window_id"
      assert_json_contains "$run_dir/02aa_content_fragment_localization.jsonl" "fragment_id"
      validate_content_details "$run_dir" "$profile"
      validate_privacy_details "$run_dir" "$profile"
      validate_full_details "$run_dir"
      ;;
    *)
      fail "Unknown profile: $profile"
      ;;
  esac
}

run_profile() {
  local profile="$1"
  local submit_json
  submit_json="$(submit_job "$profile")"
  local task_id
  task_id="$(extract_task_id "$submit_json")"
  local result_json
  result_json="$(wait_for_result "$task_id" "$profile")"
  local run_dir
  run_dir="$(extract_run_dir "$result_json")"
  check_profile_artifacts "$profile" "$run_dir"
}

main() {
  require_cmd curl
  activate_text_env
  mkdir -p "$TEMP_DIR"

  cd "$PROJECT"
  prepare_package
  probe_pii
  probe_qwen3guard
  probe_qwen35
  probe_text_api

  run_profile "privacy_only"
  run_profile "safety_only"
  run_profile "full"

  log "Local-model text compliance smoke test completed successfully."
}

main "$@"
