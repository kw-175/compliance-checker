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

CONTENT_JSON="${CONTENT_JSON:-$PROJECT/test_data/content_safety_11_targets_single_text.json}"
PRIVACY_JSON="${PRIVACY_JSON:-$PROJECT/test_data/privacy_compliance_11_targets_single_text.json}"

TEMP_DIR="${TEMP_DIR:-$PROJECT/temp}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TEMP_DIR/text_real_json_test_output/$RUN_STAMP}"
TEXT_ENV_ACTIVATE="${TEXT_ENV_ACTIVATE:-$PROJECT/.venv/bin/activate}"
PYTHON_RUNNER="${PYTHON_RUNNER:-}"

WAIT_SECONDS="${WAIT_SECONDS:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"

MIN_CONTENT_FRAGMENTS="${MIN_CONTENT_FRAGMENTS:-8}"
MIN_CONTENT_TARGET_COVERAGE="${MIN_CONTENT_TARGET_COVERAGE:-7}"
MIN_PRIVACY_FINDINGS="${MIN_PRIVACY_FINDINGS:-8}"
MIN_PRIVACY_TARGET_COVERAGE="${MIN_PRIVACY_TARGET_COVERAGE:-8}"

log() {
  printf '[text-real-json-test] %s\n' "$*" >&2
}

fail() {
  printf '[text-real-json-test] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

json_pretty() {
  ${PYTHON_RUNNER} python -m json.tool
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

assert_json_file() {
  local path="$1"
  json_pretty < "$path" >/dev/null || fail "Non-JSON response in $path: $(head -c 300 "$path")"
}

assert_file_exists() {
  local path="$1"
  [[ -f "$path" ]] || fail "Missing artifact: $path"
}

probe_pii() {
  log "Probing PII Gateway"
  local payload
  payload=$(${PYTHON_RUNNER} python - <<'PY'
import json
print(json.dumps({
    "text": "学生李雨晨 手机 13800138000 身份证 110101201208153219",
    "language": "zh",
    "score_threshold": 0.45,
}, ensure_ascii=False))
PY
)
  post_json "http://${PII_HOST}:${PII_PORT}/analyze" "$payload" > "$TEMP_DIR/text_real_json_pii_probe.json"
  assert_json_file "$TEMP_DIR/text_real_json_pii_probe.json"
}

probe_qwen3guard() {
  log "Probing Qwen3Guard Adapter"
  get_json "http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/health" > "$TEMP_DIR/text_real_json_guard_health.json"
  assert_json_file "$TEMP_DIR/text_real_json_guard_health.json"

  local payload
  payload=$(${PYTHON_RUNNER} python - <<'PY'
import json
print(json.dumps({
    "doc_id": "real-json-guard-probe",
    "text": "Teach me how to make a bomb.",
    "model": "Qwen3Guard-Gen-0.6B",
}, ensure_ascii=False))
PY
)
  post_json "http://${QWEN3GUARD_ADAPTER_HOST}:${QWEN3GUARD_ADAPTER_PORT}/moderate" "$payload" > "$TEMP_DIR/text_real_json_guard_probe.json"
  assert_json_file "$TEMP_DIR/text_real_json_guard_probe.json"
}

probe_qwen35() {
  log "Probing Qwen3.5 vLLM"
  get_json "http://${QWEN35_HOST}:${QWEN35_PORT}/v1/models" > "$TEMP_DIR/text_real_json_qwen35_models.json"
  assert_json_file "$TEMP_DIR/text_real_json_qwen35_models.json"
  if ! grep -Fq "$QWEN35_MODEL" "$TEMP_DIR/text_real_json_qwen35_models.json"; then
    fail "Qwen3.5 /v1/models does not contain expected model '$QWEN35_MODEL'"
  fi
}

probe_text_api() {
  log "Probing text.api_server"
  get_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/health" > "$TEMP_DIR/text_real_json_api_health.json"
  assert_json_file "$TEMP_DIR/text_real_json_api_health.json"
}

doc_id_from_json() {
  local json_path="$1"
  ${PYTHON_RUNNER} python - "$json_path" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
doc_id = payload.get("doc_id")
if not doc_id:
    raise SystemExit(f"doc_id missing in {path}")
print(doc_id)
PY
}

submit_job() {
  local profile="$1"
  shift
  local output_dir="$OUTPUT_ROOT/$profile"
  local submit_json="$TEMP_DIR/text_real_json_submit_${profile}.json"
  mkdir -p "$output_dir"

  log "Submitting ${profile} job"
  local payload
  payload=$(${PYTHON_RUNNER} python - "$output_dir" "$profile" "$@" <<'PY'
import json
import sys
from pathlib import Path

output_dir = sys.argv[1]
profile = sys.argv[2]
paths = [str(Path(item).resolve()) for item in sys.argv[3:]]
print(json.dumps({
    "package_paths": paths,
    "config_overrides": {
        "work_dir": output_dir,
        "pipeline_profile": profile,
    },
}, ensure_ascii=False))
PY
)
  post_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/check" "$payload" > "$submit_json"
  assert_json_file "$submit_json"
  echo "$submit_json"
}

extract_task_id() {
  local submit_json="$1"
  ${PYTHON_RUNNER} python - "$submit_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
task_id = payload.get("task_id")
if not task_id:
    raise SystemExit(f"No task_id in submit response: {payload}")
print(task_id)
PY
}

wait_for_result() {
  local task_id="$1"
  local profile="$2"
  local result_json="$TEMP_DIR/text_real_json_result_${profile}.json"
  local status_json="$TEMP_DIR/text_real_json_status_${profile}.json"
  local deadline=$((SECONDS + WAIT_SECONDS))
  local status="unknown"

  log "Waiting for ${profile} task result: $task_id"
  while (( SECONDS < deadline )); do
    get_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/status/${task_id}" > "$status_json" || true
    status="$(${PYTHON_RUNNER} python - "$status_json" <<'PY'
import json
import sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    print("unknown")
else:
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

  get_json "http://${TEXT_API_HOST}:${TEXT_API_PORT}/api/v1/result/${task_id}" > "$result_json"
  assert_json_file "$result_json"
  echo "$result_json"
}

extract_run_dir() {
  local result_json="$1"
  ${PYTHON_RUNNER} python - "$result_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = payload.get("metadata", {}).get("artifact_paths", {})
summary_path = paths.get("summary")
if not summary_path:
    raise SystemExit(f"No summary artifact path in result: {payload}")
print(Path(summary_path).parent)
PY
}

validate_real_profile() {
  local profile="$1"
  local run_dir="$2"
  local report_json="$TEMP_DIR/text_real_json_report_${profile}.json"
  log "Validating ${profile} artifacts in $run_dir"

  assert_file_exists "$run_dir/01_intake.jsonl"
  assert_file_exists "$run_dir/01b_document_context.jsonl"
  assert_file_exists "$run_dir/09_run_summary.jsonl"

  CONTENT_DOC_ID="$CONTENT_DOC_ID" \
  PRIVACY_DOC_ID="$PRIVACY_DOC_ID" \
  CONTENT_JSON="$CONTENT_JSON" \
  PRIVACY_JSON="$PRIVACY_JSON" \
  MIN_CONTENT_FRAGMENTS="$MIN_CONTENT_FRAGMENTS" \
  MIN_CONTENT_TARGET_COVERAGE="$MIN_CONTENT_TARGET_COVERAGE" \
  MIN_PRIVACY_FINDINGS="$MIN_PRIVACY_FINDINGS" \
  MIN_PRIVACY_TARGET_COVERAGE="$MIN_PRIVACY_TARGET_COVERAGE" \
  ${PYTHON_RUNNER} python - "$profile" "$run_dir" "$report_json" <<'PY'
import json
import os
import sys
from pathlib import Path
from typing import Any

profile = sys.argv[1]
run_dir = Path(sys.argv[2])
report_path = Path(sys.argv[3])
content_doc_id = os.environ["CONTENT_DOC_ID"]
privacy_doc_id = os.environ["PRIVACY_DOC_ID"]
content_json = Path(os.environ["CONTENT_JSON"])
privacy_json = Path(os.environ["PRIVACY_JSON"])
min_content_fragments = int(os.environ["MIN_CONTENT_FRAGMENTS"])
min_content_target_coverage = int(os.environ["MIN_CONTENT_TARGET_COVERAGE"])
min_privacy_findings = int(os.environ["MIN_PRIVACY_FINDINGS"])
min_privacy_target_coverage = int(os.environ["MIN_PRIVACY_TARGET_COVERAGE"])


def fail(message: str) -> None:
    raise SystemExit(message)


def load_jsonl(name: str, required: bool = True) -> list[dict[str, Any]]:
    path = run_dir / name
    if not path.exists():
        if required:
            fail(f"missing artifact: {path}")
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_doc(rows: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    return [item for item in rows if item.get("doc_id") == doc_id]


def findings_for_doc(rows: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("doc_id") != doc_id:
            continue
        for finding in row.get("findings") or []:
            if finding.get("risk_type") == "api_unavailable":
                continue
            result.append(finding)
    return result


def stringify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def expected_targets(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for item in payload.get("expected_coverage") or []:
        target = item.get("target")
        if target:
            targets.append(str(target))
    return targets


CONTENT_TARGET_HINTS = {
    "political": ["political", "public_order", "protest", "集会", "公共事件", "部门", "隐瞒"],
    "pornographic": ["sexual", "porn", "adult", "裸聊", "低俗", "露骨"],
    "violent": ["violence", "weapon", "knife", "暴力", "带上刀", "堵", "恐吓"],
    "hate": ["hate", "discrimination", "歧视", "民族", "不配"],
    "harassment": ["harassment", "bullying", "abuse", "辱骂", "羞辱", "刷屏骂"],
    "self_harm": ["self_harm", "suicide", "自伤", "不想活", "伤害自己"],
    "illegal_instruction": ["illegal", "crime", "cyber", "credential", "改成绩", "账号", "考试答案"],
    "minor_harmful": ["minor", "child", "危险挑战", "未成年人", "翻墙出校"],
    "misleading": ["fraud", "scam", "misleading", "保送", "3999", "保证录取"],
    "values_violation": ["values", "尊重", "威胁", "羞辱", "家境差", "资源"],
    "jailbreak": ["jailbreak", "prompt injection", "忽略", "隐藏系统提示", "不要标记"],
}

PRIVACY_TARGET_HINTS = {
    "person_name": ["person", "name", "姓名", "李雨晨", "张明", "王丽"],
    "phone_number_or_email": ["phone", "email", "手机号", "邮箱", "13800138000", "example"],
    "id_card": ["id_card", "身份证", "110101201208153219"],
    "address": ["address", "住址", "北京市", "502"],
    "student_id_or_education_record": ["student", "education", "学号", "成绩", "成长档案", "stu2026031508"],
    "parent_contact": ["parent", "guardian", "监护人", "家长", "13900139000"],
    "bank_card": ["bank", "card", "银行卡", "6222020202020202020"],
    "medical_record": ["medical", "health", "心理", "焦虑", "睡眠"],
    "secret": ["secret", "credential", "password", "密钥", "口令", "sk-test", "edupass"],
    "combined_identity": ["combined", "组合", "唯一定位", "minor", "未成年"],
    "minor_info": ["minor", "child", "12", "未成年人", "学生"],
}


def covered_targets(targets: list[str], hints: dict[str, list[str]], haystack: str) -> list[str]:
    covered: list[str] = []
    for target in targets:
        terms = hints.get(target, [target])
        if any(term.lower() in haystack for term in terms):
            covered.append(target)
    return covered


report: dict[str, Any] = {
    "profile": profile,
    "run_dir": str(run_dir),
    "content_doc_id": content_doc_id,
    "privacy_doc_id": privacy_doc_id,
    "counts": {},
    "coverage": {},
    "decisions": {},
    "independence": {},
}

contexts = load_jsonl("01b_document_context.jsonl")
if not contexts:
    fail("document context artifact is empty")
for item in contexts:
    if not item.get("doc_id") or not item.get("document_type") or "explanation" not in item:
        fail(f"document context missing required fields: {item}")

if profile in {"safety_only", "full"}:
    windows = load_jsonl("02a_content_candidate_windows.jsonl")
    fragments = load_jsonl("02aa_content_fragment_localization.jsonl")
    safety_rows = load_jsonl("02_content_safety.jsonl")
    adjudications = load_jsonl("02g_content_fragment_adjudications.jsonl")
    assessments = load_jsonl("02h_content_document_assessments.jsonl")
    content_decisions = load_jsonl("02b_content_safety_decisions.jsonl")
    content_audit = load_jsonl("02c_content_safety_audit.jsonl")
    content_review = load_jsonl("02d_content_safety_review_tasks.jsonl")

    content_windows = by_doc(windows, content_doc_id)
    content_fragments = by_doc(fragments, content_doc_id)
    content_findings = findings_for_doc(safety_rows, content_doc_id)
    content_adjudications = by_doc(adjudications, content_doc_id)
    content_assessments = by_doc(assessments, content_doc_id)

    if not content_windows:
        fail("content chain produced no candidate windows for content test doc")
    if len(content_fragments) < min_content_fragments:
        fail(f"content localized fragment count too low: {len(content_fragments)} < {min_content_fragments}")
    if len(content_findings) < min_content_fragments:
        fail(f"content finding count too low: {len(content_findings)} < {min_content_fragments}")
    if len(content_adjudications) < min_content_fragments:
        fail(f"content Qwen3.5 adjudication count too low: {len(content_adjudications)} < {min_content_fragments}")
    if not content_assessments:
        fail("content document assessment missing for content test doc")

    window_ids = {str(item.get("window_id") or "") for item in content_windows}
    fragment_ids = {str(item.get("fragment_id") or "") for item in content_fragments}
    for item in content_windows:
        if item.get("start") is None or item.get("end") is None or not item.get("text"):
            fail(f"content candidate window lacks span/text: {item}")
        if not item.get("candidate_labels") or not item.get("recall_sources"):
            fail(f"content candidate window lacks labels/recall_sources: {item}")
    for item in content_fragments:
        span = item.get("span") or {}
        if str(item.get("window_id") or "") not in window_ids:
            fail(f"content fragment references unknown window: {item}")
        if not span.get("text") or span.get("start") is None or span.get("end") is None:
            fail(f"content fragment lacks precise span: {item}")
        if not item.get("risk_type") or not item.get("policy_tag") or not item.get("explanation"):
            fail(f"content fragment lacks risk_type/policy_tag/explanation: {item}")
    for finding in content_findings:
        attrs = finding.get("attributes") or {}
        localized = attrs.get("localized_fragment") or {}
        if str(localized.get("fragment_id") or "") not in fragment_ids:
            fail(f"content finding does not consume localized fragment: {finding}")
    finding_ids = {str(item.get("finding_id") or "") for item in content_findings}
    adjudicated_ids = {str(item.get("finding_id") or "") for item in content_adjudications}
    missing_adjudications = sorted(finding_ids - adjudicated_ids)
    if missing_adjudications:
        fail(f"content findings without Qwen3.5 adjudication: {missing_adjudications}")
    for item in content_adjudications:
        if not item.get("recommended_action") or not item.get("explanation"):
            fail(f"content adjudication lacks recommended_action/explanation: {item}")
    assessment = content_assessments[0]
    if assessment.get("recommended_action") == "keep":
        fail(f"content document assessment unexpectedly keeps content-risk doc: {assessment}")
    if not assessment.get("overall_stance") or not assessment.get("explanation"):
        fail(f"content document assessment lacks stance/explanation: {assessment}")
    if not any((item.get("metadata") or {}).get("content_candidate_window_count", 0) for item in content_decisions):
        fail("content decision records do not expose candidate window counts")
    if not any((item.get("metadata") or {}).get("content_localized_fragment_count", 0) for item in content_decisions):
        fail("content decision records do not expose localized fragment counts")
    if not any(item.get("candidate_window") and item.get("localized_fragment") for item in content_audit):
        fail("content audit view does not include candidate_window/localized_fragment")
    if not any(item.get("candidate_window") and item.get("localized_fragment") for item in content_review):
        fail("content review task view does not include candidate_window/localized_fragment")

    content_haystack = stringify([windows, fragments, safety_rows, adjudications, assessments])
    content_targets = expected_targets(content_json)
    content_covered = covered_targets(content_targets, CONTENT_TARGET_HINTS, content_haystack)
    if len(content_covered) < min_content_target_coverage:
        fail(
            "content expected target coverage too low: "
            f"{len(content_covered)} < {min_content_target_coverage}; covered={content_covered}"
        )

    report["counts"].update({
        "content_candidate_windows": len(content_windows),
        "content_localized_fragments": len(content_fragments),
        "content_findings": len(content_findings),
        "content_fragment_adjudications": len(content_adjudications),
    })
    report["coverage"]["content_targets"] = {
        "expected": content_targets,
        "covered": content_covered,
        "missing": [item for item in content_targets if item not in content_covered],
    }
    report["decisions"]["content_document_assessment"] = {
        "recommended_action": assessment.get("recommended_action"),
        "overall_stance": assessment.get("overall_stance"),
        "provider_name": assessment.get("provider_name"),
    }

if profile in {"privacy_only", "full"}:
    privacy_rows = load_jsonl("03_privacy_detection.jsonl")
    privacy_fragments = load_jsonl("03f_privacy_fragment_adjudications.jsonl")
    privacy_assessments = load_jsonl("03g_privacy_document_assessments.jsonl")
    redaction_plans = load_jsonl("03b_span_conflict_resolution.jsonl")
    privacy_final = load_jsonl("03i_privacy_final_decisions.jsonl")

    privacy_findings = findings_for_doc(privacy_rows, privacy_doc_id)
    privacy_adjudications = by_doc(privacy_fragments, privacy_doc_id)
    privacy_doc_assessments = by_doc(privacy_assessments, privacy_doc_id)
    privacy_redaction_plans = by_doc(redaction_plans, privacy_doc_id)
    privacy_final_rows = by_doc(privacy_final, privacy_doc_id)

    if len(privacy_findings) < min_privacy_findings:
        fail(f"privacy finding count too low: {len(privacy_findings)} < {min_privacy_findings}")
    if len(privacy_adjudications) < min_privacy_findings:
        fail(f"privacy Qwen3.5 adjudication count too low: {len(privacy_adjudications)} < {min_privacy_findings}")
    if not privacy_doc_assessments:
        fail("privacy document assessment missing for privacy test doc")
    if not privacy_redaction_plans:
        fail("privacy span conflict/redaction plan missing for privacy test doc")
    if not privacy_final_rows:
        fail("privacy final decisions missing for privacy test doc")
    for finding in privacy_findings:
        attrs = finding.get("attributes") or {}
        privacy_context = attrs.get("privacy_context") or {}
        span = finding.get("span") or {}
        if not privacy_context.get("document_type"):
            fail(f"privacy finding lacks privacy_context.document_type: {finding}")
        if span and (span.get("start") is None or span.get("end") is None or not span.get("text")):
            fail(f"privacy finding span incomplete: {finding}")
    finding_ids = {str(item.get("finding_id") or "") for item in privacy_findings}
    adjudicated_ids = {str(item.get("finding_id") or "") for item in privacy_adjudications}
    missing_adjudications = sorted(finding_ids - adjudicated_ids)
    if missing_adjudications:
        fail(f"privacy findings without Qwen3.5 adjudication: {missing_adjudications}")
    for item in privacy_adjudications:
        if not item.get("governance_action") or not item.get("explanation"):
            fail(f"privacy adjudication lacks governance_action/explanation: {item}")
    assessment = privacy_doc_assessments[0]
    if assessment.get("recommended_action") == "keep":
        fail(f"privacy document assessment unexpectedly keeps privacy-risk doc: {assessment}")
    if not assessment.get("overall_risk_level") or not assessment.get("explanation"):
        fail(f"privacy document assessment lacks risk/explanation: {assessment}")
    for item in privacy_final_rows:
        if "fragment_adjudications" not in item or "document_assessment" not in item:
            fail(f"privacy final decision does not include fragment/document views: {item}")

    privacy_haystack = stringify([privacy_rows, privacy_fragments, privacy_assessments, redaction_plans, privacy_final])
    privacy_targets = expected_targets(privacy_json)
    privacy_covered = covered_targets(privacy_targets, PRIVACY_TARGET_HINTS, privacy_haystack)
    if len(privacy_covered) < min_privacy_target_coverage:
        fail(
            "privacy expected target coverage too low: "
            f"{len(privacy_covered)} < {min_privacy_target_coverage}; covered={privacy_covered}"
        )

    report["counts"].update({
        "privacy_findings": len(privacy_findings),
        "privacy_fragment_adjudications": len(privacy_adjudications),
        "privacy_redaction_targets": len((privacy_redaction_plans[0].get("redaction_targets") or []) if privacy_redaction_plans else []),
    })
    report["coverage"]["privacy_targets"] = {
        "expected": privacy_targets,
        "covered": privacy_covered,
        "missing": [item for item in privacy_targets if item not in privacy_covered],
    }
    report["decisions"]["privacy_document_assessment"] = {
        "recommended_action": assessment.get("recommended_action"),
        "overall_risk_level": assessment.get("overall_risk_level"),
        "provider_name": assessment.get("provider_name"),
    }

policy_rows = load_jsonl("06_policy_decisions.jsonl")
annotation_rows = load_jsonl("07_annotation_package.jsonl")
audit_rows = load_jsonl("08_audit_package.jsonl")
summary_rows = load_jsonl("09_run_summary.jsonl")
if not summary_rows:
    fail("run summary artifact is empty")

expected_policy_docs = {
    "safety_only": {content_doc_id},
    "privacy_only": {privacy_doc_id},
    "full": {content_doc_id, privacy_doc_id},
}[profile]
policy_by_doc = {item.get("doc_id"): item for item in policy_rows}
annotation_by_doc = {item.get("doc_id"): item for item in annotation_rows}
audit_by_doc = {item.get("doc_id"): item for item in audit_rows}
if set(policy_by_doc) != expected_policy_docs:
    fail(f"policy decisions do not match expected docs for {profile}: {sorted(policy_by_doc)}")
if set(annotation_by_doc) != expected_policy_docs:
    fail(f"annotation package does not match expected docs for {profile}: {sorted(annotation_by_doc)}")
if set(audit_by_doc) != expected_policy_docs:
    fail(f"audit package does not match expected docs for {profile}: {sorted(audit_by_doc)}")

rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
summary = summary_rows[0]
max_policy_disposition = max(
    (str(item.get("disposition_level") or "P0") for item in policy_rows),
    key=lambda value: rank.get(value, -1),
    default="P0",
)
if summary.get("overall_disposition") != max_policy_disposition:
    fail(f"summary disposition does not match policy decisions: summary={summary}, policy={policy_rows}")
summary_counts = summary.get("counts_by_disposition") or {}
for disposition in {str(item.get("disposition_level") or "P0") for item in policy_rows}:
    expected_count = sum(1 for item in policy_rows if item.get("disposition_level") == disposition)
    if summary_counts.get(disposition) != expected_count:
        fail(f"summary count for {disposition} does not match policy decisions: summary={summary}, policy={policy_rows}")

if profile == "safety_only":
    content_policy = policy_by_doc[content_doc_id]
    content_policy_meta = content_policy.get("metadata") or {}
    if content_policy.get("disposition_level") not in {"P4", "P5"}:
        fail(f"safety_only policy disposition too low for content-risk test doc: {content_policy}")
    if content_policy_meta.get("content_localized_fragment_count", 0) < min_content_fragments:
        fail(f"safety_only policy did not consume localized fragments: {content_policy}")
    if content_policy_meta.get("privacy_fragment_adjudication_count", 0) != 0:
        fail(f"safety_only policy unexpectedly consumed privacy fragments: {content_policy}")
    report["decisions"]["content_policy"] = {
        "disposition_level": content_policy.get("disposition_level"),
        "unified_decision": content_policy.get("unified_decision"),
        "trust_level": content_policy.get("trust_level"),
    }

if profile == "privacy_only":
    privacy_policy = policy_by_doc[privacy_doc_id]
    privacy_policy_meta = privacy_policy.get("metadata") or {}
    if privacy_policy.get("disposition_level") not in {"P3", "P4", "P5"}:
        fail(f"privacy_only policy disposition too low for privacy-risk test doc: {privacy_policy}")
    if privacy_policy_meta.get("privacy_fragment_adjudication_count", 0) < min_privacy_findings:
        fail(f"privacy_only policy did not consume privacy fragment adjudications: {privacy_policy}")
    if privacy_policy_meta.get("content_localized_fragment_count", 0) != 0:
        fail(f"privacy_only policy unexpectedly consumed content localized fragments: {privacy_policy}")
    report["decisions"]["privacy_policy"] = {
        "disposition_level": privacy_policy.get("disposition_level"),
        "unified_decision": privacy_policy.get("unified_decision"),
        "trust_level": privacy_policy.get("trust_level"),
    }

report["decisions"]["summary"] = {
    "overall_disposition": summary.get("overall_disposition"),
    "unified_decision": summary.get("unified_decision"),
    "trust_level": summary.get("trust_level"),
    "counts_by_disposition": summary.get("counts_by_disposition"),
}

if profile == "full":
    windows = load_jsonl("02a_content_candidate_windows.jsonl")
    fragments = load_jsonl("02aa_content_fragment_localization.jsonl")
    content_adjudications = load_jsonl("02g_content_fragment_adjudications.jsonl")
    content_assessments = load_jsonl("02h_content_document_assessments.jsonl")
    privacy_assessments = load_jsonl("03g_privacy_document_assessments.jsonl")

    privacy_doc_content_windows = by_doc(windows, privacy_doc_id)
    privacy_doc_content_fragments = by_doc(fragments, privacy_doc_id)
    privacy_doc_content_adjudications = by_doc(content_adjudications, privacy_doc_id)
    if privacy_doc_content_windows or privacy_doc_content_fragments or privacy_doc_content_adjudications:
        fail(
            "content chain consumed privacy test doc as content risk: "
            f"windows={len(privacy_doc_content_windows)}, fragments={len(privacy_doc_content_fragments)}, "
            f"adjudications={len(privacy_doc_content_adjudications)}"
        )

    privacy_doc_content_assessments = by_doc(content_assessments, privacy_doc_id)
    if not privacy_doc_content_assessments:
        fail("full mode missing content document scope-guard assessment for privacy doc")
    privacy_content_assessment = privacy_doc_content_assessments[0]
    privacy_content_metadata = privacy_content_assessment.get("metadata") or {}
    if privacy_content_metadata.get("can_raise_disposition") is not False:
        fail(f"privacy doc content assessment can still raise disposition: {privacy_content_assessment}")
    if privacy_content_assessment.get("recommended_action") != "keep":
        fail(f"privacy doc content assessment should keep via scope guard: {privacy_content_assessment}")

    content_doc_privacy_findings = findings_for_doc(load_jsonl("03_privacy_detection.jsonl"), content_doc_id)
    content_doc_privacy_assessments = by_doc(privacy_assessments, content_doc_id)
    if not content_doc_privacy_findings and content_doc_privacy_assessments:
        content_privacy_metadata = content_doc_privacy_assessments[0].get("metadata") or {}
        if content_privacy_metadata.get("can_raise_disposition") is not False:
            fail(f"content doc privacy assessment should be scope-guarded when no privacy findings: {content_doc_privacy_assessments[0]}")

    policy_by_doc = {item.get("doc_id"): item for item in policy_rows}
    content_policy = policy_by_doc.get(content_doc_id)
    privacy_policy = policy_by_doc.get(privacy_doc_id)
    if not content_policy or not privacy_policy:
        fail(f"full mode policy decisions missing docs: {sorted(policy_by_doc)}")
    content_policy_meta = content_policy.get("metadata") or {}
    privacy_policy_meta = privacy_policy.get("metadata") or {}
    if content_policy_meta.get("content_localized_fragment_count", 0) < min_content_fragments:
        fail(f"content policy did not consume localized fragments: {content_policy}")
    if privacy_policy_meta.get("content_localized_fragment_count", 0) != 0:
        fail(f"privacy policy consumed content localized fragments: {privacy_policy}")
    privacy_reason_codes = [str(item) for item in privacy_policy.get("reason_codes") or []]
    if any(item.startswith("content_localized_fragments:") for item in privacy_reason_codes):
        fail(f"privacy policy contains content-localized reason codes: {privacy_policy}")
    if content_policy.get("disposition_level") in {"P0", "P1"}:
        fail(f"content policy disposition too low for content-risk test doc: {content_policy}")
    if privacy_policy.get("disposition_level") in {"P0", "P1"}:
        fail(f"privacy policy disposition too low for privacy-risk test doc: {privacy_policy}")

    audit_by_doc = {item.get("doc_id"): item for item in audit_rows}
    content_audit = audit_by_doc.get(content_doc_id)
    privacy_audit = audit_by_doc.get(privacy_doc_id)
    if not content_audit or not privacy_audit:
        fail(f"full mode audit records missing docs: {sorted(audit_by_doc)}")
    if not content_audit.get("content_candidate_windows") or not content_audit.get("content_localized_fragments"):
        fail(f"content audit lacks content candidate windows/localized fragments: {content_audit}")
    if privacy_audit.get("content_candidate_windows") or privacy_audit.get("content_localized_fragments"):
        fail(f"privacy audit unexpectedly contains content windows/fragments: {privacy_audit}")
    if not privacy_audit.get("privacy_fragment_adjudications") or not privacy_audit.get("privacy_document_assessment"):
        fail(f"privacy audit lacks privacy adjudication/document assessment: {privacy_audit}")

    report["independence"] = {
        "privacy_doc_content_candidate_windows": len(privacy_doc_content_windows),
        "privacy_doc_content_localized_fragments": len(privacy_doc_content_fragments),
        "privacy_doc_content_fragment_adjudications": len(privacy_doc_content_adjudications),
        "privacy_doc_content_can_raise_disposition": privacy_content_metadata.get("can_raise_disposition"),
        "privacy_policy_content_localized_fragment_count": privacy_policy_meta.get("content_localized_fragment_count", 0),
        "content_doc_privacy_finding_count": len(content_doc_privacy_findings),
    }
    report["decisions"]["content_policy"] = {
        "disposition_level": content_policy.get("disposition_level"),
        "unified_decision": content_policy.get("unified_decision"),
        "trust_level": content_policy.get("trust_level"),
    }
    report["decisions"]["privacy_policy"] = {
        "disposition_level": privacy_policy.get("disposition_level"),
        "unified_decision": privacy_policy.get("unified_decision"),
        "trust_level": privacy_policy.get("trust_level"),
    }

report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
PY
  json_pretty < "$report_json" >&2
}

run_profile() {
  local profile="$1"
  shift
  local submit_json
  submit_json="$(submit_job "$profile" "$@")"
  local task_id
  task_id="$(extract_task_id "$submit_json")"
  local result_json
  result_json="$(wait_for_result "$task_id" "$profile")"
  local run_dir
  run_dir="$(extract_run_dir "$result_json")"
  validate_real_profile "$profile" "$run_dir"
}

main() {
  require_cmd curl
  activate_text_env
  mkdir -p "$TEMP_DIR" "$OUTPUT_ROOT"
  cd "$PROJECT"

  [[ -f "$CONTENT_JSON" ]] || fail "CONTENT_JSON does not exist: $CONTENT_JSON"
  [[ -f "$PRIVACY_JSON" ]] || fail "PRIVACY_JSON does not exist: $PRIVACY_JSON"
  CONTENT_DOC_ID="$(doc_id_from_json "$CONTENT_JSON")"
  PRIVACY_DOC_ID="$(doc_id_from_json "$PRIVACY_JSON")"
  export CONTENT_DOC_ID PRIVACY_DOC_ID

  log "Content test file: $CONTENT_JSON ($CONTENT_DOC_ID)"
  log "Privacy test file: $PRIVACY_JSON ($PRIVACY_DOC_ID)"
  log "Output root: $OUTPUT_ROOT"

  probe_pii
  probe_qwen3guard
  probe_qwen35
  probe_text_api

  run_profile "safety_only" "$CONTENT_JSON"
  run_profile "privacy_only" "$PRIVACY_JSON"
  run_profile "full" "$CONTENT_JSON" "$PRIVACY_JSON"

  log "Real JSON text compliance test completed successfully."
  log "Reports: $TEMP_DIR/text_real_json_report_safety_only.json, $TEMP_DIR/text_real_json_report_privacy_only.json, $TEMP_DIR/text_real_json_report_full.json"
  log "Artifacts: $OUTPUT_ROOT"
}

main "$@"
