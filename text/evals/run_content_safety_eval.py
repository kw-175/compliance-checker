from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RISK_RANK = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
DECISION_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate content-safety decision artifacts against a JSONL baseline.")
    parser.add_argument("--actual", required=True, help="Path to 02f_content_safety_final_decisions.jsonl or 02b_content_safety_decisions.jsonl")
    parser.add_argument(
        "--expected",
        default=str(Path(__file__).resolve().parent / "expected" / "content_safety_expected_decisions.jsonl"),
        help="Expected decision JSONL baseline.",
    )
    parser.add_argument("--report", default="", help="Optional path to write eval report JSON.")
    args = parser.parse_args()

    actual = {item["doc_id"]: item for item in read_jsonl(Path(args.actual))}
    expected = read_jsonl(Path(args.expected))
    failures: list[dict[str, Any]] = []
    passed = 0

    for item in expected:
        doc_id = item["doc_id"]
        observed = actual.get(doc_id)
        if not observed:
            failures.append({"doc_id": doc_id, "reason": "missing_actual_record"})
            continue
        reasons = _check_record(observed, item)
        if reasons:
            failures.append({"doc_id": doc_id, "reason": "; ".join(reasons), "actual": observed, "expected": item})
        else:
            passed += 1

    report = {
        "total": len(expected),
        "passed": passed,
        "failed": len(failures),
        "failure_rate": round(len(failures) / len(expected), 4) if expected else 0.0,
        "failures": failures,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 1 if failures else 0


def _check_record(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    decision = str(actual.get("decision") or "P0")
    risk = str(actual.get("risk_level") or "C0")
    labels = {str(item) for item in actual.get("labels", [])}

    if expected.get("expected_decision_min") and DECISION_RANK.get(decision, -1) < DECISION_RANK[str(expected["expected_decision_min"])]:
        reasons.append(f"decision {decision} below minimum {expected['expected_decision_min']}")
    if expected.get("expected_decision_max") and DECISION_RANK.get(decision, 99) > DECISION_RANK[str(expected["expected_decision_max"])]:
        reasons.append(f"decision {decision} above maximum {expected['expected_decision_max']}")
    if expected.get("expected_risk_min") and RISK_RANK.get(risk, -1) < RISK_RANK[str(expected["expected_risk_min"])]:
        reasons.append(f"risk {risk} below minimum {expected['expected_risk_min']}")
    if expected.get("expected_risk_max") and RISK_RANK.get(risk, 99) > RISK_RANK[str(expected["expected_risk_max"])]:
        reasons.append(f"risk {risk} above maximum {expected['expected_risk_max']}")

    expected_any = {str(item) for item in expected.get("expected_labels_any", [])}
    if expected_any and not _labels_overlap(labels, expected_any):
        reasons.append(f"labels {sorted(labels)} did not match any of {sorted(expected_any)}")
    return reasons


def _labels_overlap(actual: set[str], expected: set[str]) -> bool:
    for left in actual:
        for right in expected:
            if left == right or left.startswith(right + ".") or right.startswith(left + "."):
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
