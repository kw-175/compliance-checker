# OPA Rego Policy for Text Data Compliance
#
# Evaluates evidence bundles and returns per-document decisions
# along with an overall pipeline decision.
#
# Decision levels: allow > review > quarantine > reject

package compliance.decision

import rego.v1

# ─── Default ────────────────────────────────────────────────

default overall_decision := "review"

# ─── Per-document decision ──────────────────────────────────

document_decisions := [dd |
    some doc in input.documents
    dd := evaluate_document(doc)
]

evaluate_document(doc) := result if {
    reasons := array.concat(
        array.concat(
            array.concat(
                secret_reasons(doc),
                safety_reasons(doc),
            ),
            pii_reasons(doc),
        ),
        compliance_reasons(doc),
    )
    decision := decide(doc)
    scores := {
        "secrets":    secret_score(doc),
        "safety":     safety_score(doc),
        "privacy":    pii_score(doc),
        "compliance": compliance_score(doc),
        "text_scan":  text_scan_score(doc),
    }
    result := {
        "doc_id":   doc.doc_id,
        "decision": decision,
        "reasons":  reasons,
        "scores":   scores,
    }
}

# ─── Scoring functions ──────────────────────────────────────

secret_score(doc) := 0 if {
    doc.secret_count > 0
} else := 1

safety_score(doc) := 0 if {
    doc.safety_level == "unsafe"
} else := 0.5 if {
    doc.safety_level == "controversial"
} else := 1

pii_score(doc) := 0.3 if {
    doc.pii_count > 5
} else := 0.7 if {
    doc.pii_count > 0
} else := 1

compliance_score(doc) := 0.2 if {
    doc.compliance_count > 0
} else := 1

text_scan_score(doc) := 0.2 if {
    doc.keyword_count + doc.regex_count > 20
} else := 0.6 if {
    doc.keyword_count + doc.regex_count > 5
} else := 1

# ─── Reason generators ─────────────────────────────────────

secret_reasons(doc) := [sprintf("found %d leaked secret(s)", [doc.secret_count])] if {
    doc.secret_count > 0
} else := []

safety_reasons(doc) := [sprintf("content classified as %s", [upper(doc.safety_level)])] if {
    doc.safety_level != "safe"
} else := []

pii_reasons(doc) := [sprintf("high PII density: %d entities", [doc.pii_count])] if {
    doc.pii_count > 5
} else := []

compliance_reasons(doc) := [sprintf("%d license compliance issue(s)", [doc.compliance_count])] if {
    doc.compliance_count > 0
} else := []

# ─── Decision logic ────────────────────────────────────────

decide(doc) := "reject" if {
    secret_score(doc) == 0
} else := "reject" if {
    safety_score(doc) == 0
} else := "quarantine" if {
    compliance_score(doc) <= 0.3
} else := "quarantine" if {
    pii_score(doc) <= 0.3
} else := "review" if {
    text_scan_score(doc) <= 0.6
} else := "review" if {
    safety_score(doc) <= 0.5
} else := "allow"

# ─── Overall decision ──────────────────────────────────────

overall_decision := "reject" if {
    some dd in document_decisions
    dd.decision == "reject"
} else := "quarantine" if {
    some dd in document_decisions
    dd.decision == "quarantine"
} else := "review" if {
    some dd in document_decisions
    dd.decision == "review"
} else := "allow"
