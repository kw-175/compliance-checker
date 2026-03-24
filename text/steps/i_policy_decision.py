"""
Step I – Policy Decision

Primary:  OPA REST API (POST /v1/data/compliance/decision)
Fallback: Local Python rule engine

Output → decision.json
"""

from __future__ import annotations

import logging
from typing import Any

from text.config.settings import Settings
from text.models.schemas import (
    Decision,
    DocumentDecision,
    DocumentEvidence,
    EvidenceBundle,
    PolicyDecision,
    SafetyLevel,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# OPA REST API
# ────────────────────────────────────────────────────────────

def _query_opa(
    evidence_bundle: EvidenceBundle,
    settings: Settings,
) -> PolicyDecision | None:
    """Send evidence to OPA and retrieve policy decision."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; cannot query OPA")
        return None

    url = f"{settings.opa_url}/{settings.opa_policy_path}"
    payload = {
        "input": {
            "pipeline_run_id": evidence_bundle.pipeline_run_id,
            "summary": evidence_bundle.summary,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "source_id": doc.source_id,
                    "is_duplicate": doc.is_duplicate,
                    "secret_count": len(doc.secret_hits),
                    "compliance_count": len(doc.compliance_hits),
                    "keyword_count": len(doc.keyword_hits),
                    "regex_count": len(doc.regex_hits),
                    "pii_count": doc.privacy.pii_count if doc.privacy else 0,
                    "safety_level": doc.safety.safety_level.value if doc.safety else "safe",
                    "harm_categories": doc.safety.harm_categories if doc.safety else [],
                }
                for doc in evidence_bundle.documents
            ],
        }
    }

    try:
        resp = httpx.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json().get("result", {})

        # Parse OPA response into PolicyDecision
        doc_decisions = []
        for dd in result.get("document_decisions", []):
            doc_decisions.append(
                DocumentDecision(
                    doc_id=dd.get("doc_id", ""),
                    decision=Decision(dd.get("decision", "review")),
                    reasons=dd.get("reasons", []),
                    scores=dd.get("scores", {}),
                )
            )

        return PolicyDecision(
            pipeline_run_id=evidence_bundle.pipeline_run_id,
            overall_decision=Decision(result.get("overall_decision", "review")),
            document_decisions=doc_decisions,
        )
    except Exception as e:
        logger.warning("OPA query failed: %s", e)
        return None


# ────────────────────────────────────────────────────────────
# Local rule-engine fallback
# ────────────────────────────────────────────────────────────

def _evaluate_document(doc: DocumentEvidence) -> DocumentDecision:
    """Apply deterministic rules to a single document's evidence."""
    reasons: list[str] = []
    scores: dict[str, float] = {}

    # Rule 1: Secrets → reject
    secret_count = len(doc.secret_hits)
    if secret_count > 0:
        reasons.append(f"found {secret_count} leaked secret(s)")
        scores["secrets"] = 0.0
    else:
        scores["secrets"] = 1.0

    # Rule 2: Safety → unsafe → reject, controversial → review
    safety_score = 1.0
    if doc.safety:
        if doc.safety.safety_level == SafetyLevel.UNSAFE:
            reasons.append(f"content classified as UNSAFE ({doc.safety.harm_categories})")
            safety_score = 0.0
        elif doc.safety.safety_level == SafetyLevel.CONTROVERSIAL:
            reasons.append(f"content classified as CONTROVERSIAL ({doc.safety.harm_categories})")
            safety_score = 0.5
    scores["safety"] = safety_score

    # Rule 3: PII still present after redaction → review
    pii_score = 1.0
    if doc.privacy and doc.privacy.pii_count > 5:
        reasons.append(f"high PII density: {doc.privacy.pii_count} entities")
        pii_score = 0.3
    elif doc.privacy and doc.privacy.pii_count > 0:
        pii_score = 0.7
    scores["privacy"] = pii_score

    # Rule 4: License compliance issues → quarantine
    compliance_score = 1.0
    copyleft_count = 0
    for ch in doc.compliance_hits:
        for lic in ch.licenses:
            expr_lower = lic.license_expression.lower()
            if any(k in expr_lower for k in ["gpl", "agpl", "copyleft"]):
                copyleft_count += 1
    if copyleft_count > 0:
        reasons.append(f"found {copyleft_count} copyleft license(s)")
        compliance_score = 0.2
    scores["compliance"] = compliance_score

    # Rule 5: Keyword / regex hits → flag if high
    text_scan_score = 1.0
    total_text_hits = len(doc.keyword_hits) + len(doc.regex_hits)
    if total_text_hits > 20:
        reasons.append(f"{total_text_hits} keyword/regex hits (high density)")
        text_scan_score = 0.2
    elif total_text_hits > 5:
        reasons.append(f"{total_text_hits} keyword/regex hits")
        text_scan_score = 0.6
    scores["text_scan"] = text_scan_score

    # ── Decision logic ──────────────────────────────────────
    min_score = min(scores.values()) if scores else 1.0

    if min_score <= 0.0:
        decision = Decision.REJECT
    elif min_score <= 0.3:
        decision = Decision.QUARANTINE
    elif min_score <= 0.6:
        decision = Decision.REVIEW
    else:
        decision = Decision.ALLOW

    return DocumentDecision(
        doc_id=doc.doc_id,
        decision=decision,
        reasons=reasons,
        scores=scores,
    )


def _local_evaluate(evidence_bundle: EvidenceBundle) -> PolicyDecision:
    """Evaluate all documents using local rules."""
    doc_decisions = [_evaluate_document(doc) for doc in evidence_bundle.documents]

    # Overall decision = worst individual decision
    priority = {Decision.REJECT: 0, Decision.QUARANTINE: 1, Decision.REVIEW: 2, Decision.ALLOW: 3}
    if doc_decisions:
        worst = min(doc_decisions, key=lambda d: priority[d.decision])
        overall = worst.decision
    else:
        overall = Decision.ALLOW

    return PolicyDecision(
        pipeline_run_id=evidence_bundle.pipeline_run_id,
        overall_decision=overall,
        document_decisions=doc_decisions,
    )


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def run(
    evidence_bundle: EvidenceBundle,
    settings: Settings | None = None,
) -> PolicyDecision:
    """
    Make policy decisions on the evidence bundle.

    Parameters
    ----------
    evidence_bundle : EvidenceBundle
    settings : Settings, optional

    Returns
    -------
    PolicyDecision
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    # Try OPA first
    if settings.opa_enabled:
        opa_result = _query_opa(evidence_bundle, settings)
        if opa_result is not None:
            logger.info(
                "OPA decision: %s (%d doc decisions)",
                opa_result.overall_decision.value,
                len(opa_result.document_decisions),
            )
            return opa_result
        logger.info("OPA unavailable; falling back to local rule engine")

    # Fallback to local rules
    result = _local_evaluate(evidence_bundle)

    decision_counts = {}
    for dd in result.document_decisions:
        decision_counts[dd.decision.value] = decision_counts.get(dd.decision.value, 0) + 1

    logger.info(
        "Policy decision (local): overall=%s, breakdown=%s",
        result.overall_decision.value, decision_counts,
    )
    return result
