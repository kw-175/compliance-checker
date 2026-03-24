"""
Step H – Evidence Aggregation

Collects and joins results from B2 (secrets + compliance), D (dedup),
E1 (keyword + regex hits), F (privacy), and G (safety) into a unified
EvidenceBundle keyed by document ID.

Output → evidence_bundle.json
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from text.models.schemas import (
    ComplianceHit,
    DedupDocument,
    DocumentEvidence,
    EvidenceBundle,
    KeywordHit,
    PrivacyResult,
    RegexHit,
    SafetyResult,
    SecretHit,
)

logger = logging.getLogger(__name__)


def run(
    dedup_docs: list[DedupDocument],
    secret_hits: list[SecretHit],
    compliance_hits: list[ComplianceHit],
    keyword_hits: list[KeywordHit],
    regex_hits: list[RegexHit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    pipeline_run_id: str = "",
) -> EvidenceBundle:
    """
    Aggregate all scan results into a single EvidenceBundle.

    Parameters
    ----------
    dedup_docs : list[DedupDocument]
    secret_hits : list[SecretHit]
    compliance_hits : list[ComplianceHit]
    keyword_hits : list[KeywordHit]
    regex_hits : list[RegexHit]
    privacy_results : list[PrivacyResult]
    safety_results : list[SafetyResult]
    pipeline_run_id : str

    Returns
    -------
    EvidenceBundle
    """
    # Build lookup indices by doc_id
    secrets_by_source: dict[str, list[SecretHit]] = defaultdict(list)
    for h in secret_hits:
        secrets_by_source[h.source_id].append(h)

    compliance_by_source: dict[str, list[ComplianceHit]] = defaultdict(list)
    for h in compliance_hits:
        compliance_by_source[h.source_id].append(h)

    kw_by_doc: dict[str, list[KeywordHit]] = defaultdict(list)
    for h in keyword_hits:
        kw_by_doc[h.doc_id].append(h)

    regex_by_doc: dict[str, list[RegexHit]] = defaultdict(list)
    for h in regex_hits:
        regex_by_doc[h.doc_id].append(h)

    privacy_by_doc: dict[str, PrivacyResult] = {r.doc_id: r for r in privacy_results}
    safety_by_doc: dict[str, SafetyResult] = {r.doc_id: r for r in safety_results}

    # Build per-document evidence
    doc_evidences: list[DocumentEvidence] = []
    for doc in dedup_docs:
        evidence = DocumentEvidence(
            doc_id=doc.doc_id,
            source_id=doc.source_id,
            secret_hits=secrets_by_source.get(doc.source_id, []),
            compliance_hits=compliance_by_source.get(doc.source_id, []),
            is_duplicate=doc.is_duplicate,
            keyword_hits=kw_by_doc.get(doc.doc_id, []),
            regex_hits=regex_by_doc.get(doc.doc_id, []),
            privacy=privacy_by_doc.get(doc.doc_id),
            safety=safety_by_doc.get(doc.doc_id),
        )
        doc_evidences.append(evidence)

    # Summary statistics
    summary = {
        "total_documents": len(dedup_docs),
        "unique_documents": sum(1 for d in dedup_docs if not d.is_duplicate),
        "duplicate_documents": sum(1 for d in dedup_docs if d.is_duplicate),
        "total_secret_hits": len(secret_hits),
        "total_compliance_hits": len(compliance_hits),
        "total_keyword_hits": len(keyword_hits),
        "total_regex_hits": len(regex_hits),
        "total_pii_entities": sum(r.pii_count for r in privacy_results),
        "unsafe_documents": sum(
            1 for r in safety_results if r.safety_level.value == "unsafe"
        ),
        "controversial_documents": sum(
            1 for r in safety_results if r.safety_level.value == "controversial"
        ),
    }

    bundle = EvidenceBundle(
        pipeline_run_id=pipeline_run_id,
        documents=doc_evidences,
        summary=summary,
    )

    logger.info(
        "Evidence aggregation complete: %d docs, %d secret hits, "
        "%d compliance hits, %d keyword hits, %d regex hits, "
        "%d PII entities, %d unsafe, %d controversial",
        summary["total_documents"],
        summary["total_secret_hits"],
        summary["total_compliance_hits"],
        summary["total_keyword_hits"],
        summary["total_regex_hits"],
        summary["total_pii_entities"],
        summary["unsafe_documents"],
        summary["controversial_documents"],
    )
    return bundle
