"""
Step D – Early Deduplication

Primary:  Duplodocus CLI (Rust, exact + MinHash)
Fallback: Python datasketch MinHash LSH

Output → deduped_documents.jsonl + dedup_map.jsonl
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import CleanedDocument, DedupDocument, DedupMapEntry

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Exact Dedup (hash-based)
# ────────────────────────────────────────────────────────────

def _exact_dedup(
    documents: list[CleanedDocument],
) -> tuple[list[CleanedDocument], list[DedupMapEntry]]:
    """Remove exact duplicates via SHA-256 of text content."""
    seen: dict[str, str] = {}  # hash → first doc_id
    unique: list[CleanedDocument] = []
    dedup_map: list[DedupMapEntry] = []

    for doc in documents:
        text_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        if text_hash in seen:
            dedup_map.append(
                DedupMapEntry(
                    doc_id=doc.doc_id,
                    duplicate_of=seen[text_hash],
                    jaccard_similarity=1.0,
                )
            )
            logger.debug("Exact dup: %s == %s", doc.doc_id, seen[text_hash])
        else:
            seen[text_hash] = doc.doc_id
            unique.append(doc)
    return unique, dedup_map


# ────────────────────────────────────────────────────────────
# Near-Dedup via datasketch MinHash LSH (Python fallback)
# ────────────────────────────────────────────────────────────

def _shingle(text: str, k: int = 5) -> set[str]:
    """Generate character-level k-shingles from text."""
    return {text[i : i + k] for i in range(max(len(text) - k + 1, 1))}


def _minhash_dedup(
    documents: list[CleanedDocument],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> tuple[list[DedupDocument], list[DedupMapEntry]]:
    """
    Near-duplicate detection using MinHash LSH from datasketch.

    Falls back to a trivial pass-through if datasketch is not installed.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning(
            "datasketch not installed; skipping near-dedup.  "
            "pip install datasketch"
        )
        return [
            DedupDocument(
                doc_id=d.doc_id,
                source_id=d.source_id,
                text=d.text,
                is_duplicate=False,
            )
            for d in documents
        ], []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    dedup_map: list[DedupMapEntry] = []
    results: list[DedupDocument] = []

    # Build MinHash for each document
    for doc in documents:
        m = MinHash(num_perm=num_perm)
        for shingle in _shingle(doc.text):
            m.update(shingle.encode("utf-8"))
        minhashes[doc.doc_id] = m

    # Insert and query
    seen_duplicates: set[str] = set()
    for doc in documents:
        m = minhashes[doc.doc_id]
        query_result = lsh.query(m)

        duplicate_of: Optional[str] = None
        for candidate_id in query_result:
            if candidate_id != doc.doc_id and candidate_id not in seen_duplicates:
                # Compute actual Jaccard
                jaccard = m.jaccard(minhashes[candidate_id])
                if jaccard >= threshold:
                    duplicate_of = candidate_id
                    dedup_map.append(
                        DedupMapEntry(
                            doc_id=doc.doc_id,
                            duplicate_of=candidate_id,
                            jaccard_similarity=round(jaccard, 4),
                        )
                    )
                    seen_duplicates.add(doc.doc_id)
                    break

        is_dup = duplicate_of is not None
        results.append(
            DedupDocument(
                doc_id=doc.doc_id,
                source_id=doc.source_id,
                text=doc.text,
                is_duplicate=is_dup,
                duplicate_of=duplicate_of,
            )
        )

        if not is_dup:
            try:
                lsh.insert(doc.doc_id, m)
            except ValueError:
                pass  # Already exists

    return results, dedup_map


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def run(
    documents: list[CleanedDocument],
    settings: Settings | None = None,
) -> tuple[list[DedupDocument], list[DedupMapEntry]]:
    """
    Execute deduplication pipeline:
      1. Exact dedup (hash-based)
      2. Near dedup (MinHash LSH)

    Parameters
    ----------
    documents : list[CleanedDocument]
    settings : Settings, optional

    Returns
    -------
    tuple[list[DedupDocument], list[DedupMapEntry]]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    logger.info("Starting dedup on %d documents", len(documents))

    # Phase 1: exact dedup
    unique_docs, exact_map = _exact_dedup(documents)
    logger.info(
        "Exact dedup: %d → %d unique (%d duplicates removed)",
        len(documents), len(unique_docs), len(exact_map),
    )

    # Phase 2: near dedup on remaining
    dedup_results, near_map = _minhash_dedup(
        unique_docs,
        threshold=settings.dedup_threshold,
        num_perm=settings.dedup_num_perm,
    )

    # Combine maps
    all_map = exact_map + near_map
    total_dups = sum(1 for d in dedup_results if d.is_duplicate) + len(exact_map)
    logger.info(
        "Dedup complete: %d total duplicates found (%d exact, %d near)",
        total_dups, len(exact_map), len(near_map),
    )

    return dedup_results, all_map
