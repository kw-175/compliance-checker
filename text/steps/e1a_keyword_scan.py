"""
Step E1a – FlashText2 Keyword Scan

Uses FlashText2 KeywordProcessor for ultra-fast multi-keyword matching
over the deduplicated document corpus.

Output → keyword_hits.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path

from text.config.settings import Settings
from text.models.schemas import DedupDocument, KeywordHit

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 60  # chars before/after match to include as context


def _load_keywords(keywords_file: Path) -> list[str]:
    """Load keywords from a newline-delimited text file."""
    if not keywords_file.exists():
        logger.warning("Keywords file not found: %s", keywords_file)
        return []
    lines: list[str] = []
    for line in keywords_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    logger.info("Loaded %d keywords from %s", len(lines), keywords_file)
    return lines


def _extract_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
    """Return a snippet of text around the match."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


def run(
    documents: list[DedupDocument],
    settings: Settings | None = None,
) -> list[KeywordHit]:
    """
    Execute FlashText2 keyword scan.

    Parameters
    ----------
    documents : list[DedupDocument]
        Non-duplicate documents from Step D.
    settings : Settings, optional

    Returns
    -------
    list[KeywordHit]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    keywords = _load_keywords(settings.keywords_file)
    if not keywords:
        logger.warning("No keywords to scan – skipping E1a")
        return []

    # Try FlashText2, fallback to basic string search
    try:
        from flashtext2 import KeywordProcessor
        kp = KeywordProcessor(case_sensitive=False)
        kp.add_keywords_from_list(keywords)
        use_flashtext = True
        logger.info("Using FlashText2 for keyword scanning")
    except ImportError:
        logger.warning(
            "flashtext2 not installed; falling back to basic string search.  "
            "pip install flashtext2"
        )
        use_flashtext = False

    all_hits: list[KeywordHit] = []

    for doc in documents:
        if doc.is_duplicate:
            continue

        if use_flashtext:
            # FlashText2 returns list of (keyword, start, end)
            matches = kp.extract_keywords(doc.text, span_info=True)
            for keyword, start, end in matches:
                all_hits.append(
                    KeywordHit(
                        doc_id=doc.doc_id,
                        keyword=keyword,
                        start_pos=start,
                        end_pos=end,
                        context=_extract_context(doc.text, start, end),
                    )
                )
        else:
            # Basic fallback: case-insensitive search
            text_lower = doc.text.lower()
            for kw in keywords:
                kw_lower = kw.lower()
                start = 0
                while True:
                    idx = text_lower.find(kw_lower, start)
                    if idx == -1:
                        break
                    end = idx + len(kw_lower)
                    all_hits.append(
                        KeywordHit(
                            doc_id=doc.doc_id,
                            keyword=kw,
                            start_pos=idx,
                            end_pos=end,
                            context=_extract_context(doc.text, idx, end),
                        )
                    )
                    start = end

    logger.info("Keyword scan complete: %d hits across %d documents", len(all_hits), len(documents))
    return all_hits
