"""
Step E1b – Hyperscan / Regex Scan

Primary:  python-hyperscan for high-performance multi-pattern regex matching
Fallback: Python stdlib `re` module

Output → regex_hits.jsonl
"""

from __future__ import annotations

import logging
import re as re_stdlib
from pathlib import Path
from typing import Any

import yaml

from text.config.settings import Settings
from text.models.schemas import DedupDocument, RegexHit

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 60


def _load_patterns(patterns_file: Path) -> dict[str, str]:
    """Load named regex patterns from a YAML file."""
    if not patterns_file.exists():
        logger.warning("Patterns file not found: %s", patterns_file)
        return {}
    try:
        with open(patterns_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded %d regex patterns from %s", len(data), patterns_file)
        return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logger.error("Failed to load patterns file: %s", e)
        return {}


def _extract_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


# ────────────────────────────────────────────────────────────
# Hyperscan backend
# ────────────────────────────────────────────────────────────

def _scan_with_hyperscan(
    documents: list[DedupDocument],
    patterns: dict[str, str],
) -> list[RegexHit]:
    """Use python-hyperscan for multi-pattern scanning."""
    import hyperscan  # type: ignore

    pattern_names = list(patterns.keys())
    pattern_exprs = [p.encode("utf-8") for p in patterns.values()]
    pattern_ids = list(range(len(pattern_names)))
    pattern_flags = [hyperscan.HS_FLAG_DOTALL | hyperscan.HS_FLAG_SINGLEMATCH] * len(pattern_exprs)

    db = hyperscan.Database()
    db.compile(
        expressions=pattern_exprs,
        ids=pattern_ids,
        flags=pattern_flags,
    )

    all_hits: list[RegexHit] = []

    for doc in documents:
        if doc.is_duplicate:
            continue

        doc_hits: list[dict[str, Any]] = []

        def on_match(id_: int, from_: int, to: int, flags: int, context: Any = None) -> bool:
            doc_hits.append({"id": id_, "from": from_, "to": to})
            return False  # continue scanning

        text_bytes = doc.text.encode("utf-8")
        db.scan(text_bytes, match_event_handler=on_match)

        for hit in doc_hits:
            pid = hit["id"]
            name = pattern_names[pid]
            start = hit["from"]
            end = hit["to"]
            matched = text_bytes[start:end].decode("utf-8", errors="replace")
            all_hits.append(
                RegexHit(
                    doc_id=doc.doc_id,
                    pattern_name=name,
                    pattern=patterns[name],
                    matched_text=matched[:200],
                    start_pos=start,
                    end_pos=end,
                    context=_extract_context(doc.text, start, end),
                )
            )

    return all_hits


# ────────────────────────────────────────────────────────────
# Python re fallback
# ────────────────────────────────────────────────────────────

def _scan_with_re(
    documents: list[DedupDocument],
    patterns: dict[str, str],
) -> list[RegexHit]:
    """Fallback: use Python stdlib re for multi-pattern scanning."""
    compiled: list[tuple[str, re_stdlib.Pattern]] = []
    for name, expr in patterns.items():
        try:
            compiled.append((name, re_stdlib.compile(expr)))
        except re_stdlib.error as e:
            logger.warning("Invalid regex pattern '%s': %s", name, e)

    all_hits: list[RegexHit] = []

    for doc in documents:
        if doc.is_duplicate:
            continue
        for name, regex in compiled:
            for m in regex.finditer(doc.text):
                all_hits.append(
                    RegexHit(
                        doc_id=doc.doc_id,
                        pattern_name=name,
                        pattern=regex.pattern,
                        matched_text=m.group()[:200],
                        start_pos=m.start(),
                        end_pos=m.end(),
                        context=_extract_context(doc.text, m.start(), m.end()),
                    )
                )
    return all_hits


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def run(
    documents: list[DedupDocument],
    settings: Settings | None = None,
) -> list[RegexHit]:
    """
    Execute regex pattern scan.

    Parameters
    ----------
    documents : list[DedupDocument]
    settings : Settings, optional

    Returns
    -------
    list[RegexHit]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    patterns = _load_patterns(settings.patterns_file)
    if not patterns:
        logger.warning("No regex patterns loaded – skipping E1b")
        return []

    # Try Hyperscan first
    try:
        hits = _scan_with_hyperscan(documents, patterns)
        logger.info(
            "Hyperscan regex scan complete: %d hits", len(hits)
        )
        return hits
    except ImportError:
        logger.warning(
            "python-hyperscan not installed; falling back to stdlib re.  "
            "pip install hyperscan"
        )
    except Exception as e:
        logger.warning("Hyperscan scan failed (%s); falling back to re", e)

    # Fallback to stdlib re
    hits = _scan_with_re(documents, patterns)
    logger.info("Regex scan (re fallback) complete: %d hits", len(hits))
    return hits
