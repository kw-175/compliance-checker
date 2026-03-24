"""
Step C – Text Extraction & Preprocessing

Uses DataTrove (Trafilatura backend) for HTML/web text extraction,
PyMuPDF for PDF, and direct reading for plain text / code files.
Performs Unicode normalisation, whitespace cleanup, and encoding fixes.

Output → cleaned_documents.jsonl
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import CleanedDocument, SourceProfile, SourceType

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Text Extraction Helpers
# ────────────────────────────────────────────────────────────

def _extract_plain_text(file_path: str) -> Optional[str]:
    """Read plain text / code files directly."""
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to read plain text %s: %s", file_path, e)
        return None


def _extract_html_text(file_path: str) -> Optional[str]:
    """Extract text from HTML using Trafilatura (via DataTrove)."""
    try:
        import trafilatura
        html = Path(file_path).read_text(encoding="utf-8", errors="replace")
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        return text
    except ImportError:
        logger.warning(
            "trafilatura not installed; falling back to basic HTML stripping"
        )
        return _strip_html_basic(file_path)
    except Exception as e:
        logger.warning("Trafilatura extraction failed for %s: %s", file_path, e)
        return _strip_html_basic(file_path)


def _strip_html_basic(file_path: str) -> Optional[str]:
    """Naive HTML tag stripping fallback."""
    try:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", raw)
    except Exception:
        return None


def _extract_pdf_text(file_path: str) -> Optional[str]:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) not installed; skipping PDF extraction for %s", file_path
        )
        return None
    except Exception as e:
        logger.warning("PDF extraction failed for %s: %s", file_path, e)
        return None


# ────────────────────────────────────────────────────────────
# Text Cleaning
# ────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """
    Apply standard text normalisation:
    - Unicode NFC normalisation
    - Replace non-breaking spaces and zero-width chars
    - Collapse excessive whitespace / blank lines
    """
    # Unicode NFC
    text = unicodedata.normalize("NFC", raw)

    # Replace common invisible characters
    text = text.replace("\u00a0", " ")   # non-breaking space
    text = text.replace("\u200b", "")    # zero-width space
    text = text.replace("\u200c", "")    # zero-width non-joiner
    text = text.replace("\u200d", "")    # zero-width joiner
    text = text.replace("\ufeff", "")    # BOM

    # Collapse runs of whitespace within lines
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse blank lines (3+ newlines → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_language(text: str) -> str:
    """Best-effort language detection (returns ISO 639-1 code)."""
    try:
        from langdetect import detect
        return detect(text[:2000])
    except Exception:
        # Heuristic: if >30% CJK characters, label as zh
        cjk_count = sum(1 for c in text[:1000] if "\u4e00" <= c <= "\u9fff")
        if cjk_count / max(len(text[:1000]), 1) > 0.3:
            return "zh"
        return "en"


# ────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────

_EXTRACTOR_MAP = {
    SourceType.WEB_TEXT: _extract_html_text,
    SourceType.PDF_TEXT: _extract_pdf_text,
    SourceType.CODE: _extract_plain_text,
    SourceType.REPO: _extract_plain_text,
    SourceType.PACKAGE: _extract_plain_text,
    SourceType.BINARY: _extract_plain_text,
    SourceType.MIXED: _extract_plain_text,
}


def run(
    profiles: list[SourceProfile],
    settings: Settings | None = None,
) -> list[CleanedDocument]:
    """
    Extract and clean text from classified source profiles.

    Parameters
    ----------
    profiles : list[SourceProfile]
    settings : Settings, optional

    Returns
    -------
    list[CleanedDocument]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    documents: list[CleanedDocument] = []
    for profile in profiles:
        extractor = _EXTRACTOR_MAP.get(profile.source_type, _extract_plain_text)
        raw_text = extractor(profile.path)
        if raw_text is None or not raw_text.strip():
            logger.debug("No extractable text for %s", profile.source_id)
            continue

        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue

        lang = _detect_language(cleaned)

        doc = CleanedDocument(
            source_id=profile.source_id,
            text=cleaned,
            char_count=len(cleaned),
            language=lang,
            metadata={
                "source_type": profile.source_type.value,
                "original_path": profile.path,
            },
        )
        documents.append(doc)
        logger.debug(
            "Extracted doc %s from %s (%d chars, lang=%s)",
            doc.doc_id, profile.source_id, doc.char_count, lang,
        )

    logger.info(
        "Text extraction complete: %d documents from %d profiles",
        len(documents), len(profiles),
    )
    return documents
