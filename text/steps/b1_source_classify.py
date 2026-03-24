"""
Step B1 – Source Classification

Reads the source registry and classifies each source into one of:
  code | repo | package | binary | web_text | pdf_text | mixed

Output → source_profile.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path

from text.models.schemas import SourceProfile, SourceRecord, SourceType

logger = logging.getLogger(__name__)

# ── Extension → SourceType mapping ──────────────────────────

_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".lua", ".pl", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".r", ".R", ".m", ".sql", ".html", ".css", ".scss", ".sass",
    ".vue", ".svelte", ".yaml", ".yml", ".toml", ".json", ".xml",
    ".dockerfile", ".tf", ".hcl", ".proto", ".graphql", ".makefile",
}

_PACKAGE_EXTS = {
    ".whl", ".tar.gz", ".tgz", ".egg", ".gem", ".jar", ".war",
    ".zip", ".rar", ".7z", ".deb", ".rpm", ".nupkg", ".apk",
}

_BINARY_EXTS = {
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".bin", ".dat",
    ".elf", ".msi", ".img", ".iso",
}

_PDF_EXTS = {".pdf"}

_WEB_EXTS = {".html", ".htm", ".xhtml", ".mhtml"}

_TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".tsv", ".log", ".ini", ".cfg"}

# ── MIME prefix mapping ─────────────────────────────────────

_MIME_MAP: dict[str, SourceType] = {
    "text/html": SourceType.WEB_TEXT,
    "application/pdf": SourceType.PDF_TEXT,
    "application/zip": SourceType.PACKAGE,
    "application/x-tar": SourceType.PACKAGE,
    "application/gzip": SourceType.PACKAGE,
    "application/x-executable": SourceType.BINARY,
    "application/x-mach-binary": SourceType.BINARY,
    "application/x-dosexec": SourceType.BINARY,
    "application/octet-stream": SourceType.BINARY,
}


def _classify_single(record: SourceRecord) -> SourceType:
    """Determine the SourceType for a single source record."""
    # 1) Try MIME lookup
    if record.mime_type in _MIME_MAP:
        return _MIME_MAP[record.mime_type]

    # 2) Try extension lookup
    suffix = Path(record.path).suffix.lower()
    if suffix in _CODE_EXTS:
        return SourceType.CODE
    if suffix in _PACKAGE_EXTS:
        return SourceType.PACKAGE
    if suffix in _BINARY_EXTS:
        return SourceType.BINARY
    if suffix in _PDF_EXTS:
        return SourceType.PDF_TEXT
    if suffix in _WEB_EXTS:
        return SourceType.WEB_TEXT
    if suffix in _TEXT_EXTS:
        return SourceType.WEB_TEXT  # plain text treated as web_text

    # 3) Heuristic: text/* MIME → mixed
    if record.mime_type.startswith("text/"):
        return SourceType.CODE  # conservative: treat unknown text as code

    return SourceType.MIXED


def run(sources: list[SourceRecord]) -> list[SourceProfile]:
    """
    Classify each source record.

    Parameters
    ----------
    sources : list[SourceRecord]

    Returns
    -------
    list[SourceProfile]
    """
    profiles: list[SourceProfile] = []
    for src in sources:
        source_type = _classify_single(src)
        profile = SourceProfile(
            source_id=src.source_id,
            path=src.path,
            source_type=source_type,
            mime_type=src.mime_type,
            metadata={"size_bytes": src.size_bytes, "sha256": src.sha256},
        )
        profiles.append(profile)
        logger.debug(
            "Classified %s → %s (mime=%s)", src.source_id, source_type.value, src.mime_type
        )
    logger.info("Source classification complete: %d profiles", len(profiles))
    return profiles
