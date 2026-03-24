"""
Step A – Source Intake

Scans input paths (files, directories, URLs) and produces a registry of
every discrete source object with its hash, MIME type, and size.

Output → source_registry.jsonl
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path

from text.models.schemas import SourceRecord

logger = logging.getLogger(__name__)

BUFFER_SIZE = 65_536  # 64 KiB


def _sha256(file_path: Path) -> str:
    """Compute SHA-256 of a file in streaming fashion."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(BUFFER_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _detect_mime(file_path: Path) -> str:
    """Best-effort MIME type detection."""
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


def _collect_files(input_path: str) -> list[Path]:
    """Expand a single input path into concrete file paths."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file())
    logger.warning("Skipping non-existent path: %s", input_path)
    return []


def run(input_paths: list[str]) -> list[SourceRecord]:
    """
    Execute source intake.

    Parameters
    ----------
    input_paths : list[str]
        File paths, directory paths, or URLs.

    Returns
    -------
    list[SourceRecord]
    """
    records: list[SourceRecord] = []
    for raw_path in input_paths:
        files = _collect_files(raw_path)
        for fp in files:
            try:
                record = SourceRecord(
                    path=str(fp.resolve()),
                    size_bytes=fp.stat().st_size,
                    sha256=_sha256(fp),
                    mime_type=_detect_mime(fp),
                )
                records.append(record)
                logger.debug("Registered source: %s", record.source_id)
            except Exception:
                logger.exception("Failed to register source: %s", fp)
    logger.info("Source intake complete: %d sources registered", len(records))
    return records
