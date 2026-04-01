"""
Step B1: source classification.
"""

from __future__ import annotations

from pathlib import Path

from audio.models.schemas import SourceProfile, SourceRecord, SourceType

_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}
_REPO_SUFFIXES = {
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".md", ".yml", ".yaml", ".json", ".toml", ".ini", ".sh", ".ps1", ".sql", ".xml",
}


def _classify(record: SourceRecord) -> SourceType:
    # 先按后缀判断，再结合 MIME 类型兜底。
    path = Path(record.path)
    suffix = path.suffix.lower()
    mime = (record.mime_type or "").lower()
    if suffix in _AUDIO_SUFFIXES or mime.startswith("audio/"):
        return SourceType.AUDIO
    if suffix in _ARCHIVE_SUFFIXES:
        return SourceType.ARCHIVE
    if suffix in _REPO_SUFFIXES or mime.startswith("text/"):
        return SourceType.REPO
    return SourceType.MIXED


def run(sources: list[SourceRecord]) -> list[SourceProfile]:
    # 为每个 source 生成统一 profile，供下游步骤分流使用。
    return [
        SourceProfile(
            source_id=record.source_id,
            path=record.path,
            source_type=_classify(record),
            mime_type=record.mime_type,
            metadata={"size_bytes": record.size_bytes, "sha256": record.sha256},
        )
        for record in sources
    ]
