"""
Step A: source intake.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path

from audio.models.schemas import SourceRecord

logger = logging.getLogger(__name__)

_BUFFER_SIZE = 65536


def _sha256(path: Path) -> str:
    # 分块计算哈希，兼顾大文件内存占用。
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_mime(path: Path) -> str:
    # 根据扩展名推断 MIME，无法识别时回落到通用二进制类型。
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _collect_files(input_path: str) -> list[Path]:
    # 输入既可为单文件也可为目录；目录场景递归收集文件。
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(file for file in path.rglob("*") if file.is_file())
    logger.warning("Skipping missing input path: %s", input_path)
    return []


def run(input_paths: list[str]) -> list[SourceRecord]:
    # 入口步骤：将原始路径列表标准化为 SourceRecord 列表。
    records: list[SourceRecord] = []
    for raw_path in input_paths:
        for file_path in _collect_files(raw_path):
            try:
                records.append(
                    SourceRecord(
                        path=str(file_path.resolve()),
                        size_bytes=file_path.stat().st_size,
                        sha256=_sha256(file_path),
                        mime_type=_detect_mime(file_path),
                    )
                )
            except Exception:
                # 单文件失败不终止全局 intake，尽量保留可处理样本。
                logger.exception("Failed to register source: %s", file_path)
    logger.info("Source intake complete: %d sources", len(records))
    return records
