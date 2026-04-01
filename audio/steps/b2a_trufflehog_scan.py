"""
Step B2a: raw object secret scan with TruffleHog.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import SecretHit, SourceRecord
from audio.steps import run_command

logger = logging.getLogger(__name__)


def _run_trufflehog(binary: str, target_path: str) -> list[dict[str, Any]]:
    # TruffleHog 以 JSONL 逐行输出，返回码 1 可能代表发现命中，属于可接受状态。
    result = run_command(
        [binary, "filesystem", target_path, "--json", "--no-update"],
        ok_returncodes=(0, 1),
    )
    if result is None:
        return []

    findings: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            findings.append(json.loads(line))
        except json.JSONDecodeError:
            # 单行格式异常直接跳过，避免影响整体扫描结果。
            logger.debug("Skipping malformed TruffleHog line for %s", target_path)
    return findings


def _parse_finding(source_id: str, source_path: str, payload: dict[str, Any]) -> SecretHit:
    # 兼容 TruffleHog 输出层级，提取统一 SecretHit 字段。
    filesystem = payload.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {})
    return SecretHit(
        source_id=source_id,
        detector_type=str(payload.get("DetectorName", payload.get("DetectorType", ""))),
        decoder_type=str(payload.get("DecoderName", "")),
        raw_value=str(payload.get("Raw", "")),
        redacted=str(payload.get("Redacted", "")),
        file_path=str(filesystem.get("file", source_path)),
        line_number=int(filesystem.get("line", 0) or 0),
        verified=bool(payload.get("Verified", False)),
        extra=payload,
    )


def run(sources: list[SourceRecord], settings: Settings) -> list[SecretHit]:
    # 按目录扫描避免重复工作，再按文件路径映射回 source_id。
    hits: list[SecretHit] = []
    scanned_dirs: set[str] = set()
    path_to_source: dict[str, SourceRecord] = {}

    for source in sources:
        path_to_source[str(Path(source.path).resolve())] = source

    for source in sources:
        target_dir = str(Path(source.path).resolve().parent)
        if target_dir in scanned_dirs:
            # 同目录仅扫描一次，避免重复触发外部扫描器。
            continue
        scanned_dirs.add(target_dir)

        findings = _run_trufflehog(settings.trufflehog_bin, target_dir)
        for payload in findings:
            finding_file = payload.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("file", "")
            resolved_file = str(Path(finding_file).resolve()) if finding_file else ""
            # 若无法精确回溯到文件，回退关联当前 source。
            matched_source = path_to_source.get(resolved_file, source)
            hits.append(_parse_finding(matched_source.source_id, matched_source.path, payload))

    logger.info("TruffleHog scan complete: %d secret hits", len(hits))
    return hits
