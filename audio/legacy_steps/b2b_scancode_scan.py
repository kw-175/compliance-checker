"""
Step B2b: source compliance scan with ScanCode.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import ComplianceHit, LicenseMatch, SourceProfile, SourceType
from audio.steps import run_command

logger = logging.getLogger(__name__)

_ELIGIBLE_TYPES = {SourceType.REPO, SourceType.ARCHIVE, SourceType.MIXED}


def _run_scancode(binary: str, target_path: str, output_file: str) -> dict[str, Any]:
    # ScanCode 结果写入 JSON 文件，再由本步骤读取并结构化。
    result = run_command(
        [
            binary,
            "--license",
            "--copyright",
            "--info",
            "--json-pp",
            output_file,
            target_path,
            "--timeout",
            "120",
        ],
        timeout=600,
    )
    if result is None:
        return {}
    try:
        with open(output_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        # 输出文件缺失或损坏时记异常并返回空结果，交由上游降级处理。
        logger.exception("Failed to read ScanCode output for %s", target_path)
        return {}


def _parse_licenses(file_entry: dict[str, Any]) -> list[LicenseMatch]:
    # 将 ScanCode 的检测结果归一成统一 LicenseMatch 列表。
    licenses: list[LicenseMatch] = []
    for detection in file_entry.get("license_detections", []):
        matches = detection.get("matches") or [detection]
        for match in matches:
            licenses.append(
                LicenseMatch(
                    license_expression=str(detection.get("license_expression", match.get("license_expression", ""))),
                    spdx_id=str(match.get("spdx_license_expression", match.get("spdx_license_key", ""))),
                    score=float(match.get("score", 0.0) or 0.0),
                    matched_text=str(match.get("matched_text", ""))[:500],
                    start_line=int(match.get("start_line", 0) or 0),
                    end_line=int(match.get("end_line", 0) or 0),
                )
            )
    return licenses


def _parse_result(source_id: str, profile_path: str, payload: dict[str, Any]) -> list[ComplianceHit]:
    # 仅保留有合规价值的信息：许可证、版权、扫描错误。
    hits: list[ComplianceHit] = []
    for file_entry in payload.get("files", []):
        if file_entry.get("type") not in {None, "file"}:
            continue
        licenses = _parse_licenses(file_entry)
        copyrights = [
            str(item.get("copyright", ""))
            for item in file_entry.get("copyrights", [])
            if item.get("copyright")
        ]
        scan_errors = [str(err) for err in file_entry.get("scan_errors", [])]
        if not (licenses or copyrights or scan_errors):
            continue
        hits.append(
            ComplianceHit(
                source_id=source_id,
                file_path=str(file_entry.get("path", profile_path)),
                licenses=licenses,
                copyrights=copyrights,
                scan_errors=scan_errors,
            )
        )
    return hits


def run(profiles: list[SourceProfile], settings: Settings) -> list[ComplianceHit]:
    # 对可扫描类型逐个执行 ScanCode，并聚合为统一命中结果。
    hits: list[ComplianceHit] = []
    for profile in profiles:
        if profile.source_type not in _ELIGIBLE_TYPES:
            continue
        if not Path(profile.path).exists():
            logger.warning("Skipping missing source for ScanCode: %s", profile.path)
            continue
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = Path(tmp.name)
        try:
            payload = _run_scancode(settings.scancode_bin, profile.path, str(output_path))
            if payload:
                hits.extend(_parse_result(profile.source_id, profile.path, payload))
        finally:
            # 无论成功失败都清理临时文件。
            output_path.unlink(missing_ok=True)
    logger.info("ScanCode scan complete: %d compliance hits", len(hits))
    return hits
