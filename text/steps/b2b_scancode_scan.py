"""
Step B2b – ScanCode License / Copyright Scan

Runs ScanCode-toolkit on eligible sources (code, repo, package, binary, mixed)
to detect open-source licenses and copyright statements.

Output → source_compliance.jsonl
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from text.config.settings import Settings
from text.models.schemas import (
    ComplianceHit,
    LicenseMatch,
    SourceProfile,
    SourceType,
)

logger = logging.getLogger(__name__)

_ELIGIBLE_TYPES = {
    SourceType.CODE,
    SourceType.REPO,
    SourceType.PACKAGE,
    SourceType.BINARY,
    SourceType.MIXED,
}


def _run_scancode(binary: str, target_path: str, output_file: str) -> dict[str, Any]:
    """Run scancode CLI and return the parsed JSON output."""
    cmd = [
        binary,
        "--license",
        "--copyright",
        "--info",
        "--json-pp", output_file,
        target_path,
        "--timeout", "120",
    ]
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        logger.error(
            "ScanCode binary not found at '%s'. "
            "Install: pip install scancode-toolkit",
            binary,
        )
        return {}
    except subprocess.TimeoutExpired:
        logger.error("ScanCode timed out scanning %s", target_path)
        return {}

    if result.returncode != 0:
        logger.warning(
            "ScanCode exited with code %d: %s", result.returncode, result.stderr[:500]
        )

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Failed to read ScanCode output: %s", e)
        return {}


def _parse_scancode_result(
    source_id: str,
    scan_result: dict[str, Any],
) -> list[ComplianceHit]:
    """Parse ScanCode JSON output into ComplianceHit models."""
    hits: list[ComplianceHit] = []
    for file_entry in scan_result.get("files", []):
        if file_entry.get("type") != "file":
            continue

        licenses: list[LicenseMatch] = []
        for lic in file_entry.get("license_detections", []):
            for match in lic.get("matches", []):
                licenses.append(
                    LicenseMatch(
                        license_expression=lic.get("license_expression", ""),
                        spdx_id=match.get("spdx_license_expression", ""),
                        score=match.get("score", 0.0),
                        matched_text=match.get("matched_text", "")[:500],
                        start_line=match.get("start_line", 0),
                        end_line=match.get("end_line", 0),
                    )
                )

        copyrights = [
            c.get("copyright", "")
            for c in file_entry.get("copyrights", [])
            if c.get("copyright")
        ]

        scan_errors = file_entry.get("scan_errors", [])

        if licenses or copyrights or scan_errors:
            hits.append(
                ComplianceHit(
                    source_id=source_id,
                    file_path=file_entry.get("path", ""),
                    licenses=licenses,
                    copyrights=copyrights,
                    scan_errors=scan_errors,
                )
            )
    return hits


def run(
    profiles: list[SourceProfile],
    settings: Settings | None = None,
) -> list[ComplianceHit]:
    """
    Execute ScanCode on eligible source profiles.

    Parameters
    ----------
    profiles : list[SourceProfile]
        Source profiles from step B1.
    settings : Settings, optional

    Returns
    -------
    list[ComplianceHit]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    eligible = [p for p in profiles if p.source_type in _ELIGIBLE_TYPES]
    logger.info(
        "ScanCode: %d/%d sources eligible for scanning",
        len(eligible), len(profiles),
    )

    all_hits: list[ComplianceHit] = []
    for profile in eligible:
        target = profile.path
        if not Path(target).exists():
            logger.warning("Skipping non-existent path: %s", target)
            continue

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = tmp.name

        scan_result = _run_scancode(settings.scancode_bin, target, output_file)
        if scan_result:
            hits = _parse_scancode_result(profile.source_id, scan_result)
            all_hits.extend(hits)

        # Clean up temp file
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass

    logger.info("ScanCode scan complete: %d compliance hits", len(all_hits))
    return all_hits
