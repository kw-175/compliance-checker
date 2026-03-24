"""
Step B2a – TruffleHog Secret Scan

Invokes the TruffleHog v3 CLI via subprocess to scan raw source objects for
leaked secrets (API keys, tokens, passwords, etc.).

Output → raw_secret_hits.jsonl
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from text.config.settings import Settings
from text.models.schemas import SecretHit, SourceRecord

logger = logging.getLogger(__name__)


def _run_trufflehog(binary: str, target_path: str) -> list[dict[str, Any]]:
    """
    Run TruffleHog filesystem scan and return parsed JSON results.

    trufflehog filesystem <path> --json --no-update
    Each line of stdout is one JSON finding.
    """
    cmd = [binary, "filesystem", target_path, "--json", "--no-update"]
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        logger.error(
            "TruffleHog binary not found at '%s'.  "
            "Install: https://github.com/trufflesecurity/trufflehog#installation",
            binary,
        )
        return []
    except subprocess.TimeoutExpired:
        logger.error("TruffleHog timed out scanning %s", target_path)
        return []

    findings: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            findings.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Unparseable TruffleHog output line: %s", line[:120])
    if result.returncode not in (0, 1):
        logger.warning("TruffleHog exited with code %d: %s", result.returncode, result.stderr[:500])
    return findings


def _parse_finding(source_id: str, finding: dict[str, Any]) -> SecretHit:
    """Convert a single TruffleHog JSON finding into a SecretHit."""
    source_meta = finding.get("SourceMetadata", {}).get("Data", {})
    filesystem_data = source_meta.get("Filesystem", {})
    return SecretHit(
        source_id=source_id,
        detector_type=finding.get("DetectorType", ""),
        decoder_type=finding.get("DecoderType", ""),
        raw_value=finding.get("Raw", ""),
        redacted=finding.get("Redacted", ""),
        file_path=filesystem_data.get("file", ""),
        line_number=filesystem_data.get("line", 0),
        verified=finding.get("Verified", False),
        extra={
            "detector_name": finding.get("DetectorName", ""),
            "extra_data": finding.get("ExtraData", {}),
        },
    )


def run(
    sources: list[SourceRecord],
    settings: Settings | None = None,
) -> list[SecretHit]:
    """
    Execute TruffleHog scan on all source paths.

    Parameters
    ----------
    sources : list[SourceRecord]
    settings : Settings, optional

    Returns
    -------
    list[SecretHit]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    all_hits: list[SecretHit] = []
    scanned_dirs: set[str] = set()

    for src in sources:
        # Scan parent directory (TruffleHog works best on dirs)
        target = str(Path(src.path).parent)
        if target in scanned_dirs:
            continue
        scanned_dirs.add(target)

        findings = _run_trufflehog(settings.trufflehog_bin, target)
        for f in findings:
            hit = _parse_finding(src.source_id, f)
            all_hits.append(hit)

    logger.info("TruffleHog scan complete: %d secret hits found", len(all_hits))
    return all_hits
