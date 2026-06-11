"""
Step A: source intake.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from audio.models.schemas import SourceRecord

logger = logging.getLogger(__name__)

_BUFFER_SIZE = 65536
_AUDIO_MANIFEST = "audio_manifest.jsonl"
_PACKAGE_METADATA = "metadata.json"
_OPTIONAL_SIDECARS = (
    "quality_report.jsonl",
    "rejected_manifest.jsonl",
    "segments_manifest.jsonl",
    "transcript_segments.jsonl",
    "transcript_segments.json",
    "speaker_segments.jsonl",
    "lineage.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping malformed package metadata: %s", path)
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL row in %s", path)
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        logger.warning("Could not read package JSONL: %s", path)
    return rows


def _resolve_package_path(package_dir: Path, raw_ref: Any) -> Path:
    raw = str(raw_ref or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return package_dir / path


def _rel_or_abs(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _provided_files(package_dir: Path, package_metadata: dict[str, Any]) -> list[str]:
    provided: set[str] = set()
    for item in package_metadata.get("provided_files", []) or []:
        raw = str(item).strip()
        if raw:
            provided.add(raw.replace("\\", "/"))
    for name in (_PACKAGE_METADATA, _AUDIO_MANIFEST, *_OPTIONAL_SIDECARS):
        if (package_dir / name).exists():
            provided.add(name)
    return sorted(provided)


def _sidecar_paths(package_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in _OPTIONAL_SIDECARS:
        candidate = package_dir / name
        if candidate.exists():
            paths[name] = str(candidate.resolve())
    return paths


def _sidecar_metadata_key(name: str) -> str:
    if name.endswith(".jsonl"):
        return f"{name[:-6]}_path"
    if name.endswith(".json"):
        return f"{name[:-5]}_path"
    return f"{name}_path"


def _row_identifiers(row: dict[str, Any]) -> set[str]:
    return {
        str(row.get(key)).strip()
        for key in ("audio_id", "source_id", "lineage_id")
        if str(row.get(key, "")).strip()
    }


def _match_sidecar_row(
    rows: list[dict[str, Any]],
    manifest_row: dict[str, Any],
    clean_path: Path,
) -> dict[str, Any]:
    manifest_ids = _row_identifiers(manifest_row)
    for row in rows:
        if manifest_ids and manifest_ids & _row_identifiers(row):
            return row

        for key in ("clean_audio_path", "normalized_path", "output_ref", "output_path"):
            raw_ref = row.get(key)
            if not raw_ref:
                continue
            candidate = Path(str(raw_ref))
            if candidate.name == clean_path.name:
                return row
            if str(candidate).replace("\\", "/") == str(clean_path).replace("\\", "/"):
                return row
    return {}


def _manifest_audio_path(row: dict[str, Any]) -> str:
    for key in ("clean_audio_path", "normalized_audio_path", "normalized_path", "audio_path", "path"):
        raw = str(row.get(key, "")).strip()
        if raw:
            return raw
    return ""


def _source_id_for(row: dict[str, Any], clean_path: Path, clean_sha256: str) -> str:
    for key in ("audio_id", "source_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    if clean_path.stem:
        return clean_path.stem
    return clean_sha256[:12]


def _parse_cleaned_audio_package(package_dir: Path) -> list[SourceRecord]:
    manifest_path = package_dir / _AUDIO_MANIFEST
    package_metadata = _load_json(package_dir / _PACKAGE_METADATA)
    rows = _load_jsonl(manifest_path)
    provided_files = _provided_files(package_dir, package_metadata)
    sidecar_paths = _sidecar_paths(package_dir)
    quality_rows = _load_jsonl(package_dir / "quality_report.jsonl")
    lineage_rows = _load_jsonl(package_dir / "lineage.jsonl")

    package_level = str(package_metadata.get("package_level") or "").strip().lower()
    if not package_level:
        package_level = "extended" if sidecar_paths else "core"

    records: list[SourceRecord] = []
    for index, row in enumerate(rows):
        clean_ref = _manifest_audio_path(row)
        if not clean_ref:
            logger.warning("Skipping audio manifest row without audio path in %s", manifest_path)
            continue

        clean_path = _resolve_package_path(package_dir, clean_ref)
        if not clean_path.exists() or not clean_path.is_file():
            logger.warning("Skipping missing cleaned audio file: %s", clean_path)
            continue

        try:
            clean_sha256 = _sha256(clean_path)
            source_id = _source_id_for(row, clean_path, clean_sha256)
            quality_record = _match_sidecar_row(quality_rows, row, clean_path)
            lineage_record = _match_sidecar_row(lineage_rows, row, clean_path)
            package_id = str(package_metadata.get("package_id") or f"pkg_{_sha256(manifest_path)[:12]}")
            metadata = {
                "cleaned_audio_package": True,
                "package_dir": str(package_dir.resolve()),
                "package_id": package_id,
                "package_level": package_level,
                "package_contract_version": str(package_metadata.get("package_contract_version", "")),
                "package_metadata": package_metadata,
                "provided_files": provided_files,
                "sidecar_paths": sidecar_paths,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_row_number": index + 1,
                "manifest_record": row,
                "audio_id": str(row.get("audio_id") or source_id),
                "upstream_source_id": str(row.get("source_id") or ""),
                "original_ref": str(row.get("original_ref") or row.get("original_path") or ""),
                "original_sha256": str(row.get("original_sha256") or ""),
                "declared_clean_sha256": str(row.get("clean_sha256") or row.get("sha256") or ""),
                "clean_audio_relpath": _rel_or_abs(clean_path, package_dir),
                "quality_record": quality_record,
                "lineage_record": lineage_record,
                "quality_status": str(row.get("quality_status") or quality_record.get("quality_status") or "unknown"),
                "source_type": "cleaned_audio_package",
            }
            for name, path in sidecar_paths.items():
                metadata[_sidecar_metadata_key(name)] = path

            records.append(
                SourceRecord(
                    source_id=source_id,
                    path=str(clean_path.resolve()),
                    size_bytes=clean_path.stat().st_size,
                    sha256=clean_sha256,
                    mime_type=str(row.get("mime_type") or _detect_mime(clean_path)),
                    metadata=metadata,
                )
            )
        except Exception:
            logger.exception("Failed to register cleaned audio manifest row: %s", clean_ref)

    logger.info(
        "Cleaned audio package intake complete: %d sources from %s (%s)",
        len(records),
        package_dir,
        package_level,
    )
    return records


def _collect_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(file for file in path.rglob("*") if file.is_file())
    logger.warning("Skipping missing input path: %s", input_path)
    return []


def run(input_paths: list[str]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.is_dir() and (path / _AUDIO_MANIFEST).exists():
            records.extend(_parse_cleaned_audio_package(path))
            continue

        for file_path in _collect_files(raw_path):
            try:
                records.append(
                    SourceRecord(
                        path=str(file_path.resolve()),
                        size_bytes=file_path.stat().st_size,
                        sha256=_sha256(file_path),
                        mime_type=_detect_mime(file_path),
                        metadata={"source_type": "raw_path"},
                    )
                )
            except Exception:
                logger.exception("Failed to register source: %s", file_path)
    logger.info("Source intake complete: %d sources", len(records))
    return records
