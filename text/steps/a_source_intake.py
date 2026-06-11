from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from text.models.schemas import IngestUnit, PackageAsset

logger = logging.getLogger(__name__)

TEXT_FIELD_CANDIDATES = (
    "cleaned_text",
    "text",
    "content",
    "body",
    "document_text",
    "normalized_text",
    "payload_text",
)
DOC_ID_CANDIDATES = ("doc_id", "id", "record_id", "sample_id", "uid")
PACKAGE_METADATA_KEYS = ("task_id", "tenant_id", "profile_id", "source_type", "file_hash")


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65_536):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_language(text: str) -> str:
    if not text:
        return "unknown"
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if chinese and chinese >= latin:
        return "zh"
    if latin:
        return "en"
    return "unknown"


def _candidate_profiles(record: dict[str, Any], package_metadata: dict[str, Any]) -> list[str]:
    values = record.get("candidate_profiles") or package_metadata.get("candidate_profiles") or []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [str(item) for item in values if str(item).strip()]
    return []


def _normalize_doc_id(record: dict[str, Any], text: str, source_path: Path, line_no: int = 0) -> str:
    for key in DOC_ID_CANDIDATES:
        value = record.get(key)
        if value:
            return str(value)
    suffix = _sha256_text(f"{source_path}:{line_no}:{text}")[:12]
    return f"doc_{suffix}"


def _extract_text(record: dict[str, Any]) -> tuple[str, str] | None:
    for key in TEXT_FIELD_CANDIDATES:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    payload = record.get("payload")
    if isinstance(payload, dict):
        return _extract_text(payload)
    document = record.get("document")
    if isinstance(document, dict):
        return _extract_text(document)
    return None


def _package_assets(paths: list[Path], asset_type: str, role: str) -> list[PackageAsset]:
    assets: list[PackageAsset] = []
    for path in sorted({p.resolve() for p in paths}):
        metadata: dict[str, Any] = {}
        if path.exists() and path.is_file():
            metadata["sha256"] = _sha256_path(path)
        assets.append(
            PackageAsset(
                asset_type=asset_type,
                uri=str(path),
                role=role,
                metadata=metadata,
            )
        )
    return assets


def _load_json_file(path: Path) -> Any:
    return json.loads(_read_text(path))


def _load_package_metadata(paths: list[Path]) -> tuple[dict[str, Any], list[PackageAsset]]:
    metadata: dict[str, Any] = {}
    metadata_refs: list[PackageAsset] = []

    for path in paths:
        metadata_refs.append(
            PackageAsset(
                asset_type="metadata",
                uri=str(path),
                role="package_metadata",
                metadata={"sha256": _sha256_path(path)},
            )
        )
        if path.suffix.lower() == ".json":
            payload = _load_json_file(path)
            if isinstance(payload, dict):
                metadata.update(payload)
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        metadata.update(payload)
    return metadata, metadata_refs


def _build_ingest_unit(
    *,
    run_id: str,
    package_id: str,
    package_kind: str,
    parser_name: str,
    source_path: Path,
    text: str,
    doc_id: str,
    record: dict[str, Any],
    package_metadata: dict[str, Any],
    raw_text_refs: list[PackageAsset],
    cleaned_data_refs: list[PackageAsset],
    metadata_refs: list[PackageAsset],
) -> IngestUnit:
    merged_metadata = dict(package_metadata)
    record_metadata = record.get("metadata")
    if isinstance(record_metadata, dict):
        merged_metadata.update(record_metadata)
    for key in PACKAGE_METADATA_KEYS:
        if key in record and record[key] not in (None, ""):
            merged_metadata[key] = record[key]

    extensions = {
        key: value
        for key, value in record.items()
        if key not in TEXT_FIELD_CANDIDATES and key not in DOC_ID_CANDIDATES and key != "metadata"
    }

    return IngestUnit(
        run_id=run_id,
        package_id=package_id,
        doc_id=doc_id,
        source_path=str(source_path),
        text=text,
        text_hash=_sha256_text(text),
        language=_infer_language(text),
        source_type=str(merged_metadata.get("source_type", "")),
        task_id=str(merged_metadata.get("task_id", "")),
        tenant_id=str(merged_metadata.get("tenant_id", "")),
        profile_id=str(merged_metadata.get("profile_id", "")),
        file_hash=str(merged_metadata.get("file_hash", "")) or _sha256_text(text),
        package_kind=package_kind,
        parser_name=parser_name,
        candidate_profiles=_candidate_profiles(record, merged_metadata),
        raw_text_refs=raw_text_refs,
        cleaned_data_refs=cleaned_data_refs,
        metadata_refs=metadata_refs,
        metadata=merged_metadata,
        extensions=extensions,
    )


class PackageParser(Protocol):
    name: str

    def can_handle(self, path: Path) -> bool:
        ...

    def parse(self, path: Path, run_id: str) -> list[IngestUnit]:
        ...


class JsonlPackageParser:
    name = "jsonl_package"

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".jsonl"

    def parse(self, path: Path, run_id: str) -> list[IngestUnit]:
        package_id = f"pkg_{_sha256_text(str(path.resolve()))[:12]}"
        cleaned_data_refs = _package_assets([path], "cleaned_data", "primary_jsonl")
        units: list[IngestUnit] = []

        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON line %s in %s", line_no, path)
                    continue
                if not isinstance(payload, dict):
                    continue
                extracted = _extract_text(payload)
                if extracted is None:
                    continue
                text, _ = extracted
                doc_id = _normalize_doc_id(payload, text, path, line_no)
                units.append(
                    _build_ingest_unit(
                        run_id=run_id,
                        package_id=package_id,
                        package_kind="jsonl",
                        parser_name=self.name,
                        source_path=path,
                        text=text,
                        doc_id=doc_id,
                        record=payload,
                        package_metadata={},
                        raw_text_refs=[],
                        cleaned_data_refs=cleaned_data_refs,
                        metadata_refs=[],
                    )
                )
        return units


class JsonPackageParser:
    name = "json_package"

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".json"

    def parse(self, path: Path, run_id: str) -> list[IngestUnit]:
        payload = _load_json_file(path)
        records: list[dict[str, Any]] = []

        if isinstance(payload, dict):
            if isinstance(payload.get("documents"), list):
                records.extend(item for item in payload["documents"] if isinstance(item, dict))
            elif isinstance(payload.get("records"), list):
                records.extend(item for item in payload["records"] if isinstance(item, dict))
            elif _extract_text(payload):
                records.append(payload)
        elif isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))

        package_id = f"pkg_{_sha256_text(str(path.resolve()))[:12]}"
        cleaned_data_refs = _package_assets([path], "cleaned_data", "primary_json")
        units: list[IngestUnit] = []
        for index, record in enumerate(records, start=1):
            extracted = _extract_text(record)
            if extracted is None:
                continue
            text, _ = extracted
            doc_id = _normalize_doc_id(record, text, path, index)
            units.append(
                _build_ingest_unit(
                    run_id=run_id,
                    package_id=package_id,
                    package_kind="json",
                    parser_name=self.name,
                    source_path=path,
                    text=text,
                    doc_id=doc_id,
                    record=record,
                    package_metadata=payload if isinstance(payload, dict) else {},
                    raw_text_refs=[],
                    cleaned_data_refs=cleaned_data_refs,
                    metadata_refs=[],
                )
            )
        return units


class TextFilePackageParser:
    name = "text_file_package"

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {".txt", ".text", ".md"}

    def parse(self, path: Path, run_id: str) -> list[IngestUnit]:
        text = _read_text(path).strip()
        if not text:
            return []
        package_id = f"pkg_{_sha256_text(str(path.resolve()))[:12]}"
        raw_text_refs = _package_assets([path], "raw_text", "primary_text")
        record = {"doc_id": path.stem, "text": text}
        return [
            _build_ingest_unit(
                run_id=run_id,
                package_id=package_id,
                package_kind="text_file",
                parser_name=self.name,
                source_path=path,
                text=text,
                doc_id=path.stem,
                record=record,
                package_metadata={},
                raw_text_refs=raw_text_refs,
                cleaned_data_refs=[],
                metadata_refs=[],
            )
        ]


class DirectoryPackageParser:
    name = "directory_package"

    def can_handle(self, path: Path) -> bool:
        return path.is_dir()

    def parse(self, path: Path, run_id: str) -> list[IngestUnit]:
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        metadata_files = [
            candidate
            for candidate in files
            if candidate.name.lower().startswith("metadata")
            or candidate.name.lower().startswith("manifest")
        ]
        cleaned_files = [
            candidate
            for candidate in files
            if candidate.suffix.lower() in {".jsonl", ".json"} and candidate not in metadata_files
        ]
        raw_text_files = [
            candidate
            for candidate in files
            if candidate.suffix.lower() in {".txt", ".text", ".md"}
        ]

        package_metadata, metadata_refs = _load_package_metadata(metadata_files)
        raw_text_refs = _package_assets(raw_text_files, "raw_text", "package_raw_text")
        cleaned_data_refs = _package_assets(cleaned_files, "cleaned_data", "package_cleaned_data")
        package_id = f"pkg_{_sha256_text(str(path.resolve()))[:12]}"

        units: list[IngestUnit] = []
        for cleaned_path in cleaned_files:
            parser = JsonlPackageParser() if cleaned_path.suffix.lower() == ".jsonl" else JsonPackageParser()
            for unit in parser.parse(cleaned_path, run_id):
                merged_metadata = {**package_metadata, **unit.metadata}
                units.append(
                    unit.model_copy(
                        update={
                            "package_id": package_id,
                            "package_kind": "directory",
                            "parser_name": self.name,
                            "raw_text_refs": raw_text_refs,
                            "cleaned_data_refs": cleaned_data_refs,
                            "metadata_refs": metadata_refs,
                            "metadata": merged_metadata,
                            "task_id": str(merged_metadata.get("task_id", unit.task_id)),
                            "tenant_id": str(merged_metadata.get("tenant_id", unit.tenant_id)),
                            "profile_id": str(merged_metadata.get("profile_id", unit.profile_id)),
                            "source_type": str(merged_metadata.get("source_type", unit.source_type)),
                            "file_hash": str(merged_metadata.get("file_hash", unit.file_hash)),
                        }
                    )
                )

        if units:
            return units

        for raw_text in raw_text_files:
            text = _read_text(raw_text).strip()
            if not text:
                continue
            record = {"doc_id": raw_text.stem, "text": text}
            units.append(
                _build_ingest_unit(
                    run_id=run_id,
                    package_id=package_id,
                    package_kind="directory",
                    parser_name=self.name,
                    source_path=raw_text,
                    text=text,
                    doc_id=raw_text.stem,
                    record=record,
                    package_metadata=package_metadata,
                    raw_text_refs=raw_text_refs,
                    cleaned_data_refs=cleaned_data_refs,
                    metadata_refs=metadata_refs,
                )
            )
        return units


DEFAULT_PARSERS: tuple[PackageParser, ...] = (
    DirectoryPackageParser(),
    JsonlPackageParser(),
    JsonPackageParser(),
    TextFilePackageParser(),
)


def run(package_paths: list[str], run_id: str = "") -> list[IngestUnit]:
    units: list[IngestUnit] = []
    for raw_path in package_paths:
        path = Path(raw_path)
        parser = next((candidate for candidate in DEFAULT_PARSERS if candidate.can_handle(path)), None)
        if parser is None:
            logger.warning("No cleaned-package parser matched %s", raw_path)
            continue
        parsed_units = parser.parse(path, run_id)
        if not parsed_units:
            logger.warning("Parser %s found no ingest units in %s", parser.name, raw_path)
        units.extend(parsed_units)

    logger.info("Cleaned package intake completed: %d ingest units", len(units))
    return units
