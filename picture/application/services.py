"""
Picture compliance services: individual processing steps.

Each service function focuses on a single step and returns results.
No side-effects beyond what is documented.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from picture.domain.enums import FindingType, RedactionMode
from picture.domain.models import (
    BBox,
    OCRLayoutResult,
    PictureFinding,
    PictureModerationResult,
    RedactionOperation,
    RegionMask,
)
from picture.providers.base import (
    OCRLayoutProvider,
    PIIDetector,
    Preprocessor,
    Redactor,
    SafetyModerator,
    SegmentationProvider,
    VisionDetector,
)

logger = logging.getLogger(__name__)


def run_preprocess(preprocessor: Preprocessor, image_path: str, output_dir: str) -> str:
    """Run image preprocessing and return the preprocessed path."""
    start = time.monotonic()
    result = preprocessor.preprocess(image_path, output_dir)
    elapsed = (time.monotonic() - start) * 1000
    logger.info("Preprocessing completed in %.1fms", elapsed)
    return result


def run_ocr_layout(provider: OCRLayoutProvider, image_path: str) -> OCRLayoutResult:
    """Run OCR + layout analysis."""
    start = time.monotonic()
    result = provider.analyze(image_path)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "OCR completed in %.1fms: %d text blocks, %d layout regions (engine=%s)",
        elapsed, len(result.text_blocks), len(result.layout_regions), result.engine_name,
    )
    return result


def run_text_pii_detection(
    detector: PIIDetector,
    ocr_result: OCRLayoutResult,
) -> list[PictureFinding]:
    """
    Run PII detection on OCR text and map findings back to image coordinates.

    The key challenge: PII detector works on text, but we need to map the
    detected entities back to the bounding boxes in the image.
    """
    start = time.monotonic()
    findings = detector.detect(ocr_result.full_text)

    # Map text PII findings back to image regions using OCR text blocks
    mapped_findings: list[PictureFinding] = []
    for finding in findings:
        if finding.text_span:
            # Find which OCR text block(s) contain this text span
            region = _map_text_to_region(finding.text_span, ocr_result)
            if region:
                finding.region = region
        mapped_findings.append(finding)

    elapsed = (time.monotonic() - start) * 1000
    logger.info("PII detection completed in %.1fms: %d findings", elapsed, len(mapped_findings))
    return mapped_findings


def _map_text_to_region(text_span: str, ocr_result: OCRLayoutResult) -> RegionMask | None:
    """Map a text span back to image coordinates using OCR text blocks."""
    for block in ocr_result.text_blocks:
        if text_span in block.text:
            return RegionMask(
                bbox=block.bbox,
                confidence=block.confidence,
            )
    return None


def run_safety_moderation(
    moderator: SafetyModerator,
    image_path: str,
) -> PictureModerationResult:
    """Run safety moderation on the image."""
    start = time.monotonic()
    result = moderator.moderate(image_path)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "Safety moderation completed in %.1fms: safe=%s categories=%s",
        elapsed, result.is_safe, [c.value for c in result.categories],
    )
    return result


def run_vision_detection(
    detector: VisionDetector,
    image_path: str,
) -> list[PictureFinding]:
    """Run vision-based object detection."""
    start = time.monotonic()
    findings = detector.detect(image_path)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "Vision detection completed in %.1fms: %d findings",
        elapsed, len(findings),
    )
    return findings


def run_segmentation_refinement(
    provider: SegmentationProvider,
    image_path: str,
    findings: list[PictureFinding],
) -> list[PictureFinding]:
    """Refine finding regions using segmentation model."""
    start = time.monotonic()

    # Collect regions that need refinement
    regions = [f.region for f in findings if f.region is not None]
    if not regions:
        return findings

    refined_regions = provider.refine(image_path, regions)

    # Map refined regions back to findings
    region_idx = 0
    for finding in findings:
        if finding.region is not None and region_idx < len(refined_regions):
            finding.region = refined_regions[region_idx]
            region_idx += 1

    elapsed = (time.monotonic() - start) * 1000
    logger.info("Segmentation refinement completed in %.1fms", elapsed)
    return findings


def build_redaction_operations(
    findings: list[PictureFinding],
    redaction_config: dict[str, str],
) -> list[RedactionOperation]:
    """
    Build redaction operations from findings.

    Maps each finding to a RedactionOperation with the appropriate mode
    based on the finding category and configuration.
    """
    operations: list[RedactionOperation] = []

    for finding in findings:
        if finding.region is None:
            # Cannot redact without a region – skip (but it was still a finding)
            continue

        # Determine redaction mode
        mode_str = redaction_config.get(
            finding.category,
            redaction_config.get("default", "black_box"),
        )
        try:
            mode = RedactionMode(mode_str)
        except ValueError:
            mode = RedactionMode.BLACK_BOX

        operations.append(RedactionOperation(
            finding_id=finding.finding_id,
            region=finding.region,
            mode=mode,
        ))

    logger.info("Built %d redaction operations from %d findings", len(operations), len(findings))
    return operations


def run_redaction(
    redactor: Redactor,
    image_path: str,
    operations: list[RedactionOperation],
    output_path: str,
    overlay_path: str | None = None,
) -> tuple[str, str | None]:
    """
    Execute redaction and optional overlay rendering.

    Returns (compliant_image_path, overlay_image_path).
    """
    start = time.monotonic()

    compliant_path = redactor.redact(image_path, operations, output_path)

    overlay_result = None
    if overlay_path and operations:
        try:
            overlay_result = redactor.render_overlay(image_path, operations, overlay_path)
        except Exception as exc:
            logger.warning("Overlay rendering failed (non-critical): %s", exc)

    elapsed = (time.monotonic() - start) * 1000
    logger.info("Redaction completed in %.1fms: %d operations", elapsed, len(operations))
    return compliant_path, overlay_result


def merge_findings(
    *finding_lists: list[PictureFinding],
    dedup_iou_threshold: float = 0.5,
) -> list[PictureFinding]:
    """
    Merge multiple finding lists, deduplicating overlapping regions.

    Uses IoU (Intersection over Union) to detect duplicates when
    multiple providers detect the same region.
    """
    all_findings: list[PictureFinding] = []
    for flist in finding_lists:
        all_findings.extend(flist)

    if len(all_findings) <= 1:
        return all_findings

    # Simple dedup: remove findings with high IoU overlap
    result: list[PictureFinding] = []
    for finding in all_findings:
        is_dup = False
        for existing in result:
            if finding.region and existing.region:
                iou = _compute_iou(finding.region.bbox, existing.region.bbox)
                if iou > dedup_iou_threshold:
                    # Keep the one with higher score
                    if finding.score > existing.score:
                        result.remove(existing)
                        result.append(finding)
                    is_dup = True
                    break
        if not is_dup:
            result.append(finding)

    logger.info("Merged %d findings into %d (deduped)", len(all_findings), len(result))
    return result


def _compute_iou(a: BBox, b: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
