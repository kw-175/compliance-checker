"""
Picture compliance services: individual processing steps.

Each service function focuses on a single step and returns results.
No side-effects beyond what is documented.
"""
# 中文说明：该文件将编排器中的各个处理步骤拆成了可独立复用的服务函数。
# 这样做的好处是：一方面 orchestrator 更清晰，另一方面测试时也可以单测每一步。
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
    # 中文说明：预处理一般负责旋转矫正、尺寸规整、颜色空间转换等基础工作。
    # 这里统一记录耗时，便于对比不同预处理器或不同图片类型下的性能。
    start = time.monotonic()
    result = preprocessor.preprocess(image_path, output_dir)
    elapsed = (time.monotonic() - start) * 1000
    logger.info("Preprocessing completed in %.1fms", elapsed)
    return result


def run_ocr_layout(provider: OCRLayoutProvider, image_path: str) -> OCRLayoutResult:
    """Run OCR + layout analysis."""
    # 中文说明：OCR 不只提取文字，还可能同时输出文本块位置和版面区域，
    # 后续 PII 映射与版面理解都依赖这一步的结构化结果。
    start = time.monotonic()
    result = provider.analyze(image_path)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "OCR completed in %.1fms: %d text blocks, %d layout regions (engine=%s)",
        elapsed,
        len(result.text_blocks),
        len(result.layout_regions),
        result.engine_name,
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
    # 中文说明：PII 检测器输入的是纯文本，而 picture 模块最终需要的是图像区域。
    # 因此这里先检测文本中的实体，再尝试把命中的文本片段映射回 OCR block 对应的框。
    start = time.monotonic()
    findings = detector.detect(ocr_result.full_text)

    # 中文说明：mapped_findings 保留原始 finding，但尽量为其补上 region。
    mapped_findings: list[PictureFinding] = []
    for finding in findings:
        if finding.text_span:
            # 中文说明：只对带 text_span 的实体尝试回映射；
            # 没有 text_span 的结果通常无法可靠定位到图片上的具体区域。
            region = _map_text_to_region(finding.text_span, ocr_result)
            if region:
                finding.region = region
        mapped_findings.append(finding)

    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "PII detection completed in %.1fms: %d findings",
        elapsed,
        len(mapped_findings),
    )
    return mapped_findings


def _map_text_to_region(text_span: str, ocr_result: OCRLayoutResult) -> RegionMask | None:
    """Map a text span back to image coordinates using OCR text blocks."""
    # 中文说明：这里采用的是最简单的包含匹配策略：
    # 只要 text_span 出现在某个 OCR block 的文本里，就把该 block 的 bbox 作为实体区域。
    # 这种实现成本低，但对跨 block 文本、重复文本、多次出现文本的情况会比较粗糙。
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
    # 中文说明：安全审核通常识别色情、暴力、仇恨、违法等高风险内容，
    # 它对最终策略是否直接 DROP 往往有决定性影响。
    start = time.monotonic()
    result = moderator.moderate(image_path)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "Safety moderation completed in %.1fms: safe=%s categories=%s",
        elapsed,
        result.is_safe,
        [c.value for c in result.categories],
    )
    return result


def run_vision_detection(
    detector: VisionDetector,
    image_path: str,
) -> list[PictureFinding]:
    """Run vision-based object detection."""
    # 中文说明：视觉检测负责发现人脸、二维码、印章、工牌、车牌等结构化目标，
    # 是自然图和混合截图链路中的主能力之一。
    start = time.monotonic()
    findings = detector.detect(image_path)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "Vision detection completed in %.1fms: %d findings",
        elapsed,
        len(findings),
    )
    return findings


def run_segmentation_refinement(
    provider: SegmentationProvider,
    image_path: str,
    findings: list[PictureFinding],
) -> list[PictureFinding]:
    """Refine finding regions using segmentation model."""
    start = time.monotonic()

    # 中文说明：只有已经有 region 的 finding 才有必要进入分割细化。
    # 完全没有空间信息的 finding 无法从分割模型中获益。
    regions = [f.region for f in findings if f.region is not None]
    if not regions:
        return findings

    # 中文说明：分割模型接收原图和粗粒度区域，返回更精确的区域结果。
    refined_regions = provider.refine(image_path, regions)

    # 中文说明：refined_regions 与输入 regions 的顺序一一对应，
    # 因此这里按顺序写回到原 findings 中。
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
            # 中文说明：没有 region 就无法执行视觉层面的脱敏，
            # 这类 finding 仍然会留在报告中，但不会生成 redaction operation。
            continue

        # 中文说明：优先按类别查找脱敏模式，找不到时退回 default。
        mode_str = redaction_config.get(
            finding.category,
            redaction_config.get("default", "black_box"),
        )
        try:
            mode = RedactionMode(mode_str)
        except ValueError:
            # 中文说明：如果配置里写了无效模式，不让流程失败，
            # 而是保底退回最稳妥的黑框模式。
            mode = RedactionMode.BLACK_BOX

        operations.append(
            RedactionOperation(
                finding_id=finding.finding_id,
                region=finding.region,
                mode=mode,
            )
        )

    logger.info(
        "Built %d redaction operations from %d findings",
        len(operations),
        len(findings),
    )
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

    # 中文说明：主脱敏输出是必做步骤，overlay 只是辅助审计材料。
    compliant_path = redactor.redact(image_path, operations, output_path)

    overlay_result = None
    if overlay_path and operations:
        try:
            # 中文说明：overlay 一般用于调试和审计，
            # 即使绘制失败也不应影响主脱敏结果交付。
            overlay_result = redactor.render_overlay(
                image_path, operations, overlay_path
            )
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
    # 中文说明：先把多个来源的 finding 平铺成一个列表。
    all_findings: list[PictureFinding] = []
    for flist in finding_lists:
        all_findings.extend(flist)

    if len(all_findings) <= 1:
        return all_findings

    # 中文说明：这里的去重逻辑比较轻量，按 bbox 的 IoU 判断是否重复，
    # 适合作为 provider 聚合时的基础去重策略。
    result: list[PictureFinding] = []
    for finding in all_findings:
        is_dup = False
        for existing in result:
            if finding.region and existing.region:
                iou = _compute_iou(finding.region.bbox, existing.region.bbox)
                if iou > dedup_iou_threshold:
                    # 中文说明：同一位置如果重复命中，保留 score 更高的那个结果。
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
    # 中文说明：先求两个框的相交矩形。
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)

    # 中文说明：没有重叠时 IoU 直接为 0。
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - intersection

    # 中文说明：union 为 0 理论上不常见，但这里仍做保护，避免除零异常。
    return intersection / union if union > 0 else 0.0
