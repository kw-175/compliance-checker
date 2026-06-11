"""
Picture compliance services: individual processing steps.

Each service function focuses on a single step and returns results.
No side-effects beyond what is documented.
"""
# 中文说明：该文件将编排器中的各个处理步骤拆成了可独立复用的服务函数。
# 这样做的好处是：一方面 orchestrator 更清晰，另一方面测试时也可以单测每一步。
from __future__ import annotations

import logging
import re
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

_TEXT_CONTENT_PATTERNS: list[tuple[str, str, float, str, re.Pattern[str]]] = [
    ("violence", "OCR_TEXT_VIOLENCE", 0.92, "violent_or_dangerous_text", re.compile(r"(炸弹|恐怖|杀人|爆炸|bomb|terror|kill|shoot)", re.IGNORECASE)),
    ("sexual_content", "OCR_TEXT_SEXUAL", 0.92, "sexual_or_explicit_text", re.compile(r"(色情|成人视频|裸聊|porn|explicit sex|sexual service)", re.IGNORECASE)),
    ("self_harm", "OCR_TEXT_SELF_HARM", 0.92, "self_harm_or_suicide_text", re.compile(r"(自杀|自残|结束生命|suicide|self harm|end my life)", re.IGNORECASE)),
    ("hate_speech", "OCR_TEXT_HATE", 0.88, "hate_or_discrimination_text", re.compile(r"(灭绝|种族仇恨|racial hatred|genocide|仇恨)", re.IGNORECASE)),
    (
        "illegal_instruction",
        "OCR_TEXT_ILLEGAL",
        0.90,
        "illegal_or_dangerous_instruction_text",
        re.compile(
            r"((制作|合成|获取|绕过|破解|仿制).{0,12}(炸弹|爆炸物|枪支|毒品|身份证|银行卡|考试答案)|绕过安全|bypass safety|jailbreak|忽略之前的指令)",
            re.IGNORECASE,
        ),
    ),
]


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
    precomputed = ocr_result.metadata.get("precomputed_pii_findings")
    if isinstance(precomputed, list) and all(
        isinstance(item, PictureFinding) for item in precomputed
    ):
        logger.info(
            "PII detection reused %d precomputed findings from OCR metadata",
            len(precomputed),
        )
        return precomputed

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


def run_text_content_detection(ocr_result: OCRLayoutResult) -> list[PictureFinding]:
    """Detect unsafe OCR text using lightweight pattern matching."""
    text = (ocr_result.full_text or "").strip()
    if not text:
        return []

    findings: list[PictureFinding] = []
    for category, reason_code, score, explanation, pattern in _TEXT_CONTENT_PATTERNS:
        for match in pattern.finditer(text):
            region = _map_text_to_region(match.group(), ocr_result)
            findings.append(
                PictureFinding(
                    finding_type=FindingType.TEXT_CONTENT,
                    category=category,
                    label=f"OCR text content: {category}",
                    score=score,
                    region=region,
                    text_span=match.group(),
                    reason_code=reason_code,
                    provider="TextContentHeuristic",
                    explanation=explanation,
                    metadata={
                        "char_start": match.start(),
                        "char_end": match.end(),
                    },
                )
            )
    logger.info("Text content detection completed: %d findings", len(findings))
    return findings


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
    target_types: list[str] | None = None,
) -> list[PictureFinding]:
    """Run vision-based object detection."""
    # 中文说明：视觉检测负责发现人脸、二维码、印章、工牌、车牌等结构化目标，
    # 是自然图和混合截图链路中的主能力之一。
    start = time.monotonic()
    try:
        findings = detector.detect(image_path, target_types=target_types)  # type: ignore[call-arg]
    except TypeError:
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

        for planned_region, planned_mode, metadata in _plan_minimal_redaction(
            finding,
            redaction_config,
        ):
            operations.append(
                RedactionOperation(
                    finding_id=finding.finding_id,
                    region=planned_region,
                    mode=planned_mode,
                    metadata=metadata,
                )
            )

    logger.info(
        "Built %d redaction operations from %d findings",
        len(operations),
        len(findings),
    )
    return operations


def _plan_minimal_redaction(
    finding: PictureFinding,
    redaction_config: dict[str, str],
) -> list[tuple[RegionMask, RedactionMode, dict[str, Any]]]:
    category = str(finding.category or "").lower()
    bbox = finding.region.bbox
    original = {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h}

    def item(
        strategy: str,
        region: RegionMask,
        mode: RedactionMode,
        semantic_preserved: bool = True,
    ) -> tuple[RegionMask, RedactionMode, dict[str, Any]]:
        return (
            region,
            mode,
            {
                "source_category": category,
                "original_bbox": original,
                "redaction_bbox": {
                    "x": region.bbox.x,
                    "y": region.bbox.y,
                    "w": region.bbox.w,
                    "h": region.bbox.h,
                },
                "minimization_strategy": strategy,
                "semantic_preservation": semantic_preserved,
                "identity_reidentification_risk": "low",
            },
        )

    if finding.finding_type in {FindingType.TEXT_PII, FindingType.TEXT_CONTENT}:
        return [
            item(
                "ocr_text_region_redaction",
                finding.region,
                _mode_from_config(category, redaction_config),
                semantic_preserved=False,
            )
        ]

    if category == "face":
        # 中文说明：人脸保留头部轮廓和姿态，只破坏眼鼻等身份识别核心区域。
        return [
            item(
                "face_identity_core_blur",
                _scale_region(finding.region, 0.18, 0.22, 0.64, 0.38),
                RedactionMode.GAUSSIAN_BLUR,
            )
        ]

    if category in {"qr_code", "barcode"}:
        # 中文说明：码类包含可跳转或可编码信息，命中后对完整局部区域做确定性脱敏。
        if category == "qr_code":
            region = _pad_region(finding.region, 0.05, 0.05)
            strategy = "qr_full_region_redaction"
        else:
            region = _pad_region(finding.region, 0.08, 0.04)
            strategy = "barcode_full_region_redaction"
        return [item(strategy, region, _mode_from_config(category, redaction_config), semantic_preserved=False)]

    if category == "license_plate":
        return [
            item(
                "license_plate_number_band",
                _scale_region(finding.region, 0.12, 0.22, 0.76, 0.56),
                RedactionMode.BLACK_BOX,
            )
        ]

    if category == "badge":
        return [
            item(
                "badge_text_core_pixelate",
                _scale_region(finding.region, 0.08, 0.25, 0.84, 0.55),
                RedactionMode.PIXELATE,
            )
        ]

    if category in {"id_card", "student_id_card"}:
        # 中文说明：证件保留卡片轮廓，遮挡主要文字/号码区域。
        return [
            item(
                "id_card_text_band",
                _scale_region(finding.region, 0.08, 0.28, 0.84, 0.48),
                RedactionMode.BLACK_BOX,
            )
        ]

    if category == "stamp":
        return [
            item(
                "stamp_center_text_blur",
                _scale_region(finding.region, 0.18, 0.18, 0.64, 0.64),
                RedactionMode.GAUSSIAN_BLUR,
            )
        ]

    if category == "signature":
        return [
            item(
                "signature_stroke_pixelate",
                _scale_region(finding.region, 0.04, 0.12, 0.92, 0.76),
                RedactionMode.PIXELATE,
            )
        ]

    if category in {"avatar", "account_region", "school_class_identifier"}:
        return [
            item(
                f"{category}_semantic_preserving_pixelate",
                _scale_region(finding.region, 0.08, 0.12, 0.84, 0.76),
                RedactionMode.PIXELATE,
            )
        ]

    return [
        item(
            "configured_full_region_redaction",
            finding.region,
            _mode_from_config(category, redaction_config),
            semantic_preserved=False,
        )
    ]


def _scale_region(
    region: RegionMask,
    rel_x: float,
    rel_y: float,
    rel_w: float,
    rel_h: float,
) -> RegionMask:
    bbox = region.bbox
    w = max(1.0, bbox.w * rel_w)
    h = max(1.0, bbox.h * rel_h)
    return region.model_copy(
        update={
            "bbox": BBox(
                x=bbox.x + bbox.w * rel_x,
                y=bbox.y + bbox.h * rel_y,
                w=w,
                h=h,
            ),
            "polygon": None,
            "mask_path": None,
        }
    )


def _pad_region(region: RegionMask, pad_x_ratio: float, pad_y_ratio: float) -> RegionMask:
    bbox = region.bbox
    pad_x = max(4.0, bbox.w * pad_x_ratio)
    pad_y = max(4.0, bbox.h * pad_y_ratio)
    polygon = region.polygon
    if polygon is not None and polygon.points:
        cx = sum(point[0] for point in polygon.points) / len(polygon.points)
        cy = sum(point[1] for point in polygon.points) / len(polygon.points)
        scale_x = (bbox.w + 2.0 * pad_x) / max(1.0, bbox.w)
        scale_y = (bbox.h + 2.0 * pad_y) / max(1.0, bbox.h)
        polygon = polygon.model_copy(
            update={
                "points": [
                    (
                        cx + (float(point[0]) - cx) * scale_x,
                        cy + (float(point[1]) - cy) * scale_y,
                    )
                    for point in polygon.points
                ]
            }
        )
    return region.model_copy(
        update={
            "bbox": BBox(
                x=bbox.x - pad_x,
                y=bbox.y - pad_y,
                w=bbox.w + 2.0 * pad_x,
                h=bbox.h + 2.0 * pad_y,
            ),
            "polygon": polygon,
        }
    )


def _mode_from_config(category: str, redaction_config: dict[str, str]) -> RedactionMode:
    mode_str = redaction_config.get(category, redaction_config.get("default", "black_box"))
    try:
        return RedactionMode(mode_str)
    except ValueError:
        return RedactionMode.BLACK_BOX


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
            if not _same_finding_semantics(finding, existing):
                continue
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


def suppress_code_adjacent_text_findings(
    pii_findings: list[PictureFinding],
    vision_findings: list[PictureFinding],
) -> list[PictureFinding]:
    """Drop OCR PII findings that are human-readable text attached to QR/barcode regions."""
    code_findings = [
        finding
        for finding in vision_findings
        if finding.finding_type == FindingType.VISION_OBJECT
        and str(finding.category or "").lower() in {"qr_code", "barcode"}
        and finding.region is not None
    ]
    if not code_findings or not pii_findings:
        return pii_findings

    filtered: list[PictureFinding] = []
    for finding in pii_findings:
        if not _code_text_suppressible(finding):
            filtered.append(finding)
            continue
        matched_code = next((code for code in code_findings if _text_attached_to_code(finding, code)), None)
        if matched_code is None:
            filtered.append(finding)
            continue
        logger.info(
            "Suppress OCR PII finding attached to visual code region: pii=%s code=%s",
            finding.category,
            matched_code.category,
        )
    return filtered


def _same_finding_semantics(a: PictureFinding, b: PictureFinding) -> bool:
    """Only dedupe duplicate detections of the same risk, not overlapping different risks."""
    a_violation_id = str((a.metadata or {}).get("violation_id") or "")
    b_violation_id = str((b.metadata or {}).get("violation_id") or "")
    if a_violation_id and b_violation_id and a_violation_id != b_violation_id:
        return False
    return (
        a.finding_type == b.finding_type
        and a.category == b.category
        and (a.reason_code or "") == (b.reason_code or "")
    )


def _code_text_suppressible(finding: PictureFinding) -> bool:
    if finding.finding_type != FindingType.TEXT_PII or finding.region is None:
        return False
    category = str(finding.category or "").lower()
    if category not in {
        "phone_number",
        "bank_card",
        "student_id",
        "account",
        "social_account",
        "bank_account",
        "pii_entity",
        "combined_identity",
    }:
        return False
    text = str(finding.text_span or finding.label or "").strip()
    if not text:
        text = str((finding.metadata or {}).get("text") or (finding.metadata or {}).get("raw_text") or "").strip()
    compact = re.sub(r"[\s\-_.]", "", text)
    if len(compact) < 6:
        return False
    alnum_count = len(re.findall(r"[A-Za-z0-9]", compact))
    digit_count = len(re.findall(r"\d", compact))
    return alnum_count >= 6 and digit_count / max(1, alnum_count) >= 0.65


def suppress_textual_visual_findings(
    vision_findings: list[PictureFinding],
    pii_findings: list[PictureFinding],
) -> list[PictureFinding]:
    """Prefer OCR text regions over SAM-style localization for textual PII objects."""
    if not vision_findings or not pii_findings:
        return vision_findings
    ocr_categories = {
        str(finding.category or "").lower()
        for finding in pii_findings
        if finding.finding_type == FindingType.TEXT_PII and finding.region is not None
    }
    if not ocr_categories:
        return vision_findings

    filtered: list[PictureFinding] = []
    for finding in vision_findings:
        if finding.finding_type != FindingType.VISION_OBJECT or not _visual_finding_is_textual_duplicate(finding, ocr_categories):
            filtered.append(finding)
            continue
        logger.info(
            "Suppress textual visual finding because OCR PII provides a stronger region: category=%s metadata=%s",
            finding.category,
            finding.metadata,
        )
    return filtered


def _visual_finding_is_textual_duplicate(finding: PictureFinding, ocr_categories: set[str]) -> bool:
    category = str(finding.category or "").lower()
    metadata = finding.metadata or {}
    haystack = " ".join(
        str(value or "")
        for value in (
            category,
            finding.label,
            finding.explanation,
            metadata.get("qwen_object_name_zh"),
            metadata.get("qwen_description"),
            metadata.get("qwen_sam3_prompt"),
            metadata.get("prompt"),
        )
    ).lower()
    text_like_categories = {"account_region", "screen_text", "text_region"}
    if category not in text_like_categories and not any(token in haystack for token in ("phone", "email", "姓名", "名字", "手机", "电话", "邮箱")):
        return False
    hints = {
        "phone_number": ("phone", "mobile", "手机", "电话", "号码", "联系方式"),
        "email": ("email", "mail", "邮箱"),
        "person_name": ("person", "name", "姓名", "名字", "人名"),
    }
    for pii_category, tokens in hints.items():
        if pii_category in ocr_categories and any(token in haystack for token in tokens):
            return True
    return False


def _text_attached_to_code(text_finding: PictureFinding, code_finding: PictureFinding) -> bool:
    if text_finding.region is None or code_finding.region is None:
        return False
    text_box = text_finding.region.bbox
    code_box = code_finding.region.bbox
    horizontal_overlap = _horizontal_overlap_ratio(text_box, code_box)
    if horizontal_overlap < 0.45:
        return False
    if _bbox_contained(text_box, code_box, expand_x=0.15, expand_y=0.25):
        return True
    text_top = text_box.y
    text_bottom = text_box.y + text_box.h
    code_top = code_box.y
    code_bottom = code_box.y + code_box.h
    near_below = text_top >= code_top and text_top <= code_bottom + max(8.0, code_box.h * 0.55)
    near_above = text_bottom <= code_bottom and text_bottom >= code_top - max(6.0, code_box.h * 0.25)
    return near_below or near_above


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    smaller = min(max(1.0, a.w), max(1.0, b.w))
    return overlap / smaller


def _bbox_contained(inner: BBox, outer: BBox, expand_x: float = 0.0, expand_y: float = 0.0) -> bool:
    pad_x = outer.w * expand_x
    pad_y = outer.h * expand_y
    return (
        inner.x >= outer.x - pad_x
        and inner.y >= outer.y - pad_y
        and inner.x + inner.w <= outer.x + outer.w + pad_x
        and inner.y + inner.h <= outer.y + outer.h + pad_y
    )


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
