"""Tests for picture service-level helpers."""
from __future__ import annotations

from types import SimpleNamespace

import picture.providers.ocr.paddleocr_vl as paddleocr_vl
import picture.providers.safety.qwen35_vl as qwen35_vl
from picture.application.services import build_redaction_operations, merge_findings, run_text_content_detection, suppress_code_adjacent_text_findings, suppress_textual_visual_findings
from picture.application.orchestrator import PictureComplianceOrchestrator
from picture.domain.enums import FindingType, RedactionMode, SafetyCategory
from picture.domain.models import BBox, OCRLayoutResult, OCRTextBlock, PictureFinding, PictureJob, PictureModerationResult, Polygon, RedactionOperation, RegionMask
from picture.providers.text_compliance import _build_ocr_block_spans, _extract_findings, _isolated_ocr_name_findings, _map_text_to_region, run_text_pipeline_for_ocr
from picture.providers.safety.qwen_sam3_fusion import QwenSAM3SafetyFusionModerator
import picture.providers.safety.qwen_sam3_fusion as qwen_safety_fusion
from picture.providers.safety.qwen35_vl import Qwen35VLSafetyModerator
from picture.providers.ocr.paddleocr_vl import (
    PaddleOCRVLProvider,
    _ocr_pass_tasks,
    _official_prompt_labels,
    _parse_text_blocks,
    _validate_ocr_text,
)
from picture.providers.ocr.paddleocr_vl_api import PaddleOCRVLAPIProvider, _parse_layout_parsing_response
from picture.providers.vision.qwen_sam3_fusion import (
    QwenSAM3FusionVisionDetector,
    SemanticObject,
    _dynamic_prompts,
    _parse_semantic_objects,
)
import picture.providers.vision.privacy_specialists as privacy_specialists
from picture.providers.vision.privacy_specialists import MediaPipeFullRangeFaceDetector, OpenCVBarcodeDetector, OpenCVQRCodeDetector, VisualPrivacySpecialistDetectors, _face_identifiability, _scrfd_blob
from picture.providers.vision.sam3 import DEFAULT_PROMPTS, _dedupe_findings
from picture.providers.vision.sam3_api import SAM3APIVisionDetector
from picture.providers.segmentation.sam3_api import SAM3APISegmentationProvider


def _ocr_result(text: str) -> OCRLayoutResult:
    return OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=0, y=0, w=320, h=80),
                confidence=0.98,
                language="zh",
            )
        ],
        engine_name="test",
    )


def test_text_content_does_not_flag_generic_tutorial() -> None:
    findings = run_text_content_detection(_ocr_result("数学几何证明教程：三角形全等"))

    assert findings == []


def test_text_content_flags_specific_dangerous_instruction() -> None:
    findings = run_text_content_detection(_ocr_result("制作爆炸物教程"))

    assert any(
        finding.finding_type == FindingType.TEXT_CONTENT
        and finding.category == "illegal_instruction"
        for finding in findings
    )


def test_ocr_block_spans_use_canonical_virtual_newlines() -> None:
    result = OCRLayoutResult(
        full_text="被OCR服务改写过的全文",
        text_blocks=[
            OCRTextBlock(text="姓名：张三", bbox=BBox(x=0, y=0, w=80, h=20), confidence=0.9),
            OCRTextBlock(text="电话：18502304582", bbox=BBox(x=0, y=30, w=180, h=20), confidence=0.92),
        ],
    )

    spans = _build_ocr_block_spans(result)

    assert spans[0]["start"] == 0
    assert spans[0]["end"] == len("姓名：张三")
    assert spans[1]["start"] == len("姓名：张三") + 1
    assert spans[1]["end"] == spans[1]["start"] + len("电话：18502304582")


def test_ocr_text_span_maps_to_partial_region_with_padding() -> None:
    block_text = "联系电话：18502304582 班级：三年二班"
    phone_start = block_text.index("18502304582")
    phone_end = phone_start + len("18502304582")
    result = OCRLayoutResult(
        full_text=block_text,
        text_blocks=[
            OCRTextBlock(
                text=block_text,
                bbox=BBox(x=100, y=50, w=300, h=24),
                polygon=Polygon(points=[(100, 50), (400, 50), (400, 74), (100, 74)]),
                confidence=0.96,
            )
        ],
        metadata={"image_width": 800, "image_height": 600},
    )

    region = _map_text_to_region("18502304582", result, start=phone_start, end=phone_end)

    assert region is not None
    assert region.bbox.x > 150
    assert region.bbox.x + region.bbox.w < 360
    assert region.bbox.w < 220
    assert region.polygon is not None


def test_ocr_text_span_maps_inside_large_block_before_region_filter() -> None:
    text = "4||893913||999600"
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=296, y=245, w=430, h=297),
                polygon=Polygon(points=[(296, 245), (726, 245), (726, 542), (296, 542)]),
                confidence=0.7,
            )
        ],
        metadata={"image_width": 1024, "image_height": 768},
    )

    region = _map_text_to_region("999600", result, start=11, end=17)

    assert region is not None
    assert region.bbox.x > 500
    assert region.bbox.w < 230


def test_ocr_multiline_block_maps_span_to_line_level_region() -> None:
    text = "姓名：张三\n联系电话：18502304582\n班级：三年二班"
    phone_start = text.index("18502304582")
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=100, y=60, w=360, h=120),
                confidence=0.9,
            )
        ],
        metadata={"image_width": 800, "image_height": 600},
    )

    spans = _build_ocr_block_spans(result)
    region = _map_text_to_region("18502304582", result, start=phone_start, end=phone_start + 11)

    assert len(spans) == 3
    assert spans[1]["unit_kind"] == "ocr_line"
    assert spans[1]["text"] == "联系电话：18502304582"
    assert region is not None
    assert 90 <= region.bbox.y <= 110
    assert region.bbox.h < 80


def test_ocr_text_finding_carries_region_trace_metadata() -> None:
    text = "联系电话：18502304582"
    start = text.index("18502304582")
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=100, y=50, w=220, h=30),
                confidence=0.9,
            )
        ],
        metadata={"image_width": 800, "image_height": 600},
    )
    document = {
        "findings": [
            {
                "finding_type": "privacy",
                "risk_type": "phone_number",
                "text": "18502304582",
                "start": start,
                "end": start + 11,
                "confidence": 0.95,
            }
        ]
    }

    findings = _extract_findings(document, result)

    assert len(findings) == 1
    metadata = findings[0].metadata
    assert findings[0].region is not None
    assert metadata["ocr_unit_ids"] == ["ocr_block_0001"]
    assert metadata["ocr_region_source"] == "ocr_block_weighted_span"
    assert metadata["ocr_region_quality"] == "medium"


def test_standalone_cn_mobile_is_not_suppressed_as_machine_code() -> None:
    text = "13307692576"
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=340, y=550, w=530, h=45),
                confidence=0.9,
            )
        ],
        metadata={"image_width": 1280, "image_height": 720},
    )
    document = {
        "findings": [
            {
                "finding_type": "privacy",
                "risk_type": "phone_number",
                "text": text,
                "start": 0,
                "end": len(text),
                "confidence": 0.88,
            }
        ]
    }

    findings = _extract_findings(document, result)

    assert len(findings) == 1
    assert findings[0].category == "phone_number"
    assert findings[0].region is not None


def test_picture_ocr_recalls_isolated_chinese_name_with_identity_context() -> None:
    text = "梅媛\n23护本1-5班、助产本第三小组\n2025年12月8日\n13307692576"
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(text="梅媛", bbox=BBox(x=343, y=374, w=526, h=45), confidence=0.9),
            OCRTextBlock(text="23护本1-5班、助产本第三小组", bbox=BBox(x=343, y=419, w=526, h=45), confidence=0.9),
            OCRTextBlock(text="2025年12月8日", bbox=BBox(x=343, y=509, w=526, h=45), confidence=0.9),
            OCRTextBlock(text="13307692576", bbox=BBox(x=343, y=554, w=526, h=45), confidence=0.9),
        ],
        metadata={"image_width": 1280, "image_height": 720},
    )

    findings = _isolated_ocr_name_findings(result, {})

    assert len(findings) == 1
    assert findings[0].category == "person_name"
    assert findings[0].text_span == "梅媛"
    assert findings[0].region is not None


def test_ocr_pipeline_skips_full_text_without_spatial_units(tmp_path) -> None:
    result = OCRLayoutResult(
        full_text="联系电话：18502304582",
        text_blocks=[],
        metadata={"valid_text": True, "spatially_mappable_text": False},
    )

    findings = run_text_pipeline_for_ocr(result, profile="full", run_id="job_no_units", work_dir=tmp_path)

    assert findings == []
    assert not (tmp_path / "ocr_text_compliance" / "cleaned_docs.jsonl").exists()


def test_ocr_char_polys_ignore_virtual_newlines() -> None:
    text = "姓名：张三\n电话：18502304582"
    phone_start = text.index("18502304582")
    visible_chars = [char for char in text if char not in {"\n", "\r"}]
    char_polys = []
    for index, _char in enumerate(visible_chars):
        x = float(index * 10)
        char_polys.append([[x, 0.0], [x + 8.0, 0.0], [x + 8.0, 10.0], [x, 10.0]])
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=0, y=0, w=240, h=40),
                confidence=0.9,
                metadata={"char_polys": char_polys},
            )
        ],
        metadata={"image_width": 400, "image_height": 200},
    )

    region = _map_text_to_region("18502304582", result, start=phone_start, end=phone_start + 11)

    assert region is not None
    assert region.bbox.x >= 75
    assert region.bbox.x < 100
    assert region.bbox.w > 90


def test_ocr_text_mapping_prefers_paddleocr_spotting_instances() -> None:
    text = "Tax Id: 945-82-2137"
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=100, y=100, w=500, h=40),
                confidence=0.7,
            )
        ],
        metadata={
            "image_width": 800,
            "image_height": 600,
            "text_instances": [
                {
                    "unit_id": "spot_0001",
                    "text": "945-82-2137",
                    "bbox": {"x": 310, "y": 100, "w": 150, "h": 40},
                    "polygon": {"points": [[310, 100], [460, 100], [460, 140], [310, 140]]},
                    "confidence": 0.96,
                    "source": "paddleocr_spotting_rec_poly",
                    "quality": "high",
                }
            ],
        },
    )

    document = {
        "findings": [
            {
                "finding_type": "privacy",
                "risk_type": "id_card",
                "text": "945-82-2137",
                "start": 8,
                "end": 19,
                "confidence": 0.95,
            }
        ]
    }

    findings = _extract_findings(document, result)

    assert len(findings) == 1
    assert findings[0].region is not None
    assert 306 <= findings[0].region.bbox.x <= 308
    assert findings[0].metadata["ocr_region_source"] == "paddleocr_spotting_rec_poly"
    assert findings[0].metadata["ocr_region_quality"] == "high"


def test_ocr_address_fragment_expands_to_address_group() -> None:
    text = "Seller:\nAndrews Ltd\n58861 Gonzalez Prairie\nLake Daniellefurt, IN 57228\nTax Id: 945-82-2137"
    start = text.index("Prairie")
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=100, y=100, w=420, h=200),
                confidence=0.8,
            )
        ],
        metadata={"image_width": 800, "image_height": 600},
    )
    document = {
        "findings": [
            {
                "finding_type": "privacy",
                "risk_type": "address",
                "text": "Prairie",
                "start": start,
                "end": start + len("Prairie"),
                "confidence": 0.8,
            }
        ]
    }

    findings = _extract_findings(document, result)

    assert len(findings) == 1
    assert findings[0].region is not None
    assert "58861 Gonzalez Prairie" in findings[0].metadata["ocr_unit_texts"]
    assert "Lake Daniellefurt, IN 57228" in findings[0].metadata["ocr_unit_texts"]


def test_ocr_document_level_address_assessment_recalls_address_group() -> None:
    text = "Client:\nDuncan PLC\nUnit 8799 Box 0703\nDPO AP 81970\nTax Id: 911-82-7132"
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=100, y=100, w=360, h=160),
                confidence=0.8,
            )
        ],
        metadata={"image_width": 800, "image_height": 600},
    )
    document = {
        "document_assessment": {
            "explanation": "This document contains business addresses and tax identifiers.",
        },
        "findings": [],
    }

    findings = _extract_findings(document, result)

    assert len(findings) == 1
    assert findings[0].category == "address"
    assert "Unit 8799 Box 0703" in (findings[0].text_span or "")
    assert findings[0].metadata["ocr_region_source"] == "ocr_address_recall"


def test_ocr_machine_code_social_account_findings_are_suppressed() -> None:
    text = "4||893913||999600"
    result = OCRLayoutResult(
        full_text=text,
        text_blocks=[
            OCRTextBlock(
                text=text,
                bbox=BBox(x=296, y=245, w=430, h=297),
                confidence=0.7,
            )
        ],
        metadata={"image_width": 1024, "image_height": 768},
    )
    document = {
        "findings": [
            {
                "finding_type": "privacy",
                "risk_type": "social_account",
                "text": "999600",
                "start": 11,
                "end": 17,
                "confidence": 0.59,
            }
        ]
    }

    findings = _extract_findings(document, result)

    assert findings == []


def test_sam3_api_segmentation_sends_polygon_prompt(monkeypatch) -> None:
    captured = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "regions": [
                    {
                        "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                        "polygon": [[1, 2], [4, 2], [4, 6], [1, 6]],
                        "confidence": 0.99,
                    }
                ]
            }

    def _post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    import picture.providers.segmentation.sam3_api as sam3_segmentation_api

    monkeypatch.setattr(sam3_segmentation_api.httpx, "post", _post)
    provider = SAM3APISegmentationProvider(base_url="http://sam3.test", timeout_seconds=12)
    region = RegionMask(
        bbox=BBox(x=10, y=20, w=30, h=12),
        polygon=Polygon(points=[(10, 20), (40, 20), (40, 32), (10, 32)]),
        confidence=0.8,
    )

    refined = provider.refine("image.png", [region])

    assert captured["url"] == "http://sam3.test/v1/sam3/refine"
    assert captured["json"]["regions"][0]["polygon"] == [[10.0, 20.0], [40.0, 20.0], [40.0, 32.0], [10.0, 32.0]]
    assert captured["json"]["regions"][0]["region_kind"] == "ocr_text"
    assert captured["json"]["regions"][0]["text_prompt"] == "printed text characters"
    assert captured["json"]["regions"][0]["refine_mode"] == "text_region"
    assert refined[0].polygon is not None


def test_sam3_api_vision_preserves_mask_and_polygon_metadata() -> None:
    provider = SAM3APIVisionDetector(base_url="http://sam3.test")

    findings = provider._findings_from_detections(
        [
            {
                "category": "dangerous",
                "prompt": "knife",
                "prompt_type": "multi_point_box_proxy",
                "score": 0.88,
                "threshold": 0.25,
                "bbox": {"x": 10, "y": 12, "w": 30, "h": 14},
                "polygon": [[10, 12], [40, 12], [38, 26], [10, 26]],
                "polygons": [
                    [[10, 12], [40, 12], [38, 26], [10, 26]],
                    [[50, 52], [54, 52], [54, 56], [50, 56]],
                ],
                "mask_path": "/tmp/knife_mask.png",
                "mask_area": 380,
                "mask_area_ratio": 0.04,
                "mask_bbox_fill_ratio": 0.9,
                "point_prompts": [{"x": 22, "y": 18, "label": True}],
            }
        ]
    )

    assert findings[0].region is not None
    assert findings[0].region.mask_path == "/tmp/knife_mask.png"
    assert findings[0].region.polygon is not None
    assert findings[0].metadata["polygons"][1] == [[50, 52], [54, 52], [54, 56], [50, 56]]
    assert findings[0].metadata["mask_bbox_fill_ratio"] == 0.9
    assert findings[0].metadata["point_prompts"] == [{"x": 22, "y": 18, "label": True}]


def test_qwen35_vl_extract_json_repairs_missing_comma() -> None:
    payload = qwen35_vl._extract_json(
        """
        {
          "is_safe": false,
          "categories": ["dangerous"]
          "scores": {"dangerous": 0.91},
          "review_required": true,
          "explanation": "图片中存在疑似危险物品。"
        }
        """
    )

    assert payload["is_safe"] is False
    assert payload["categories"] == ["dangerous"]
    assert payload["scores"]["dangerous"] == 0.91


def test_qwen35_vl_bad_json_degrades_to_manual_review(monkeypatch) -> None:
    moderator = Qwen35VLSafetyModerator()
    moderator._provider = SimpleNamespace(
        base_url="http://qwen.test/v1",
        model="Qwen3.5-9B",
        api_key="",
        mode="local_model",
        max_tokens=1024,
    )
    monkeypatch.setattr(qwen35_vl, "_image_data_url", lambda *args, **kwargs: "data:image/png;base64,AAAA")
    monkeypatch.setattr(moderator, "_request_content", lambda *args, **kwargs: '{"is_safe": false "categories": [')

    result = moderator.moderate("sample.png")

    assert result.is_safe is False
    assert result.categories == [SafetyCategory.OTHER_NSFW]
    assert result.reason_codes == ["VISUAL_SAFETY_MODEL_JSON_INVALID"]
    assert result.metadata["review_required"] is True
    assert result.metadata["degraded"] is True


def test_visual_safety_filter_accepts_content_label_namespace_for_video() -> None:
    orchestrator = PictureComplianceOrchestrator.__new__(PictureComplianceOrchestrator)
    job = PictureJob(options={"visual_safety_target_labels": ["content.violent"]})
    moderation = PictureModerationResult(
        is_safe=False,
        categories=[SafetyCategory.DANGEROUS],
        scores={"dangerous": 0.91},
        metadata={"explanation": "图片显示多人肢体冲突，属于暴力行为。"},
    )

    filtered = orchestrator._filter_selected_moderation(job, moderation)

    assert filtered.is_safe is False
    assert SafetyCategory.DANGEROUS in filtered.categories
    assert "SAFETY_DANGEROUS" in filtered.reason_codes


def test_visual_safety_filter_preserves_qwen_metadata_risk_when_category_was_safe() -> None:
    orchestrator = PictureComplianceOrchestrator.__new__(PictureComplianceOrchestrator)
    job = PictureJob(options={"visual_safety_target_labels": ["content.violent"]})
    moderation = PictureModerationResult(
        is_safe=True,
        categories=[SafetyCategory.SAFE],
        scores={},
        metadata={
            "explanation": "图片显示多人肢体冲突，属于暴力行为，不符合教育数据合规要求。",
            "category_details": {
                "dangerous": {
                    "risk_subtype_zh": "肢体冲突/暴力行为",
                    "object_name_zh": "多人肢体冲突",
                }
            },
            "review_required": True,
        },
    )

    filtered = orchestrator._filter_selected_moderation(job, moderation)

    assert filtered.is_safe is False
    assert SafetyCategory.DANGEROUS in filtered.categories
    assert "SAFETY_DANGEROUS" in filtered.reason_codes


def test_merge_findings_keeps_distinct_risks_in_same_region() -> None:
    region = RegionMask(bbox=BBox(x=10, y=10, w=100, h=100), confidence=0.9)
    phone = PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category="phone_number",
        score=0.95,
        region=region,
        reason_code="PII_PHONE",
    )
    face = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.98,
        region=region,
        reason_code="VISION_FACE",
    )

    findings = merge_findings([phone], [face])

    assert len(findings) == 2
    assert {finding.reason_code for finding in findings} == {"PII_PHONE", "VISION_FACE"}


def test_merge_findings_dedupes_same_risk_by_score() -> None:
    region = RegionMask(bbox=BBox(x=10, y=10, w=100, h=100), confidence=0.9)
    lower_score = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.50,
        region=region,
        reason_code="VISION_FACE",
    )
    higher_score = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.90,
        region=region,
        reason_code="VISION_FACE",
    )

    findings = merge_findings([lower_score], [higher_score])

    assert len(findings) == 1
    assert findings[0].score == 0.90


def test_merge_findings_keeps_distinct_safety_violation_ids() -> None:
    first = PictureFinding(
        finding_type=FindingType.SAFETY,
        category="dangerous",
        score=0.9,
        region=RegionMask(bbox=BBox(x=10, y=10, w=50, h=30), confidence=0.9),
        reason_code="SAFETY_DANGEROUS",
        metadata={"violation_id": "dangerous_1"},
    )
    second = PictureFinding(
        finding_type=FindingType.SAFETY,
        category="dangerous",
        score=0.88,
        region=RegionMask(bbox=BBox(x=15, y=12, w=48, h=30), confidence=0.88),
        reason_code="SAFETY_DANGEROUS",
        metadata={"violation_id": "dangerous_2"},
    )

    findings = merge_findings([first], [second])

    assert len(findings) == 2


def test_suppress_code_adjacent_barcode_digits() -> None:
    barcode = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="barcode",
        score=0.98,
        region=RegionMask(bbox=BBox(x=100, y=100, w=220, h=60), confidence=0.98),
        reason_code="VISION_BARCODE",
    )
    digits = PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category="phone_number",
        label="6901234567890",
        text_span="6901234567890",
        score=0.80,
        region=RegionMask(bbox=BBox(x=115, y=165, w=190, h=22), confidence=0.80),
        reason_code="PII_PHONE",
    )

    filtered = suppress_code_adjacent_text_findings([digits], [barcode])

    assert filtered == []


def test_suppress_code_adjacent_text_keeps_remote_phone_number() -> None:
    barcode = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="barcode",
        score=0.98,
        region=RegionMask(bbox=BBox(x=100, y=100, w=220, h=60), confidence=0.98),
        reason_code="VISION_BARCODE",
    )
    phone = PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category="phone_number",
        label="13800138000",
        text_span="13800138000",
        score=0.90,
        region=RegionMask(bbox=BBox(x=20, y=20, w=110, h=24), confidence=0.90),
        reason_code="PII_PHONE",
    )

    filtered = suppress_code_adjacent_text_findings([phone], [barcode])

    assert filtered == [phone]


def test_suppress_textual_visual_phone_when_ocr_phone_has_region() -> None:
    ocr_phone = PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category="phone_number",
        label="联系方式检测",
        text_span="13307692576",
        score=0.88,
        region=RegionMask(bbox=BBox(x=343, y=554, w=526, h=45), confidence=0.88),
        reason_code="OCR_TEXT_PII_PHONE_NUMBER",
    )
    visual_phone = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="account_region",
        label="SAM3 detected account_region",
        score=0.99,
        region=RegionMask(bbox=BBox(x=401, y=643, w=409, h=44), confidence=0.43),
        reason_code="VISION_ACCOUNT_REGION",
        metadata={"qwen_object_name_zh": "手机号码", "qwen_description": "显示手机号码 13307692576"},
    )
    stamp = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="stamp",
        label="印章",
        score=0.8,
        region=RegionMask(bbox=BBox(x=900, y=120, w=200, h=180), confidence=0.8),
        reason_code="VISION_STAMP",
    )

    filtered = suppress_textual_visual_findings([visual_phone, stamp], [ocr_phone])

    assert filtered == [stamp]


def test_face_redaction_preserves_semantics_with_identity_core() -> None:
    from picture.application.services import build_redaction_operations

    finding = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.9,
        region=RegionMask(bbox=BBox(x=100, y=50, w=200, h=300), confidence=0.9),
        reason_code="VISION_FACE",
    )

    operations = build_redaction_operations([finding], {"face": "gaussian_blur", "default": "black_box"})

    assert len(operations) == 1
    op = operations[0]
    assert op.mode == RedactionMode.GAUSSIAN_BLUR
    assert op.region.bbox.x == 136
    assert op.region.bbox.y == 116
    assert op.region.bbox.w == 128
    assert op.region.bbox.h == 114
    assert op.metadata["minimization_strategy"] == "face_identity_core_blur"
    assert op.metadata["semantic_preservation"] is True


def test_qr_redaction_uses_full_padded_region() -> None:
    from picture.application.services import build_redaction_operations

    finding = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="qr_code",
        score=0.9,
        region=RegionMask(bbox=BBox(x=10, y=20, w=100, h=100), confidence=0.9),
        reason_code="VISION_QR_CODE",
    )

    operations = build_redaction_operations([finding], {"qr_code": "black_box", "default": "black_box"})

    assert len(operations) == 1
    assert operations[0].mode == RedactionMode.BLACK_BOX
    assert operations[0].region.bbox.x == 5
    assert operations[0].region.bbox.y == 15
    assert operations[0].region.bbox.w == 110
    assert operations[0].region.bbox.h == 110
    assert operations[0].metadata["minimization_strategy"] == "qr_full_region_redaction"
    assert operations[0].metadata["semantic_preservation"] is False


def test_redactor_uses_mask_instead_of_full_bbox(tmp_path) -> None:
    from PIL import Image

    from picture.providers.redaction.opencv_redactor import OpenCVRedactor

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "redacted.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    mask = Image.new("L", (20, 20), 0)
    for x in range(8, 12):
        for y in range(8, 12):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    op = RedactionOperation(
        finding_id="masked",
        region=RegionMask(
            bbox=BBox(x=5, y=5, w=10, h=10),
            mask_path=str(mask_path),
            confidence=0.9,
        ),
        mode=RedactionMode.BLACK_BOX,
    )

    OpenCVRedactor().redact(str(image_path), [op], str(output_path))
    result = Image.open(output_path).convert("RGB")

    assert result.getpixel((9, 9)) == (0, 0, 0)
    assert result.getpixel((6, 6)) == (255, 255, 255)


def test_redactor_uses_polygon_instead_of_full_bbox(tmp_path) -> None:
    from PIL import Image

    from picture.domain.models import Polygon
    from picture.providers.redaction.opencv_redactor import OpenCVRedactor

    image_path = tmp_path / "image.png"
    output_path = tmp_path / "redacted_polygon.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    op = RedactionOperation(
        finding_id="polygon",
        region=RegionMask(
            bbox=BBox(x=4, y=4, w=12, h=12),
            polygon=Polygon(points=[(8, 8), (12, 8), (10, 12)]),
            confidence=0.9,
        ),
        mode=RedactionMode.BLACK_BOX,
    )

    OpenCVRedactor().redact(str(image_path), [op], str(output_path))
    result = Image.open(output_path).convert("RGB")

    assert result.getpixel((10, 9)) == (0, 0, 0)
    assert result.getpixel((5, 5)) == (255, 255, 255)


def test_paddleocr_vl_provider_requires_local_model_files(tmp_path) -> None:
    provider = PaddleOCRVLProvider(model_dir=str(tmp_path), use_gpu=False)

    try:
        provider._validate_model_dir()
    except Exception as exc:
        assert "PaddleOCR-VL local model files" in str(exc)
    else:
        raise AssertionError("missing local PaddleOCR-VL files should fail")


def test_paddleocr_vl_defaults_to_document_quality_ocr_and_spotting_passes() -> None:
    provider = PaddleOCRVLProvider(model_dir="/tmp/model", use_gpu=False)

    assert provider._backend == "transformers"
    assert provider._task == "spotting"
    assert provider._max_new_tokens == 768
    assert _ocr_pass_tasks("ocr") == ["ocr"]
    assert _ocr_pass_tasks("spotting") == ["ocr", "spotting"]
    assert _official_prompt_labels("ocr") == ["ocr"]


def test_paddleocr_vl_uses_qwen_fallback_when_primary_ocr_is_empty(monkeypatch) -> None:
    provider = PaddleOCRVLProvider(model_dir="/tmp/model", use_gpu=False)
    paddle_result = OCRLayoutResult(
        full_text="",
        text_blocks=[],
        engine_name="PaddleOCR-VL-1.5(local)",
        metadata={"backend": "transformers", "valid_text": False, "invalid_reason": "empty_text"},
    )
    monkeypatch.setattr(
        paddleocr_vl,
        "_resolve_qwen_provider",
        lambda: SimpleNamespace(
            base_url="http://qwen.test/v1",
            model="Qwen3.5-9B",
            api_key="",
            mode="local_model",
        ),
    )
    monkeypatch.setattr(paddleocr_vl, "_image_data_url", lambda *args, **kwargs: "data:image/png;base64,AAAA")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"full_text":"Invoice no: 51109338\\nTax Id: 945-82-2137",'
                                '"text_blocks":[{"text":"Tax Id: 945-82-2137","bbox":[140,650,280,30],"confidence":0.9}]}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(paddleocr_vl.httpx, "post", lambda *args, **kwargs: _Response())

    result = provider._analyze_with_qwen_fallback(
        "sample.png",
        1654,
        2339,
        "spotting",
        paddle_result=paddle_result,
    )

    assert result.metadata["backend"] == "qwen35_vl_fallback"
    assert result.metadata["qwen_fallback_success"] is True
    assert "Tax Id" in result.full_text
    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].bbox.x == 140


def test_paddleocr_vl_parser_handles_json_blocks() -> None:
    blocks = _parse_text_blocks(
        '[{"text":"张三 13800138000","bbox":[10,20,210,60],"confidence":0.91}]',
        width=400,
        height=300,
    )

    assert len(blocks) == 1
    assert blocks[0].text == "张三 13800138000"
    assert blocks[0].bbox.x == 10
    assert blocks[0].bbox.w == 200


def test_paddleocr_vl_rejects_formula_hallucination_as_ocr_text() -> None:
    hallucination = r"\( \text{C}_{6}\text{H}_{12}\text{O}_{6} \) " * 12

    valid, reason = _validate_ocr_text(hallucination)
    blocks = _parse_text_blocks(hallucination, width=400, height=300)

    assert valid is False
    assert reason == "formula_hallucination"
    assert blocks == []


def test_paddleocr_vl_accepts_invoice_like_plain_ocr_text() -> None:
    text = "Invoice no: 12847181 Seller: Fitzpatrick and Sons Tax Id: 998-99-5253 IBAN: GB92PBPQ73499358975916"

    valid, reason = _validate_ocr_text(text)
    blocks = _parse_text_blocks(text, width=400, height=300)

    assert valid is True
    assert reason == ""
    assert len(blocks) == 1
    assert "Tax Id" in blocks[0].text


def test_paddleocr_vl_rejects_language_model_placeholder_ocr_text() -> None:
    placeholder = r"\( \) " * 40 + "The quick brown fox jumps over the\n" + "1." + "0" * 300

    valid, reason = _validate_ocr_text(placeholder)
    blocks = _parse_text_blocks(placeholder, width=400, height=300)

    assert valid is False
    assert reason in {"language_model_placeholder", "repetitive_numeric_hallucination", "low_diversity_hallucination"}
    assert blocks == []


def test_paddleocr_vl_api_provider_maps_layout_parsing_response(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "errorCode": 0,
                "result": {
                    "layoutParsingResults": [
                        {
                            "markdown": {"text": "姓名：张三\n手机号：13800138000"},
                            "prunedResult": {
                                "parsing_res_list": [
                                    {
                                        "block_label": "text",
                                        "block_content": "姓名：张三",
                                        "block_bbox": [10, 20, 110, 50],
                                        "score": 0.91,
                                    },
                                    {
                                        "block_label": "text",
                                        "block_content": "手机号：13800138000",
                                        "block_bbox": [[12, 60], [220, 60], [220, 90], [12, 90]],
                                        "score": 0.93,
                                    },
                                ]
                            },
                        }
                    ]
                },
            }

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, json):
            assert url == "http://paddle.test/layout-parsing"
            assert json["fileType"] == 1
            assert json["useLayoutDetection"] is True
            assert json["useChartRecognition"] is True
            assert json["useSealRecognition"] is True
            assert json["visualize"] is False
            return _Response()

    import picture.providers.ocr.paddleocr_vl_api as paddleocr_vl_api

    monkeypatch.setattr(paddleocr_vl_api.httpx, "Client", _Client)

    provider = PaddleOCRVLAPIProvider(base_url="http://paddle.test")
    result = provider.analyze(str(image_path))

    assert result.engine_name == "PaddleOCR-VL-1.5(api)"
    assert "13800138000" in result.full_text
    assert len(result.text_blocks) == 2
    assert result.text_blocks[0].bbox.w == 100
    assert result.text_blocks[1].bbox.x == 12
    assert len(result.layout_regions) == 2
    assert result.metadata["backend"] == "paddlex_serving"
    assert result.metadata["stages"] == ["layout_analysis", "vlm_recognition", "reading_order_merge"]


def test_paddleocr_vl_api_filters_image_placeholder_markdown() -> None:
    result = _parse_layout_parsing_response(
        {
            "errorCode": 0,
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {
                            "text": '<div style="text-align: center;"><img src="imgs/img_in_image_box_1_1_797_450.jpg" alt=""/></div>'
                        },
                        "prunedResult": {"parsing_res_list": []},
                    }
                ]
            },
        }
    )

    assert result.full_text == ""
    assert result.text_blocks == []
    assert result.metadata["valid_text"] is False
    assert result.metadata["invalid_reason"] == "image_placeholder_only"
    assert result.metadata["spatially_mappable_text"] is False
    assert result.metadata["placeholder_text_count"] == 1


def test_qwen_sam3_semantic_parser_filters_allowed_categories() -> None:
    objects = _parse_semantic_objects(
        {
            "sensitive_objects": [
                {"category": "face", "present": True, "confidence": 0.9, "requires_redaction": True},
                {"category": "unknown", "present": True, "confidence": 0.9},
            ]
        },
        {"face"},
    )

    assert len(objects) == 1
    assert objects[0].category == "face"
    assert objects[0].confidence == 0.9


def test_sam3_face_prompts_do_not_use_person_or_head() -> None:
    prompts = DEFAULT_PROMPTS["face"]

    assert "person" not in prompts
    assert "head" not in prompts


def test_sam3_dedupe_removes_nested_face_body_box() -> None:
    large = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.92,
        region=RegionMask(bbox=BBox(x=0, y=0, w=100, h=200), confidence=0.92),
        reason_code="VISION_FACE",
    )
    small = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.86,
        region=RegionMask(bbox=BBox(x=30, y=20, w=36, h=42), confidence=0.86),
        reason_code="VISION_FACE",
    )

    findings = _dedupe_findings([large, small])

    assert len(findings) == 1
    assert findings[0].region is not None
    assert findings[0].region.bbox.w == 36
    assert findings[0].region.bbox.h == 42


def test_qwen_dynamic_face_prompts_filter_location_and_body_terms() -> None:
    prompts = _dynamic_prompts(
        {
            "face": SemanticObject(
                category="face",
                present=True,
                confidence=0.91,
                requires_redaction=True,
                object_name_zh="人物面部",
                location_hint="左上区域",
                description="虽然戴着帽子和耳罩，但面部轮廓和特征仍可辨识，属于可识别身份的对象。",
                suggested_prompt_for_sam3="person",
            )
        }
    )

    face_prompts = prompts["face"]
    assert "左上区域" not in face_prompts
    assert "person" not in face_prompts
    assert "人物面部" in face_prompts


def test_qwen_sam3_fusion_adds_unlocalized_review_finding(monkeypatch, tmp_path) -> None:
    detector = QwenSAM3FusionVisionDetector(model_dir=str(tmp_path))
    monkeypatch.setattr(
        detector,
        "_semantic_detect",
        lambda image_path, targets: (
            [SemanticObject(category="face", present=True, confidence=0.91, requires_redaction=True)],
            "",
        ),
    )
    monkeypatch.setattr(detector._sam3, "detect", lambda image_path, target_types=None: [])

    findings = detector.detect("sample.png", target_types=["face"])

    assert len(findings) == 1
    assert findings[0].category == "face"
    assert findings[0].region is None
    assert findings[0].metadata["localization_required"] is True


def test_qwen_sam3_fusion_drops_sam3_only_face_when_qwen_fails(monkeypatch, tmp_path) -> None:
    detector = QwenSAM3FusionVisionDetector(model_dir=str(tmp_path), sam3_keep_without_qwen_threshold=0.75)
    monkeypatch.setattr(detector, "_semantic_detect", lambda image_path, targets: ([], "timeout"))
    region = RegionMask(bbox=BBox(x=1, y=2, w=3, h=4), confidence=0.9)
    monkeypatch.setattr(
        detector._sam3,
        "detect",
        lambda image_path, target_types=None: [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="face",
                score=0.88,
                region=region,
                reason_code="VISION_FACE",
            )
        ],
    )

    findings = detector.detect("sample.png", target_types=["face"])

    assert findings == []


def test_qwen_sam3_fusion_qr_fast_path_does_not_call_qwen_or_sam3(tmp_path) -> None:
    region = RegionMask(bbox=BBox(x=10, y=20, w=40, h=40), confidence=0.99)
    qr = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="qr_code",
        score=0.99,
        region=region,
        reason_code="VISION_QR_CODE",
        metadata={"source_detectors": ["opencv_wechat_qrcode"]},
    )
    specialists = SimpleNamespace(detect=lambda image_path, targets: [qr])
    detector = QwenSAM3FusionVisionDetector(model_dir=str(tmp_path), specialist_detectors=specialists)
    detector._semantic_detect = lambda image_path, targets: (_ for _ in ()).throw(AssertionError("qwen should not run"))
    detector._sam3.detect = lambda image_path, target_types=None: (_ for _ in ()).throw(AssertionError("sam3 should not run"))

    findings = detector.detect("sample.png", target_types=["qr_code"])

    assert len(findings) == 1
    assert findings[0].category == "qr_code"
    assert findings[0].metadata["operator_id"] == "VPI_006"
    assert findings[0].metadata["qwen_semantic_confirmed"] is True


def test_qwen_sam3_fusion_face_specialist_respects_selected_categories(tmp_path) -> None:
    region = RegionMask(bbox=BBox(x=10, y=20, w=30, h=35), confidence=0.96)
    face = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.96,
        region=region,
        reason_code="VISION_FACE",
        metadata={"source_detectors": ["scrfd"]},
    )
    observed_targets: list[list[str]] = []

    def specialist_detect(image_path: str, targets: list[str]) -> list[PictureFinding]:
        observed_targets.append(list(targets))
        return [face]

    specialists = SimpleNamespace(detect=specialist_detect)
    detector = QwenSAM3FusionVisionDetector(model_dir=str(tmp_path), specialist_detectors=specialists)
    detector._semantic_detect = lambda image_path, targets: ([], "")
    detector._sam3.detect = lambda image_path, target_types=None: []

    findings = detector.detect("sample.png", target_types=["face"])

    assert observed_targets == [["face"]]
    assert len(findings) == 1
    assert findings[0].category == "face"
    assert findings[0].metadata["operator_id"] == "VPI_001"
    assert findings[0].metadata["source_detectors"] == ["scrfd"]


def test_qwen_sam3_fusion_face_specialist_blocks_sam3_only_face(tmp_path) -> None:
    specialist_region = RegionMask(bbox=BBox(x=10, y=20, w=30, h=35), confidence=0.96)
    sam3_region = RegionMask(bbox=BBox(x=80, y=40, w=34, h=38), confidence=0.88)
    specialists = SimpleNamespace(
        detect=lambda image_path, targets: [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="face",
                score=0.96,
                region=specialist_region,
                reason_code="VISION_FACE",
                metadata={"source_detectors": ["mediapipe_full_range"]},
            )
        ]
    )
    detector = QwenSAM3FusionVisionDetector(model_dir=str(tmp_path), specialist_detectors=specialists)
    detector._semantic_detect = lambda image_path, targets: ([], "timeout")
    detector._sam3.detect = lambda image_path, target_types=None: [
        PictureFinding(
            finding_type=FindingType.VISION_OBJECT,
            category="face",
            score=0.88,
            region=sam3_region,
            reason_code="VISION_FACE",
        )
    ]

    findings = detector.detect("sample.png", target_types=["face"])

    assert len(findings) == 1
    assert findings[0].metadata.get("source_detectors") == ["mediapipe_full_range"]


def test_visual_privacy_specialists_runs_mediapipe_and_scrfd_for_face() -> None:
    mediapipe = SimpleNamespace(
        detect=lambda image_path: [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="face",
                score=0.91,
                region=RegionMask(bbox=BBox(x=10, y=10, w=20, h=20), confidence=0.91),
                reason_code="VISION_FACE",
                metadata={"source_detectors": ["mediapipe_full_range"], "face_keypoints": [[12, 12], [24, 12], [18, 18]]},
            )
        ]
    )
    scrfd = SimpleNamespace(
        detect=lambda image_path: [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="face",
                score=0.93,
                region=RegionMask(bbox=BBox(x=80, y=80, w=20, h=20), confidence=0.93),
                reason_code="VISION_FACE",
                metadata={"source_detectors": ["scrfd"], "face_keypoints": [[82, 82], [94, 82], [88, 88]]},
            )
        ]
    )
    specialists = VisualPrivacySpecialistDetectors(
        qr_detector=SimpleNamespace(detect=lambda image_path: []),
        barcode_detector=SimpleNamespace(detect=lambda image_path: []),
        face_detector=scrfd,
        mediapipe_face_detector=mediapipe,
    )

    findings = specialists.detect("sample.png", ["face"])

    assert len(findings) == 2
    assert {tuple(finding.metadata.get("source_detectors", [])) for finding in findings} == {
        ("mediapipe_full_range",),
        ("scrfd",),
    }


def test_face_identifiability_rejects_back_head_like_candidate() -> None:
    decision = _face_identifiability(
        BBox(x=20, y=20, w=80, h=80),
        keypoints=[],
        score=0.95,
        image_width=400,
        image_height=300,
    )

    assert decision["is_identifiable_face"] is False
    assert decision["face_filter_decision"] == "drop"
    assert "关键点不足" in decision["face_filter_reason"]


def test_face_identifiability_keeps_clear_face_candidate() -> None:
    decision = _face_identifiability(
        BBox(x=20, y=20, w=80, h=80),
        keypoints=[[35, 40], [65, 40], [50, 55], [40, 75], [60, 75]],
        score=0.92,
        image_width=400,
        image_height=300,
    )

    assert decision["is_identifiable_face"] is True
    assert decision["face_filter_decision"] == "keep"
    assert decision["face_visible_keypoint_count"] == 5


def test_face_identifiability_rejects_thin_hand_like_candidate() -> None:
    decision = _face_identifiability(
        BBox(x=100, y=100, w=20, h=74),
        keypoints=[[105, 110], [110, 120], [115, 130]],
        score=0.92,
        image_width=400,
        image_height=300,
    )

    assert decision["is_identifiable_face"] is False
    assert "长宽比异常" in decision["face_filter_reason"]


def test_face_findings_are_not_eligible_for_generic_segmentation_refine() -> None:
    finding = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="face",
        score=0.96,
        region=RegionMask(bbox=BBox(x=10, y=10, w=50, h=50), confidence=0.96),
        reason_code="VISION_FACE",
    )
    orchestrator = PictureComplianceOrchestrator.__new__(PictureComplianceOrchestrator)

    assert orchestrator._should_refine_with_segmentation(finding) is False


def test_barcode_detector_requires_decoded_text(monkeypatch, tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "barcode_like_wall.png"
    Image.new("RGB", (100, 60), "white").save(image_path)

    class FakeCV2:
        @staticmethod
        def imread(path):
            return object()

    class FakeDetector:
        def detectAndDecodeWithType(self, image):
            points = [[[5.0, 8.0], [90.0, 8.0], [90.0, 28.0], [5.0, 28.0]]]
            return True, ("",), (), points

    detector = OpenCVBarcodeDetector()
    detector._detector = FakeDetector()
    monkeypatch.setattr(privacy_specialists, "_import_cv2", lambda: FakeCV2)

    assert detector.detect(str(image_path)) == []


def test_barcode_detector_keeps_decoded_text(monkeypatch, tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "barcode.png"
    Image.new("RGB", (100, 60), "white").save(image_path)

    class FakeCV2:
        @staticmethod
        def imread(path):
            return object()

    class FakeDetector:
        def detectAndDecodeWithType(self, image):
            points = [[[5.0, 8.0], [90.0, 8.0], [90.0, 28.0], [5.0, 28.0]]]
            return True, ("ABC123",), ("EAN_13",), points

    detector = OpenCVBarcodeDetector()
    detector._detector = FakeDetector()
    monkeypatch.setattr(privacy_specialists, "_import_cv2", lambda: FakeCV2)

    findings = detector.detect(str(image_path))

    assert len(findings) == 1
    assert findings[0].category == "barcode"
    assert findings[0].metadata["decoded_text_present"] is True


def test_qr_detector_requires_decoded_text(monkeypatch, tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "qr_like_pattern.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FakeCV2:
        @staticmethod
        def imread(path):
            return object()

    class FakeDetector:
        def detectAndDecode(self, image):
            points = [[[10.0, 10.0], [80.0, 10.0], [80.0, 80.0], [10.0, 80.0]]]
            return ("",), points

    detector = OpenCVQRCodeDetector()
    detector._detector = FakeDetector()
    monkeypatch.setattr(privacy_specialists, "_import_cv2", lambda: FakeCV2)

    assert detector.detect(str(image_path)) == []


def test_qr_detector_keeps_decoded_text(monkeypatch, tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "qr.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FakeCV2:
        @staticmethod
        def imread(path):
            return object()

    class FakeDetector:
        def detectAndDecode(self, image):
            points = [[[10.0, 10.0], [80.0, 10.0], [80.0, 80.0], [10.0, 80.0]]]
            return ("https://example.test",), points

    detector = OpenCVQRCodeDetector()
    detector._detector = FakeDetector()
    monkeypatch.setattr(privacy_specialists, "_import_cv2", lambda: FakeCV2)

    findings = detector.detect(str(image_path))

    assert len(findings) == 1
    assert findings[0].category == "qr_code"
    assert findings[0].metadata["decoded_text_present"] is True


def test_scrfd_blob_matches_insightface_preprocess() -> None:
    import numpy as np

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    blob, scale, pad_left, pad_top = _scrfd_blob(image, 640)

    assert blob.shape == (1, 3, 640, 640)
    assert scale == 3.2
    assert pad_left == 0
    assert pad_top == 0
    # OpenCV reads BGR images; official InsightFace preprocessing uses swapRB=True.
    assert blob[0, 0, 0, 0] > 0.9
    assert blob[0, 2, 0, 0] < -0.9


def test_qwen_sam3_safety_fusion_does_not_fake_region_when_unlocalized(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "unsafe.png"
    from PIL import Image

    Image.new("RGB", (32, 16), "white").save(image_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.DANGEROUS],
            scores={"dangerous": 0.95},
            reason_codes=["SAFETY_DANGEROUS"],
            provider="qwen",
            metadata={
                "category_details": {"dangerous": {"object_name_zh": "疑似手枪"}},
                "evidence_regions": [
                    {
                        "category": "dangerous",
                        "label": "疑似手枪",
                        "bbox": [3, 4, 8, 9],
                        "confidence": 0.6,
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(moderator._sam3, "detect_with_prompts", lambda *args, **kwargs: [])

    result = moderator.moderate(str(image_path))

    assert result.metadata["evidence_regions"] == []
    assert result.metadata["qwen_evidence_hints"][0]["source"] == "qwen_hint_only"
    assert result.metadata["qwen_evidence_hints"][0]["bbox"] == [3.0, 4.0, 8.0, 9.0]
    assert result.metadata["localization_status"] == "unlocalized"
    assert result.metadata["review_required"] is True


def test_safety_finding_with_region_builds_redaction_even_if_drop() -> None:
    finding = PictureFinding(
        finding_type=FindingType.SAFETY,
        category="dangerous",
        score=0.96,
        region=RegionMask(bbox=BBox(x=0, y=0, w=100, h=80), confidence=0.96),
        reason_code="SAFETY_DANGEROUS",
    )

    operations = build_redaction_operations([finding], {"default": "black_box"})

    assert len(operations) == 1
    assert operations[0].mode == RedactionMode.BLACK_BOX


def test_dangerous_safety_findings_are_eligible_for_segmentation_refine() -> None:
    finding = PictureFinding(
        finding_type=FindingType.SAFETY,
        category="dangerous",
        score=0.96,
        region=RegionMask(bbox=BBox(x=10, y=10, w=40, h=20), confidence=0.96),
        reason_code="SAFETY_DANGEROUS",
    )
    orchestrator = PictureComplianceOrchestrator.__new__(PictureComplianceOrchestrator)

    assert orchestrator._should_refine_with_segmentation(finding) is True


def test_ocr_text_findings_are_not_eligible_for_segmentation_refine() -> None:
    finding = PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category="phone_number",
        score=0.96,
        region=RegionMask(bbox=BBox(x=10, y=10, w=80, h=20), confidence=0.96),
        reason_code="OCR_TEXT_PII_PHONE_NUMBER",
    )
    orchestrator = PictureComplianceOrchestrator.__new__(PictureComplianceOrchestrator)

    assert orchestrator._should_refine_with_segmentation(finding) is False


def test_ocr_text_id_card_redaction_uses_ocr_region_without_visual_band_scaling() -> None:
    finding = PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category="id_card",
        score=0.96,
        region=RegionMask(bbox=BBox(x=100, y=200, w=300, h=40), confidence=0.96),
        reason_code="OCR_TEXT_PII_ID_CARD",
    )

    operations = build_redaction_operations([finding], {"default": "black_box"})

    assert len(operations) == 1
    assert operations[0].mode == RedactionMode.BLACK_BOX
    assert operations[0].region.bbox == finding.region.bbox
    assert operations[0].metadata["minimization_strategy"] == "ocr_text_region_redaction"
    assert operations[0].metadata["semantic_preservation"] is False


def test_qwen_sam3_safety_fusion_uses_sam3_region_when_localized(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "unsafe.png"
    from PIL import Image

    Image.new("RGB", (64, 64), "white").save(image_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.DANGEROUS],
            scores={"dangerous": 0.95},
            reason_codes=["SAFETY_DANGEROUS"],
            provider="qwen",
            metadata={
                "category_details": {"dangerous": {"object_name_zh": "疑似手枪"}},
                "evidence_regions": [{"category": "dangerous", "bbox": [8, 8, 10, 10]}],
            },
        ),
    )
    monkeypatch.setattr(
        moderator._sam3,
        "detect_with_prompts",
        lambda *args, **kwargs: [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="dangerous",
                score=0.88,
                region=RegionMask(bbox=BBox(x=10, y=11, w=12, h=13), confidence=0.88),
                metadata={"prompt": "pistol"},
            )
        ],
    )

    result = moderator.moderate(str(image_path))

    evidence = result.metadata["evidence_regions"]
    assert evidence[0]["source"] == "sam3_safety_localization"
    assert evidence[0]["bbox"] == [10.0, 11.0, 12.0, 13.0]
    assert evidence[0]["localization_status"] == "localized_by_sam3"
    assert result.metadata["localization_status"] == "localized_by_sam3"


def test_safety_dangerous_prompts_are_specific() -> None:
    prompts = qwen_safety_fusion.SAFETY_SAM3_PROMPTS["dangerous"]

    assert "weapon" not in prompts
    assert "dangerous object" not in prompts
    assert "pistol" in prompts
    assert "firearm" in prompts


def test_safety_prompt_rounds_do_not_use_scene_descriptions() -> None:
    rounds = qwen_safety_fusion._safety_prompt_rounds(
        ["dangerous"],
        {
            "dangerous": {
                "object_name_zh": "疑似手枪",
                "risk_subtype_zh": "枪械",
                "scene_description_zh": "人物手持疑似枪械，位于画面左侧。",
                "risk_reason_zh": "该画面存在危险行为，不适合作为普通教育数据交付。",
            }
        },
        {"evidence_regions": [{"category": "dangerous", "label": "枪械", "description": "人物手持枪械"}]},
    )

    flat = [prompt for item in rounds for values in item.values() for prompt in values]
    assert "pistol" in flat
    assert "gun" in flat
    assert "人物手持疑似枪械" not in flat
    assert "该画面存在危险行为" not in flat


def test_safety_localization_rejects_far_dangerous_box(tmp_path) -> None:
    image_path = tmp_path / "gun.png"
    from PIL import Image

    Image.new("RGB", (200, 100), "white").save(image_path)
    finding = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="dangerous",
        label="SAM3 detected dangerous",
        score=0.92,
        region=RegionMask(bbox=BBox(x=150, y=70, w=35, h=20), confidence=0.92),
        metadata={"prompt": "pistol"},
    )

    selected, rejected = qwen_safety_fusion._select_reliable_sam3_evidence(
        [finding],
        [{"category": "dangerous", "bbox": [10, 10, 30, 20], "source": "qwen_hint_only"}],
        str(image_path),
    )

    assert selected == []
    assert rejected[0]["rejection_reason"] == "far_from_qwen_hint"


def test_safety_localization_prefers_specific_gun_box_near_qwen_hint(tmp_path) -> None:
    image_path = tmp_path / "gun.png"
    from PIL import Image

    Image.new("RGB", (300, 200), "white").save(image_path)
    large_generic = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="dangerous",
        label="weapon",
        score=0.96,
        region=RegionMask(bbox=BBox(x=0, y=0, w=180, h=160), confidence=0.96),
        metadata={"prompt": "weapon"},
    )
    far_specific = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="dangerous",
        label="pistol",
        score=0.94,
        region=RegionMask(bbox=BBox(x=230, y=150, w=40, h=28), confidence=0.94),
        metadata={"prompt": "pistol"},
    )
    near_specific = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="dangerous",
        label="pistol",
        score=0.82,
        region=RegionMask(bbox=BBox(x=80, y=75, w=46, h=24), confidence=0.82),
        metadata={"prompt": "pistol"},
    )

    selected, rejected = qwen_safety_fusion._select_reliable_sam3_evidence(
        [large_generic, far_specific, near_specific],
        [{"category": "dangerous", "bbox": [78, 72, 52, 32], "source": "qwen_hint_only"}],
        str(image_path),
    )

    assert len(selected) == 1
    assert selected[0]["bbox"] == [80.0, 75.0, 46.0, 24.0]
    assert selected[0]["label"] == "pistol"
    assert {item["rejection_reason"] for item in rejected} == {
        "generic_dangerous_prompt",
        "far_from_qwen_hint",
    }


def test_safety_localization_allows_large_specific_dangerous_object(tmp_path) -> None:
    image_path = tmp_path / "large_gun.png"
    from PIL import Image

    Image.new("RGB", (512, 512), "white").save(image_path)
    finding = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="dangerous",
        label="gun",
        score=0.96,
        region=RegionMask(bbox=BBox(x=0, y=12, w=468, h=490), confidence=0.96),
        metadata={"prompt": "gun"},
    )

    selected, rejected = qwen_safety_fusion._select_reliable_sam3_evidence(
        [finding],
        [{"category": "dangerous", "bbox": [0, 18, 512, 494], "source": "qwen_hint_only"}],
        str(image_path),
    )

    assert rejected == []
    assert len(selected) == 1
    assert selected[0]["label"] == "gun"
    assert selected[0]["bbox"] == [0.0, 12.0, 468.0, 490.0]


def test_explicit_body_localization_allows_large_human_body(tmp_path) -> None:
    image_path = tmp_path / "explicit_body.png"
    from PIL import Image

    Image.new("RGB", (1000, 1000), "white").save(image_path)
    finding = PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category="explicit",
        label="human body",
        score=0.87,
        region=RegionMask(bbox=BBox(x=35, y=0, w=940, h=900), confidence=0.87),
        metadata={"prompt": "human body"},
    )
    violation = {
        "category": "explicit",
        "entity_label_en": "naked body",
        "sam_prompt_texts": ["naked body", "human body"],
        "center_point": [500, 500],
    }

    selected, rejected = qwen_safety_fusion._select_violation_candidates(
        [finding],
        violation,
        str(image_path),
        require_center_proximity=False,
    )

    assert rejected == []
    assert len(selected) == 1
    assert selected[0]["label"] == "human body"


def test_explicit_prompts_filter_clothing_context() -> None:
    prompts = qwen_safety_fusion._expand_category_prompts(
        "explicit",
        ["fishnet stockings", "high heels", "naked body"],
    )

    assert "fishnet stockings" not in prompts
    assert "high heels" not in prompts
    assert "naked body" in prompts
    assert "human body" in prompts


def test_other_nsfw_upper_body_prompts_expand_for_sam3() -> None:
    prompts = qwen_safety_fusion._expand_category_prompts(
        "other_nsfw",
        ["bare torso"],
    )

    assert prompts[:6] == ["bare torso", "nude torso", "chest and abdomen", "torso without head", "skin area", "human body"]
    assert prompts.index("upper body") > prompts.index("human body")


def test_qwen35_backfills_upper_body_as_redact_only() -> None:
    violations = qwen35_vl._violations(
        [
            {
                "category": "other_nsfw",
                "entity_label_en": "bare torso",
                "entity_label_zh": "裸露上身",
                "center_point": [530, 550],
                "confidence": 0.65,
            }
        ]
    )

    assert violations[0]["risk_subtype"] == "exposed_upper_body"
    assert violations[0]["decision_hint"] == "redact_only"
    assert "chest and abdomen" in violations[0]["sam_prompt_texts"]


def test_safety_violation_geometry_no_longer_scales_inside_fusion() -> None:
    violation = {
        "category": "other_nsfw",
        "entity_label_en": "bare torso",
        "center_point": [530, 550],
        "center_points": [[530, 550]],
        "rough_bbox": [280, 100, 780, 800],
    }

    normalized = qwen_safety_fusion._normalize_violation_geometry(violation, 2316, 3352)

    assert normalized is violation
    assert "qwen_geometry_normalized_from_1000" not in normalized


def test_qwen35_scales_visible_image_geometry_to_original_space() -> None:
    geometry = {
        "original_size": [2316, 3352],
        "qwen_input_size": [884, 1280],
        "max_side": 1280,
    }

    point = qwen35_vl._scale_point_from_qwen_input([530, 600], geometry)
    bbox = qwen35_vl._scale_bbox_from_qwen_input([300, 400, 450, 500], geometry)

    assert [round(value, 2) for value in point] == [1388.55, 1571.25]
    assert [round(value, 2) for value in bbox] == [785.97, 1047.5, 1178.96, 1309.38]


def test_safety_violation_geometry_does_not_scale_dangerous_objects() -> None:
    violation = {
        "category": "dangerous",
        "entity_label_en": "pistol",
        "center_point": [530, 550],
        "rough_bbox": [500, 530, 120, 60],
    }

    normalized = qwen_safety_fusion._normalize_violation_geometry(violation, 2316, 3352)

    assert normalized is violation
    assert "qwen_geometry_normalized_from_1000" not in normalized


def test_upper_body_local_bbox_never_overrides_sam3_geometry() -> None:
    assert not qwen_safety_fusion._local_bbox_is_usable(
        [200, 250, 600, 800],
        {"crop_bbox": [234, 0, 2007, 3352]},
        [297, 340, 466, 441],
        {
            "category": "other_nsfw",
            "entity_label_en": "bare torso",
            "risk_subtype": "exposed_upper_body",
            "center_point": [530, 550],
        },
    )


def test_upper_body_sam3_candidate_survives_qwen_local_review_rejection(tmp_path) -> None:
    image_path = tmp_path / "upper_body.png"
    from PIL import Image

    Image.new("RGB", (1000, 1000), "white").save(image_path)
    evidence = {
        "category": "other_nsfw",
        "label": "upper body",
        "bbox": [297.0, 340.0, 466.0, 441.0],
        "confidence": 0.80,
    }
    violation = {
        "category": "other_nsfw",
        "entity_label_en": "bare torso",
        "risk_subtype": "exposed_upper_body",
        "decision_hint": "redact_only",
        "center_point": [530.0, 550.0],
    }

    class DummyQwen:
        def verify_local_violation(self, crop_path, violation):
            return {"is_authentic_violation": False, "confidence": 0.20, "boundary_status": "uncertain"}

    class DummySAM:
        def refine_regions(self, image_path, regions):
            return regions

    result = qwen_safety_fusion._review_and_refine_candidate(
        DummySAM(),
        DummyQwen(),
        str(image_path),
        evidence,
        violation,
        1000,
        1000,
    )

    assert result is not None
    assert result["local_review_bypassed"] is True
    assert result["bbox"] == [297.0, 340.0, 466.0, 441.0]


def test_upper_body_limb_like_candidate_is_rejected() -> None:
    evidence = {
        "category": "other_nsfw",
        "label": "skin area",
        "bbox": [300.0, 100.0, 70.0, 430.0],
        "confidence": 0.9,
    }
    violation = {
        "category": "other_nsfw",
        "entity_label_en": "bare torso",
        "risk_subtype": "exposed_upper_body",
        "center_point": [335.0, 280.0],
    }

    ok, reason, _ = qwen_safety_fusion._violation_candidate_quality(
        evidence,
        violation,
        1000,
        1000,
        require_center_proximity=False,
    )

    assert not ok
    assert reason == "upper_body_candidate_limb_like"


def test_upper_body_partial_global_coverage_marks_review_required() -> None:
    evidence = {
        "category": "other_nsfw",
        "label": "bare torso",
        "bbox": [1073, 1827, 353, 414],
        "local_review": {"boundary_status": "complete"},
    }
    violation = {
        "category": "other_nsfw",
        "entity_label_en": "bare torso",
        "risk_subtype": "exposed_upper_body",
        "rough_bbox": [785.97, 1047.5, 1178.96, 1309.38],
        "center_points": [[1388.55, 1571.25]],
    }

    qwen_safety_fusion._apply_upper_body_global_coverage(evidence, violation)

    assert evidence["global_target_coverage"] == "partial"
    assert evidence["boundary_status"] == "truncated"
    assert evidence["review_required"] is True


def test_upper_body_sam3_candidate_does_not_bypass_arm_only_review(tmp_path) -> None:
    image_path = tmp_path / "upper_body.png"
    from PIL import Image

    Image.new("RGB", (1000, 1000), "white").save(image_path)
    evidence = {
        "category": "other_nsfw",
        "label": "upper body",
        "bbox": [297.0, 340.0, 466.0, 441.0],
        "confidence": 0.80,
    }
    violation = {
        "category": "other_nsfw",
        "entity_label_en": "bare torso",
        "risk_subtype": "exposed_upper_body",
        "decision_hint": "redact_only",
        "center_point": [530.0, 550.0],
    }

    class DummyQwen:
        def verify_local_violation(self, crop_path, violation):
            return {"is_authentic_violation": False, "confidence": 0.0, "reason_zh": "局部图中仅显示一只手臂，未包含躯干或裸露上身区域。"}

    class DummySAM:
        def refine_regions(self, image_path, regions):
            return regions

    result = qwen_safety_fusion._review_and_refine_candidate(
        DummySAM(),
        DummyQwen(),
        str(image_path),
        evidence,
        violation,
        1000,
        1000,
    )

    assert result is None


def test_explicit_body_violation_rejects_tiny_local_part_candidate() -> None:
    evidence = {"category": "explicit", "label": "genital area", "bbox": [456, 457, 136, 90]}
    violation = {
        "category": "explicit",
        "entity_label_en": "naked body",
        "sam_prompt_texts": ["naked body", "human body", "genital area"],
        "rough_bbox": [285, 10, 720, 985],
    }

    assert qwen_safety_fusion._explicit_body_violation_has_tiny_local_candidate(
        evidence,
        violation,
        1000,
        1000,
    )


def test_qwen_sam3_safety_fusion_uses_point_guided_object_violation(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "gun.png"
    mask_path = tmp_path / "mask.png"
    from PIL import Image

    Image.new("RGB", (240, 160), "white").save(image_path)
    Image.new("L", (240, 160), 255).save(mask_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.DANGEROUS],
            scores={"dangerous": 0.95},
            reason_codes=["SAFETY_DANGEROUS"],
            provider="qwen",
            metadata={
                "violations": [
                    {
                        "violation_id": "dangerous_1",
                        "category": "dangerous",
                        "entity_label_en": "pistol",
                        "entity_label_zh": "疑似手枪",
                        "center_point": [105, 88],
                        "confidence": 0.93,
                    }
                ],
                "category_details": {"dangerous": {"object_name_zh": "疑似手枪"}},
            },
        ),
    )
    point_calls = []

    def _detect_with_points(path, point_prompts):
        point_calls.extend(point_prompts)
        return [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="dangerous",
                score=0.89,
                region=RegionMask(
                    bbox=BBox(x=88, y=78, w=38, h=20),
                    mask_path=str(mask_path),
                    confidence=0.89,
                ),
                metadata={"prompt": "pistol", "point": {"x": 105, "y": 88, "label": True}},
            )
        ]

    monkeypatch.setattr(moderator._sam3, "detect_with_points", _detect_with_points)
    monkeypatch.setattr(
        moderator._sam3,
        "refine_regions",
        lambda path, regions: [
            regions[0].model_copy(
                update={
                    "bbox": BBox(x=90, y=80, w=32, h=16),
                    "mask_path": str(mask_path),
                    "confidence": 0.91,
                }
            )
        ],
    )
    monkeypatch.setattr(
        moderator._qwen,
        "verify_local_violation",
        lambda crop_path, violation: {
            "is_authentic_violation": True,
            "entity_label_en": "pistol",
            "entity_label_zh": "疑似手枪",
            "boundary_status": "complete",
            "local_bbox": [45.6, 24, 38, 20],
            "confidence": 0.86,
            "reason_zh": "局部图中可见疑似手枪。",
        },
    )

    result = moderator.moderate(str(image_path))

    assert point_calls
    assert point_calls[0]["point"] == [105.0, 88.0]
    evidence = result.metadata["evidence_regions"]
    assert len(evidence) == 1
    assert evidence[0]["violation_id"] == "dangerous_1"
    assert evidence[0]["mask_path"] == str(mask_path)
    assert evidence[0]["bbox"] == [90.0, 80.0, 32.0, 16.0]
    assert evidence[0]["sam3_refined"] is True
    assert evidence[0]["has_mask"] is True
    assert evidence[0]["mask_quality_score"] > 0.75
    assert evidence[0]["localization_status"] == "localized_by_qwen_point_sam3_refined_mask_verified"


def test_qwen_sam3_safety_fusion_falls_back_to_text_prompt_when_point_fails(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "gun.png"
    from PIL import Image

    Image.new("RGB", (240, 160), "white").save(image_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.DANGEROUS],
            scores={"dangerous": 0.95},
            reason_codes=["SAFETY_DANGEROUS"],
            provider="qwen",
            metadata={
                "violations": [
                    {
                        "violation_id": "dangerous_1",
                        "category": "dangerous",
                        "entity_label_en": "pistol",
                        "entity_label_zh": "疑似手枪",
                        "sam_prompt_text": "pistol",
                        "center_point": [30, 30],
                        "confidence": 0.93,
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(moderator._sam3, "detect_with_points", lambda path, point_prompts: [])
    exact_calls = []

    def _detect_exact(path, prompts):
        exact_calls.extend(prompts)
        return [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="dangerous",
                score=0.89,
                region=RegionMask(bbox=BBox(x=90, y=70, w=44, h=24), confidence=0.89),
                metadata={"prompt": "pistol"},
            )
        ]

    monkeypatch.setattr(moderator._sam3, "detect_exact_prompts", _detect_exact)
    monkeypatch.setattr(moderator._sam3, "detect_with_prompts", lambda *args, **kwargs: [])
    monkeypatch.setattr(moderator._sam3, "refine_regions", lambda path, regions: regions)
    monkeypatch.setattr(
        moderator._qwen,
        "verify_local_violation",
        lambda crop_path, violation: {
            "is_authentic_violation": True,
            "entity_label_en": "pistol",
            "entity_label_zh": "疑似手枪",
            "boundary_status": "complete",
            "local_bbox": [10, 8, 34, 18],
            "confidence": 0.86,
            "reason_zh": "局部图中可见疑似手枪。",
        },
    )

    result = moderator.moderate(str(image_path))

    assert exact_calls
    evidence = result.metadata["evidence_regions"]
    assert len(evidence) == 1
    assert evidence[0]["localization_attempt"] == "exact_text_prompt"
    assert evidence[0]["localization_status"] == "localized_by_qwen_point_sam3_verified"


def test_qwen_sam3_safety_fusion_rejects_drifting_refine(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "gun.png"
    from PIL import Image

    Image.new("RGB", (240, 160), "white").save(image_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.DANGEROUS],
            scores={"dangerous": 0.95},
            reason_codes=["SAFETY_DANGEROUS"],
            provider="qwen",
            metadata={
                "violations": [
                    {
                        "violation_id": "dangerous_1",
                        "category": "dangerous",
                        "entity_label_en": "pistol",
                        "entity_label_zh": "疑似手枪",
                        "center_point": [105, 88],
                        "confidence": 0.93,
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(
        moderator._sam3,
        "detect_with_points",
        lambda path, point_prompts: [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="dangerous",
                score=0.89,
                region=RegionMask(bbox=BBox(x=88, y=78, w=38, h=20), confidence=0.89),
                metadata={"prompt": "pistol"},
            )
        ],
    )
    monkeypatch.setattr(
        moderator._sam3,
        "refine_regions",
        lambda path, regions: [
            regions[0].model_copy(update={"bbox": BBox(x=210, y=5, w=6, h=5), "confidence": 0.95})
        ],
    )
    monkeypatch.setattr(
        moderator._qwen,
        "verify_local_violation",
        lambda crop_path, violation: {
            "is_authentic_violation": True,
            "entity_label_en": "pistol",
            "entity_label_zh": "疑似手枪",
            "boundary_status": "complete",
            "local_bbox": [5, 5, 30, 15],
            "confidence": 0.86,
            "reason_zh": "局部图中可见疑似手枪。",
        },
    )

    result = moderator.moderate(str(image_path))

    evidence = result.metadata["evidence_regions"][0]
    assert evidence["sam3_refine_rejected"] is True
    assert evidence["bbox"] != [210.0, 5.0, 6.0, 5.0]


def test_qwen_sam3_safety_fusion_explicit_uses_concrete_prompts_and_keeps_sam3_mask(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "explicit.png"
    mask_path = tmp_path / "explicit_mask.png"
    from PIL import Image

    Image.new("RGB", (320, 240), "white").save(image_path)
    Image.new("L", (320, 240), 255).save(mask_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.EXPLICIT],
            scores={"explicit": 0.96},
            reason_codes=["SAFETY_EXPLICIT"],
            provider="qwen",
            metadata={
                "violations": [
                    {
                        "violation_id": "explicit_1",
                        "category": "explicit",
                        "entity_label_en": "nude male",
                        "entity_label_zh": "裸露男性",
                        "sam_prompt_texts": ["naked body"],
                        "center_points": [[160, 120]],
                        "rough_bbox": [80, 20, 160, 200],
                        "confidence": 0.96,
                    }
                ],
            },
        ),
    )
    point_calls = []

    def _detect_with_points(path, point_prompts):
        point_calls.extend(point_prompts)
        return [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="explicit",
                score=0.82,
                region=RegionMask(
                    bbox=BBox(x=120, y=90, w=70, h=60),
                    mask_path=str(mask_path),
                    confidence=0.82,
                ),
                metadata={"prompt": "naked body"},
            )
        ]

    monkeypatch.setattr(moderator._sam3, "detect_with_points", _detect_with_points)
    monkeypatch.setattr(moderator._sam3, "detect_exact_prompts", lambda path, prompts: [])
    monkeypatch.setattr(moderator._sam3, "detect_with_prompts", lambda *args, **kwargs: [])
    monkeypatch.setattr(moderator._sam3, "refine_regions", lambda path, regions: regions)
    monkeypatch.setattr(
        moderator._qwen,
        "verify_local_violation",
        lambda crop_path, violation: {
            "is_authentic_violation": True,
            "entity_label_en": "nude male",
            "entity_label_zh": "裸露男性",
            "boundary_status": "complete",
            "local_bbox": None,
            "confidence": 0.78,
            "reason_zh": "局部图包含裸露违规区域。",
        },
    )

    result = moderator.moderate(str(image_path))

    prompts = {item["text"] for item in point_calls}
    assert {"human body", "naked body", "nude torso"} <= prompts
    evidence = result.metadata["evidence_regions"][0]
    assert evidence["bbox"] == [120.0, 90.0, 70.0, 60.0]
    assert evidence["mask_path"] == str(mask_path)
    assert evidence["localization_status"] == "localized_by_qwen_point_sam3_refined_mask_verified"


def test_qwen_sam3_safety_fusion_filters_abstract_sam3_prompts(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "explicit.png"
    from PIL import Image

    Image.new("RGB", (320, 240), "white").save(image_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.EXPLICIT],
            scores={"explicit": 0.96},
            reason_codes=["SAFETY_EXPLICIT"],
            provider="qwen",
            metadata={
                "violations": [
                    {
                        "violation_id": "explicit_1",
                        "category": "explicit",
                        "entity_label_en": "nude male except head",
                        "entity_label_zh": "裸露男性",
                        "sam_prompt_texts": ["body below head", "色情裸露区域", "person holding weapon", "sex toy", "naked body"],
                        "center_points": [[160, 120]],
                        "rough_bbox": [80, 20, 160, 200],
                        "confidence": 0.96,
                    }
                ],
            },
        ),
    )
    point_calls = []

    def _detect_with_points(path, point_prompts):
        point_calls.extend(point_prompts)
        return []

    monkeypatch.setattr(moderator._sam3, "detect_with_points", _detect_with_points)
    monkeypatch.setattr(moderator._sam3, "detect_exact_prompts", lambda path, prompts: [])
    monkeypatch.setattr(moderator._sam3, "detect_with_prompts", lambda *args, **kwargs: [])

    result = moderator.moderate(str(image_path))

    prompts = {item["text"] for item in point_calls}
    assert "body below head" not in prompts
    assert "色情裸露区域" not in prompts
    assert "nude male except head" not in prompts
    assert "person holding weapon" not in prompts
    assert "sex toy" in prompts
    assert {"human body", "exposed body part", "sex toy"} <= prompts
    assert result.metadata["evidence_regions"][0]["localization_status"] == "coarse_localization_from_qwen_rough_bbox"


def test_qwen_sam3_safety_fusion_allows_new_short_noun_prompts(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "dangerous.png"
    from PIL import Image

    Image.new("RGB", (320, 240), "white").save(image_path)
    moderator = QwenSAM3SafetyFusionModerator()
    monkeypatch.setattr(
        moderator._qwen,
        "moderate",
        lambda path: PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.DANGEROUS],
            scores={"dangerous": 0.9},
            reason_codes=["SAFETY_DANGEROUS"],
            provider="qwen",
            metadata={
                "violations": [
                    {
                        "violation_id": "dangerous_1",
                        "category": "dangerous",
                        "entity_label_en": "crossbow",
                        "entity_label_zh": "弩",
                        "sam_prompt_texts": ["crossbow", "person holding weapon", "dangerous object"],
                        "center_points": [[160, 120]],
                        "rough_bbox": [100, 80, 100, 80],
                        "confidence": 0.9,
                    }
                ],
            },
        ),
    )
    point_calls = []
    monkeypatch.setattr(moderator._sam3, "detect_with_points", lambda path, prompts: point_calls.extend(prompts) or [])
    monkeypatch.setattr(moderator._sam3, "detect_exact_prompts", lambda path, prompts: [])
    monkeypatch.setattr(moderator._sam3, "detect_with_prompts", lambda *args, **kwargs: [])

    moderator.moderate(str(image_path))

    prompts = {item["text"] for item in point_calls}
    assert "crossbow" in prompts
    assert "person holding weapon" not in prompts
    assert "dangerous object" not in prompts
