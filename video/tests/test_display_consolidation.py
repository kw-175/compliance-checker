from __future__ import annotations

from pathlib import Path

from PIL import Image

from video.application.clip_moderator import ClipModerationConfig, _event_instance_records, moderate_clip_windows
from video.application.display_consolidation import consolidate_display_risks
from video.domain.models import FrameReference, TimeSpan, VideoRiskAnnotation


def _risk(
    category: str,
    *,
    source_modality: str = "visual",
    start_ms: int = 0,
    end_ms: int = 240,
    confidence: float = 0.95,
    severity: str = "high",
    regions: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> VideoRiskAnnotation:
    return VideoRiskAnnotation(
        source_modality=source_modality,
        category=category,
        operator_id="VVIS_CSA_003" if category.startswith("content.") else "",
        source_operator_id="CSA_003" if category.startswith("content.") else "",
        target_type=category,
        severity=severity,
        confidence=confidence,
        span=TimeSpan(start_ms=start_ms, end_ms=end_ms),
        frame_ids=[f"frame_{start_ms}"],
        regions=regions or [],
        reason_codes=["SAFETY_GRAPHIC_VIOLENCE"] if category.startswith("content.") else [],
        metadata=metadata or {},
    )


def test_physical_conflict_display_uses_qwen_event_without_suppressing_raw_candidates() -> None:
    risks = [
        _risk("content.graphic_violence", source_modality="video_clip", start_ms=0, end_ms=2120, severity="critical"),
        _risk("content.violence", start_ms=0, end_ms=240),
        _risk(
            "visual.dangerous",
            start_ms=240,
            end_ms=480,
            metadata={"risk_subtype_zh": "肢体冲突/暴力行为", "object_name_zh": "多人肢体冲突"},
        ),
        _risk(
            "visual.dangerous",
            start_ms=1200,
            end_ms=1440,
            regions=[
                {
                    "frame_id": "frame_1200",
                    "bbox": {"x": 357, "y": 279, "w": 121, "h": 56},
                    "source": "qwen_rough_bbox_fallback",
                    "mask_quality_score": 0.15,
                    "instance_label_zh": "疑似手枪",
                    "instance_label_en": "pistol",
                }
            ],
            metadata={"instance_label_zh": "疑似手枪"},
        ),
        _risk(
            "privacy.face",
            start_ms=0,
            end_ms=480,
            confidence=0.61,
            severity="medium",
            regions=[
                {
                    "frame_id": "frame_0",
                    "bbox": {"x": 200, "y": 185, "w": 25, "h": 38},
                    "confidence": 0.61,
                }
            ],
            metadata={"face_filter_decision": "keep", "face_visible_keypoint_count": 5, "instance_label_zh": "人脸"},
        ),
    ]

    display = consolidate_display_risks(risks)

    assert len(display) == 1
    assert display[0].category == "content.graphic_violence"
    assert display[0].metadata["display_label_zh"] == "暴力斗殴"
    assert display[0].span.start_ms == 0
    assert display[0].span.end_ms == 2120
    assert "display_role" not in risks[1].metadata
    assert "display_role" not in risks[2].metadata
    assert "display_role" not in risks[3].metadata
    assert "display_role" not in risks[4].metadata
    assert risks[4].eligible_for_redaction is True


def test_identifiable_face_is_kept_as_display_privacy_risk() -> None:
    risks = [
        _risk("content.graphic_violence", source_modality="video_clip", start_ms=0, end_ms=1000, severity="critical"),
        _risk(
            "privacy.face",
            start_ms=200,
            end_ms=900,
            confidence=0.91,
            severity="medium",
            regions=[
                {
                    "frame_id": "frame_200",
                    "bbox": {"x": 120, "y": 80, "w": 96, "h": 112},
                    "confidence": 0.91,
                }
            ],
            metadata={
                "face_filter_decision": "keep",
                "face_visible_keypoint_count": 5,
                "identifiability_score": 0.9,
                "instance_label_zh": "人脸",
                "tracking": {"redaction_ready": True, "redaction_scope": "sam3_video_track"},
            },
        ),
    ]

    display = consolidate_display_risks(risks)

    assert [risk.metadata["display_role"] for risk in display] == ["primary_event", "primary_privacy"]
    assert display[1].category == "privacy.face"


def test_qwen_physical_conflict_instances_merge_to_one_track_seed() -> None:
    event = {
        "category": "physical_conflict",
        "confidence": 0.93,
        "start_time": 0.0,
        "end_time": 2.0,
        "instances": [
            {
                "category": "participant",
                "entity_label_zh": "参与打斗的人",
                "keyframes": [{"frame_index": 1, "bbox": [10, 20, 30, 40], "confidence": 0.8}],
            },
            {
                "category": "person",
                "entity_label_zh": "参与打斗的人",
                "keyframes": [{"frame_index": 1, "bbox": [35, 25, 20, 30], "confidence": 0.7}],
            },
            {
                "category": "weapon",
                "entity_label_zh": "刀具",
                "keyframes": [{"frame_index": 1, "bbox": [90, 20, 15, 25], "confidence": 0.9}],
            },
        ],
    }

    records = _event_instance_records(event)

    assert records[0]["category"] == "conflict_region"
    assert records[0]["entity_label_zh"] == "暴力斗殴区域"
    assert records[0]["keyframes"][0]["bbox"] == [10.0, 20.0, 45.0, 40.0]
    assert records[1]["category"] == "weapon"


def test_physical_conflict_display_event_attaches_concrete_spatial_track() -> None:
    risks = [
        _risk("content.graphic_violence", source_modality="video_clip", start_ms=0, end_ms=2000, severity="critical"),
        _risk(
            "content.violence",
            source_modality="visual",
            start_ms=0,
            end_ms=1000,
            regions=[
                {
                    "frame_id": "frame_0",
                    "pts_ms": 0,
                    "bbox": {"x": 10, "y": 20, "w": 80, "h": 90},
                    "confidence": 0.9,
                    "source": "qwen_point_sam3_local_review",
                    "instance_label_zh": "打斗人群",
                    "instance_label_en": "conflict_region",
                },
                {
                    "frame_id": "frame_1",
                    "pts_ms": 250,
                    "bbox": {"x": 16, "y": 24, "w": 78, "h": 88},
                    "confidence": 0.94,
                    "source": "sam3_video_tracker",
                    "instance_label_zh": "打斗人群",
                    "instance_label_en": "conflict_region",
                },
            ],
            metadata={
                "video_role": "object_instance",
                "instance_label_zh": "暴力斗殴区域",
                "instance_label_en": "conflict_region",
                "trackable_target_type": "conflict_region",
                "tracking_backend": "sam3_video_tracker",
                "tracking": {
                    "tracking_backend": "sam3_video_tracker",
                    "bbox_series": [
                        {
                            "frame_id": "frame_1",
                            "pts_ms": 250,
                            "bbox": {"x": 16, "y": 24, "w": 78, "h": 88},
                            "confidence": 0.94,
                            "source": "sam3_video_tracker",
                        }
                    ],
                    "redaction_series": [
                        {
                            "frame_id": "frame_1",
                            "pts_ms": 250,
                            "bbox": {"x": 16, "y": 24, "w": 78, "h": 88},
                            "confidence": 0.94,
                            "source": "sam3_video_tracker",
                        }
                    ],
                },
            },
        ),
    ]

    display = consolidate_display_risks(risks)

    assert len(display) == 1
    assert display[0].metadata["display_label_zh"] == "暴力斗殴"
    assert display[0].target_type == "conflict_region"
    assert display[0].spatial_precision == "bbox"
    assert display[0].metadata["tracking"]["tracking_backend"] == "sam3_video_tracker"
    assert len(display[0].metadata["tracking"]["bbox_series"]) >= 1
    assert len(display[0].regions) == 2


def test_physical_conflict_display_event_filters_rejected_regions_not_entire_track() -> None:
    risks = [
        _risk(
            "content.graphic_violence",
            source_modality="video_clip",
            start_ms=0,
            end_ms=2400,
            severity="critical",
            metadata={"event_subtype": "physical_conflict"},
        ),
        _risk(
            "content.violence",
            source_modality="visual",
            start_ms=0,
            end_ms=2400,
            regions=[
                {
                    "frame_id": "frame_0",
                    "pts_ms": 0,
                    "bbox": {"x": 20, "y": 30, "w": 140, "h": 90},
                    "confidence": 0.86,
                    "source": "qwen_point_sam3_local_review",
                    "localization_status": "localized_by_qwen_point_sam3_refined_mask_verified",
                    "mask_quality_score": 0.78,
                    "instance_label_zh": "暴力斗殴区域",
                    "instance_label_en": "conflict_region",
                },
                {
                    "frame_id": "frame_1",
                    "pts_ms": 400,
                    "bbox": {"x": 18, "y": 28, "w": 148, "h": 96},
                    "confidence": 0.15,
                    "source": "qwen_rough_bbox_fallback",
                    "localization_status": "coarse_localization_after_sam3_rejections",
                    "mask_quality_score": 0.15,
                    "instance_label_zh": "暴力斗殴区域",
                    "instance_label_en": "conflict_region",
                },
                {
                    "frame_id": "frame_2",
                    "pts_ms": 800,
                    "bbox": {"x": 24, "y": 36, "w": 136, "h": 84},
                    "confidence": 0.9,
                    "source": "sam3_video_tracker",
                    "localization_status": "localized_by_qwen_point_sam3_refined_mask_verified",
                    "mask_quality_score": 0.88,
                    "instance_label_zh": "暴力斗殴区域",
                    "instance_label_en": "conflict_region",
                },
            ],
            metadata={
                "video_role": "object_instance",
                "instance_label_zh": "暴力斗殴区域",
                "instance_label_en": "conflict_region",
                "trackable_target_type": "conflict_region",
                "tracking_backend": "sam3_video_tracker",
                "tracking": {
                    "tracking_backend": "sam3_video_tracker",
                    "quality_flags": ["low_quality_coarse_localization"],
                    "bbox_series": [
                        {
                            "frame_id": "frame_2",
                            "pts_ms": 800,
                            "bbox": {"x": 24, "y": 36, "w": 136, "h": 84},
                            "confidence": 0.9,
                            "source": "sam3_video_tracker",
                        }
                    ],
                    "redaction_series": [
                        {
                            "frame_id": "frame_2",
                            "pts_ms": 800,
                            "bbox": {"x": 24, "y": 36, "w": 136, "h": 84},
                            "confidence": 0.9,
                            "source": "sam3_video_tracker",
                        }
                    ],
                },
            },
        ),
    ]

    display = consolidate_display_risks(risks)

    assert len(display) == 1
    assert display[0].spatial_precision == "bbox"
    assert display[0].localization_status == "event_with_tracked_regions"
    assert len(display[0].regions) == 2
    assert all(region["source"] != "qwen_rough_bbox_fallback" for region in display[0].regions)
    assert len(display[0].metadata["tracking"]["bbox_series"]) >= 2


def test_clip_moderation_second_pass_localizes_text_only_physical_conflict(monkeypatch, tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (120, 90), (255, 255, 255)).save(frame_path)
    frame = FrameReference(frame_index=0, pts_ms=0, image_uri=str(frame_path), metadata={"duration_ms": 250})
    calls = []

    def fake_post_json(config, payload):
        calls.append(payload["task"])
        if payload["task"] == "video_safety_detection_with_temporal_localization":
            return {
                "events": [
                    {
                        "category": "physical_conflict",
                        "confidence": 0.95,
                        "start_time": 0.0,
                        "end_time": 0.25,
                        "evidence": "左下角两人发生肢体纠缠。",
                    }
                ]
            }
        assert payload["task"] == "video_safety_event_region_localization"
        return {
            "instances": [
                {
                    "category": "conflict_region",
                    "entity_label_zh": "暴力斗殴区域",
                    "entity_label_en": "conflict_region",
                    "confidence": 0.88,
                    "keyframes": [{"frame_index": 0, "bbox": [5, 40, 50, 35], "confidence": 0.88}],
                }
            ]
        }

    monkeypatch.setattr("video.application.clip_moderator._post_json", fake_post_json)

    risks, audits = moderate_clip_windows(
        [frame],
        [{"window_id": "clip_0001", "start_ms": 0, "end_ms": 250, "frame_ids": [frame.frame_id]}],
        ClipModerationConfig(max_frames=1),
    )

    assert calls == ["video_safety_detection_with_temporal_localization", "video_safety_event_region_localization"]
    assert len(risks) == 2
    assert risks[1].source_modality == "video_object"
    assert risks[1].target_type == "conflict_region"
    assert risks[1].regions[0]["bbox"] == {"x": 5.0, "y": 40.0, "w": 50.0, "h": 35.0}
    assert audits[0]["localization_attempts"][0]["status"] == "completed"
