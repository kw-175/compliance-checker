from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import uuid
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from audio.models.schemas import Decision as AudioDecision
from picture.domain.enums import DecisionType, FindingType, JobStatus, RouteType, SafetyCategory
from picture.domain.models import BBox, PictureFinding, PictureJob, PictureModerationResult, PicturePolicyResult, RegionMask, SourceSpec
from video.application import services
from video.application import picture_api_client
from video.application.action_planner import build_action_plan
from video.application.clip_moderator import ClipModerationConfig, moderate_clip_windows
from video.application.risk_builder import build_frame_risks
from video.application.sam3_video_tracker import Sam3VideoTrackerConfig, enrich_with_sam3_video_tracking
from video.application.temporal_aggregation import aggregate_risks
from video.application.tracking import enrich_risk_tracks
from video.config.settings import Settings
from video.domain.enums import VideoDecisionType, VideoGovernanceDecision, VideoJobStatus
from video.domain.models import TaskContext, TimeSpan, VideoActionPlan, VideoGovernancePolicyResult, VideoRedactionOperation, VideoRiskAnnotation
from video.domain.taxonomy import map_moderation_category
from video.pipeline import VideoCompliancePipeline
from ops import sam3_api


def _make_frame(color: tuple[int, int, int], offset: int) -> Image.Image:
    image = Image.new("RGB", (800, 600), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([100 + offset, 60, 350 + offset, 240], fill=(20, 20, 20))
    draw.rectangle([420, 360, 600, 520], fill=(240, 240, 240))
    return image


def _write_gif(path: Path) -> None:
    frames = [_make_frame((240, 240, 240), 0), _make_frame((230, 230, 230), 20), _make_frame((220, 220, 220), 40)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[150, 150, 150], loop=0)


def _write_frame_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index, color in enumerate([(240, 240, 240), (230, 230, 230), (220, 220, 220)]):
        _make_frame(color, index * 15).save(path / f"frame_{index:03d}.png")


def _write_sidecar_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def _make_case_dir(name: str) -> Path:
    base = Path("video_test_runs") / f"{name}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _make_settings(base: Path) -> Settings:
    return Settings(
        work_dir=base / "video_work",
        storage_base_path=base / "video_storage",
        max_workers=1,
        clip_moderation_enabled=False,
        sam3_video_tracking_enabled=False,
    )


def test_video_temporal_and_tracking_capabilities_are_enabled_by_default():
    assert Settings.model_fields["clip_moderation_enabled"].default is True
    assert Settings.model_fields["sam3_video_tracking_enabled"].default is True


def _mock_picture_options(route_hint: str) -> dict[str, object]:
    return {
        "route_hint": route_hint,
        "picture_ocr_provider": "mock",
        "picture_pii_provider": "mock",
        "picture_text_compliance_provider": "none",
        "picture_safety_provider": "mock",
        "picture_vision_provider": "mock",
        "picture_segmentation_provider": "mock",
    }


class _MockPictureComplianceApiClient:
    calls: list[dict[str, object]] = []

    def __init__(self, config=None):
        self.config = config

    def check_health(self) -> None:
        return None

    def run_frame(self, *, image_uri: str, tenant_id: str, profile: str, options: dict[str, object]) -> PictureJob:
        self.calls.append({"image_uri": image_uri, "tenant_id": tenant_id, "profile": profile, "options": dict(options)})
        route_hint = str(options.get("route_hint") or "mixed").lower()
        route = {
            "document": RouteType.DOCUMENT,
            "natural": RouteType.NATURAL,
            "mixed": RouteType.MIXED,
        }.get(route_hint, RouteType.MIXED)
        findings = [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="qr_code",
                label="qr_code",
                score=0.92,
                region=RegionMask(bbox=BBox(x=420, y=360, w=180, h=160), confidence=0.92),
                reason_code="VISION_QR_CODE",
                provider="mock_picture_api",
                metadata={"operator_id": "VPI_006"},
            ),
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="signature",
                label="signature",
                score=0.88,
                region=RegionMask(bbox=BBox(x=100, y=60, w=250, h=180), confidence=0.88),
                reason_code="VISION_SIGNATURE",
                provider="mock_picture_api",
                metadata={"operator_id": "VPI_004"},
            ),
        ]
        enabled_visual_types = set(str(item).lower() for item in (options.get("visual_sensitive_object_types") or []))
        if not enabled_visual_types or "face" in enabled_visual_types:
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category="face",
                    label="face",
                    score=0.9,
                    region=RegionMask(bbox=BBox(x=160, y=120, w=120, h=120), confidence=0.9),
                    reason_code="VISION_FACE",
                    provider="mock_picture_api",
                    metadata={"operator_id": "VPI_001"},
                )
            )
        if route == RouteType.NATURAL:
            findings.append(
                PictureFinding(
                    finding_type=FindingType.SAFETY,
                    category="explicit",
                    label="explicit",
                    score=0.86,
                    reason_code="SAFETY_EXPLICIT",
                    provider="mock_picture_api",
                    metadata={"operator_id": "CSA_002"},
                )
            )
        return PictureJob(
            tenant_id=tenant_id,
            status=JobStatus.DONE,
            source=SourceSpec(uri=image_uri, mime_type="image/png"),
            route=route,
            profile=profile,
            options=dict(options),
            findings=findings,
            policy_result=PicturePolicyResult(
                decision=DecisionType.PASS_REDACTED,
                reason_codes=[finding.reason_code for finding in findings if finding.reason_code],
            ),
            compliant_image_uri=image_uri,
            overlay_image_uri=image_uri,
            provider_versions={"picture_api": "mock"},
        )


@pytest.fixture(autouse=True)
def _mock_picture_api(monkeypatch: pytest.MonkeyPatch):
    _MockPictureComplianceApiClient.calls = []
    monkeypatch.setattr(services, "PictureComplianceApiClient", _MockPictureComplianceApiClient)
    monkeypatch.setattr(picture_api_client, "PictureComplianceApiClient", _MockPictureComplianceApiClient)


def test_gif_pipeline_produces_redacted_asset_by_default():
    base = _make_case_dir("gif")
    input_path = base / "sample_video.gif"
    _write_gif(input_path)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options=_mock_picture_options("mixed"))
    assert result.status == VideoJobStatus.DONE
    assert result.policy_result is not None
    assert result.policy_result.decision == VideoDecisionType.PASS_REDACTED
    assert result.governance_result is not None
    assert result.action_plan is not None
    assert result.action_plan.render_redacted_asset is True
    assert result.asset is not None
    assert result.asset.compliant_video_uri is not None
    assert result.asset.report_uri is not None
    assert len(result.risk_annotations) > 0
    assert len(result.findings) > 0


def test_gif_pipeline_builds_operator_driven_plan_for_detected_risks():
    base = _make_case_dir("unsafe")
    input_path = base / "sample_unsafe_explicit.gif"
    _write_gif(input_path)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options=_mock_picture_options("natural"))
    assert result.status == VideoJobStatus.DONE
    assert result.governance_result is not None
    assert result.governance_result.decision in {
        VideoGovernanceDecision.TRANSFORM_REQUIRED,
        VideoGovernanceDecision.REVIEW_REQUIRED,
    }
    assert result.action_plan is not None
    assert any(risk.operator_id for risk in result.risk_annotations)


def test_directory_input_and_audio_sidecar():
    base = _make_case_dir("dir_audio")
    frame_dir = base / "sequence"
    _write_frame_directory(frame_dir)
    _write_sidecar_wav(frame_dir / "audio.wav")
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(frame_dir), tenant_id="test-tenant", options=_mock_picture_options("document"))
    assert result.status == VideoJobStatus.DONE
    assert result.asset is not None
    assert result.asset.audio_uri is not None
    assert "audio_pipeline" in result.step_latencies
    assert result.governance_result is not None
    assert result.action_plan is not None


def test_mp4_sequence_support_with_mocked_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    base = _make_case_dir("mp4_sequence")
    input_path = base / "sample_video.mp4"
    input_path.write_bytes(b"fake-mp4")

    def fake_run_command(command, timeout=300, ok_returncodes=(0,)):
        if "ffprobe" in command[0]:
            payload = {
                "streams": [
                    {"codec_type": "video", "avg_frame_rate": "12/1", "width": 800, "height": 600, "nb_frames": "24"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
                "format": {"duration": "2.0"},
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if "ffmpeg" in command[0]:
            pattern = Path(command[-1])
            pattern.parent.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                _make_frame((240 - index * 10, 240, 240), index * 10).save(pattern.parent / f"sample_video_frame_{index:05d}.png")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return None

    monkeypatch.setattr(services, "run_command", fake_run_command)
    sequence = services.load_sequence(str(input_path), base / "work", frame_stride=4, ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe")
    assert sequence.source_kind == "video_container"
    assert sequence.has_native_audio is True
    assert len(sequence.frames) == 3
    assert sequence.total_duration_ms == 2000


def test_mp4_render_support_with_mocked_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    base = _make_case_dir("mp4_render")
    frames_dir = base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    frame_refs = []
    for index in range(2):
        frame_path = frames_dir / f"frame_{index:03d}.png"
        _make_frame((240, 240 - index * 10, 240), index * 20).save(frame_path)
        frame_paths.append(str(frame_path.resolve()))
        frame_refs.append(services.FrameReference(frame_index=index, pts_ms=index * 500, image_uri=str(frame_path.resolve()), metadata={"duration_ms": 500}))
    sequence = services.SequenceBundle("video_container", frame_refs, [500, 500], 2, 1000, str(base / "sample.mp4"), 2.0, True)
    frame_jobs = [PictureJob(source=SourceSpec(uri=frame_path, mime_type="image/png")) for frame_path in frame_paths]

    def fake_run_command(command, timeout=300, ok_returncodes=(0,)):
        if "ffmpeg" in command[0]:
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return None

    monkeypatch.setattr(services, "run_command", fake_run_command)
    compliant_path, preview_path = services.render_sequence_outputs(sequence, frame_jobs, base / "rendered_output", decision=VideoDecisionType.PASS_REDACTED, ffmpeg_bin="ffmpeg")
    assert compliant_path is not None and compliant_path.endswith(".mp4")
    assert Path(compliant_path).exists()
    assert preview_path is not None and Path(preview_path).exists()


def test_render_sequence_outputs_applies_track_redaction_series() -> None:
    base = _make_case_dir("track_redaction_render")
    frame_path = base / "frame.png"
    Image.new("RGB", (120, 90), (255, 255, 255)).save(frame_path)
    frame = services.FrameReference(frame_index=0, pts_ms=0, image_uri=str(frame_path.resolve()), metadata={"duration_ms": 500})
    sequence = services.SequenceBundle(
        source_kind="frame_directory",
        frames=[frame],
        frame_durations_ms=[500],
        total_input_frames=1,
        total_duration_ms=500,
    )
    frame_job = PictureJob(source=SourceSpec(uri=str(frame_path.resolve()), mime_type="image/png"))
    action_plan = VideoActionPlan(
        render_redacted_asset=True,
        operations=[
            VideoRedactionOperation(
                risk_id="risk_face",
                modality="visual",
                operation="black_box",
                start_ms=0,
                end_ms=500,
                regions=[],
                metadata={
                    "redaction_ready": True,
                    "redaction_scope": "sampled_frame",
                    "redaction_series": [
                        {
                            "frame_id": frame.frame_id,
                            "pts_ms": 0,
                            "bbox": {"x": 20, "y": 15, "w": 40, "h": 30},
                            "confidence": 0.9,
                            "source": "detected",
                        }
                    ],
                },
            )
        ],
    )

    compliant_path, _ = services.render_sequence_outputs(
        sequence,
        [frame_job],
        base / "rendered_output",
        decision=VideoDecisionType.PASS_REDACTED,
        render_preview=False,
        action_plan=action_plan,
    )

    assert compliant_path is not None
    image = Image.open(compliant_path).convert("RGB")
    assert image.getpixel((30, 25)) == (0, 0, 0)


def test_mp4_pipeline_support_with_mocked_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    base = _make_case_dir("mp4_pipeline")
    input_path = base / "sample_video.mp4"
    input_path.write_bytes(b"fake-mp4")

    def fake_run_command(command, timeout=300, ok_returncodes=(0,)):
        if "ffprobe" in command[0]:
            payload = {
                "streams": [
                    {"codec_type": "video", "avg_frame_rate": "10/1", "width": 800, "height": 600, "nb_frames": "20"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
                "format": {"duration": "2.0"},
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if "ffmpeg" in command[0]:
            output = Path(command[-1])
            if "%05d" in str(output):
                output.parent.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    _make_frame((240, 240 - index * 10, 240), index * 20).save(output.parent / f"sample_video_frame_{index:05d}.png")
            elif output.suffix == ".wav":
                output.parent.mkdir(parents=True, exist_ok=True)
                _write_sidecar_wav(output)
            elif output.suffix == ".mp4":
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"mp4")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return None

    monkeypatch.setattr(services, "run_command", fake_run_command)
    monkeypatch.setattr("video.pipeline.run_audio_sidecar", lambda audio_path, work_dir, config_overrides=None: (AudioDecision.ALLOW, work_dir, None))
    pipeline = VideoCompliancePipeline(settings=_make_settings(base).model_copy(update={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"}))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options=_mock_picture_options("mixed"))
    assert result.status == VideoJobStatus.DONE
    assert result.asset is not None
    assert result.asset.compliant_video_uri is not None
    assert result.governance_result is not None
    assert result.action_plan is not None
    assert result.action_plan.render_redacted_asset is True


def test_operator_selection_excludes_face_for_face_training():
    base = _make_case_dir("face_operator_selection")
    input_path = base / "sample_video.gif"
    _write_gif(input_path)
    options = _mock_picture_options("mixed")
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(
        str(input_path),
        tenant_id="test-tenant",
        operator_selection={
            "disabled_operator_ids": ["VPI_001"],
            "disabled_target_types": ["face"],
            "preserved_training_targets": ["face"],
        },
        options=options,
    )
    assert result.status == VideoJobStatus.DONE
    assert all(risk.category != "privacy.face" for risk in result.risk_annotations)
    assert any(risk.category == "privacy.qr_code" for risk in result.risk_annotations)
    assert any(risk.category == "privacy.signature" for risk in result.risk_annotations)
    assert any(target.target_type == "face" for target in result.preserved_training_targets)


def test_video_operator_selection_is_forwarded_to_picture_api():
    base = _make_case_dir("operator_forwarding")
    input_path = base / "sample_video.gif"
    _write_gif(input_path)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(
        str(input_path),
        tenant_id="test-tenant",
        operator_selection={
            "visual_sensitive_object_operator_ids": ["VVIS_VPI_006"],
            "visual_safety_operator_ids": ["VVIS_CSA_003"],
            "privacy_operator_ids": ["VTXT_PII_002"],
            "content_safety_operator_ids": ["VTXT_CSA_001"],
        },
        options={"route_hint": "mixed"},
    )
    assert result.status == VideoJobStatus.DONE
    assert _MockPictureComplianceApiClient.calls
    frame_options = _MockPictureComplianceApiClient.calls[0]["options"]
    assert frame_options["visual_sensitive_object_operator_ids"] == ["VPI_006"]
    assert frame_options["visual_sensitive_object_types"] == ["qr_code"]
    assert frame_options["visual_safety_operator_ids"] == ["CSA_003"]
    assert frame_options["visual_safety_target_labels"] == ["visual.violent"]
    assert frame_options["privacy_operator_ids"] == ["PII_002"]
    assert set(frame_options["privacy_target_types"]) >= {"phone", "email"}
    assert frame_options["content_safety_operator_ids"] == ["CSA_001"]
    assert frame_options["content_safety_target_labels"] == ["content.political"]
    assert frame_options["enable_visual_sensitive_object_detection"] is True
    assert frame_options["enable_visual_safety_detection"] is True
    assert frame_options["enable_text_privacy_detection"] is True
    assert frame_options["enable_text_content_detection"] is True


def test_video_maps_fight_metadata_to_violence_before_dangerous() -> None:
    moderation = PictureModerationResult(
        is_safe=False,
        categories=[SafetyCategory.DANGEROUS],
        metadata={
            "explanation": "图片显示多人肢体冲突，属于暴力行为。",
            "category_details": {
                "dangerous": {
                    "risk_subtype_zh": "肢体冲突/暴力行为",
                    "object_name_zh": "多人肢体冲突",
                }
            },
        },
    )

    assert map_moderation_category(moderation, "SAFETY_DANGEROUS") == "content.graphic_violence"


def test_frame_safety_evidence_becomes_trackable_object_instance() -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=0, image_uri="/tmp/frame.png", metadata={"duration_ms": 250})
    finding = PictureFinding(
        finding_type=FindingType.SAFETY,
        category="dangerous",
        label="违法危险内容：疑似手枪",
        score=0.91,
        region=RegionMask(bbox=BBox(x=100, y=80, w=40, h=24), confidence=0.91),
        reason_code="SAFETY_DANGEROUS",
        provider="qwen35_sam3_fusion",
        metadata={
            "entity_label_en": "pistol",
            "entity_label_zh": "疑似手枪",
            "localization_status": "localized_by_qwen_point_sam3_verified",
            "mask_quality_score": 0.82,
            "evidence_regions": [
                {
                    "bbox": [100, 80, 40, 24],
                    "source": "qwen_point_sam3_local_review",
                    "entity_label_en": "pistol",
                    "entity_label_zh": "疑似手枪",
                    "localization_status": "localized_by_qwen_point_sam3_verified",
                    "mask_quality_score": 0.82,
                }
            ],
        },
    )
    frame_job = PictureJob(
        tenant_id="test-tenant",
        status=JobStatus.DONE,
        source=SourceSpec(uri=frame.image_uri, mime_type="image/png"),
        findings=[finding],
    )

    risks = build_frame_risks([frame], [frame_job])

    assert len(risks) == 1
    assert risks[0].metadata["video_role"] == "object_instance"
    assert risks[0].metadata["instance_label_zh"] == "疑似手枪"
    assert risks[0].regions[0]["source"] == "qwen_point_sam3_local_review"
    assert risks[0].regions[0]["mask_quality_score"] == 0.82


def test_clip_moderation_instances_create_video_object_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _make_case_dir("clip_instances")
    frame_a_path = base / "000.png"
    frame_b_path = base / "001.png"
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_a_path)
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_b_path)
    frame_a = services.FrameReference(frame_index=0, pts_ms=0, image_uri=str(frame_a_path), metadata={"duration_ms": 250})
    frame_b = services.FrameReference(frame_index=1, pts_ms=250, image_uri=str(frame_b_path), metadata={"duration_ms": 250})

    def fake_post_json(config, payload):
        assert payload["localization_required"] is True
        assert payload["tracking_required"] is True
        assert payload["coordinate_format"] == "xywh_pixels"
        assert "sexual_exposure" in payload["localization_targets"]
        assert "hate_symbol" in payload["labels"]
        assert payload["output_schema"]["events"][0]["instances"][0]["keyframes"][0]["bbox"] == [0, 0, 0, 0]
        return {
            "events": [
                {
                    "category": "weapon_threat",
                    "confidence": 0.9,
                    "start_time": 0.0,
                    "end_time": 0.5,
                    "evidence": "人物手持疑似手枪",
                    "instances": [
                        {
                            "entity_label_en": "pistol",
                            "entity_label_zh": "疑似手枪",
                            "confidence": 0.86,
                            "keyframes": [
                                {"frame_id": frame_a.frame_id, "bbox": [10, 12, 20, 10], "confidence": 0.86},
                                {"frame_id": frame_b.frame_id, "bbox": [24, 18, 20, 10], "confidence": 0.84},
                            ],
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr("video.application.clip_moderator._post_json", fake_post_json)

    risks, audits = moderate_clip_windows(
        [frame_a, frame_b],
        [{"window_id": "clip_0001", "start_ms": 0, "end_ms": 500, "frame_ids": [frame_a.frame_id, frame_b.frame_id]}],
        ClipModerationConfig(confidence_threshold=0.55),
    )

    assert audits[0]["status"] == "completed"
    assert any(risk.source_modality == "video_clip" for risk in risks)
    instance = next(risk for risk in risks if risk.source_modality == "video_object")
    assert instance.metadata["video_role"] == "object_instance"
    assert instance.metadata["instance_label_zh"] == "疑似手枪"
    assert len(instance.regions) == 2


def test_clip_moderation_event_level_bbox_is_kept_for_sexual_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _make_case_dir("clip_event_bbox")
    frame_path = base / "000.png"
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_path)
    frame = services.FrameReference(frame_index=0, pts_ms=1000, image_uri=str(frame_path), metadata={"duration_ms": 250})

    def fake_post_json(config, payload):
        return {
            "events": [
                {
                    "category": "sexual_exposure",
                    "confidence": 0.87,
                    "time_span": [1.0, 1.2],
                    "evidence": "画面存在裸露敏感区域",
                    "entity_label_en": "exposed_body",
                    "entity_label_zh": "裸露敏感区域",
                    "bbox": [15, 18, 30, 22],
                }
            ]
        }

    monkeypatch.setattr("video.application.clip_moderator._post_json", fake_post_json)

    risks, _ = moderate_clip_windows(
        [frame],
        [{"window_id": "clip_0001", "start_ms": 1000, "end_ms": 1250, "frame_ids": [frame.frame_id]}],
        ClipModerationConfig(confidence_threshold=0.55),
    )

    event = next(risk for risk in risks if risk.source_modality == "video_clip")
    instance = next(risk for risk in risks if risk.source_modality == "video_object")
    assert event.metadata["video_role"] == "event"
    assert event.span.start_ms == 1000
    assert instance.category == "content.sexual"
    assert instance.metadata["instance_label_zh"] == "裸露敏感区域"
    assert instance.regions[0]["bbox"] == {"x": 15.0, "y": 18.0, "w": 30.0, "h": 22.0}


def test_clip_moderation_text_only_conflict_gets_motion_seed_after_empty_localization(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _make_case_dir("clip_motion_seed")
    frame_a_path = base / "000.png"
    frame_b_path = base / "001.png"
    frame_c_path = base / "002.png"
    for path, offset in ((frame_a_path, 0), (frame_b_path, 18), (frame_c_path, 36)):
        image = Image.new("RGB", (160, 120), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle([32 + offset, 42, 82 + offset, 82], fill=(20, 20, 20))
        image.save(path)
    frame_a = services.FrameReference(frame_index=0, pts_ms=0, image_uri=str(frame_a_path), metadata={"duration_ms": 250})
    frame_b = services.FrameReference(frame_index=1, pts_ms=250, image_uri=str(frame_b_path), metadata={"duration_ms": 250})
    frame_c = services.FrameReference(frame_index=2, pts_ms=500, image_uri=str(frame_c_path), metadata={"duration_ms": 250})
    calls = []

    def fake_post_json(config, payload):
        calls.append(payload["task"])
        if payload["task"] == "video_safety_detection_with_temporal_localization":
            return {
                "events": [
                    {
                        "category": "physical_conflict",
                        "confidence": 0.92,
                        "start_time": 0.0,
                        "end_time": 0.75,
                        "evidence": "多人发生肢体推搡和拉扯。",
                    }
                ]
            }
        assert payload["task"] == "video_safety_event_region_localization"
        return {"events": [{"category": "physical_conflict", "confidence": 0.9, "evidence": "仍只给出文本"}]}

    monkeypatch.setattr("video.application.clip_moderator._post_json", fake_post_json)

    risks, audits = moderate_clip_windows(
        [frame_a, frame_b, frame_c],
        [{"window_id": "clip_0001", "start_ms": 0, "end_ms": 750, "frame_ids": [frame_a.frame_id, frame_b.frame_id, frame_c.frame_id]}],
        ClipModerationConfig(confidence_threshold=0.55),
    )

    assert calls == ["video_safety_detection_with_temporal_localization", "video_safety_event_region_localization"]
    assert audits[0]["localization_attempts"][0]["status"] == "fallback_motion_seeded"
    event = next(risk for risk in risks if risk.source_modality == "video_clip")
    instance = next(risk for risk in risks if risk.source_modality == "video_object")
    assert event.metadata["raw_event"]["localization_status"] == "motion_saliency_seeded_after_secondary_empty"
    assert instance.target_type == "conflict_region"
    assert instance.metadata["tracking_seed_source"] == "frame_motion_saliency_seed"
    assert instance.regions
    assert instance.regions[0]["source"] == "frame_motion_saliency_seed"
    assert instance.regions[0]["bbox"]["w"] > 0
    assert instance.regions[0]["bbox"]["h"] > 0


def test_event_risk_is_not_sent_to_sam3_but_instance_is(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=0, image_uri="/tmp/frame.png", metadata={"duration_ms": 250})
    event = VideoRiskAnnotation(
        source_modality="video_clip",
        category="content.graphic_violence",
        span=TimeSpan(start_ms=0, end_ms=250),
        frame_ids=[frame.frame_id],
        metadata={"video_role": "event"},
    )
    instance = VideoRiskAnnotation(
        source_modality="video_object",
        category="content.graphic_violence",
        span=TimeSpan(start_ms=0, end_ms=250),
        frame_ids=[frame.frame_id],
        regions=[{"frame_id": frame.frame_id, "bbox": {"x": 10, "y": 10, "w": 20, "h": 12}, "confidence": 0.9}],
        metadata={"video_role": "object_instance", "instance_label_en": "pistol"},
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"tracks": []}

    def fake_post(url, json, timeout):
        assert len(json["tracks"]) == 1
        assert json["tracks"][0]["risk_id"] == instance.risk_id
        return _Response()

    monkeypatch.setitem(sys.modules, "httpx", type("_Httpx", (), {"post": staticmethod(fake_post)}))

    report = enrich_with_sam3_video_tracking(
        [event, instance],
        [frame],
        Sam3VideoTrackerConfig(base_url="http://sam3.test"),
    )

    assert report["candidate_count"] == 1


def test_unlocalized_visual_risks_do_not_merge_across_frames() -> None:
    left = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=200),
        frame_ids=["frame_a"],
    )
    right = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=240, end_ms=440),
        frame_ids=["frame_b"],
    )

    risks = aggregate_risks([left, right], gap_tolerance_ms=1000)

    assert len(risks) == 2


def test_tracking_adds_display_span_and_localization_precision() -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=480, image_uri="", metadata={"duration_ms": 240})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=2000),
        frame_ids=[frame.frame_id],
    )

    enrich_risk_tracks([risk], [frame])

    assert risk.display_span is not None
    assert risk.display_span.start_ms == 480
    assert risk.display_span.end_ms == 720
    assert risk.temporal_precision == "frame"
    assert risk.spatial_precision == "full_frame"
    assert risk.localization_status == "frame_review"


def test_action_plan_uses_track_redaction_series_for_localized_privacy() -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=480, image_uri="", metadata={"duration_ms": 240})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=2000),
        frame_ids=[frame.frame_id],
        regions=[{"frame_id": frame.frame_id, "bbox": {"x": 10, "y": 20, "w": 100, "h": 80}, "confidence": 0.91}],
    )
    enrich_risk_tracks([risk], [frame])
    policy = VideoGovernancePolicyResult(
        decision=VideoGovernanceDecision.TRANSFORM_REQUIRED,
        requires_transformation=True,
    )

    plan = build_action_plan([risk], policy, TaskContext(), options={"render_redacted_asset": True})

    assert plan.render_redacted_asset is True
    assert len(plan.operations) == 1
    operation = plan.operations[0]
    assert operation.start_ms == 480
    assert operation.end_ms == 720
    assert operation.metadata["redaction_ready"] is True
    assert operation.metadata["redaction_scope"] == "sampled_frame"
    assert operation.metadata["redaction_series"]


def test_sam3_video_tracker_expands_seeded_visual_track(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_a = services.FrameReference(frame_index=0, pts_ms=0, image_uri="/tmp/frame_a.png", metadata={"duration_ms": 250})
    frame_b = services.FrameReference(frame_index=1, pts_ms=250, image_uri="/tmp/frame_b.png", metadata={"duration_ms": 250})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=500),
        frame_ids=[frame_a.frame_id],
        track_id="track_face_0001",
        regions=[{"frame_id": frame_a.frame_id, "bbox": {"x": 10, "y": 20, "w": 60, "h": 50}, "confidence": 0.9}],
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "tracks": [
                    {
                        "risk_id": risk.risk_id,
                        "track_id": risk.track_id,
                        "points": [
                            {
                                "frame_id": frame_b.frame_id,
                                "bbox": {"x": 14, "y": 22, "w": 60, "h": 50},
                                "confidence": 0.87,
                                "mask_path": "/tmp/mask_b.png",
                            }
                        ],
                    }
                ]
            }

    def _post(url, json, timeout):
        assert url == "http://sam3.test/v1/sam3/video-track"
        assert json["tracks"][0]["seed_regions"]
        return _Response()

    monkeypatch.setitem(sys.modules, "httpx", type("_Httpx", (), {"post": staticmethod(_post)}))

    report = enrich_with_sam3_video_tracking(
        [risk],
        [frame_a, frame_b],
        Sam3VideoTrackerConfig(base_url="http://sam3.test"),
    )
    enrich_risk_tracks([risk], [frame_a, frame_b])

    assert report["applied"] is True
    assert any(region.get("frame_id") == frame_b.frame_id for region in risk.regions)
    assert risk.metadata["tracking"]["tracking_backend"] == "sam3_video_tracker"
    assert risk.metadata["tracking"]["redaction_scope"] == "sam3_video_track"
    assert risk.metadata["tracking"]["redaction_ready"] is True


def test_sam3_video_tracker_failure_keeps_fallback_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=0, image_uri="/tmp/frame.png", metadata={"duration_ms": 250})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=250),
        frame_ids=[frame.frame_id],
        regions=[{"frame_id": frame.frame_id, "bbox": {"x": 10, "y": 20, "w": 60, "h": 50}, "confidence": 0.9}],
    )

    def _post(url, json, timeout):
        raise RuntimeError("tracker offline")

    monkeypatch.setitem(sys.modules, "httpx", type("_Httpx", (), {"post": staticmethod(_post)}))

    report = enrich_with_sam3_video_tracking(
        [risk],
        [frame],
        Sam3VideoTrackerConfig(base_url="http://sam3.test"),
    )
    enrich_risk_tracks([risk], [frame])

    assert report["applied"] is False
    assert report["reason"] == "tracker_request_failed"
    assert risk.metadata["tracking"]["tracking_backend"] == "sam3_image_iou_interpolation"
    assert risk.metadata["sam3_video_tracking"]["fallback"] == "sampled_frame_tracking"


def test_sam3_video_tracker_rejects_ambiguous_same_frame_points(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_a = services.FrameReference(frame_index=0, pts_ms=0, image_uri="/tmp/frame_a.png", metadata={"duration_ms": 250})
    frame_b = services.FrameReference(frame_index=1, pts_ms=250, image_uri="/tmp/frame_b.png", metadata={"duration_ms": 250})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=500),
        frame_ids=[frame_a.frame_id],
        track_id="track_face_ambiguous",
        regions=[{"frame_id": frame_a.frame_id, "bbox": {"x": 10, "y": 20, "w": 60, "h": 50}, "confidence": 0.9}],
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "tracks": [
                    {
                        "risk_id": risk.risk_id,
                        "track_id": risk.track_id,
                        "points": [
                            {
                                "frame_id": frame_b.frame_id,
                                "bbox": {"x": 14, "y": 22, "w": 60, "h": 50},
                                "confidence": 0.87,
                            },
                            {
                                "frame_id": frame_b.frame_id,
                                "bbox": {"x": 320, "y": 180, "w": 30, "h": 40},
                                "confidence": 0.95,
                            },
                        ],
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "httpx", type("_Httpx", (), {"post": staticmethod(lambda url, json, timeout: _Response())}))

    report = enrich_with_sam3_video_tracking(
        [risk],
        [frame_a, frame_b],
        Sam3VideoTrackerConfig(base_url="http://sam3.test"),
    )
    enrich_risk_tracks([risk], [frame_a, frame_b])

    assert report["applied"] is True
    assert sum(1 for region in risk.regions if region.get("frame_id") == frame_b.frame_id) == 1
    assert risk.metadata["tracking"]["redaction_scope"] == "manual_review"
    assert risk.metadata["tracking"]["redaction_ready"] is False
    assert "ambiguous_sam3_points_same_frame" in risk.metadata["tracking"]["quality_flags"]


def test_coarse_fallback_localization_never_becomes_auto_redaction() -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=0, image_uri="/tmp/frame.png", metadata={"duration_ms": 250})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.screen_sensitive",
        span=TimeSpan(start_ms=0, end_ms=250),
        frame_ids=[frame.frame_id],
        regions=[
            {
                "frame_id": frame.frame_id,
                "bbox": {"x": 100, "y": 90, "w": 200, "h": 120},
                "confidence": 0.7,
                "source": "qwen_rough_bbox_fallback",
            }
        ],
        metadata={"localization_status": "coarse_localization_after_sam3_rejections", "mask_quality_score": 0.15},
    )

    enrich_risk_tracks([risk], [frame])

    assert risk.metadata["tracking"]["redaction_scope"] == "manual_review"
    assert risk.metadata["tracking"]["redaction_ready"] is False
    assert "low_quality_coarse_localization" in risk.metadata["tracking"]["quality_flags"]


def test_adjacent_unlocalized_visual_safety_risks_merge_as_event() -> None:
    risks = [
        VideoRiskAnnotation(
            source_modality="safety",
            category="content.graphic_violence",
            severity="critical",
            confidence=0.81,
            span=TimeSpan(start_ms=0, end_ms=240),
            frame_ids=["frame_a"],
            reason_codes=["SAFETY_GRAPHIC_VIOLENCE"],
        ),
        VideoRiskAnnotation(
            source_modality="safety",
            category="content.graphic_violence",
            severity="critical",
            confidence=0.88,
            span=TimeSpan(start_ms=240, end_ms=480),
            frame_ids=["frame_b"],
            reason_codes=["SAFETY_GRAPHIC_VIOLENCE"],
        ),
    ]

    aggregated = aggregate_risks(risks, gap_tolerance_ms=1000)

    assert len(aggregated) == 1
    assert aggregated[0].span.start_ms == 0
    assert aggregated[0].span.end_ms == 480
    assert aggregated[0].frame_ids == ["frame_a", "frame_b"]


def test_sam3_video_track_endpoint_returns_track_points(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _make_case_dir("sam3_video_endpoint")
    frame_a = base / "000.png"
    frame_b = base / "001.png"
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_a)
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_b)

    class _FakeVideoModel:
        def init_state(self, **kwargs):
            assert len(kwargs["resource_path"]) == 2
            return {}

        def add_prompt(self, inference_state, frame_idx, boxes_xywh, box_labels):
            assert frame_idx == 0
            assert boxes_xywh == [[0.1, 0.125, 0.3, 0.25]]
            return 0, {
                "out_obj_ids": [1],
                "out_probs": [0.9],
                "out_boxes_xywh": [[0.1, 0.125, 0.3, 0.25]],
                "out_binary_masks": [],
            }

        def propagate_in_video(self, inference_state, start_frame_idx, max_frame_num_to_track, reverse=False):
            if reverse:
                return iter(())
            return iter(
                [
                    (
                        1,
                        {
                            "out_obj_ids": [1],
                            "out_probs": [0.86],
                            "out_boxes_xywh": [[0.2, 0.25, 0.3, 0.25]],
                            "out_binary_masks": [],
                        },
                    )
                ]
            )

    monkeypatch.setattr(sam3_api, "_get_video_runtime", lambda: {"model": _FakeVideoModel(), "device": "cuda", "checkpoint_path": "fake"})
    monkeypatch.setattr(sam3_api, "_inference_context", sam3_api.nullcontext)

    response = asyncio.run(
        sam3_api.video_track(
            sam3_api.VideoTrackRequest(
                frames=[
                    sam3_api.VideoFrameModel(frame_id="frame_a", frame_index=0, pts_ms=0, image_path=str(frame_a)),
                    sam3_api.VideoFrameModel(frame_id="frame_b", frame_index=1, pts_ms=250, image_path=str(frame_b)),
                ],
                tracks=[
                    sam3_api.VideoTrackSeedModel(
                        risk_id="risk_face",
                        track_id="track_face",
                        category="privacy.face",
                        seed_regions=[
                            sam3_api.RegionModel(
                                frame_id="frame_a",
                                bbox=sam3_api.BBoxModel(x=10, y=10, w=30, h=20),
                                confidence=0.9,
                            )
                        ],
                    )
                ],
                return_masks=False,
                return_polygons=False,
            )
        )
    )

    points = response["tracks"][0]["points"]
    assert response["metadata"]["backend"] == "official_sam3_video"
    assert [point["frame_id"] for point in points] == ["frame_a", "frame_b"]
    assert points[1]["bbox"] == {"x": 20.0, "y": 20.0, "w": 30.0, "h": 20.0}


def test_sam3_video_track_endpoint_keeps_only_seed_object(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _make_case_dir("sam3_video_seed_object")
    frame_a = base / "000.png"
    frame_b = base / "001.png"
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_a)
    Image.new("RGB", (100, 80), (255, 255, 255)).save(frame_b)

    class _FakeVideoModel:
        def init_state(self, **kwargs):
            return {}

        def add_prompt(self, inference_state, frame_idx, boxes_xywh, box_labels):
            return 0, {
                "out_obj_ids": [7, 8],
                "out_probs": [0.9, 0.99],
                "out_boxes_xywh": [[0.1, 0.125, 0.3, 0.25], [0.75, 0.625, 0.15, 0.2]],
                "out_binary_masks": [],
            }

        def propagate_in_video(self, inference_state, start_frame_idx, max_frame_num_to_track, reverse=False):
            if reverse:
                return iter(())
            return iter(
                [
                    (
                        1,
                        {
                            "out_obj_ids": [7, 8],
                            "out_probs": [0.86, 0.99],
                            "out_boxes_xywh": [[0.2, 0.25, 0.3, 0.25], [0.72, 0.6, 0.15, 0.2]],
                            "out_binary_masks": [],
                        },
                    )
                ]
            )

    monkeypatch.setattr(sam3_api, "_get_video_runtime", lambda: {"model": _FakeVideoModel(), "device": "cuda", "checkpoint_path": "fake"})
    monkeypatch.setattr(sam3_api, "_inference_context", sam3_api.nullcontext)

    response = asyncio.run(
        sam3_api.video_track(
            sam3_api.VideoTrackRequest(
                frames=[
                    sam3_api.VideoFrameModel(frame_id="frame_a", frame_index=0, pts_ms=0, image_path=str(frame_a)),
                    sam3_api.VideoFrameModel(frame_id="frame_b", frame_index=1, pts_ms=250, image_path=str(frame_b)),
                ],
                tracks=[
                    sam3_api.VideoTrackSeedModel(
                        risk_id="risk_face",
                        track_id="track_face",
                        category="privacy.face",
                        seed_regions=[
                            sam3_api.RegionModel(
                                frame_id="frame_a",
                                bbox=sam3_api.BBoxModel(x=10, y=10, w=30, h=20),
                                confidence=0.9,
                            )
                        ],
                    )
                ],
                return_masks=False,
                return_polygons=False,
            )
        )
    )

    points = response["tracks"][0]["points"]
    assert response["tracks"][0]["metadata"]["seed_obj_id"] == 7
    assert [point["obj_id"] for point in points] == [7, 7]
    assert [point["frame_id"] for point in points] == ["frame_a", "frame_b"]


def test_action_plan_does_not_auto_redact_unlocalized_visual_privacy() -> None:
    frame = services.FrameReference(frame_index=0, pts_ms=480, image_uri="", metadata={"duration_ms": 240})
    risk = VideoRiskAnnotation(
        source_modality="visual",
        category="privacy.face",
        span=TimeSpan(start_ms=0, end_ms=2000),
        frame_ids=[frame.frame_id],
    )
    enrich_risk_tracks([risk], [frame])
    policy = VideoGovernancePolicyResult(
        decision=VideoGovernanceDecision.TRANSFORM_REQUIRED,
        requires_transformation=True,
    )

    plan = build_action_plan([risk], policy, TaskContext(), options={"render_redacted_asset": True})

    assert plan.render_redacted_asset is False
    assert plan.operations == []
    assert plan.metadata["manual_review_redaction_count"] == 1


def test_pipeline_writes_scene_clip_quality_and_review_artifacts():
    base = _make_case_dir("scene_quality")
    input_path = base / "sample_video.gif"
    _write_gif(input_path)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options=_mock_picture_options("mixed"))

    assert result.status == VideoJobStatus.DONE
    output_dir = pipeline.output_dir
    assert (output_dir / "scene_manifest.jsonl").exists()
    assert (output_dir / "clip_windows.jsonl").exists()
    assert (output_dir / "quality_report.json").exists()
    assert (output_dir / "review_queue.jsonl").exists()
    assert result.risk_tracks
    assert any("bbox_series" in track for track in result.risk_tracks)


def test_clip_moderation_risks_are_merged_when_enabled(monkeypatch: pytest.MonkeyPatch):
    base = _make_case_dir("clip_moderation")
    input_path = base / "sample_video.gif"
    _write_gif(input_path)

    def fake_moderate_clip_windows(frames, clip_windows, config, asset_id="", operator_selection=None):
        return [
            VideoRiskAnnotation(
                asset_id=asset_id,
                source_modality="video_clip",
                category="content.graphic_violence",
                operator_id="VVIS_CSA_003",
                source_operator_id="CSA_003",
                target_type="content.graphic_violence",
                severity="critical",
                confidence=0.93,
                span=TimeSpan(start_ms=0, end_ms=300),
                frame_ids=[frames[0].frame_id],
                text_span="多人肢体冲突",
                provider="fake_clip_moderator",
                reason_codes=["CLIP_CSA_003"],
            )
        ], [{"window_id": "clip_0001", "status": "completed", "event_count": 1}]

    monkeypatch.setattr("video.pipeline.moderate_clip_windows", fake_moderate_clip_windows)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(
        str(input_path),
        tenant_id="test-tenant",
        options={**_mock_picture_options("mixed"), "enable_clip_moderation": True},
    )

    assert result.status == VideoJobStatus.DONE
    assert any(risk.source_modality == "video_clip" and risk.category == "content.graphic_violence" for risk in result.risk_annotations)
    assert any(track["source_modality"] == "video_clip" for track in result.risk_tracks)
