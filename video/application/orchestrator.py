"""Execution blueprint for the video compliance module.

This file intentionally provides a design-time orchestrator blueprint instead of
an end-to-end executable pipeline. The goal is to make the reuse strategy for
`picture`, `audio`, and `text` explicit before implementation starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from video.domain.enums import VideoRouteType


@dataclass(frozen=True)
class PipelineStage:
    """A single planned stage in the video compliance pipeline."""

    name: str
    description: str
    reuses: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parallelizable: bool = False


@dataclass
class VideoComplianceBlueprint:
    """Build a route-aware video compliance plan around existing modules."""

    default_route: VideoRouteType = VideoRouteType.MIXED
    shared_outputs: tuple[str, ...] = field(
        default_factory=lambda: (
            "frame_manifest.jsonl",
            "segment_manifest.jsonl",
            "video_findings.jsonl",
            "evidence_bundle.json",
            "decision.json",
            "report.json",
        )
    )

    def build(self, route: VideoRouteType | None = None) -> list[PipelineStage]:
        """Return the staged execution plan for the requested route."""
        resolved_route = route or self.default_route
        stages = [
            PipelineStage(
                name="source_prepare",
                description="标准化视频输入，沉淀本地工作副本和媒体元数据。",
                reuses=("picture/infra/storage.py",),
                outputs=("normalized_video.mp4", "source_metadata.json"),
            ),
            PipelineStage(
                name="frame_sampling",
                description="按关键帧、固定采样率或镜头切换抽帧，为后续单帧合规分析做准备。",
                reuses=("picture/providers/preprocess.py",),
                outputs=("frame_manifest.jsonl",),
            ),
            PipelineStage(
                name="scene_segmentation",
                description="按镜头或片段切分视频，并给每个片段打上 route 候选。",
                reuses=("picture/providers/router.py",),
                outputs=("segment_manifest.jsonl",),
            ),
        ]
        stages.extend(self._route_stages(resolved_route))
        stages.extend(
            [
                PipelineStage(
                    name="temporal_tracking",
                    description="把逐帧发现合并为可跨时间段生效的时序发现。",
                    reuses=("picture/application/services.py",),
                    outputs=("video_findings.jsonl",),
                ),
                PipelineStage(
                    name="audio_pipeline",
                    description="抽取音轨并复用 audio 模块完成 ASR、PII、安全审核与音频脱敏。",
                    reuses=("audio/pipeline.py",),
                    outputs=("audio_evidence_bundle.json", "redacted_audio_manifest.jsonl"),
                    parallelizable=True,
                ),
                PipelineStage(
                    name="policy_decision",
                    description="把视觉、音轨、文本证据聚合为视频级最终决策。",
                    reuses=("picture/domain/policy.py", "text/pipeline.py", "audio/pipeline.py"),
                    outputs=("evidence_bundle.json", "decision.json"),
                ),
                PipelineStage(
                    name="video_render",
                    description="将时序遮挡和音轨脱敏结果回写为可交付视频产物。",
                    reuses=("picture/providers/redaction/opencv_redactor.py",),
                    outputs=("compliant_video.mp4", "preview.mp4", "report.json"),
                ),
            ]
        )
        return stages

    def reuse_map(self) -> dict[str, tuple[str, ...]]:
        """Summarize which existing modules should be reused by video."""
        return {
            "single_frame_visual_compliance": (
                "picture/application/orchestrator.py",
                "picture/application/services.py",
                "picture/providers/base.py",
                "picture/providers/router.py",
                "picture/providers/preprocess.py",
                "picture/domain/models.py",
                "picture/domain/policy.py",
            ),
            "audio_track_compliance": (
                "audio/pipeline.py",
                "audio/steps/c0_audio_normalize.py",
                "audio/steps/c1_asr_transcribe.py",
                "audio/steps/f_privacy_detection.py",
                "audio/steps/g_safety_moderation.py",
                "audio/steps/k_audio_redaction.py",
            ),
            "evidence_and_audit_style": (
                "text/pipeline.py",
                "audio/pipeline.py",
                "picture/infra/repository.py",
                "picture/infra/storage.py",
            ),
        }

    def _route_stages(self, route: VideoRouteType) -> list[PipelineStage]:
        common_picture_reuse = (
            "picture/application/orchestrator.py",
            "picture/application/services.py",
        )
        if route == VideoRouteType.SCREENCAST:
            return [
                PipelineStage(
                    name="frame_document_analysis",
                    description="以文档/界面帧为主，优先跑 OCR、文本 PII、二维码、印章、工牌检测。",
                    reuses=common_picture_reuse,
                    outputs=("picture_jobs_document.jsonl",),
                    parallelizable=True,
                )
            ]
        if route == VideoRouteType.NATURAL:
            return [
                PipelineStage(
                    name="frame_natural_analysis",
                    description="以自然场景帧为主，优先跑视觉检测、人物区域、安全审核和分割精修。",
                    reuses=common_picture_reuse,
                    outputs=("picture_jobs_natural.jsonl",),
                    parallelizable=True,
                )
            ]
        return [
            PipelineStage(
                name="frame_mixed_analysis",
                description="对混合场景帧复用 picture 的 mixed 双链路并支持并行执行。",
                reuses=common_picture_reuse,
                outputs=("picture_jobs_mixed.jsonl",),
                parallelizable=True,
            )
        ]
