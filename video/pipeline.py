"""Pipeline orchestrator for video compliance checking."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from audio.models.schemas import Decision as AudioDecision
from picture.infra.storage import LocalFileStorageBackend
from video.application.services import (
    aggregate_policy,
    analyze_frames,
    build_picture_settings,
    build_video_findings,
    derive_segments,
    extract_audio_track,
    load_sequence,
    prepare_input_source,
    probe_media,
    render_sequence_outputs,
    resolve_sidecar_audio,
    resolve_video_route,
    run_audio_sidecar,
    write_json,
    write_jsonl,
)
from video.config.settings import Settings, get_settings
from video.domain.enums import VideoDecisionType, VideoJobStatus
from video.domain.models import VideoAsset, VideoJob, VideoReport

# 统一契约层导入
from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision
from common.runtime import PipelineExecutionContext, TrustEvaluator

logger = logging.getLogger(__name__)


class VideoCompliancePipeline:
    """A runnable video compliance pipeline built on top of picture/audio."""

    def __init__(self, settings: Settings | None = None):
        # 使用显式配置或环境配置创建运行实例。
        self.settings = settings or get_settings()
        # 每次运行独立 run_id，保证产物目录与任务一一对应。
        self.run_id = uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id
        # 统一通过本地存储后端持久化可交付产物。
        self.storage = LocalFileStorageBackend(self.settings.storage_base_path)
        # 统一执行上下文（记录步骤状态、降级事件、失败信息）
        self.exec_ctx: PipelineExecutionContext | None = None

    def execute(
        self,
        input_path: str,
        tenant_id: str = "default",
        profile: str | None = None,
        options: dict[str, object] | None = None,
    ) -> VideoJob:
        # options 使用可变副本，避免调用方对象被意外修改。
        options = dict(options or {})
        profile = profile or self.settings.default_profile
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建顶层任务对象，后续逐步填充帧、发现、策略与产物信息。
        job = VideoJob(tenant_id=tenant_id, source_uri=input_path, profile=profile, options=options)
        total_start = time.monotonic()

        try:
            # 初始化执行上下文
            self.exec_ctx = PipelineExecutionContext(pipeline_run_id=job.job_id)
            # 1) 统一准备输入源（复制文件或引用目录）。
            prepared_source = prepare_input_source(input_path, self.output_dir / "input")
            job.asset = VideoAsset(original_uri=prepared_source)
            self._set_status(job, VideoJobStatus.PREPROCESSING)

            # 2) 加载序列：按输入类型走目录帧/动画图/视频容器抽帧路径。
            sequence = load_sequence(
                prepared_source,
                self.output_dir,
                frame_stride=int(options.get("frame_stride", self.settings.frame_stride)),
                max_frames=int(options.get("max_frames", self.settings.max_frames)),
                default_frame_duration_ms=int(options.get("default_frame_duration_ms", self.settings.default_frame_duration_ms)),
                ffmpeg_bin=self.settings.ffmpeg_bin,
                ffprobe_bin=self.settings.ffprobe_bin,
            )
            job.frames = sequence.frames
            job.asset.normalized_uri = prepared_source
            job.asset.metadata.update({
                "source_kind": sequence.source_kind,
                "fps": sequence.fps,
                "total_duration_ms": sequence.total_duration_ms,
                "has_native_audio": sequence.has_native_audio,
            })
            if sequence.source_kind == "video_container":
                # 容器输入补充 ffprobe 元数据，便于审计与调参。
                media_info = probe_media(Path(prepared_source), ffprobe_bin=self.settings.ffprobe_bin)
                job.asset.metadata.update({key: value for key, value in media_info.items() if value not in (None, "")})
                job.provider_versions["video_demux"] = "ffmpeg"
            else:
                job.provider_versions["video_demux"] = sequence.source_kind
            job.step_latencies["source_prepare"] = (time.monotonic() - total_start) * 1000

            # 3) 逐帧复用 picture 引擎完成视觉合规分析。
            picture_settings = build_picture_settings(self.output_dir)
            self._set_status(job, VideoJobStatus.DETECTING)
            detect_start = time.monotonic()
            frame_jobs = analyze_frames(
                sequence.frames,
                tenant_id=tenant_id,
                profile=profile,
                picture_settings=picture_settings,
                options=options,
                max_workers=int(options.get("max_workers", self.settings.max_workers)),
            )
            job.step_latencies["frame_detect"] = (time.monotonic() - detect_start) * 1000

            # 4) 将逐帧结果提升为视频级路由、片段和时序发现。
            job.route = resolve_video_route(frame_jobs)
            job.segments = derive_segments(sequence.frames, frame_jobs)
            job.findings = build_video_findings(sequence.frames, frame_jobs)
            job.provider_versions.update(_collect_provider_versions(frame_jobs))

            # 记录帧级摘要清单，便于回放与审计。
            frame_manifest = [
                {
                    "frame_id": frame.frame_id,
                    "frame_index": frame.frame_index,
                    "pts_ms": frame.pts_ms,
                    "duration_ms": frame.metadata.get("duration_ms", 0),
                    "route": frame_job.route.value if frame_job.route else None,
                    "decision": frame_job.policy_result.decision.value if frame_job.policy_result else None,
                    "reason_codes": frame_job.policy_result.reason_codes if frame_job.policy_result else [],
                    "findings": len(frame_job.findings),
                }
                for frame, frame_job in zip(sequence.frames, frame_jobs)
            ]
            write_jsonl(frame_manifest, self.output_dir / "frame_manifest.jsonl")
            write_jsonl(job.segments, self.output_dir / "segment_manifest.jsonl")
            write_jsonl(job.findings, self.output_dir / "video_findings.jsonl")

            # 5) 处理音轨：优先 sidecar，其次（可选）抽取容器原生音轨。
            audio_decision = None
            audio_work_dir = self.output_dir / "audio_work"
            render_audio_path = None
            audio_path = resolve_sidecar_audio(prepared_source, options, enabled=bool(options.get("enable_audio_sidecar", self.settings.enable_audio_sidecar)), sidecar_extensions=list(self.settings.sidecar_audio_extensions))
            if not audio_path and sequence.source_kind == "video_container" and bool(options.get("extract_native_audio", self.settings.extract_native_audio)):
                extract_start = time.monotonic()
                audio_path = extract_audio_track(prepared_source, self.output_dir / "native_audio", ffmpeg_bin=self.settings.ffmpeg_bin)
                job.step_latencies["native_audio_extract"] = (time.monotonic() - extract_start) * 1000
                if audio_path:
                    job.asset.metadata["native_audio_extracted"] = True

            if audio_path:
                self._set_status(job, VideoJobStatus.AUDIO_PROCESSING)
                audio_start = time.monotonic()
                try:
                    # 复用 audio 子流水线输出音频决策与脱敏音轨。
                    audio_decision, audio_output_dir, redacted_audio_path = run_audio_sidecar(audio_path, audio_work_dir)
                    job.provider_versions["audio_pipeline"] = "AudioCompliancePipeline"
                    job.asset.metadata["audio_output_dir"] = str(audio_output_dir.resolve())
                    job.asset.audio_uri = audio_path
                    if redacted_audio_path:
                        job.asset.metadata["redacted_audio_path"] = redacted_audio_path
                    # 渲染阶段优先使用脱敏后音轨。
                    render_audio_path = _choose_audio_for_render(audio_path, redacted_audio_path, audio_decision)
                except Exception:
                    # 是否中断由配置控制：严格模式抛错，宽松模式记录后继续。
                    if self.settings.fail_on_audio_error:
                        raise
                    logger.exception("Audio sidecar processing failed for %s", audio_path)
                    job.asset.metadata["audio_error"] = "sidecar_processing_failed"
                    # 记录音轨处理降级事件
                    if self.exec_ctx:
                        self.exec_ctx.record_step_failure(
                            "audio_sidecar",
                            error="sidecar_processing_failed",
                        )
                job.step_latencies["audio_pipeline"] = (time.monotonic() - audio_start) * 1000

            # 6) 聚合 picture+audio 的决策得到视频级最终策略。
            self._set_status(job, VideoJobStatus.POLICY_EVALUATING)
            policy_start = time.monotonic()
            job.policy_result = aggregate_policy(frame_jobs, profile, audio_decision=audio_decision)
            job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

            compliant_path = None
            preview_path = None
            # 7) 非 DROP 才执行回写渲染，DROP 仅输出报告与索引。
            self._set_status(job, VideoJobStatus.RENDERING)
            render_start = time.monotonic()
            if job.policy_result.decision != VideoDecisionType.DROP:
                compliant_path, preview_path = render_sequence_outputs(
                    sequence,
                    frame_jobs,
                    self.output_dir,
                    decision=job.policy_result.decision,
                    render_preview=bool(options.get("render_preview", self.settings.render_preview)),
                    ffmpeg_bin=self.settings.ffmpeg_bin,
                    audio_path=render_audio_path,
                )
            job.step_latencies["render"] = (time.monotonic() - render_start) * 1000

            if compliant_path:
                # 保留原始输出后缀，兼容 GIF/MP4 两类渲染产物。
                compliant_suffix = Path(compliant_path).suffix or ".gif"
                job.asset.compliant_video_uri = self.storage.save(compliant_path, f"{job.job_id}/compliant_video{compliant_suffix}")
            if preview_path:
                preview_suffix = Path(preview_path).suffix or ".gif"
                job.asset.metadata["preview_uri"] = self.storage.save(preview_path, f"{job.job_id}/preview{preview_suffix}")
            job.asset.frame_manifest_uri = self.storage.save(str((self.output_dir / "frame_manifest.jsonl").resolve()), f"{job.job_id}/frame_manifest.jsonl")

            # 8) 生成审计报告并落盘存储。
            report = VideoReport(
                job_id=job.job_id,
                route=job.route or resolve_video_route(frame_jobs),
                decision=job.policy_result.decision if job.policy_result else VideoDecisionType.PASS_RAW,
                findings=job.findings,
                provider_info=job.provider_versions,
                reason_codes=job.policy_result.reason_codes if job.policy_result else [],
                artifacts={
                    "source": prepared_source,
                    "frame_manifest": str((self.output_dir / "frame_manifest.jsonl").resolve()),
                    "segment_manifest": str((self.output_dir / "segment_manifest.jsonl").resolve()),
                    "video_findings": str((self.output_dir / "video_findings.jsonl").resolve()),
                    "compliant_video": compliant_path or "",
                    "preview": preview_path or "",
                    "audio_work_dir": str(audio_work_dir.resolve()) if audio_path else "",
                    "render_audio_path": render_audio_path or "",
                },
                latency_ms=dict(job.step_latencies),
            )
            report_path = self.output_dir / "report.json"
            write_json(report, report_path)
            job.asset.report_uri = self.storage.save(str(report_path.resolve()), f"{job.job_id}/report.json")

            job.completed_at = datetime.now(timezone.utc)
            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            self._set_status(job, VideoJobStatus.DROPPED if job.policy_result and job.policy_result.decision == VideoDecisionType.DROP else VideoJobStatus.DONE)

            # ── 统一契约输出构建 ────────────────────────
            compliance_output = self._build_compliance_output(job)
            job._compliance_output = compliance_output

            return job

        except Exception as exc:
            # 顶层兜底：固化错误信息与耗时，返回 FAILED 任务对象。
            logger.exception("Video pipeline failed for %s", input_path)
            job.error = str(exc)
            job.error_detail = type(exc).__name__
            job.completed_at = datetime.now(timezone.utc)
            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            self._set_status(job, VideoJobStatus.FAILED)
            return job

    def _set_status(self, job: VideoJob, status: VideoJobStatus) -> None:
        # 同步更新时间戳，便于外部查询任务最新状态。
        job.status = status
        job.updated_at = datetime.now(timezone.utc)

    def _build_compliance_output(self, job: VideoJob) -> ComplianceOutput:
        """构建统一契约输出（双轨交付物）。"""
        from common.adapters import (
            build_annotation_package,
            build_audit_package,
            build_compliance_output,
            build_release_package,
            deduplicate_evidence_units,
            map_video_decision_to_unified,
            video_finding_to_evidence,
        )
        from common.policy import evaluate_with_profile, load_policy_profile

        # 1. 转换 VideoFinding → EvidenceUnit
        evidence_units = [video_finding_to_evidence(f) for f in job.findings]
        evidence_units = deduplicate_evidence_units(evidence_units)

        # 2. Profile 化策略评估
        ctx = self.exec_ctx or PipelineExecutionContext(pipeline_run_id=job.job_id)
        profile = load_policy_profile("default")
        policy_result = evaluate_with_profile(
            evidence_units, profile=profile,
            degrade_events=ctx.degrade_events,
        )
        trust_level = TrustEvaluator.evaluate(ctx)
        unified_decision = policy_result.decision

        # 3. 标注样本包
        content_uri = job.asset.compliant_video_uri or job.asset.original_uri if job.asset else ""
        annotation_pkg = build_annotation_package(
            modality=Modality.VIDEO,
            pipeline_run_id=job.job_id,
            clean_content_uri=content_uri,
            content_format=job.mime_type or "video/mp4",
            evidence_units=evidence_units,
            decision=unified_decision,
            trust_level=trust_level,
        )

        # 4. 审计证据包
        audit_pkg = build_audit_package(
            modality=Modality.VIDEO,
            pipeline_run_id=job.job_id,
            evidence_units=evidence_units,
            degrade_events=ctx.degrade_events,
            policy_result=policy_result,
            ctx=ctx,
        )

        # 5. 组装发布包
        release_pkg = build_release_package(
            modality=Modality.VIDEO,
            pipeline_run_id=job.job_id,
            annotation_package=annotation_pkg,
            audit_package=audit_pkg,
            decision=unified_decision,
            trust_level=trust_level,
        )

        # 6. 填充双轨交付物 URI 到 job
        job.annotation_package_uri = content_uri
        job.audit_package_uri = job.asset.report_uri if job.asset else ""
        job.trust_level = trust_level.value

        legacy = job.policy_result.model_dump() if job.policy_result else None
        return build_compliance_output(
            pipeline_run_id=job.job_id,
            modality=Modality.VIDEO,
            decision=unified_decision,
            trust_level=trust_level,
            release_package=release_pkg,
            degrade_summary=policy_result.degrade_summary,
            review_suggestions=policy_result.review_suggestions,
            explanation_summary=audit_pkg.review_summary,
            legacy_decision=legacy,
        )

def _choose_audio_for_render(original_audio_path: str, redacted_audio_path: str | None, audio_decision: AudioDecision | None) -> str | None:
    # 脱敏音轨优先；其次仅在允许场景下回退到原始音轨。
    if redacted_audio_path and Path(redacted_audio_path).exists():
        return redacted_audio_path
    if audio_decision in {None, AudioDecision.ALLOW} and Path(original_audio_path).exists():
        return original_audio_path
    return None


def _collect_provider_versions(frame_jobs) -> dict[str, str]:  # type: ignore[no-untyped-def]
    # 收集每帧 provider 版本信息并统一加上 picture_ 前缀。
    providers: dict[str, str] = {}
    for frame_job in frame_jobs:
        for key, value in frame_job.provider_versions.items():
            providers[f"picture_{key}"] = value
    providers["video_pipeline"] = "VideoCompliancePipeline"
    return providers
