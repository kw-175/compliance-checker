"""Pipeline orchestrator for video compliance checking."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from audio.models.schemas import Decision as AudioDecision
from picture.infra.storage import LocalFileStorageBackend
from video.application.action_planner import build_action_plan
from video.application.clip_moderator import ClipModerationConfig, moderate_clip_windows
from video.application.display_consolidation import consolidate_display_risks
from video.application.exporter import export_governance_artifacts
from video.application.operator_selection import resolve_operator_selection
from video.application.redaction import render_redacted_derivative
from video.application.risk_builder import build_audio_risks, build_frame_risks
from video.application.picture_api_client import PictureApiConfig
from video.application.quality import build_quality_report, build_review_queue
from video.application.scene_sampling import build_clip_windows, detect_scene_windows
from video.application.sam3_video_tracker import Sam3VideoTrackerConfig, enrich_with_sam3_video_tracking
from video.application.services import (
    aggregate_policy,
    analyze_frames,
    build_video_findings,
    derive_segments,
    extract_audio_track,
    load_sequence,
    prepare_input_source,
    probe_media,
    resolve_sidecar_audio,
    resolve_video_route,
    run_audio_sidecar,
    write_json,
    write_jsonl,
)
from video.application.policy import evaluate_policy
from video.application.temporal_aggregation import aggregate_risks, risk_summary
from video.application.tracking import build_risk_tracks, enrich_risk_tracks
from video.config.settings import Settings, get_settings
from video.domain.enums import VideoDecisionType, VideoJobStatus
from video.domain.models import ComplianceOperatorSelection, TaskContext, VideoAsset, VideoJob, VideoReport

logger = logging.getLogger(__name__)


class VideoCompliancePipeline:
    """A runnable video compliance pipeline built on top of picture/audio."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.run_id = uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id
        self.storage = LocalFileStorageBackend(self.settings.storage_base_path)

    def execute(
        self,
        input_path: str,
        tenant_id: str = "default",
        profile: str | None = None,
        dataset_id: str = "",
        asset_id: str = "",
        cleaning_run_id: str = "",
        task_context: TaskContext | dict[str, object] | None = None,
        operator_selection: ComplianceOperatorSelection | dict[str, object] | None = None,
        options: dict[str, object] | None = None,
    ) -> VideoJob:
        options = dict(options or {})
        profile = profile or self.settings.default_profile
        context = task_context if isinstance(task_context, TaskContext) else TaskContext(**dict(task_context or {}))
        resolved_selection = resolve_operator_selection(operator_selection, options)
        downstream_options = {**options, **resolved_selection.picture_options()}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        job = VideoJob(tenant_id=tenant_id, source_uri=input_path, profile=profile, options=options)
        job.task_context = context
        job.operator_selection = resolved_selection.selection
        job.preserved_training_targets = resolved_selection.preserved_targets(task_type=context.task_type)
        total_start = time.monotonic()

        try:
            prepared_source = prepare_input_source(input_path, self.output_dir / "input")
            job.asset = VideoAsset(original_uri=prepared_source)
            if asset_id:
                job.asset.metadata["asset_id"] = asset_id
            if dataset_id:
                job.asset.metadata["dataset_id"] = dataset_id
            if cleaning_run_id:
                job.asset.metadata["cleaning_run_id"] = cleaning_run_id
            self._set_status(job, VideoJobStatus.PREPROCESSING)

            sequence = load_sequence(
                prepared_source,
                self.output_dir,
                frame_stride=int(options.get("frame_stride", self.settings.frame_stride)),
                sample_fps=float(options.get("sample_fps", self.settings.sample_fps)),
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
                media_info = probe_media(Path(prepared_source), ffprobe_bin=self.settings.ffprobe_bin)
                job.asset.metadata.update({key: value for key, value in media_info.items() if value not in (None, "")})
                job.provider_versions["video_demux"] = "ffmpeg"
            else:
                job.provider_versions["video_demux"] = sequence.source_kind
            job.step_latencies["source_prepare"] = (time.monotonic() - total_start) * 1000

            scene_start = time.monotonic()
            scene_windows = detect_scene_windows(
                sequence.frames,
                change_threshold=float(options.get("scene_change_threshold", self.settings.scene_change_threshold)),
                min_scene_duration_ms=int(options.get("scene_min_duration_ms", self.settings.scene_min_duration_ms)),
            ) if bool(options.get("scene_detection_enabled", self.settings.scene_detection_enabled)) else []
            clip_windows = build_clip_windows(
                sequence.frames,
                scene_windows,
                max_window_ms=int(options.get("clip_window_ms", self.settings.clip_window_ms)),
                overlap_ms=int(options.get("clip_window_overlap_ms", self.settings.clip_window_overlap_ms)),
            )
            write_jsonl(scene_windows, self.output_dir / "scene_manifest.jsonl")
            write_jsonl(clip_windows, self.output_dir / "clip_windows.jsonl")
            job.asset.metadata["scene_count"] = len(scene_windows)
            job.asset.metadata["clip_window_count"] = len(clip_windows)
            job.step_latencies["scene_sampling"] = (time.monotonic() - scene_start) * 1000

            self._set_status(job, VideoJobStatus.DETECTING)
            detect_start = time.monotonic()
            frame_jobs = analyze_frames(
                sequence.frames,
                tenant_id=tenant_id,
                profile=profile,
                options=downstream_options,
                max_workers=int(options.get("max_workers", self.settings.max_workers)),
                picture_api_config=self._picture_api_config(options),
            )
            job.step_latencies["frame_detect"] = (time.monotonic() - detect_start) * 1000

            job.route = resolve_video_route(frame_jobs)
            job.segments = derive_segments(sequence.frames, frame_jobs)
            job.findings = build_video_findings(sequence.frames, frame_jobs)
            job.provider_versions.update(_collect_provider_versions(frame_jobs))

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
                    "analysis_hash": frame.metadata.get("analysis_hash", ""),
                    "cached_from_frame_id": frame.metadata.get("cached_from_frame_id", ""),
                    "cache_reason": frame.metadata.get("cache_reason", ""),
                }
                for frame, frame_job in zip(sequence.frames, frame_jobs)
            ]
            write_jsonl(frame_manifest, self.output_dir / "frame_manifest.jsonl")
            write_jsonl(job.segments, self.output_dir / "segment_manifest.jsonl")
            write_jsonl(job.findings, self.output_dir / "video_findings.jsonl")

            audio_decision = None
            audio_work_dir = self.output_dir / "audio_work"
            render_audio_path = None
            audio_output_dir = None
            audio_path = resolve_sidecar_audio(prepared_source, options, enabled=bool(options.get("enable_audio_sidecar", self.settings.enable_audio_sidecar)), sidecar_extensions=list(self.settings.sidecar_audio_extensions))
            if not audio_path and sequence.source_kind == "video_container" and bool(options.get("extract_native_audio", self.settings.extract_native_audio)):
                extract_start = time.monotonic()
                audio_path = extract_audio_track(prepared_source, self.output_dir / "native_audio", ffmpeg_bin=self.settings.ffmpeg_bin)
                job.step_latencies["native_audio_extract"] = (time.monotonic() - extract_start) * 1000
                if audio_path:
                    job.asset.metadata["native_audio_extracted"] = True

            if audio_path:
                job.asset.audio_uri = audio_path
                self._set_status(job, VideoJobStatus.AUDIO_PROCESSING)
                audio_start = time.monotonic()
                try:
                    audio_overrides = {
                        **resolved_selection.audio_config_overrides(),
                        "operator_id": str(options.get("operator_id") or "CMP_008"),
                        "dataset_name": f"{str(options.get('operator_id') or 'CMP_008')}-{dataset_id or Path(prepared_source).stem}",
                    }
                    audio_decision, audio_output_dir, redacted_audio_path = run_audio_sidecar(
                        audio_path,
                        audio_work_dir,
                        config_overrides=audio_overrides,
                    )
                    job.provider_versions["audio_pipeline"] = "AudioTextApiBridge"
                    job.asset.metadata["audio_output_dir"] = str(audio_output_dir.resolve())
                    if redacted_audio_path:
                        job.asset.metadata["redacted_audio_path"] = redacted_audio_path
                    render_audio_path = _choose_audio_for_render(audio_path, redacted_audio_path, audio_decision)
                except Exception:
                    if self.settings.fail_on_audio_error:
                        raise
                    logger.exception("Audio sidecar processing failed for %s", audio_path)
                    job.asset.metadata["audio_error"] = "sidecar_processing_failed"
                job.step_latencies["audio_pipeline"] = (time.monotonic() - audio_start) * 1000

            self._set_status(job, VideoJobStatus.POLICY_EVALUATING)
            policy_start = time.monotonic()
            job.policy_result = aggregate_policy(frame_jobs, profile, audio_decision=audio_decision)
            frame_risks = build_frame_risks(
                sequence.frames,
                frame_jobs,
                asset_id=asset_id or job.asset.asset_id,
                operator_selection=resolved_selection,
            )
            audio_risks = build_audio_risks(
                audio_output_dir,
                asset_id=asset_id or job.asset.asset_id,
                operator_selection=resolved_selection,
            )
            clip_risks = []
            clip_moderation_audits = []
            if bool(options.get("enable_clip_moderation", self.settings.clip_moderation_enabled)):
                clip_start = time.monotonic()
                clip_risks, clip_moderation_audits = moderate_clip_windows(
                    sequence.frames,
                    clip_windows,
                    ClipModerationConfig(
                        base_url=str(options.get("clip_moderation_base_url") or self.settings.clip_moderation_base_url),
                        endpoint=str(options.get("clip_moderation_endpoint") or self.settings.clip_moderation_endpoint),
                        timeout_seconds=int(options.get("clip_moderation_timeout_seconds", self.settings.clip_moderation_timeout_seconds)),
                        max_frames=int(options.get("clip_moderation_max_frames", self.settings.clip_moderation_max_frames)),
                        confidence_threshold=float(options.get("clip_moderation_confidence_threshold", self.settings.clip_moderation_confidence_threshold)),
                        fail_on_error=bool(options.get("clip_moderation_fail_on_error", self.settings.clip_moderation_fail_on_error)),
                    ),
                    asset_id=asset_id or job.asset.asset_id,
                    operator_selection=resolved_selection,
                )
                write_jsonl(clip_moderation_audits, self.output_dir / "clip_moderation_audit.jsonl")
                job.provider_versions["clip_moderator"] = "qwen_video_action_recognition"
                job.step_latencies["clip_moderation"] = (time.monotonic() - clip_start) * 1000
            else:
                write_jsonl([], self.output_dir / "clip_moderation_audit.jsonl")
                job.step_latencies["clip_moderation"] = 0.0
            job.risk_annotations = aggregate_risks(
                frame_risks + audio_risks + clip_risks,
                gap_tolerance_ms=int(options.get("risk_gap_tolerance_ms", self.settings.track_gap_tolerance_ms)),
                iou_threshold=float(options.get("risk_iou_threshold", self.settings.track_iou_threshold)),
            )
            _attach_representative_frames(job.risk_annotations, sequence.frames)
            sam3_tracking_report = self._sam3_video_tracking_report(job.risk_annotations, sequence.frames, options)
            job.asset.metadata["sam3_video_tracking"] = sam3_tracking_report
            if sam3_tracking_report.get("applied"):
                job.provider_versions["sam3_video_tracker"] = "sam3_video_tracker_api"
            enrich_risk_tracks(
                job.risk_annotations,
                sequence.frames,
                gap_tolerance_ms=int(options.get("risk_gap_tolerance_ms", self.settings.track_gap_tolerance_ms)),
                iou_threshold=float(options.get("risk_iou_threshold", self.settings.track_iou_threshold)),
            )
            job.display_risks = consolidate_display_risks(job.risk_annotations)
            job.risk_tracks = build_risk_tracks(job.risk_annotations, sequence.frames)
            job.governance_result = evaluate_policy(
                job.risk_annotations,
                context,
                profile=profile,
                operator_selection=resolved_selection.selection,
                preserved_targets=job.preserved_training_targets,
            )
            job.action_plan = build_action_plan(job.risk_annotations, job.governance_result, context, options=options)
            review_queue = build_review_queue(job.risk_annotations, job.governance_result, job.action_plan)
            quality_report = build_quality_report(
                sequence,
                frame_jobs,
                job.risk_annotations,
                scene_windows,
                clip_windows,
                clip_moderation_audits,
            )
            write_jsonl(review_queue, self.output_dir / "review_queue.jsonl")
            write_json(quality_report, self.output_dir / "quality_report.json")
            job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

            compliant_path = None
            preview_path = None
            if job.action_plan.render_redacted_asset:
                self._set_status(job, VideoJobStatus.RENDERING)
                render_start = time.monotonic()
                compliant_path, preview_path = render_redacted_derivative(
                    sequence,
                    frame_jobs,
                    job.action_plan,
                    self.output_dir,
                    render_preview=bool(options.get("render_preview", self.settings.render_preview)),
                    ffmpeg_bin=self.settings.ffmpeg_bin,
                    audio_path=render_audio_path,
                )
                job.step_latencies["render"] = (time.monotonic() - render_start) * 1000
            else:
                job.step_latencies["render"] = 0.0

            if compliant_path:
                compliant_suffix = Path(compliant_path).suffix or ".gif"
                job.asset.compliant_video_uri = self.storage.save(compliant_path, f"{job.job_id}/compliant_video{compliant_suffix}")
            if preview_path:
                preview_suffix = Path(preview_path).suffix or ".gif"
                job.asset.metadata["preview_uri"] = self.storage.save(preview_path, f"{job.job_id}/preview{preview_suffix}")
            job.asset.frame_manifest_uri = self.storage.save(str((self.output_dir / "frame_manifest.jsonl").resolve()), f"{job.job_id}/frame_manifest.jsonl")
            compliance_output = export_governance_artifacts(
                self.output_dir,
                self.run_id,
                job.risk_annotations,
                job.governance_result,
                job.action_plan,
                context,
                operator_selection=resolved_selection.selection,
                preserved_targets=job.preserved_training_targets,
                display_risks=job.display_risks,
                extra_artifacts={
                    "frame_manifest": str((self.output_dir / "frame_manifest.jsonl").resolve()),
                    "scene_manifest": str((self.output_dir / "scene_manifest.jsonl").resolve()),
                    "clip_windows": str((self.output_dir / "clip_windows.jsonl").resolve()),
                    "clip_moderation_audit": str((self.output_dir / "clip_moderation_audit.jsonl").resolve()),
                    "segment_manifest": str((self.output_dir / "segment_manifest.jsonl").resolve()),
                    "video_findings": str((self.output_dir / "video_findings.jsonl").resolve()),
                    "risk_tracks": str((self.output_dir / "risk_tracks.json").resolve()),
                    "display_risks": str((self.output_dir / "display_risks.jsonl").resolve()),
                    "review_queue": str((self.output_dir / "review_queue.jsonl").resolve()),
                    "quality_report": str((self.output_dir / "quality_report.json").resolve()),
                },
            )
            write_json(job.risk_tracks, self.output_dir / "risk_tracks.json")
            job.asset.metadata["compliance_output_uri"] = str((self.output_dir / "compliance_output.json").resolve())

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
                    "scene_manifest": str((self.output_dir / "scene_manifest.jsonl").resolve()),
                    "clip_windows": str((self.output_dir / "clip_windows.jsonl").resolve()),
                    "clip_moderation_audit": str((self.output_dir / "clip_moderation_audit.jsonl").resolve()),
                    "segment_manifest": str((self.output_dir / "segment_manifest.jsonl").resolve()),
                    "video_findings": str((self.output_dir / "video_findings.jsonl").resolve()),
                    "compliant_video": compliant_path or "",
                    "preview": preview_path or "",
                    "audio_work_dir": str(audio_work_dir.resolve()) if audio_path else "",
                    "render_audio_path": render_audio_path or "",
                    "risk_annotations": str((self.output_dir / "risk_annotations.jsonl").resolve()),
                    "display_risks": str((self.output_dir / "display_risks.jsonl").resolve()),
                    "risk_tracks": str((self.output_dir / "risk_tracks.json").resolve()),
                    "review_queue": str((self.output_dir / "review_queue.jsonl").resolve()),
                    "quality_report": str((self.output_dir / "quality_report.json").resolve()),
                    "preserved_training_targets": str((self.output_dir / "preserved_training_targets.jsonl").resolve()),
                    "operator_selection_snapshot": str((self.output_dir / "operator_selection_snapshot.json").resolve()),
                    "operator_catalog_snapshot": str((self.output_dir / "operator_catalog_snapshot.json").resolve()),
                    "policy_decision": str((self.output_dir / "policy_decision.json").resolve()),
                    "action_plan": str((self.output_dir / "action_plan.json").resolve()),
                    "annotation_overlay": str((self.output_dir / "annotation_overlay.json").resolve()),
                    "audit_package": str((self.output_dir / "audit_package.jsonl").resolve()),
                    "compliance_output": str((self.output_dir / "compliance_output.json").resolve()),
                },
                latency_ms=dict(job.step_latencies),
                risk_summary={
                    **risk_summary(job.risk_annotations),
                    "display": risk_summary(job.display_risks),
                },
                display_risks=job.display_risks,
                governance_decision=job.governance_result.model_dump(mode="json") if job.governance_result else {},
                action_plan=job.action_plan.model_dump(mode="json") if job.action_plan else {},
                preserved_training_targets=[item.model_dump(mode="json") for item in job.preserved_training_targets],
            )
            report_path = self.output_dir / "report.json"
            write_json(report, report_path)
            job.asset.report_uri = self.storage.save(str(report_path.resolve()), f"{job.job_id}/report.json")

            job.completed_at = datetime.now(timezone.utc)
            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            self._set_status(job, VideoJobStatus.DROPPED if job.policy_result and job.policy_result.decision == VideoDecisionType.DROP else VideoJobStatus.DONE)
            return job

        except Exception as exc:
            logger.exception("Video pipeline failed for %s", input_path)
            job.error = str(exc)
            job.error_detail = type(exc).__name__
            job.completed_at = datetime.now(timezone.utc)
            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            self._set_status(job, VideoJobStatus.FAILED)
            return job

    def _set_status(self, job: VideoJob, status: VideoJobStatus) -> None:
        job.status = status
        job.updated_at = datetime.now(timezone.utc)

    def _picture_api_config(self, options: dict[str, object]) -> PictureApiConfig:
        return PictureApiConfig(
            base_url=str(options.get("picture_api_base_url") or self.settings.picture_api_base_url),
            submit_path=str(options.get("picture_api_submit_path") or self.settings.picture_api_submit_path),
            status_path=str(options.get("picture_api_status_path") or self.settings.picture_api_status_path),
            report_path=str(options.get("picture_api_report_path") or self.settings.picture_api_report_path),
            health_path=str(options.get("picture_api_health_path") or self.settings.picture_api_health_path),
            timeout_seconds=int(options.get("picture_api_timeout_seconds") or self.settings.picture_api_timeout_seconds),
            task_timeout_seconds=int(options.get("picture_api_task_timeout_seconds") or self.settings.picture_api_task_timeout_seconds),
            poll_interval_seconds=float(options.get("picture_api_poll_interval_seconds") or self.settings.picture_api_poll_interval_seconds),
        )

    def _sam3_video_tracking_report(self, risks, frames, options: dict[str, object]) -> dict[str, object]:  # type: ignore[no-untyped-def]
        enabled = bool(options.get("sam3_video_tracking_enabled", self.settings.sam3_video_tracking_enabled))
        if not enabled:
            return {"enabled": False, "applied": False, "backend": "sam3_video_tracker"}
        return enrich_with_sam3_video_tracking(
            risks,
            frames,
            Sam3VideoTrackerConfig(
                base_url=str(options.get("sam3_video_tracker_base_url") or self.settings.sam3_video_tracker_base_url),
                endpoint=str(options.get("sam3_video_tracker_endpoint") or self.settings.sam3_video_tracker_endpoint),
                timeout_seconds=int(options.get("sam3_video_tracker_timeout_seconds") or self.settings.sam3_video_tracker_timeout_seconds),
                fail_on_error=bool(options.get("sam3_video_tracker_fail_on_error", self.settings.sam3_video_tracker_fail_on_error)),
                return_masks=bool(options.get("sam3_video_tracker_return_masks", self.settings.sam3_video_tracker_return_masks)),
            ),
        )


def _choose_audio_for_render(original_audio_path: str, redacted_audio_path: str | None, audio_decision: AudioDecision | None) -> str | None:
    if redacted_audio_path and Path(redacted_audio_path).exists():
        return redacted_audio_path
    if audio_decision in {None, AudioDecision.ALLOW} and Path(original_audio_path).exists():
        return original_audio_path
    return None


def _attach_representative_frames(risks, frames) -> None:  # type: ignore[no-untyped-def]
    if not frames:
        return
    frame_by_id = {frame.frame_id: frame for frame in frames}
    for risk in risks:
        frame = frame_by_id.get(risk.frame_ids[0]) if risk.frame_ids else None
        if frame is None:
            span_start = getattr(risk.span, "start_ms", 0) or 0
            frame = min(frames, key=lambda item: abs((item.pts_ms or 0) - span_start))
        if frame is None:
            continue
        risk.metadata["representative_frame_id"] = frame.frame_id
        risk.metadata["representative_frame_uri"] = frame.image_uri
        risk.metadata["representative_frame_pts_ms"] = frame.pts_ms


def _build_risk_tracks(risks) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    tracks: list[dict[str, object]] = []
    for index, risk in enumerate(risks):
        if not risk.track_id:
            risk.track_id = f"track_{index + 1:04d}"
        tracks.append({
            "track_id": risk.track_id,
            "risk_id": risk.risk_id,
            "category": risk.category,
            "source_modality": risk.source_modality,
            "operator_id": risk.operator_id,
            "source_operator_id": risk.source_operator_id,
            "severity": risk.severity,
            "confidence": risk.confidence,
            "span": risk.span.model_dump(mode="json"),
            "frame_ids": risk.frame_ids,
            "representative_frame_id": risk.metadata.get("representative_frame_id", ""),
            "representative_frame_uri": risk.metadata.get("representative_frame_uri", ""),
            "region_count": len(risk.regions),
            "audio_text": (risk.audio_segment or {}).get("text", "") if risk.audio_segment else "",
            "text_span": risk.text_span or "",
        })
    return tracks


def _collect_provider_versions(frame_jobs) -> dict[str, str]:  # type: ignore[no-untyped-def]
    providers: dict[str, str] = {}
    for frame_job in frame_jobs:
        for key, value in frame_job.provider_versions.items():
            providers[f"picture_{key}"] = value
    providers["video_pipeline"] = "VideoCompliancePipeline"
    return providers
