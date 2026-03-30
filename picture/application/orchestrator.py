"""
Picture compliance orchestrator.

Implements three processing chains:
A. Document image chain   (document)
B. Natural image chain    (natural)
C. Mixed screenshot chain (mixed) — runs OCR + safety in parallel

The orchestrator manages job lifecycle, provider injection,
error handling, timing, and report generation.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from picture.application.services import (
    build_redaction_operations,
    merge_findings,
    run_ocr_layout,
    run_preprocess,
    run_redaction,
    run_safety_moderation,
    run_segmentation_refinement,
    run_text_pii_detection,
    run_vision_detection,
)
from picture.domain.enums import DecisionType, JobStatus, RouteType
from picture.domain.exceptions import PictureError, UnsupportedMediaError
from picture.domain.models import (
    PictureAsset,
    PictureFinding,
    PictureJob,
    PictureModerationResult,
    PictureReport,
)
from picture.domain.policy import ConfigurablePolicyEngine
from picture.providers.base import (
    JobRepository,
    OCRLayoutProvider,
    PIIDetector,
    Preprocessor,
    Redactor,
    Router,
    SafetyModerator,
    SegmentationProvider,
    StorageBackend,
    VisionDetector,
)

logger = logging.getLogger(__name__)

# Supported MIME types
_SUPPORTED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/tiff", "image/bmp", "image/gif",
    "application/pdf",
}


class PictureComplianceOrchestrator:
    """
    Main orchestrator for picture compliance processing.

    Manages the full lifecycle of a compliance job:
    1. Validate input & create job
    2. Route to the appropriate chain
    3. Execute processing steps
    4. Generate report and persist results
    """

    def __init__(
        self,
        router: Router,
        preprocessor: Preprocessor,
        ocr_provider: OCRLayoutProvider,
        pii_detector: PIIDetector,
        safety_moderator: SafetyModerator,
        vision_detector: VisionDetector,
        segmentation_provider: SegmentationProvider,
        redactor: Redactor,
        policy_engine: ConfigurablePolicyEngine,
        storage: StorageBackend,
        repository: JobRepository,
        settings: Any | None = None,
    ) -> None:
        self._router = router
        self._preprocessor = preprocessor
        self._ocr = ocr_provider
        self._pii = pii_detector
        self._safety = safety_moderator
        self._vision = vision_detector
        self._segmentation = segmentation_provider
        self._redactor = redactor
        self._policy = policy_engine
        self._storage = storage
        self._repo = repository
        self._settings = settings

    def execute(self, job: PictureJob) -> PictureJob:
        """
        Execute the full compliance pipeline for a job.

        This is the main entry point. It:
        1. Validates the input
        2. Preprocesses the image
        3. Routes to the correct chain
        4. Executes the chain
        5. Generates report
        6. Persists results
        """
        total_start = time.monotonic()
        try:
            self._update_status(job, JobStatus.PREPROCESSING)

            # ── Validate ────────────────────────────────────────────
            self._validate_input(job)

            # ── Resolve input file ──────────────────────────────────
            image_path = self._resolve_source(job)

            # ── Preprocess ──────────────────────────────────────────
            work_dir = self._get_work_dir(job)
            preprocess_start = time.monotonic()
            preprocessed_path = run_preprocess(
                self._preprocessor, image_path, str(work_dir / "preprocess")
            )
            job.step_latencies["preprocess"] = (time.monotonic() - preprocess_start) * 1000
            job.asset = PictureAsset(
                original_uri=job.source.uri,
                preprocessed_uri=preprocessed_path,
                mime_type=job.source.mime_type,
            )

            # ── Route ───────────────────────────────────────────────
            route_start = time.monotonic()
            route_hint = job.options.get("route_hint", "auto")
            route = self._router.classify(preprocessed_path, {"route_hint": route_hint})
            job.route = route
            job.step_latencies["route"] = (time.monotonic() - route_start) * 1000
            self._update_status(job, JobStatus.ROUTED)

            # ── Execute chain ───────────────────────────────────────
            if route == RouteType.DOCUMENT:
                self._execute_document_chain(job, preprocessed_path, work_dir)
            elif route == RouteType.NATURAL:
                self._execute_natural_chain(job, preprocessed_path, work_dir)
            else:
                self._execute_mixed_chain(job, preprocessed_path, work_dir)

            # ── Finalize ────────────────────────────────────────────
            total_elapsed = (time.monotonic() - total_start) * 1000
            job.step_latencies["total"] = total_elapsed

            if job.status not in (JobStatus.DROPPED, JobStatus.FAILED):
                self._update_status(job, JobStatus.DONE)

            job.completed_at = datetime.now(timezone.utc)

            # ── Generate report ─────────────────────────────────────
            self._generate_report(job, work_dir)

            self._repo.save_job(job)
            logger.info(
                "Job %s completed: decision=%s, route=%s, findings=%d, latency=%.1fms",
                job.job_id,
                job.policy_result.decision.value if job.policy_result else "N/A",
                route.value,
                len(job.findings),
                total_elapsed,
            )

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.error_detail = type(exc).__name__
            job.completed_at = datetime.now(timezone.utc)
            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            self._repo.save_job(job)
            logger.exception("Job %s failed: %s", job.job_id, exc)

        return job

    # ─────────────────────────────────────────────────────────────────
    # Chain A: Document image
    # ─────────────────────────────────────────────────────────────────

    def _execute_document_chain(
        self, job: PictureJob, image_path: str, work_dir: Path
    ) -> None:
        """
        Document image chain:
        1. OCR/layout → 2. Text PII detect → 3. Vision detect →
        4. Segmentation refine → 5. Redaction → 6. Policy evaluate → 7. Output
        """
        logger.info("Executing DOCUMENT chain for job %s", job.job_id)
        self._update_status(job, JobStatus.DETECTING)

        # 1. OCR + layout
        ocr_start = time.monotonic()
        ocr_result = run_ocr_layout(self._ocr, image_path)
        job.ocr_result = ocr_result
        job.step_latencies["ocr_layout"] = (time.monotonic() - ocr_start) * 1000
        job.provider_versions["ocr"] = self._ocr.name

        # 2. Text PII detection
        pii_start = time.monotonic()
        pii_findings = run_text_pii_detection(self._pii, ocr_result)
        job.step_latencies["text_pii"] = (time.monotonic() - pii_start) * 1000
        job.provider_versions["pii"] = self._pii.name

        # 3. Vision detection
        vision_start = time.monotonic()
        vision_findings = run_vision_detection(self._vision, image_path)
        job.step_latencies["vision_detect"] = (time.monotonic() - vision_start) * 1000
        job.provider_versions["vision"] = self._vision.name

        # 4. Merge findings
        all_findings = merge_findings(pii_findings, vision_findings)
        job.findings = all_findings

        # 5. Segmentation refinement
        self._update_status(job, JobStatus.SEGMENTING)
        seg_start = time.monotonic()
        all_findings = run_segmentation_refinement(
            self._segmentation, image_path, all_findings
        )
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000
        job.provider_versions["segmentation"] = self._segmentation.name

        # 6. Policy evaluation
        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_start = time.monotonic()
        policy_result = self._policy.evaluate(all_findings, None, job.profile)
        job.policy_result = policy_result
        job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

        # 7. Redaction or drop
        self._apply_decision(job, image_path, all_findings, work_dir)

    # ─────────────────────────────────────────────────────────────────
    # Chain B: Natural image
    # ─────────────────────────────────────────────────────────────────

    def _execute_natural_chain(
        self, job: PictureJob, image_path: str, work_dir: Path
    ) -> None:
        """
        Natural image chain:
        1. Safety moderation → 2. Vision detect → 3. Segmentation refine →
        4. Redaction or drop → 5. Policy evaluate → 6. Output
        """
        logger.info("Executing NATURAL chain for job %s", job.job_id)
        self._update_status(job, JobStatus.DETECTING)

        # 1. Safety moderation
        safety_start = time.monotonic()
        moderation_result = run_safety_moderation(self._safety, image_path)
        job.moderation_result = moderation_result
        job.step_latencies["safety"] = (time.monotonic() - safety_start) * 1000
        job.provider_versions["safety"] = self._safety.name

        # 2. Vision detection
        vision_start = time.monotonic()
        vision_findings = run_vision_detection(self._vision, image_path)
        job.step_latencies["vision_detect"] = (time.monotonic() - vision_start) * 1000
        job.provider_versions["vision"] = self._vision.name

        job.findings = vision_findings

        # 3. Segmentation refinement
        self._update_status(job, JobStatus.SEGMENTING)
        seg_start = time.monotonic()
        vision_findings = run_segmentation_refinement(
            self._segmentation, image_path, vision_findings
        )
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000
        job.provider_versions["segmentation"] = self._segmentation.name

        # 4. Policy evaluation (includes safety moderation in decision)
        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_start = time.monotonic()
        policy_result = self._policy.evaluate(
            vision_findings, moderation_result, job.profile
        )
        job.policy_result = policy_result
        job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

        # 5. Redaction or drop
        self._apply_decision(job, image_path, vision_findings, work_dir)

    # ─────────────────────────────────────────────────────────────────
    # Chain C: Mixed screenshot
    # ─────────────────────────────────────────────────────────────────

    def _execute_mixed_chain(
        self, job: PictureJob, image_path: str, work_dir: Path
    ) -> None:
        """
        Mixed screenshot chain (dual parallel execution):
        Phase 1: OCR/layout AND safety moderation in parallel
        Phase 2: Text PII detect AND vision detect in parallel
        Phase 3: Merge → segmentation → redaction → policy → output
        """
        logger.info("Executing MIXED chain for job %s", job.job_id)
        self._update_status(job, JobStatus.DETECTING)

        # ── Phase 1: OCR + Safety in parallel ───────────────────────
        ocr_result = None
        moderation_result = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(run_ocr_layout, self._ocr, image_path)
            safety_future = executor.submit(run_safety_moderation, self._safety, image_path)

            phase1_start = time.monotonic()
            for future in as_completed([ocr_future, safety_future]):
                try:
                    result = future.result()
                    if hasattr(result, "full_text"):
                        ocr_result = result
                    else:
                        moderation_result = result
                except Exception as exc:
                    logger.warning("Phase 1 provider failed: %s", exc)

            job.step_latencies["phase1_parallel"] = (time.monotonic() - phase1_start) * 1000

        job.ocr_result = ocr_result
        job.moderation_result = moderation_result
        job.provider_versions["ocr"] = self._ocr.name
        job.provider_versions["safety"] = self._safety.name

        # ── Phase 2: PII + Vision in parallel ───────────────────────
        pii_findings: list[PictureFinding] = []
        vision_findings: list[PictureFinding] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if ocr_result:
                futures["pii"] = executor.submit(
                    run_text_pii_detection, self._pii, ocr_result
                )
            futures["vision"] = executor.submit(
                run_vision_detection, self._vision, image_path
            )

            phase2_start = time.monotonic()
            for key, future in futures.items():
                try:
                    result = future.result()
                    if key == "pii":
                        pii_findings = result
                    else:
                        vision_findings = result
                except Exception as exc:
                    logger.warning("Phase 2 provider '%s' failed: %s", key, exc)

            job.step_latencies["phase2_parallel"] = (time.monotonic() - phase2_start) * 1000

        job.provider_versions["pii"] = self._pii.name
        job.provider_versions["vision"] = self._vision.name

        # ── Phase 3: Merge → Segment → Policy → Redact ─────────────
        all_findings = merge_findings(pii_findings, vision_findings)
        job.findings = all_findings

        # Segmentation
        self._update_status(job, JobStatus.SEGMENTING)
        seg_start = time.monotonic()
        all_findings = run_segmentation_refinement(
            self._segmentation, image_path, all_findings
        )
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000
        job.provider_versions["segmentation"] = self._segmentation.name

        # Policy
        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_start = time.monotonic()
        policy_result = self._policy.evaluate(
            all_findings, moderation_result, job.profile
        )
        job.policy_result = policy_result
        job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

        # Redaction or drop
        self._apply_decision(job, image_path, all_findings, work_dir)

    # ─────────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────────

    def _validate_input(self, job: PictureJob) -> None:
        """Validate the job input."""
        mime = job.source.mime_type.lower()
        if mime and mime not in _SUPPORTED_MIME_TYPES:
            raise UnsupportedMediaError(mime)

    def _resolve_source(self, job: PictureJob) -> str:
        """Resolve the source URI to a local file path."""
        uri = job.source.uri
        if uri.startswith("local://"):
            return uri.replace("local://", "")
        if uri.startswith("s3://"):
            work_dir = self._get_work_dir(job)
            local_path = str(work_dir / "input" / Path(uri).name)
            return self._storage.load(uri, local_path)
        # Assume local file path
        return uri

    def _get_work_dir(self, job: PictureJob) -> Path:
        """Get the working directory for a job."""
        if self._settings:
            base = Path(self._settings.work_dir)
        else:
            base = Path("./compliance_output_picture")
        work_dir = base / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _update_status(self, job: PictureJob, status: JobStatus) -> None:
        """Update job status and persist."""
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        self._repo.save_job(job)

    def _apply_decision(
        self,
        job: PictureJob,
        image_path: str,
        findings: list[PictureFinding],
        work_dir: Path,
    ) -> None:
        """Apply the policy decision: redact, pass raw, or drop."""
        if job.policy_result is None:
            return

        decision = job.policy_result.decision

        if decision == DecisionType.DROP:
            self._update_status(job, JobStatus.DROPPED)
            logger.info("Job %s DROPPED by policy", job.job_id)
            return

        if decision == DecisionType.PASS_RAW:
            # No redaction needed
            compliant_uri = self._storage.save(
                image_path, f"{job.job_id}/compliant.png"
            )
            job.compliant_image_uri = compliant_uri
            return

        # PASS_REDACTED: apply redactions
        self._update_status(job, JobStatus.REDACTING)

        # Build redaction config from settings or job options
        redaction_config = self._get_redaction_config(job)

        redact_start = time.monotonic()
        operations = build_redaction_operations(findings, redaction_config)
        job.redaction_operations = operations

        output_path = str(work_dir / "compliant.png")
        overlay_path = str(work_dir / "overlay.png")

        compliant_path, overlay_result = run_redaction(
            self._redactor, image_path, operations, output_path, overlay_path
        )
        job.step_latencies["redaction"] = (time.monotonic() - redact_start) * 1000

        # Persist to storage
        job.compliant_image_uri = self._storage.save(
            compliant_path, f"{job.job_id}/compliant.png"
        )
        if overlay_result:
            job.overlay_image_uri = self._storage.save(
                overlay_result, f"{job.job_id}/overlay.png"
            )

    def _get_redaction_config(self, job: PictureJob) -> dict[str, str]:
        """Build redaction mode mapping from settings and job options."""
        if self._settings:
            config = {
                "person_name": self._settings.redaction_mode_text,
                "phone_number": self._settings.redaction_mode_text,
                "email": self._settings.redaction_mode_text,
                "id_card": self._settings.redaction_mode_text,
                "bank_card": self._settings.redaction_mode_text,
                "address": self._settings.redaction_mode_text,
                "face": self._settings.redaction_mode_face,
                "qr_code": self._settings.redaction_mode_qr,
                "barcode": self._settings.redaction_mode_qr,
                "signature": self._settings.redaction_mode_signature,
                "stamp": self._settings.redaction_mode_signature,
                "license_plate": self._settings.redaction_mode_default,
                "badge": self._settings.redaction_mode_default,
                "default": self._settings.redaction_mode_default,
            }
        else:
            config = {
                "default": "black_box",
                "face": "gaussian_blur",
                "signature": "solid_fill",
                "stamp": "solid_fill",
            }

        # Override from job options
        for key in ("redaction_mode_text", "redaction_mode_face"):
            if key in job.options:
                if key == "redaction_mode_text":
                    for cat in ("person_name", "phone_number", "email", "id_card", "bank_card", "address"):
                        config[cat] = job.options[key]
                elif key == "redaction_mode_face":
                    config["face"] = job.options[key]

        return config

    def _generate_report(self, job: PictureJob, work_dir: Path) -> None:
        """Generate and persist the audit report JSON."""
        report = PictureReport(
            job_id=job.job_id,
            route=job.route or RouteType.MIXED,
            decision=job.policy_result.decision if job.policy_result else DecisionType.PASS_RAW,
            findings=job.findings,
            moderation=job.moderation_result,
            redaction_operations=job.redaction_operations,
            provider_info=job.provider_versions,
            reason_codes=job.policy_result.reason_codes if job.policy_result else [],
            timestamps={
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else "",
            },
            latency_ms=job.step_latencies,
        )

        report_path = work_dir / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        job.report_uri = self._storage.save(
            str(report_path), f"{job.job_id}/report.json"
        )
        logger.info("Report saved to %s", job.report_uri)
