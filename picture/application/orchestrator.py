"""
Picture compliance orchestrator.

Implements a unified image entry with smart gates:
light precheck -> OCR gate -> text gates -> visual gates -> merge ->
minimal redaction -> education value check -> policy decision -> output.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor
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
    run_text_content_detection,
    run_text_pii_detection,
    run_vision_detection,
    suppress_code_adjacent_text_findings,
    suppress_textual_visual_findings,
)
from picture.providers.text_compliance import run_text_pipeline_for_ocr
from picture.domain.enums import DecisionType, FindingType, JobStatus, RouteType, SafetyCategory
from picture.domain.exceptions import UnsupportedMediaError
from picture.domain.models import BBox, PictureAsset, PictureFinding, PictureJob, PictureReport, Polygon, RegionMask
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

from common.contracts import ComplianceOutput
from common.enums import Modality
from common.runtime import PipelineExecutionContext, TrustEvaluator

logger = logging.getLogger(__name__)

_SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "image/gif",
    "application/pdf",
}

_HIGH_RISK_TEXT_CONTENT_CODES = {
    "OCR_TEXT_VIOLENCE",
    "OCR_TEXT_SEXUAL",
    "OCR_TEXT_SELF_HARM",
    "OCR_TEXT_HATE",
    "OCR_TEXT_ILLEGAL",
}

_TEXT_CONTENT_LABEL_ALIASES = {
    "content.political": {"political", "politics", "ocr_text_political"},
    "content.pornographic": {"sexual", "sexual_content", "pornographic", "explicit", "ocr_text_sexual"},
    "content.violent": {"violence", "violent", "graphic_violence", "terror", "ocr_text_violence"},
    "content.hate": {"hate", "hate_speech", "discrimination", "ocr_text_hate"},
    "content.harassment": {"harassment", "abuse", "insult"},
    "content.self_harm": {"self_harm", "suicide", "ocr_text_self_harm"},
    "content.illegal_instruction": {"illegal", "illegal_instruction", "dangerous", "ocr_text_illegal"},
    "content.minor_harmful": {"minor_harmful", "minor", "child_harm"},
    "content.misleading": {"misleading", "fraud", "scam"},
    "content.values_violation": {"values_violation", "value_risk"},
    "content.jailbreak": {"jailbreak", "prompt_injection", "prompt_attack"},
}

_VISUAL_SAFETY_LABEL_ALIASES = {
    "visual.political": {"political"},
    "visual.pornographic": {"explicit", "sexual", "nudity", "other_nsfw", SafetyCategory.EXPLICIT.value, SafetyCategory.OTHER_NSFW.value},
    "visual.violent": {
        "violence",
        "violent",
        "graphic_violence",
        "dangerous",
        "fight",
        "fighting",
        "assault",
        "physical_conflict",
        "crowd_conflict",
        "bodily_conflict",
        "肢体冲突",
        "斗殴",
        "打架",
        "暴力行为",
        SafetyCategory.GRAPHIC_VIOLENCE.value,
        SafetyCategory.DANGEROUS.value,
    },
    "visual.hate": {"hate", "hate_symbol", SafetyCategory.HATE_SYMBOL.value},
    "visual.harassment": {"harassment", "abuse"},
    "visual.self_harm": {"self_harm", "suicide", SafetyCategory.SELF_HARM.value},
    "visual.illegal_instruction": {"dangerous", "weapon", "drug", SafetyCategory.DANGEROUS.value},
    "visual.minor_harmful": {"minor_harmful", "minor"},
    "visual.misleading": {"misleading", "fraud", "scam"},
    "visual.values_violation": {"values_violation"},
    "visual.jailbreak": {"jailbreak", "prompt_injection"},
}
_VISUAL_SAFETY_LABEL_ALIASES.update({
    label.replace("visual.", "content.", 1): set(aliases)
    for label, aliases in list(_VISUAL_SAFETY_LABEL_ALIASES.items())
})


class PictureComplianceOrchestrator:
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
        self.exec_ctx: PipelineExecutionContext | None = None

    def execute(self, job: PictureJob) -> PictureJob:
        total_start = time.monotonic()
        self.exec_ctx = PipelineExecutionContext(pipeline_run_id=job.job_id)
        try:
            self._update_status(job, JobStatus.PREPROCESSING)
            self._validate_input(job)
            source_path = self._resolve_source(job)
            work_dir = self._get_work_dir(job)

            preprocess_start = time.monotonic()
            preprocessed_path = run_preprocess(
                self._preprocessor,
                source_path,
                str(work_dir / "preprocess"),
            )
            job.step_latencies["preprocess"] = (time.monotonic() - preprocess_start) * 1000
            job.asset = PictureAsset(
                original_uri=job.source.uri,
                preprocessed_uri=preprocessed_path,
                mime_type=job.source.mime_type,
            )

            job.route = RouteType.UNIFIED
            job.precheck = self._build_precheck(job, preprocessed_path)
            self._record_step_audit(job, "light_precheck", True, "", job.precheck)
            self._update_status(job, JobStatus.ROUTED)
            self._execute_unified_chain(job, preprocessed_path, work_dir)

            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            if job.status not in {JobStatus.DROPPED, JobStatus.FAILED}:
                self._update_status(job, JobStatus.DONE)
            job.completed_at = datetime.now(timezone.utc)
            self._generate_report(job, work_dir)
            job._compliance_output = self._build_compliance_output(job)
            self._repo.save_job(job)
            logger.info(
                "Job %s completed: decision=%s action=%s findings=%d",
                job.job_id,
                job.policy_result.decision.value if job.policy_result else "N/A",
                job.policy_result.dataset_action if job.policy_result else "N/A",
                len(job.findings),
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

    def _execute_unified_chain(self, job: PictureJob, image_path: str, work_dir: Path) -> None:
        self._update_status(job, JobStatus.DETECTING)

        ocr_required, ocr_skip_reason = self._should_run_ocr(job)
        ocr_result = None
        if ocr_required:
            self._mark_current_step(job, "ocr", self._ocr.name)
            start = time.monotonic()
            ocr_result = run_ocr_layout(self._ocr, image_path)
            ocr_result.metadata = {
                **dict(ocr_result.metadata or {}),
                "image_width": job.precheck.get("image_width", 0),
                "image_height": job.precheck.get("image_height", 0),
            }
            job.step_latencies["ocr_layout"] = (time.monotonic() - start) * 1000
            job.provider_versions["ocr"] = self._ocr.name
            ocr_signals = self._ocr_signals(ocr_result)
            job.precheck.update(ocr_signals)
            self._record_step_audit(job, "ocr", True, "", ocr_signals)
            job.ocr_result = ocr_result
        else:
            job.precheck.update({"ocr_executed": False, "ocr_skip_reason": ocr_skip_reason})
            self._record_step_audit(job, "ocr", False, ocr_skip_reason, job.precheck)

        text_gate_passed, text_gate_reason, text_gate_signals = self._text_gate(job, ocr_result)
        self._record_step_audit(
            job,
            "text_compliance_gate",
            text_gate_passed,
            text_gate_reason,
            text_gate_signals,
        )

        pii_findings: list[PictureFinding] = []
        text_content_findings: list[PictureFinding] = []
        text_pipeline_enabled = self._text_pipeline_enabled()
        moderation_result = None
        safety_findings: list[PictureFinding] = []
        vision_findings: list[PictureFinding] = []
        ocr_quality_findings: list[PictureFinding] = []
        futures: dict[str, Any] = {}
        self._mark_current_step(job, "parallel_detection")
        with ThreadPoolExecutor(max_workers=3) as pool:
            if text_pipeline_enabled and text_gate_passed and ocr_result is not None and (
                self._should_run_text_privacy(job) or self._should_run_text_content(job)
            ):
                futures["text_compliance"] = pool.submit(
                    self._run_ocr_text_compliance_step,
                    job,
                    ocr_result,
                    work_dir,
                )
            elif text_pipeline_enabled:
                self._record_step_audit(
                    job,
                    "ocr_text_compliance_reuse",
                    False,
                    text_gate_reason or "文本合规闸门未通过，未调用文本合规流水线。",
                    text_gate_signals,
                )
            if (not text_pipeline_enabled) and self._should_run_text_privacy(job) and text_gate_passed and ocr_result is not None:
                futures["text_pii"] = pool.submit(self._run_local_text_pii_step, job, ocr_result)
            if (not text_pipeline_enabled) and self._should_run_text_content(job) and text_gate_passed and ocr_result is not None:
                futures["text_content"] = pool.submit(self._run_local_text_content_step, job, ocr_result)
            if self._should_run_visual_safety(job):
                futures["visual_safety"] = pool.submit(self._run_visual_safety_step, job, image_path)
            else:
                self._record_step_audit(
                    job,
                    "visual_content_safety",
                    False,
                    "用户关闭视觉内容安全或当前流程不要求视觉内容安全检测。",
                    {},
                )
            if self._should_run_visual_sensitive_objects(job):
                futures["vision_detect"] = pool.submit(self._run_vision_detection_step, job, image_path)
            else:
                self._record_step_audit(
                    job,
                    "visual_sensitive_object_detection",
                    False,
                    "用户关闭视觉敏感对象检测或当前流程不需要隐私治理。",
                    {},
                )

            if "text_compliance" in futures:
                result = futures["text_compliance"].result()
                self._apply_step_result(job, result)
                pii_findings = result["pii_findings"]
                text_content_findings = result["text_content_findings"]
            if "text_pii" in futures:
                result = futures["text_pii"].result()
                self._apply_step_result(job, result)
                pii_findings = result["findings"]
            elif not (text_pipeline_enabled and text_gate_passed and ocr_result is not None and self._should_run_text_privacy(job)):
                skip_reason = self._skip_reason_for_text_step(job, "privacy", text_gate_passed, text_gate_reason)
                self._record_step_audit(job, "ocr_text_privacy_detection", False, skip_reason, text_gate_signals)
            if "text_content" in futures:
                result = futures["text_content"].result()
                self._apply_step_result(job, result)
                text_content_findings = result["findings"]
            elif not (text_pipeline_enabled and text_gate_passed and ocr_result is not None and self._should_run_text_content(job)):
                skip_reason = self._skip_reason_for_text_step(job, "content", text_gate_passed, text_gate_reason)
                self._record_step_audit(job, "ocr_text_content_safety", False, skip_reason, text_gate_signals)
            if "visual_safety" in futures:
                result = futures["visual_safety"].result()
                self._apply_step_result(job, result)
                moderation_result = result["moderation_result"]
                safety_findings = self._moderation_to_findings(moderation_result)
            if "vision_detect" in futures:
                result = futures["vision_detect"].result()
                self._apply_step_result(job, result)
                vision_findings = result["findings"]

        ocr_quality_findings = self._ocr_quality_findings(job, ocr_result, moderation_result)
        if ocr_quality_findings:
            self._record_step_audit(
                job,
                "ocr_quality_guard",
                True,
                "",
                {
                    "finding_count": len(ocr_quality_findings),
                    "reason": "文档类图片 OCR 为空，禁止按无风险图片直接放行。",
                },
            )

        pii_count_before_code_suppression = len(pii_findings)
        pii_findings = suppress_code_adjacent_text_findings(pii_findings, vision_findings)
        suppressed_code_text_count = pii_count_before_code_suppression - len(pii_findings)
        if suppressed_code_text_count > 0:
            self._record_step_audit(
                job,
                "ocr_code_adjacent_text_suppression",
                True,
                "",
                {
                    "suppressed_count": suppressed_code_text_count,
                    "reason": "条形码/二维码视觉区域附近的 OCR 数字段归并为码类隐私风险，不再作为账号或电话等独立文本隐私风险展示。",
                },
            )

        vision_count_before_textual_suppression = len(vision_findings)
        vision_findings = suppress_textual_visual_findings(vision_findings, pii_findings)
        suppressed_textual_visual_count = vision_count_before_textual_suppression - len(vision_findings)
        if suppressed_textual_visual_count > 0:
            self._record_step_audit(
                job,
                "visual_textual_duplicate_suppression",
                True,
                "",
                {
                    "suppressed_count": suppressed_textual_visual_count,
                    "reason": "手机号、邮箱、姓名等文字型隐私目标优先使用 OCR 文本定位，抑制不可靠的视觉分割重复结果。",
                },
            )

        all_findings = merge_findings(
            pii_findings,
            text_content_findings,
            safety_findings,
            vision_findings,
            ocr_quality_findings,
        )
        job.findings = all_findings

        self._update_status(job, JobStatus.SEGMENTING)
        seg_candidates = [
            finding for finding in all_findings
            if self._should_refine_with_segmentation(finding)
        ]
        self._mark_current_step(job, "segmentation", self._segmentation.name)
        seg_start = time.monotonic()
        try:
            redaction_findings = run_segmentation_refinement(
                self._segmentation,
                image_path,
                seg_candidates,
            )
        except Exception as exc:
            job.degrade_events.append(
                {
                    "step": "segmentation",
                    "reason": f"segmentation failed, fallback to bbox redaction: {type(exc).__name__}: {exc}",
                    "fallback": "bbox_redaction",
                }
            )
            redaction_findings = seg_candidates
        self._merge_refined_regions(all_findings, redaction_findings)
        all_findings = merge_findings([self._clip_finding_region(finding, job) for finding in all_findings])
        job.findings = all_findings
        job.provider_versions["segmentation"] = self._segmentation.name
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000

        redaction_config = self._get_redaction_config(job)
        operations = build_redaction_operations(all_findings, redaction_config)
        job.redaction_operations = operations

        output_path = str(work_dir / "compliant.png")
        overlay_path = str(work_dir / "overlay.png")
        if operations:
            self._update_status(job, JobStatus.REDACTING)
            start = time.monotonic()
            compliant_path, overlay_result = run_redaction(
                self._redactor,
                image_path,
                operations,
                output_path,
                overlay_path,
            )
            job.step_latencies["redaction"] = (time.monotonic() - start) * 1000
            job.compliant_image_uri = self._storage.save(compliant_path, f"{job.job_id}/compliant.png")
            if overlay_result:
                job.overlay_image_uri = self._storage.save(overlay_result, f"{job.job_id}/overlay.png")
            self._record_step_audit(
                job,
                "minimal_redaction",
                True,
                "",
                {"redaction_count": len(operations)},
            )
        else:
            job.compliant_image_uri = self._storage.save(image_path, f"{job.job_id}/compliant.png")
            self._record_step_audit(job, "minimal_redaction", False, "未发现需要脱敏的对象级区域。", {})

        education_value_preserved = self._education_value_check(job, image_path)
        self._record_step_audit(
            job,
            "education_value_check",
            True,
            "",
            {"education_value_preserved": education_value_preserved},
        )

        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_context = {
            "ordinary_dataset_enabled": self._ordinary_dataset_enabled(job),
            "restricted_dataset_enabled": self._restricted_dataset_enabled(job),
            "restricted_use_case": str(job.options.get("restricted_use_case", "") or "").strip(),
            "authorized_sensitive_use": bool(job.options.get("authorized_sensitive_use", False)),
            "education_value_preserved": education_value_preserved,
            "ocr_executed": bool(ocr_result is not None),
            "executed_steps": [item["step"] for item in job.step_audits if item.get("executed")],
            "skipped_steps": [
                {"step": item["step"], "reason": item.get("skip_reason", "")}
                for item in job.step_audits
                if not item.get("executed")
            ],
        }
        job.policy_result = self._policy.evaluate(
            all_findings,
            moderation_result,
            job.profile,
            context=policy_context,
        )

        if job.policy_result.decision == DecisionType.DROP:
            self._update_status(job, JobStatus.DROPPED)

    def _build_precheck(self, job: PictureJob, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        width, height = 0, 0
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            pass
        mime_type = job.source.mime_type or mimetypes.guess_type(path.name)[0] or ""
        file_size = path.stat().st_size if path.exists() else 0
        return {
            "image_width": width,
            "image_height": height,
            "file_size": file_size,
            "mime_type": mime_type,
            "enable_total_compliance": self._enable_total_compliance(job),
            "ocr_disable_requested": bool(job.options.get("disable_ocr", False)),
            "ocr_policy": "always_run",
            "visual_safety_disabled": bool(job.options.get("disable_visual_safety", False)),
            "visual_sensitive_objects_disabled": bool(job.options.get("disable_visual_sensitive_objects", False)),
            "ordinary_dataset_enabled": self._ordinary_dataset_enabled(job),
            "restricted_dataset_enabled": self._restricted_dataset_enabled(job),
            "picture_mode": str(job.options.get("picture_mode", "") or ""),
            "selected_operators": {
                "privacy": self._selected_list(job, "privacy_operator_ids"),
                "content_safety": self._selected_list(job, "content_safety_operator_ids"),
                "visual_safety": self._selected_list(job, "visual_safety_operator_ids"),
                "visual_sensitive_object": self._selected_list(job, "visual_sensitive_object_operator_ids"),
            },
        }

    def _ocr_signals(self, ocr_result: Any) -> dict[str, Any]:
        blocks = list(ocr_result.text_blocks)
        text_length = len((ocr_result.full_text or "").strip())
        block_count = len(blocks)
        metadata = dict(getattr(ocr_result, "metadata", {}) or {})
        valid_text = bool(metadata.get("valid_text", text_length > 0))
        spatially_mappable_text = bool(metadata.get("spatially_mappable_text", block_count > 0))
        mean_conf = (
            sum(float(block.confidence) for block in blocks) / block_count if block_count else 0.0
        )
        return {
            "ocr_executed": True,
            "ocr_skip_reason": "",
            "ocr_text_length": text_length,
            "ocr_block_count": block_count,
            "ocr_mean_confidence": round(mean_conf, 4),
            "ocr_valid_text": valid_text,
            "ocr_spatially_mappable_text": spatially_mappable_text,
            "ocr_invalid_reason": str(metadata.get("invalid_reason") or ""),
            "ocr_generation_passes": list(metadata.get("generation_passes") or []),
            "ocr_effective_text_preview": str(metadata.get("effective_text_preview") or "")[:300],
        }

    def _text_gate(self, job: PictureJob, ocr_result: Any) -> tuple[bool, str, dict[str, Any]]:
        if ocr_result is None:
            return False, "OCR 未执行，无法进入文本合规闸门。", {}

        signals = self._ocr_signals(ocr_result)
        signals.update({
            "text_gate_policy": "run_text_compliance_when_ocr_text_is_valid_non_empty_and_spatially_mappable",
        })
        if not (self._should_run_text_privacy(job) or self._should_run_text_content(job)):
            return False, "用户未选择任何文字类检测。", signals
        if not signals.get("ocr_valid_text", signals["ocr_text_length"] > 0):
            reason = signals.get("ocr_invalid_reason") or "invalid_ocr_text"
            return False, f"OCR 未提取出有效文字，原因：{reason}，因此跳过文本合规检测。", signals
        if signals["ocr_text_length"] <= 0:
            return False, "OCR 未识别出文字，文本长度为 0，因此跳过文本合规检测。", signals
        if not signals.get("ocr_spatially_mappable_text", signals.get("ocr_block_count", 0) > 0):
            return False, "OCR 文本缺少可回贴到图片的空间索引，因此跳过自动文本合规检测。", signals
        return True, "", signals

    def _ocr_quality_findings(
        self,
        job: PictureJob,
        ocr_result: Any,
        moderation_result: Any,
    ) -> list[PictureFinding]:
        if ocr_result is None:
            return []
        signals = self._ocr_signals(ocr_result)
        if signals.get("ocr_text_length", 0) > 0 and signals.get("ocr_valid_text", True):
            return []
        if not self._looks_like_text_document(job, moderation_result):
            return []

        explanation = (
            "图片具有明显文档/票据/表格特征，但 PaddleOCR-VL 未提取出有效文字。"
            "为避免漏检地址、税号、账号、电话等隐私信息，禁止按无风险图片直接放行，需人工复核或重新 OCR。"
        )
        return [
            PictureFinding(
                finding_type=FindingType.TEXT_CONTENT,
                category="ocr_extraction_failed",
                label="OCR文字提取失败",
                score=1.0,
                reason_code="OCR_TEXT_EXTRACTION_FAILED",
                provider=getattr(self._ocr, "name", "OCR"),
                explanation=explanation,
                metadata={
                    "ocr_text_length": signals.get("ocr_text_length", 0),
                    "ocr_block_count": signals.get("ocr_block_count", 0),
                    "ocr_invalid_reason": signals.get("ocr_invalid_reason", ""),
                    "ocr_generation_passes": signals.get("ocr_generation_passes", []),
                    "requires_manual_review": True,
                    "source": "ocr_quality_guard",
                },
            )
        ]

    def _looks_like_text_document(self, job: PictureJob, moderation_result: Any) -> bool:
        mode = str(job.options.get("picture_mode") or "").strip().lower()
        if mode in {"ocr", "ocr_text", "document", "document_ocr", "text_only"}:
            return True

        source_name = Path(str(job.source.uri or "")).name.lower()
        name_markers = (
            "document",
            "ocr",
            "invoice",
            "receipt",
            "bill",
            "form",
            "table",
            "privacy",
            "idcard",
            "passport",
            "tax",
        )
        if any(marker in source_name for marker in name_markers):
            return True

        metadata = dict(getattr(moderation_result, "metadata", {}) or {})
        explanation = str(metadata.get("explanation") or "").lower()
        doc_markers = (
            "发票",
            "账单",
            "票据",
            "表格",
            "文档",
            "单据",
            "收据",
            "证件",
            "合同",
            "invoice",
            "receipt",
            "bill",
            "table",
            "document",
            "form",
            "tax",
        )
        return any(marker in explanation for marker in doc_markers)

    def _skip_reason_for_text_step(
        self,
        job: PictureJob,
        step_kind: str,
        text_gate_passed: bool,
        text_gate_reason: str,
    ) -> str:
        if step_kind == "privacy" and not self._should_run_text_privacy(job):
            return "用户关闭隐私检测或当前流程不要求 OCR 文本隐私检测。"
        if step_kind == "content" and not self._should_run_text_content(job):
            return "用户未选择 OCR 文本内容合规检测。"
        if not text_gate_passed:
            return text_gate_reason
        return "当前流程无需执行该步骤。"

    def _record_step_audit(
        self,
        job: PictureJob,
        step: str,
        executed: bool,
        skip_reason: str,
        input_signals: dict[str, Any],
    ) -> None:
        job.step_audits.append(
            {
                "step": step,
                "executed": executed,
                "skip_reason": skip_reason,
                "input_signals": input_signals,
            }
        )

    def _mark_current_step(self, job: PictureJob, step: str, provider: str = "") -> None:
        job.precheck["current_step"] = step
        if provider:
            job.precheck["current_provider"] = provider
        job.precheck["current_step_started_at"] = datetime.now(timezone.utc).isoformat()
        self._repo.save_job(job)

    def _apply_step_result(self, job: PictureJob, result: dict[str, Any]) -> None:
        for key, value in dict(result.get("latencies") or {}).items():
            job.step_latencies[key] = value
        for key, value in dict(result.get("provider_versions") or {}).items():
            job.provider_versions[key] = value
        for audit in list(result.get("audits") or []):
            if isinstance(audit, dict):
                self._record_step_audit(
                    job,
                    str(audit.get("step") or ""),
                    bool(audit.get("executed", False)),
                    str(audit.get("skip_reason") or ""),
                    dict(audit.get("input_signals") or {}),
                )
        if "moderation_result" in result:
            job.moderation_result = result["moderation_result"]
        if "text_content_findings" in result:
            job.text_content_findings = result["text_content_findings"]

    def _run_ocr_text_compliance_step(
        self,
        job: PictureJob,
        ocr_result: Any,
        work_dir: Path,
    ) -> dict[str, Any]:
        start = time.monotonic()
        profile = self._text_pipeline_profile(job)
        text_findings = run_text_pipeline_for_ocr(
            ocr_result,
            profile=profile,
            run_id=job.job_id,
            work_dir=work_dir,
            text_api_base_url=str(getattr(self._settings, "text_api_base_url", "") or ""),
            timeout_seconds=float(getattr(self._settings, "text_api_timeout_seconds", 300.0)),
            poll_interval_seconds=float(getattr(self._settings, "text_api_poll_interval_seconds", 2.0)),
            config_overrides={
                "privacy_operator_ids": self._selected_list(job, "privacy_operator_ids"),
                "privacy_target_types": self._selected_list(job, "privacy_target_types"),
                "content_safety_operator_ids": self._selected_list(job, "content_safety_operator_ids"),
                "content_safety_target_labels": self._selected_list(job, "content_safety_target_labels"),
            },
        )
        provider_name = "text.api_pipeline.APICompliancePipeline"
        text_findings = self._filter_selected_text_findings(job, text_findings)
        pii_findings = [
            finding for finding in text_findings
            if finding.finding_type == FindingType.TEXT_PII and self._should_run_text_privacy(job)
        ]
        text_content_findings = [
            finding for finding in text_findings
            if finding.finding_type == FindingType.TEXT_CONTENT and self._should_run_text_content(job)
        ]
        provider_versions = {"ocr_text_compliance": provider_name}
        if self._should_run_text_privacy(job):
            provider_versions["pii"] = provider_name
        audits = [
            {
                "step": "ocr_text_compliance_reuse",
                "executed": True,
                "skip_reason": "",
                "input_signals": {
                    "profile": profile,
                    "finding_count": len(text_findings),
                    "pii_finding_count": len(pii_findings),
                    "content_finding_count": len(text_content_findings),
                    "mapped_region_count": sum(1 for finding in text_findings if finding.region is not None),
                    "unmapped_region_count": sum(1 for finding in text_findings if finding.region is None),
                    "artifact_imported_count": sum(
                        1 for finding in text_findings
                        if (finding.metadata or {}).get("source_kind") in {
                            "privacy_audit",
                            "redaction_target",
                            "content_safety_artifact",
                        }
                    ),
                    "privacy_operator_ids": self._selected_list(job, "privacy_operator_ids"),
                    "content_safety_operator_ids": self._selected_list(job, "content_safety_operator_ids"),
                    "qwen_reused_from_text_compliance": True,
                    "degraded": False,
                },
            },
        ]
        if self._should_run_text_privacy(job):
            audits.append({"step": "ocr_text_privacy_detection", "executed": True, "skip_reason": "", "input_signals": {"finding_count": len(pii_findings), "provider": "text_compliance_pipeline", "privacy_operator_ids": self._selected_list(job, "privacy_operator_ids")}})
        if self._should_run_text_content(job):
            audits.append({"step": "ocr_text_content_safety", "executed": True, "skip_reason": "", "input_signals": {"finding_count": len(text_content_findings), "provider": "text_compliance_pipeline", "content_safety_operator_ids": self._selected_list(job, "content_safety_operator_ids")}})
        return {
            "pii_findings": pii_findings,
            "text_content_findings": text_content_findings,
            "latencies": {"ocr_text_compliance_reuse": (time.monotonic() - start) * 1000},
            "provider_versions": provider_versions,
            "audits": audits,
        }

    def _run_local_text_pii_step(self, job: PictureJob, ocr_result: Any) -> dict[str, Any]:
        start = time.monotonic()
        pii_findings = run_text_pii_detection(self._pii, ocr_result)
        pii_findings = self._filter_selected_text_findings(job, pii_findings)
        return {
            "findings": pii_findings,
            "latencies": {"text_pii": (time.monotonic() - start) * 1000},
            "provider_versions": {"pii": self._pii.name},
            "audits": [{"step": "ocr_text_privacy_detection", "executed": True, "skip_reason": "", "input_signals": {"finding_count": len(pii_findings), "privacy_operator_ids": self._selected_list(job, "privacy_operator_ids")}}],
        }

    def _run_local_text_content_step(self, job: PictureJob, ocr_result: Any) -> dict[str, Any]:
        start = time.monotonic()
        text_content_findings = run_text_content_detection(ocr_result)
        text_content_findings = self._filter_selected_text_findings(job, text_content_findings)
        return {
            "findings": text_content_findings,
            "text_content_findings": text_content_findings,
            "latencies": {"text_content": (time.monotonic() - start) * 1000},
            "audits": [{"step": "ocr_text_content_safety", "executed": True, "skip_reason": "", "input_signals": {"finding_count": len(text_content_findings), "content_safety_operator_ids": self._selected_list(job, "content_safety_operator_ids")}}],
        }

    def _run_visual_safety_step(self, job: PictureJob, image_path: str) -> dict[str, Any]:
        start = time.monotonic()
        moderation_result = run_safety_moderation(self._safety, image_path)
        moderation_result = self._filter_selected_moderation(job, moderation_result)
        return {
            "moderation_result": moderation_result,
            "latencies": {"visual_safety": (time.monotonic() - start) * 1000},
            "provider_versions": {"safety": self._safety.name},
            "audits": [{
                "step": "visual_content_safety",
                "executed": True,
                "skip_reason": "",
                "input_signals": {
                "is_safe": moderation_result.is_safe,
                "reason_codes": moderation_result.reason_codes,
                "visual_safety_operator_ids": self._selected_list(job, "visual_safety_operator_ids"),
                "review_required": bool((moderation_result.metadata or {}).get("review_required", False)),
                "evidence_region_count": len((moderation_result.metadata or {}).get("evidence_regions") or []),
                },
            }],
        }

    def _run_vision_detection_step(self, job: PictureJob, image_path: str) -> dict[str, Any]:
        start = time.monotonic()
        target_types = self._selected_list(job, "visual_sensitive_object_types") or self._default_visual_sensitive_object_types(job)
        vision_findings = run_vision_detection(self._vision, image_path, target_types=target_types)
        vision_findings = self._filter_visual_findings_by_types(vision_findings, target_types)
        qwen_confirmed_count = sum(
            1 for finding in vision_findings
            if bool((finding.metadata or {}).get("qwen_semantic_confirmed", False))
        )
        localized_count = sum(1 for finding in vision_findings if finding.region is not None)
        unlocalized_count = sum(
            1 for finding in vision_findings
            if bool((finding.metadata or {}).get("localization_required", False)) or finding.region is None
        )
        return {
            "findings": vision_findings,
            "latencies": {"vision_detect": (time.monotonic() - start) * 1000},
            "provider_versions": {"vision": self._vision.name},
            "audits": [{
                "step": "visual_sensitive_object_detection",
                "executed": True,
                "skip_reason": "",
                "input_signals": {
                    "finding_count": len(vision_findings),
                    "qwen_sensitive_present_count": qwen_confirmed_count,
                    "sam3_localized_count": localized_count,
                    "unlocalized_sensitive_count": unlocalized_count,
                    "manual_region_required_count": unlocalized_count,
                    "visual_sensitive_object_operator_ids": self._selected_list(job, "visual_sensitive_object_operator_ids"),
                    "visual_sensitive_object_types": target_types,
                },
            }],
        }

    def _moderation_to_findings(self, moderation: Any) -> list[PictureFinding]:
        if moderation is None or moderation.is_safe:
            return []
        findings: list[PictureFinding] = []
        categories = list(moderation.categories) or [SafetyCategory.OTHER_NSFW]
        evidence_regions = list((moderation.metadata or {}).get("evidence_regions") or [])
        localized_violations = list((moderation.metadata or {}).get("localized_violations") or [])
        object_level_safety = bool(localized_violations) or any(
            isinstance(item, dict) and (item.get("violation_id") or item.get("entity_label_en") or item.get("mask_path"))
            for item in evidence_regions
        )
        qwen_evidence_hints = list((moderation.metadata or {}).get("qwen_evidence_hints") or [])
        unlocalized_categories = {
            str(item).lower()
            for item in ((moderation.metadata or {}).get("unlocalized_safety_categories") or [])
        }
        explanation = str((moderation.metadata or {}).get("explanation") or "")
        category_details = dict((moderation.metadata or {}).get("category_details") or {})
        if object_level_safety:
            handled_violation_ids: set[str] = set()
            for item in evidence_regions:
                if not isinstance(item, dict):
                    continue
                cat_value = str(item.get("category") or SafetyCategory.OTHER_NSFW.value)
                if cat_value == SafetyCategory.SAFE.value:
                    continue
                violation_id = str(item.get("violation_id") or "")
                if violation_id:
                    handled_violation_ids.add(violation_id)
                object_name = str(item.get("entity_label_zh") or item.get("label") or item.get("entity_label_en") or "").strip()
                decision_hint = str(item.get("decision_hint") or "").strip().lower()
                risk_subtype = str(item.get("risk_subtype") or "").strip()
                region = _region_from_evidence_item(item)
                review_required = bool(item.get("review_required", False)) or region is None
                findings.append(
                    PictureFinding(
                        finding_type=FindingType.SAFETY,
                        category=cat_value,
                        label=_safety_finding_label(cat_value, object_name, risk_subtype=risk_subtype, decision_hint=decision_hint),
                        score=float(item.get("confidence") or moderation.scores.get(cat_value, 1.0)),
                        region=region,
                        reason_code=f"SAFETY_{cat_value.upper()}",
                        provider=moderation.provider,
                        threshold_used=0.7,
                        explanation=str((item.get("local_review") or {}).get("reason_zh") or item.get("description") or explanation or _default_safety_explanation(cat_value, object_name)),
                        metadata={
                            "moderation_result": True,
                            "review_required": review_required,
                            "evidence_regions": [item],
                            "localization_status": str(item.get("localization_status") or ("localized_by_sam3_mask" if region and region.mask_path else "localized_by_sam3")),
                            "localization_required": region is None,
                            "violation_id": violation_id,
                            "entity_label_en": str(item.get("entity_label_en") or ""),
                            "entity_label_zh": object_name,
                            "risk_subtype": risk_subtype,
                            "decision_hint": decision_hint,
                            "center_point": item.get("center_point"),
                            "boundary_status": item.get("boundary_status"),
                            "mask_path": item.get("mask_path"),
                            "polygons": item.get("polygons"),
                            "mask_area": item.get("mask_area"),
                            "mask_area_ratio": item.get("mask_area_ratio"),
                            "mask_bbox_fill_ratio": item.get("mask_bbox_fill_ratio"),
                            "mask_quality_score": item.get("mask_quality_score"),
                            "sam3_refined": bool(item.get("sam3_refined", False)),
                            "sam3_refine_rejected": bool(item.get("sam3_refine_rejected", False)),
                            "fallback_to_original_sam3_mask": bool(item.get("fallback_to_original_sam3_mask", False)),
                            "localization_attempt": item.get("localization_attempt"),
                        },
                    )
                )
            for violation in localized_violations:
                if not isinstance(violation, dict):
                    continue
                violation_id = str(violation.get("violation_id") or "")
                if violation_id and violation_id in handled_violation_ids:
                    continue
                if str(violation.get("localization_status") or "") != "unlocalized":
                    continue
                cat_value = str(violation.get("category") or SafetyCategory.OTHER_NSFW.value)
                object_name = str(violation.get("entity_label_zh") or violation.get("entity_label_en") or "").strip()
                decision_hint = str(violation.get("decision_hint") or "").strip().lower()
                risk_subtype = str(violation.get("risk_subtype") or "").strip()
                findings.append(
                    PictureFinding(
                        finding_type=FindingType.SAFETY,
                        category=cat_value,
                        label=_safety_finding_label(cat_value, object_name, risk_subtype=risk_subtype, decision_hint=decision_hint),
                        score=float(violation.get("confidence") or moderation.scores.get(cat_value, 1.0)),
                        region=None,
                        reason_code=f"SAFETY_{cat_value.upper()}",
                        provider=moderation.provider,
                        threshold_used=0.7,
                        explanation=explanation or _default_safety_explanation(cat_value, object_name),
                        metadata={
                            "moderation_result": True,
                            "review_required": True,
                            "localization_status": "unlocalized",
                            "localization_required": True,
                            "violation_id": violation_id,
                            "entity_label_en": str(violation.get("entity_label_en") or ""),
                            "entity_label_zh": object_name,
                            "risk_subtype": risk_subtype,
                            "decision_hint": decision_hint,
                            "center_point": violation.get("center_point"),
                        },
                    )
                )
            return findings
        for category in categories:
            cat_value = getattr(category, "value", str(category))
            if cat_value == SafetyCategory.SAFE.value:
                continue
            score = float(moderation.scores.get(cat_value, 1.0))
            detail = dict(category_details.get(cat_value) or {})
            object_name = str(detail.get("object_name_zh") or detail.get("risk_subtype_zh") or "").strip()
            risk_subtype = str(detail.get("risk_subtype") or detail.get("risk_subtype_zh") or "").strip()
            decision_hint = str(detail.get("decision_hint") or "").strip().lower()
            risk_reason = str(detail.get("risk_reason_zh") or detail.get("scene_description_zh") or "").strip()
            evidence_for_category = [
                item for item in evidence_regions
                if str(item.get("category") or item.get("label") or "").lower() in {cat_value.lower(), object_name.lower()}
            ] or evidence_regions
            qwen_hints_for_category = [
                item for item in qwen_evidence_hints
                if str(item.get("category") or "").lower() in {"", cat_value.lower()}
            ]
            label = _safety_finding_label(cat_value, object_name, risk_subtype=risk_subtype, decision_hint=decision_hint)
            finding_explanation = risk_reason or explanation or _default_safety_explanation(cat_value, object_name)
            region = _region_from_evidence_regions(evidence_for_category)
            localization_status = "localized_by_sam3" if region is not None else (
                "qwen_hint_only" if qwen_hints_for_category else "unlocalized"
            )
            review_required = bool((moderation.metadata or {}).get("review_required", False)) or region is None or cat_value.lower() in unlocalized_categories
            findings.append(
                PictureFinding(
                    finding_type=FindingType.SAFETY,
                    category=cat_value,
                    label=label,
                    score=score,
                    region=region,
                    reason_code=f"SAFETY_{cat_value.upper()}",
                    provider=moderation.provider,
                    threshold_used=0.7,
                    explanation=finding_explanation,
                    metadata={
                        "moderation_result": True,
                        "review_required": review_required,
                        "evidence_regions": evidence_for_category,
                        "qwen_evidence_hints": qwen_hints_for_category,
                        "localization_status": localization_status,
                        "localization_required": region is None,
                        "object_name_zh": object_name,
                        "risk_subtype_zh": str(detail.get("risk_subtype_zh") or ""),
                        "risk_subtype": risk_subtype,
                        "decision_hint": decision_hint,
                        "scene_description_zh": str(detail.get("scene_description_zh") or ""),
                        "risk_reason_zh": risk_reason,
                        "mask_quality_score": _best_mask_quality(evidence_for_category),
                    },
                )
            )
        return findings

    def _clip_finding_region(self, finding: PictureFinding, job: PictureJob) -> PictureFinding:
        if finding.region is None:
            return finding
        width = float(job.precheck.get("image_width", 0) or 0)
        height = float(job.precheck.get("image_height", 0) or 0)
        if width <= 0 or height <= 0:
            return finding
        bbox = finding.region.bbox
        x = max(0.0, min(float(bbox.x), width - 1.0))
        y = max(0.0, min(float(bbox.y), height - 1.0))
        right = max(x + 1.0, min(float(bbox.x + bbox.w), width))
        bottom = max(y + 1.0, min(float(bbox.y + bbox.h), height))
        return finding.model_copy(
            update={
                "region": finding.region.model_copy(
                    update={"bbox": BBox(x=x, y=y, w=max(1.0, right - x), h=max(1.0, bottom - y))}
                )
            }
        )

    def _should_refine_with_segmentation(self, finding: PictureFinding) -> bool:
        if finding.region is None:
            return False
        category = str(finding.category or "").lower()
        if finding.finding_type == FindingType.VISION_OBJECT:
            return category in {"id_card", "badge", "signature", "stamp", "avatar", "account_region"}
        if finding.finding_type == FindingType.SAFETY:
            metadata = finding.metadata or {}
            if bool(metadata.get("sam3_refine_rejected", False)):
                return False
            if bool(metadata.get("sam3_refined", False)) and finding.region.mask_path:
                return False
            return category in {"dangerous", "explicit", "graphic_violence", "hate_symbol", "self_harm", "other_nsfw"}
        if finding.finding_type in {FindingType.TEXT_PII, FindingType.TEXT_CONTENT}:
            return False
        return False

    def _merge_refined_regions(
        self,
        all_findings: list[PictureFinding],
        refined_findings: list[PictureFinding],
    ) -> None:
        refined_by_id = {
            finding.finding_id: finding.region
            for finding in refined_findings
            if finding.region is not None
        }
        if not refined_by_id:
            return
        for finding in all_findings:
            refined_region = refined_by_id.get(finding.finding_id)
            if refined_region is not None:
                if finding.finding_type in {FindingType.TEXT_PII, FindingType.TEXT_CONTENT}:
                    if finding.region is not None and not _ocr_refined_region_is_reliable(finding.region, refined_region):
                        finding.metadata = {
                            **dict(finding.metadata or {}),
                            "sam3_refined": False,
                            "sam3_refine_rejected": True,
                            "sam3_refine_rejected_bbox": refined_region.bbox.model_dump(mode="json"),
                            "region_source": (finding.metadata or {}).get("region_source", ""),
                            "ocr_region_source": (finding.metadata or {}).get("ocr_region_source", ""),
                            "ocr_region_quality": (finding.metadata or {}).get("ocr_region_quality", "medium"),
                        }
                        continue
                    previous_source = str((finding.metadata or {}).get("region_source") or "")
                    finding.metadata = {
                        **dict(finding.metadata or {}),
                        "pre_refine_region_source": previous_source,
                        "region_source": "sam3_refine" if refined_region.mask_path or refined_region.polygon else previous_source,
                        "ocr_region_source": "sam3_refine" if refined_region.mask_path or refined_region.polygon else previous_source,
                        "ocr_region_quality": "high" if refined_region.mask_path or refined_region.polygon else (finding.metadata or {}).get("ocr_region_quality", "medium"),
                        "sam3_refined": bool(refined_region.mask_path or refined_region.polygon),
                        "requires_manual_region_review": False,
                    }
                finding.region = refined_region

    def _enable_total_compliance(self, job: PictureJob) -> bool:
        return bool(job.options.get("enable_total_compliance", True))

    def _ordinary_dataset_enabled(self, job: PictureJob) -> bool:
        return bool(job.options.get("ordinary_dataset_enabled", True))

    def _restricted_dataset_enabled(self, job: PictureJob) -> bool:
        return bool(job.options.get("restricted_dataset_enabled", False))

    def _should_run_ocr(self, job: PictureJob) -> tuple[bool, str]:
        return True, "OCR 是图片合规检测的必执行步骤；只有 OCR 文本长度为 0 时才跳过文本合规检测。"

    def _default_visual_sensitive_object_types(self, job: PictureJob) -> list[str]:
        mode = str(job.options.get("picture_mode") or "").strip().lower()
        if mode in {"ocr", "ocr_text", "document", "document_ocr", "text_only"}:
            return ["id_card", "signature", "stamp", "qr_code", "barcode"]
        if mode in {"privacy", "privacy_only"}:
            return ["face", "id_card", "badge", "license_plate", "account_region"]
        if mode in {"visual", "visual_only"}:
            return ["face", "id_card", "badge", "qr_code", "barcode", "signature", "stamp"]
        return ["face", "id_card", "badge", "qr_code", "barcode", "signature", "stamp"]

    def _should_run_text_privacy(self, job: PictureJob) -> bool:
        explicit = job.options.get("enable_text_privacy_detection")
        if explicit is not None:
            return bool(explicit)
        return self._enable_total_compliance(job) or self._ordinary_dataset_enabled(job)

    def _should_run_text_content(self, job: PictureJob) -> bool:
        explicit = job.options.get("enable_text_content_detection")
        if explicit is not None:
            return bool(explicit)
        return self._enable_total_compliance(job)

    def _should_run_visual_safety(self, job: PictureJob) -> bool:
        if bool(job.options.get("disable_visual_safety", False)):
            return False
        explicit = job.options.get("enable_visual_safety_detection")
        if explicit is not None:
            return bool(explicit)
        return self._enable_total_compliance(job) or self._ordinary_dataset_enabled(job)

    def _should_run_visual_sensitive_objects(self, job: PictureJob) -> bool:
        if bool(job.options.get("disable_visual_sensitive_objects", False)):
            return False
        explicit = job.options.get("enable_visual_sensitive_object_detection")
        if explicit is not None:
            return bool(explicit)
        return self._enable_total_compliance(job) or self._ordinary_dataset_enabled(job)

    def _text_pipeline_enabled(self) -> bool:
        if not self._settings:
            return False
        return str(getattr(self._settings, "text_compliance_provider", "") or "").lower() in {"text_api", "text_pipeline"}

    def _text_pipeline_profile(self, job: PictureJob) -> str:
        privacy = self._should_run_text_privacy(job)
        content = self._should_run_text_content(job)
        if privacy and content:
            return "full"
        if privacy:
            return "privacy_only"
        if content:
            return "safety_only"
        return "full"

    def _selected_list(self, job: PictureJob, key: str) -> list[str]:
        value = job.options.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _filter_selected_text_findings(
        self,
        job: PictureJob,
        findings: list[PictureFinding],
    ) -> list[PictureFinding]:
        privacy_targets = set(self._selected_list(job, "privacy_target_types"))
        content_labels = set(self._selected_list(job, "content_safety_target_labels"))
        if not privacy_targets and not content_labels:
            return findings

        filtered: list[PictureFinding] = []
        content_aliases = self._expanded_aliases(content_labels, _TEXT_CONTENT_LABEL_ALIASES)
        for finding in findings:
            if finding.finding_type == FindingType.TEXT_PII:
                if not privacy_targets or self._finding_matches(finding, privacy_targets):
                    filtered.append(finding)
            elif finding.finding_type == FindingType.TEXT_CONTENT:
                if not content_labels or self._finding_matches(finding, content_aliases):
                    filtered.append(finding)
            else:
                filtered.append(finding)
        return filtered

    def _filter_selected_visual_findings(
        self,
        job: PictureJob,
        findings: list[PictureFinding],
    ) -> list[PictureFinding]:
        target_types = set(self._selected_list(job, "visual_sensitive_object_types"))
        if not target_types:
            return findings
        return [finding for finding in findings if self._finding_matches(finding, target_types)]

    def _filter_visual_findings_by_types(
        self,
        findings: list[PictureFinding],
        target_types: list[str],
    ) -> list[PictureFinding]:
        targets = {item for item in target_types if item}
        if not targets:
            return findings
        return [finding for finding in findings if self._finding_matches(finding, targets)]

    def _filter_selected_moderation(
        self,
        job: PictureJob,
        moderation: Any,
    ) -> Any:
        labels = set(self._selected_list(job, "visual_safety_target_labels"))
        if not labels or moderation is None:
            return moderation
        aliases = self._expanded_aliases(labels, _VISUAL_SAFETY_LABEL_ALIASES)
        categories = [category for category in moderation.categories if str(category.value if hasattr(category, "value") else category).lower() in aliases]
        metadata_categories = self._moderation_metadata_categories(moderation.metadata or {})
        for category in metadata_categories:
            if category.value in aliases and category not in categories:
                categories.append(category)
        scores = {
            key: value for key, value in moderation.scores.items()
            if str(key).lower() in aliases
        }
        for category in categories:
            if category != SafetyCategory.SAFE:
                scores.setdefault(category.value, max(moderation.scores.values(), default=1.0))
        reason_codes = [
            code for code in moderation.reason_codes
            if any(alias in str(code).lower() for alias in aliases)
        ]
        if categories and not reason_codes:
            reason_codes = [
                f"SAFETY_{category.value.upper()}"
                for category in categories
                if category != SafetyCategory.SAFE
            ]
        metadata = dict(moderation.metadata or {})
        if not categories and not reason_codes and not moderation.is_safe and (
            metadata.get("degraded") or metadata.get("review_required")
        ):
            categories = [category for category in moderation.categories if category != SafetyCategory.SAFE] or [SafetyCategory.OTHER_NSFW]
            for category in categories:
                scores.setdefault(category.value, max(moderation.scores.values(), default=1.0))
            reason_codes = list(moderation.reason_codes) or [
                f"SAFETY_{category.value.upper()}"
                for category in categories
                if category != SafetyCategory.SAFE
            ]
        is_safe = not categories and not reason_codes
        return moderation.model_copy(update={
            "is_safe": is_safe,
            "categories": categories or ([SafetyCategory.SAFE] if is_safe else []),
            "scores": scores,
            "reason_codes": reason_codes,
            "metadata": {
                **metadata,
                "selected_visual_safety_labels": sorted(labels),
            },
        })

    def _moderation_metadata_categories(self, metadata: dict[str, Any]) -> list[SafetyCategory]:
        raw_values: list[Any] = []
        category_details = metadata.get("category_details") or {}
        raw_values.extend(category_details.keys())
        for detail in category_details.values():
            if isinstance(detail, dict):
                raw_values.extend(detail.values())
        for key in ("violations", "localized_violations", "evidence_regions"):
            for item in metadata.get(key) or []:
                if isinstance(item, dict):
                    raw_values.append(item.get("category"))
                    raw_values.append(item.get("risk_subtype"))
                    raw_values.append(item.get("risk_subtype_zh"))
                    raw_values.append(item.get("object_name_zh"))
        explanation = str(metadata.get("explanation") or "")
        if any(token in explanation for token in ("肢体冲突", "斗殴", "打架", "暴力")):
            raw_values.append(SafetyCategory.DANGEROUS.value)
            raw_values.append(SafetyCategory.GRAPHIC_VIOLENCE.value)
        return self._metadata_values_to_safety_categories(raw_values)

    def _metadata_values_to_safety_categories(self, values: list[Any]) -> list[SafetyCategory]:
        categories: list[SafetyCategory] = []
        for value in values:
            text = str(value or "").strip().lower()
            if not text:
                continue
            category = None
            if text in {"explicit", "sexual", "nudity", "pornographic", "色情", "裸露"}:
                category = SafetyCategory.EXPLICIT
            elif text in {"graphic_violence", "violence", "violent", "fight", "fighting", "assault", "physical_conflict", "肢体冲突", "斗殴", "打架", "暴力行为"}:
                category = SafetyCategory.GRAPHIC_VIOLENCE
            elif text in {"dangerous", "weapon", "firearm", "gun", "pistol", "knife", "drug", "疑似手枪", "枪械", "刀具", "毒品"}:
                category = SafetyCategory.DANGEROUS
            elif text in {"hate", "hate_symbol"}:
                category = SafetyCategory.HATE_SYMBOL
            elif text in {"self_harm", "suicide"}:
                category = SafetyCategory.SELF_HARM
            elif text in {"other_nsfw", "nsfw"}:
                category = SafetyCategory.OTHER_NSFW
            if category and category not in categories:
                categories.append(category)
        return categories

    def _expanded_aliases(self, selected: set[str], alias_map: dict[str, set[str]]) -> set[str]:
        aliases = {item.lower() for item in selected}
        for item in selected:
            aliases.update(alias.lower() for alias in alias_map.get(item, set()))
        return aliases

    def _finding_matches(self, finding: PictureFinding, accepted: set[str]) -> bool:
        tokens = {
            str(finding.category or "").lower(),
            str(finding.label or "").lower(),
            str(finding.reason_code or "").lower(),
        }
        metadata = finding.metadata or {}
        for key in ("risk_type", "category", "label", "policy_tag", "entity_type"):
            value = metadata.get(key)
            if value:
                tokens.add(str(value).lower())
        raw = metadata.get("text_pipeline_finding")
        if isinstance(raw, dict):
            for key in ("risk_type", "category", "label", "policy_tag", "entity_type", "type", "finding_type"):
                value = raw.get(key)
                if value:
                    tokens.add(str(value).lower().replace(".", "_"))
        normalized = {token.replace(".", "_").replace("-", "_") for token in tokens}
        accepted_normalized = {item.lower().replace(".", "_").replace("-", "_") for item in accepted}
        return bool(normalized & accepted_normalized) or any(
            accepted_item in token or token in accepted_item
            for token in normalized
            for accepted_item in accepted_normalized
            if token and accepted_item
        )

    def _education_value_check(self, job: PictureJob, image_path: str) -> bool:
        if not job.redaction_operations:
            return True
        width = int(job.precheck.get("image_width", 0) or 0)
        height = int(job.precheck.get("image_height", 0) or 0)
        if width <= 0 or height <= 0:
            return True
        total_area = width * height
        redaction_area = 0.0
        for operation in job.redaction_operations:
            bbox = operation.region.bbox
            redaction_area += max(0.0, bbox.w) * max(0.0, bbox.h)
        ratio = redaction_area / total_area if total_area else 0.0
        max_ratio = float(job.options.get("max_redaction_area_ratio") or (getattr(self._settings, "max_redaction_area_ratio", 0.45) if self._settings else 0.45))
        if ratio > max_ratio:
            return False
        return True

    def _validate_input(self, job: PictureJob) -> None:
        mime = job.source.mime_type.lower()
        if mime and mime not in _SUPPORTED_MIME_TYPES:
            raise UnsupportedMediaError(mime)

    def _resolve_source(self, job: PictureJob) -> str:
        uri = job.source.uri
        if uri.startswith("local://"):
            return uri.replace("local://", "")
        if uri.startswith("s3://"):
            work_dir = self._get_work_dir(job)
            local_path = str(work_dir / "input" / Path(uri).name)
            return self._storage.load(uri, local_path)
        return uri

    def _get_work_dir(self, job: PictureJob) -> Path:
        base = Path(self._settings.work_dir) if self._settings else Path("./compliance_output_picture")
        work_dir = base / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _update_status(self, job: PictureJob, status: JobStatus) -> None:
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        self._repo.save_job(job)

    def _get_redaction_config(self, job: PictureJob) -> dict[str, str]:
        if self._settings:
            config = {
                "person_name": self._settings.redaction_mode_text,
                "phone_number": self._settings.redaction_mode_text,
                "email": self._settings.redaction_mode_text,
                "id_card": self._settings.redaction_mode_text,
                "bank_card": self._settings.redaction_mode_text,
                "bank_account": self._settings.redaction_mode_text,
                "address": self._settings.redaction_mode_text,
                "student_id": self._settings.redaction_mode_text,
                "date_time": self._settings.redaction_mode_text,
                "pii_entity": self._settings.redaction_mode_text,
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
        for key in (
            "redaction_mode_text",
            "redaction_mode_face",
            "redaction_mode_qr",
            "redaction_mode_signature",
            "redaction_mode_default",
        ):
            if key in job.options:
                if key == "redaction_mode_text":
                    for category in (
                        "person_name",
                        "phone_number",
                        "email",
                        "id_card",
                        "bank_card",
                        "bank_account",
                        "address",
                        "student_id",
                        "date_time",
                        "pii_entity",
                    ):
                        config[category] = job.options[key]
                elif key == "redaction_mode_face":
                    config["face"] = job.options[key]
                elif key == "redaction_mode_qr":
                    config["qr_code"] = job.options[key]
                    config["barcode"] = job.options[key]
                elif key == "redaction_mode_signature":
                    config["signature"] = job.options[key]
                    config["stamp"] = job.options[key]
                elif key == "redaction_mode_default":
                    config["default"] = job.options[key]
                    for category in ("license_plate", "badge", "avatar", "account_region", "school_class_identifier"):
                        config[category] = job.options[key]
        return config

    def _generate_report(self, job: PictureJob, work_dir: Path) -> None:
        report = PictureReport(
            job_id=job.job_id,
            route=job.route or RouteType.UNIFIED,
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
            precheck=job.precheck,
            step_audits=job.step_audits,
            policy_snapshot=job.policy_result.model_dump(mode="json") if job.policy_result else {},
        )
        report_path = work_dir / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2))
        job.report_uri = self._storage.save(str(report_path), f"{job.job_id}/report.json")

    def _build_compliance_output(self, job: PictureJob) -> ComplianceOutput:
        from common.adapters import (
            build_annotation_package,
            build_audit_package,
            build_compliance_output,
            build_release_package,
            deduplicate_evidence_units,
            picture_finding_to_evidence,
        )
        from common.policy import evaluate_with_profile, load_policy_profile

        evidence_units = [picture_finding_to_evidence(finding) for finding in job.findings]
        evidence_units = deduplicate_evidence_units(evidence_units)

        ctx = self.exec_ctx or PipelineExecutionContext(pipeline_run_id=job.job_id)
        profile = load_policy_profile("default")
        policy_result = evaluate_with_profile(
            evidence_units,
            profile=profile,
            degrade_events=ctx.degrade_events,
        )
        trust_level = TrustEvaluator.evaluate(ctx)
        unified_decision = policy_result.decision

        content_uri = (
            job.compliant_image_uri
            or (job.asset.original_uri if job.asset else "")
            or job.source.uri
        )
        annotation_pkg = build_annotation_package(
            modality=Modality.PICTURE,
            pipeline_run_id=job.job_id,
            clean_content_uri=content_uri,
            content_format=job.source.mime_type if job.source else "image/png",
            evidence_units=evidence_units,
            decision=unified_decision,
            trust_level=trust_level,
        )
        audit_pkg = build_audit_package(
            modality=Modality.PICTURE,
            pipeline_run_id=job.job_id,
            evidence_units=evidence_units,
            degrade_events=ctx.degrade_events,
            policy_result=policy_result,
            ctx=ctx,
        )
        release_pkg = build_release_package(
            modality=Modality.PICTURE,
            pipeline_run_id=job.job_id,
            annotation_package=annotation_pkg,
            audit_package=audit_pkg,
            decision=unified_decision,
            trust_level=trust_level,
        )

        job.annotation_package_uri = content_uri
        job.audit_package_uri = job.report_uri
        job.trust_level = trust_level.value

        legacy = job.policy_result.model_dump(mode="json") if job.policy_result else None
        return build_compliance_output(
            pipeline_run_id=job.job_id,
            modality=Modality.PICTURE,
            decision=unified_decision,
            trust_level=trust_level,
            release_package=release_pkg,
            degrade_summary=policy_result.degrade_summary,
            review_suggestions=policy_result.review_suggestions,
            explanation_summary=audit_pkg.review_summary,
            legacy_decision=legacy,
        )


def _safety_finding_label(category: str, object_name: str, *, risk_subtype: str = "", decision_hint: str = "") -> str:
    if decision_hint == "redact_only" and (risk_subtype == "exposed_upper_body" or object_name in {"裸露上身", "裸露躯干", "裸露胸腹", "裸露背部"}):
        return f"裸露身体区域：{object_name or '裸露上身'}"
    category_label = {
        "explicit": "色情裸露内容",
        "graphic_violence": "血腥暴力内容",
        "hate_symbol": "仇恨极端符号",
        "self_harm": "自伤自杀内容",
        "dangerous": "违法危险内容",
        "other_nsfw": "低俗不适内容",
    }.get(category, "视觉内容安全风险")
    return f"{category_label}：{object_name}" if object_name else category_label


def _region_from_evidence_regions(evidence_regions: list[dict[str, Any]]) -> RegionMask | None:
    boxes: list[tuple[float, float, float, float, float]] = []
    for item in evidence_regions:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((x, y, w, h, confidence))
    if not boxes:
        return None
    left = min(item[0] for item in boxes)
    top = min(item[1] for item in boxes)
    right = max(item[0] + item[2] for item in boxes)
    bottom = max(item[1] + item[3] for item in boxes)
    confidence = sum(item[4] for item in boxes) / len(boxes)
    return RegionMask(
        bbox=BBox(x=left, y=top, w=max(1.0, right - left), h=max(1.0, bottom - top)),
        confidence=confidence,
    )


def _region_from_evidence_item(item: dict[str, Any]) -> RegionMask | None:
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    polygon = None
    raw_polygon = item.get("polygon")
    if not isinstance(raw_polygon, list):
        raw_polygons = item.get("polygons")
        if isinstance(raw_polygons, list) and raw_polygons:
            raw_polygon = raw_polygons[0]
    if isinstance(raw_polygon, list):
        points: list[tuple[float, float]] = []
        for point in raw_polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if points:
            polygon = Polygon(points=points)
    return RegionMask(
        bbox=BBox(x=x, y=y, w=w, h=h),
        polygon=polygon,
        mask_path=str(item.get("mask_path") or "") or None,
        confidence=confidence,
    )


def _best_mask_quality(items: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for item in items:
        value = item.get("mask_quality_score")
        try:
            if value is not None:
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _ocr_refined_region_is_reliable(original: RegionMask, refined: RegionMask) -> bool:
    original_box = original.bbox
    refined_box = refined.bbox
    original_area = max(1.0, float(original_box.w) * float(original_box.h))
    refined_area = max(1.0, float(refined_box.w) * float(refined_box.h))
    if refined_area < original_area * 0.12:
        return False
    if refined_area > original_area * 6.0:
        return False
    if _bbox_iou_region(original_box, refined_box) < 0.08:
        return False
    original_cx = float(original_box.x) + float(original_box.w) / 2.0
    original_cy = float(original_box.y) + float(original_box.h) / 2.0
    refined_cx = float(refined_box.x) + float(refined_box.w) / 2.0
    refined_cy = float(refined_box.y) + float(refined_box.h) / 2.0
    max_shift = max(float(original_box.w), float(original_box.h), 1.0) * 1.25
    return ((original_cx - refined_cx) ** 2 + (original_cy - refined_cy) ** 2) ** 0.5 <= max_shift


def _bbox_iou_region(a: BBox, b: BBox) -> float:
    ax2 = float(a.x) + float(a.w)
    ay2 = float(a.y) + float(a.h)
    bx2 = float(b.x) + float(b.w)
    by2 = float(b.y) + float(b.h)
    inter_w = max(0.0, min(ax2, bx2) - max(float(a.x), float(b.x)))
    inter_h = max(0.0, min(ay2, by2) - max(float(a.y), float(b.y)))
    inter = inter_w * inter_h
    area_a = max(1.0, float(a.w) * float(a.h))
    area_b = max(1.0, float(b.w) * float(b.h))
    return inter / max(1.0, area_a + area_b - inter)


def _default_safety_explanation(category: str, object_name: str) -> str:
    if object_name in {"裸露上身", "裸露躯干", "裸露胸腹", "裸露背部"}:
        return f"视觉多模态模型识别到{object_name}，按数据交付策略需要局部脱敏。"
    target = object_name or "图片内容"
    category_label = {
        "explicit": "色情裸露",
        "graphic_violence": "血腥暴力",
        "hate_symbol": "仇恨极端符号",
        "self_harm": "自伤自杀",
        "dangerous": "违法危险",
        "other_nsfw": "低俗不适",
    }.get(category, "内容安全")
    return f"视觉多模态模型识别到{target}，属于{category_label}风险，需要按图片合规策略处置。"
