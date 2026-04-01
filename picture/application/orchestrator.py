"""
Picture compliance orchestrator.

Implements three processing chains:
A. Document image chain   (document)
B. Natural image chain    (natural)
C. Mixed screenshot chain (mixed) runs OCR + safety in parallel

The orchestrator manages job lifecycle, provider injection,
error handling, timing, and report generation.
"""
# 中文说明：该文件是 picture 模块的总编排器，负责把路由、预处理、OCR、PII、
# 安全审核、目标检测、分割、脱敏、策略判断、存储与仓储串成一条完整流水线。
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

# 中文说明：这里集中声明 picture 模块当前接受的输入 MIME 类型。
# 后续如果支持新的图片或文档格式，只需要在这里补充白名单即可。
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
        # 中文说明：编排器本身不关心具体模型实现，它只依赖抽象接口；
        # 因此可以在测试环境注入 mock provider，在生产环境注入真实模型 provider。
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
        # 中文说明：total_start 记录整个任务的端到端耗时，用于最终审计与性能分析。
        total_start = time.monotonic()
        try:
            # 中文说明：任务一开始先进入预处理状态，便于前端或 API 查询当前进度。
            self._update_status(job, JobStatus.PREPROCESSING)

            # 中文说明：先做最基础的输入合法性校验，尽早拦截不支持的格式。
            self._validate_input(job)

            # 中文说明：统一把输入解析成当前机器可访问的本地路径；
            # 对于 local:// 直接去前缀，对于 s3:// 则拉取到当前任务工作目录。
            image_path = self._resolve_source(job)

            # 中文说明：每个任务都有独立的工作目录，用来保存预处理结果、
            # 中间产物、脱敏输出和最终报告，避免任务之间互相覆盖。
            work_dir = self._get_work_dir(job)

            # 中文说明：预处理负责尺寸规整、方向修正、颜色空间处理等基础工作，
            # 后续所有视觉与 OCR provider 都基于预处理后的结果运行。
            preprocess_start = time.monotonic()
            preprocessed_path = run_preprocess(
                self._preprocessor, image_path, str(work_dir / "preprocess")
            )
            job.step_latencies["preprocess"] = (
                time.monotonic() - preprocess_start
            ) * 1000

            # 中文说明：asset 保存原始资源与预处理资源之间的映射关系，
            # 方便后续报告回溯“原始输入是什么、实际参与检测的是哪个文件”。
            job.asset = PictureAsset(
                original_uri=job.source.uri,
                preprocessed_uri=preprocessed_path,
                mime_type=job.source.mime_type,
            )

            # 中文说明：路由器决定当前输入更像文档图、自然图还是混合截图；
            # route_hint 允许调用方通过 options 提示路由器优先选择某条链路。
            route_start = time.monotonic()
            route_hint = job.options.get("route_hint", "auto")
            route = self._router.classify(preprocessed_path, {"route_hint": route_hint})
            job.route = route
            job.step_latencies["route"] = (time.monotonic() - route_start) * 1000
            self._update_status(job, JobStatus.ROUTED)

            # 中文说明：不同图像类型走不同链路。
            # 文档图更偏重 OCR+文字 PII，自然图更偏重安全审核与目标检测，
            # 混合截图则两类能力并行执行。
            if route == RouteType.DOCUMENT:
                self._execute_document_chain(job, preprocessed_path, work_dir)
            elif route == RouteType.NATURAL:
                self._execute_natural_chain(job, preprocessed_path, work_dir)
            else:
                self._execute_mixed_chain(job, preprocessed_path, work_dir)

            # 中文说明：无论走哪条链路，最后都补充总耗时统计。
            total_elapsed = (time.monotonic() - total_start) * 1000
            job.step_latencies["total"] = total_elapsed

            # 中文说明：只有未被丢弃、未失败的任务才置为 DONE；
            # DROPPED 和 FAILED 会保留自己的终态，避免被覆盖。
            if job.status not in (JobStatus.DROPPED, JobStatus.FAILED):
                self._update_status(job, JobStatus.DONE)

            job.completed_at = datetime.now(timezone.utc)

            # 中文说明：报告生成放在末尾，确保其中的 findings、策略结果、
            # provider 版本、耗时等字段都已经完整填充。
            self._generate_report(job, work_dir)

            # 中文说明：最后再次持久化一次完整任务快照，确保仓储中的状态与报告一致。
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
            # 中文说明：任何未捕获异常都会被统一转换为 FAILED 终态，
            # 同时记录错误类型和错误信息，便于后续排查。
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.error_detail = type(exc).__name__
            job.completed_at = datetime.now(timezone.utc)
            job.step_latencies["total"] = (time.monotonic() - total_start) * 1000
            self._repo.save_job(job)
            logger.exception("Job %s failed: %s", job.job_id, exc)

        return job

    # 中文说明：A 链路面向“文档类图像”，典型场景是扫描件、合同、表单、证件、
    # 发票或带大量文本的截图。此时 OCR 与文本 PII 是主能力，视觉检测用于补足章、签名等非纯文字敏感对象。
    def _execute_document_chain(
        self, job: PictureJob, image_path: str, work_dir: Path
    ) -> None:
        """
        Document image chain:
        1. OCR/layout
        2. Text PII detect
        3. Vision detect
        4. Segmentation refine
        5. Redaction
        6. Policy evaluate
        7. Output
        """
        logger.info("Executing DOCUMENT chain for job %s", job.job_id)
        self._update_status(job, JobStatus.DETECTING)

        # 中文说明：先做 OCR 和版面分析，产出文本块、版面块和全文拼接结果；
        # 这是文档链路后续文字检测的基础输入。
        ocr_start = time.monotonic()
        ocr_result = run_ocr_layout(self._ocr, image_path)
        job.ocr_result = ocr_result
        job.step_latencies["ocr_layout"] = (time.monotonic() - ocr_start) * 1000
        job.provider_versions["ocr"] = self._ocr.name

        # 中文说明：PII 检测器直接基于 OCR 的全文进行实体识别，
        # 然后再通过 OCR block 映射回图像区域。
        pii_start = time.monotonic()
        pii_findings = run_text_pii_detection(self._pii, ocr_result)
        job.step_latencies["text_pii"] = (time.monotonic() - pii_start) * 1000
        job.provider_versions["pii"] = self._pii.name

        # 中文说明：视觉检测用于补足 OCR 不擅长的对象，比如人脸、印章、二维码、
        # 工牌、车牌等具有空间结构但未必能稳定转成文本的目标。
        vision_start = time.monotonic()
        vision_findings = run_vision_detection(self._vision, image_path)
        job.step_latencies["vision_detect"] = (
            time.monotonic() - vision_start
        ) * 1000
        job.provider_versions["vision"] = self._vision.name

        # 中文说明：把文本侧和视觉侧结果合并并做去重，形成统一 findings 视图。
        all_findings = merge_findings(pii_findings, vision_findings)
        job.findings = all_findings

        # 中文说明：分割细化会把粗粒度框进一步修正成更精确的区域，
        # 减少脱敏时对正常内容的误伤。
        self._update_status(job, JobStatus.SEGMENTING)
        seg_start = time.monotonic()
        all_findings = run_segmentation_refinement(
            self._segmentation, image_path, all_findings
        )
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000
        job.provider_versions["segmentation"] = self._segmentation.name

        # 中文说明：文档链路一般不依赖安全审核结果，因此 moderation 参数传 None。
        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_start = time.monotonic()
        policy_result = self._policy.evaluate(all_findings, None, job.profile)
        job.policy_result = policy_result
        job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

        # 中文说明：策略判断完成后统一进入“通过原图 / 脱敏输出 / 直接丢弃”的决策落地阶段。
        self._apply_decision(job, image_path, all_findings, work_dir)

    # 中文说明：B 链路面向“自然图像”，典型场景是实拍照片、监控抓拍、街景、
    # 人像或商品图。这里安全审核和视觉检测是主流程，不依赖 OCR 作为核心前提。
    def _execute_natural_chain(
        self, job: PictureJob, image_path: str, work_dir: Path
    ) -> None:
        """
        Natural image chain:
        1. Safety moderation
        2. Vision detect
        3. Segmentation refine
        4. Redaction or drop
        5. Policy evaluate
        6. Output
        """
        logger.info("Executing NATURAL chain for job %s", job.job_id)
        self._update_status(job, JobStatus.DETECTING)

        # 中文说明：自然图首先做安全审核，因为像涉黄、暴力等内容可以直接影响最终策略决策。
        safety_start = time.monotonic()
        moderation_result = run_safety_moderation(self._safety, image_path)
        job.moderation_result = moderation_result
        job.step_latencies["safety"] = (time.monotonic() - safety_start) * 1000
        job.provider_versions["safety"] = self._safety.name

        # 中文说明：视觉检测负责识别人脸、车牌、二维码等空间对象。
        vision_start = time.monotonic()
        vision_findings = run_vision_detection(self._vision, image_path)
        job.step_latencies["vision_detect"] = (
            time.monotonic() - vision_start
        ) * 1000
        job.provider_versions["vision"] = self._vision.name

        job.findings = vision_findings

        # 中文说明：将粗检测框交给分割模块进一步细化，提高脱敏区域边界质量。
        self._update_status(job, JobStatus.SEGMENTING)
        seg_start = time.monotonic()
        vision_findings = run_segmentation_refinement(
            self._segmentation, image_path, vision_findings
        )
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000
        job.provider_versions["segmentation"] = self._segmentation.name

        # 中文说明：自然图的策略评估会同时综合视觉 findings 与安全审核结果。
        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_start = time.monotonic()
        policy_result = self._policy.evaluate(
            vision_findings, moderation_result, job.profile
        )
        job.policy_result = policy_result
        job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

        # 中文说明：策略可能要求直接丢弃，也可能要求对敏感区域脱敏后输出。
        self._apply_decision(job, image_path, vision_findings, work_dir)

    # 中文说明：C 链路面向“混合截图”，比如聊天截图、网页截图、App 页面截图。
    # 这类图片通常既有大量文本，也可能包含头像、二维码、缩略图等视觉对象，
    # 因此把 OCR/PII 与 safety/vision 做并行处理更高效。
    def _execute_mixed_chain(
        self, job: PictureJob, image_path: str, work_dir: Path
    ) -> None:
        """
        Mixed screenshot chain (dual parallel execution):
        Phase 1: OCR/layout AND safety moderation in parallel
        Phase 2: Text PII detect AND vision detect in parallel
        Phase 3: Merge, segmentation, redaction, policy, output
        """
        logger.info("Executing MIXED chain for job %s", job.job_id)
        self._update_status(job, JobStatus.DETECTING)

        # 中文说明：第一阶段并行跑 OCR 与安全审核。
        # 这两者输入相同但彼此独立，适合并发执行以降低总延迟。
        ocr_result = None
        moderation_result = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(run_ocr_layout, self._ocr, image_path)
            safety_future = executor.submit(run_safety_moderation, self._safety, image_path)

            phase1_start = time.monotonic()
            for future in as_completed([ocr_future, safety_future]):
                try:
                    result = future.result()

                    # 中文说明：这里通过是否带有 full_text 属性来区分 OCR 结果与审核结果。
                    # OCRLayoutResult 具备 full_text，而安全审核结果不具备。
                    if hasattr(result, "full_text"):
                        ocr_result = result
                    else:
                        moderation_result = result
                except Exception as exc:
                    # 中文说明：混合链路对单个 provider 失败采取尽量降级而非整体失败，
                    # 只要剩余能力还能继续，就继续往下跑。
                    logger.warning("Phase 1 provider failed: %s", exc)

            job.step_latencies["phase1_parallel"] = (
                time.monotonic() - phase1_start
            ) * 1000

        job.ocr_result = ocr_result
        job.moderation_result = moderation_result
        job.provider_versions["ocr"] = self._ocr.name
        job.provider_versions["safety"] = self._safety.name

        # 中文说明：第二阶段并行跑“文字 PII 检测”和“视觉目标检测”。
        # 如果 OCR 失败，则跳过 PII，但 vision 仍然执行。
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

            job.step_latencies["phase2_parallel"] = (
                time.monotonic() - phase2_start
            ) * 1000

        job.provider_versions["pii"] = self._pii.name
        job.provider_versions["vision"] = self._vision.name

        # 中文说明：并行阶段的结果在这里汇总，形成统一 findings 列表。
        all_findings = merge_findings(pii_findings, vision_findings)
        job.findings = all_findings

        # 中文说明：对统一 findings 做一次区域精修，得到更适合脱敏的 mask 或 bbox。
        self._update_status(job, JobStatus.SEGMENTING)
        seg_start = time.monotonic()
        all_findings = run_segmentation_refinement(
            self._segmentation, image_path, all_findings
        )
        job.step_latencies["segmentation"] = (time.monotonic() - seg_start) * 1000
        job.provider_versions["segmentation"] = self._segmentation.name

        # 中文说明：混合链路的策略会同时参考 findings 与安全审核结果。
        self._update_status(job, JobStatus.POLICY_EVALUATING)
        policy_start = time.monotonic()
        policy_result = self._policy.evaluate(
            all_findings, moderation_result, job.profile
        )
        job.policy_result = policy_result
        job.step_latencies["policy"] = (time.monotonic() - policy_start) * 1000

        # 中文说明：最后统一落地策略决策。
        self._apply_decision(job, image_path, all_findings, work_dir)

    def _validate_input(self, job: PictureJob) -> None:
        """Validate the job input."""
        # 中文说明：mime_type 允许为空字符串，但如果传了且不在白名单内，就直接报错。
        mime = job.source.mime_type.lower()
        if mime and mime not in _SUPPORTED_MIME_TYPES:
            raise UnsupportedMediaError(mime)

    def _resolve_source(self, job: PictureJob) -> str:
        """Resolve the source URI to a local file path."""
        uri = job.source.uri

        # 中文说明：local:// 语义是“这是一个当前机器上的本地文件”，只需去掉协议头。
        if uri.startswith("local://"):
            return uri.replace("local://", "")

        # 中文说明：s3:// 语义是远端对象存储资源，需要先下载到任务目录后再处理。
        if uri.startswith("s3://"):
            work_dir = self._get_work_dir(job)
            local_path = str(work_dir / "input" / Path(uri).name)
            return self._storage.load(uri, local_path)

        # 中文说明：其余情况默认认为调用方传入的已经是可直接访问的路径。
        return uri

    def _get_work_dir(self, job: PictureJob) -> Path:
        """Get the working directory for a job."""
        # 中文说明：优先使用配置中的 work_dir，未提供时退化到本地默认目录。
        if self._settings:
            base = Path(self._settings.work_dir)
        else:
            base = Path("./compliance_output_picture")

        work_dir = base / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _update_status(self, job: PictureJob, status: JobStatus) -> None:
        """Update job status and persist."""
        # 中文说明：状态更新与仓储落库绑定在一起，
        # 这样外部在查询任务时总能看到最新状态。
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
        # 中文说明：没有策略结果时无法决策，直接返回。
        if job.policy_result is None:
            return

        decision = job.policy_result.decision

        if decision == DecisionType.DROP:
            # 中文说明：DROP 表示该资源不应对外发布，因此不生成合规输出图。
            self._update_status(job, JobStatus.DROPPED)
            logger.info("Job %s DROPPED by policy", job.job_id)
            return

        if decision == DecisionType.PASS_RAW:
            # 中文说明：PASS_RAW 表示原图可直接通过，但仍然会复制到输出位置，
            # 让下游统一从 compliant_image_uri 获取“最终可交付产物”。
            compliant_uri = self._storage.save(
                image_path, f"{job.job_id}/compliant.png"
            )
            job.compliant_image_uri = compliant_uri
            return

        # 中文说明：其余情况默认进入脱敏路径，即 PASS_REDACTED。
        self._update_status(job, JobStatus.REDACTING)

        # 中文说明：先构建“敏感类别 -> 脱敏模式”的配置映射，
        # 配置可来源于全局 settings，也可被单任务 options 覆盖。
        redaction_config = self._get_redaction_config(job)

        redact_start = time.monotonic()

        # 中文说明：把 finding 转成 redaction operation，
        # operation 是真正交给 redactor 执行的结构化指令。
        operations = build_redaction_operations(findings, redaction_config)
        job.redaction_operations = operations

        output_path = str(work_dir / "compliant.png")
        overlay_path = str(work_dir / "overlay.png")

        compliant_path, overlay_result = run_redaction(
            self._redactor, image_path, operations, output_path, overlay_path
        )
        job.step_latencies["redaction"] = (time.monotonic() - redact_start) * 1000

        # 中文说明：主输出图是脱敏后的正式合规图；
        # overlay 图是辅助审计材料，用于查看命中了哪些区域。
        job.compliant_image_uri = self._storage.save(
            compliant_path, f"{job.job_id}/compliant.png"
        )
        if overlay_result:
            job.overlay_image_uri = self._storage.save(
                overlay_result, f"{job.job_id}/overlay.png"
            )

    def _get_redaction_config(self, job: PictureJob) -> dict[str, str]:
        """Build redaction mode mapping from settings and job options."""
        # 中文说明：优先从 settings 构造全量类别映射；
        # 这里的 key 必须与 finding.category 保持一致。
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
            # 中文说明：没有 settings 时使用一组可工作的保底默认值。
            config = {
                "default": "black_box",
                "face": "gaussian_blur",
                "signature": "solid_fill",
                "stamp": "solid_fill",
            }

        # 中文说明：任务级 options 可以覆盖部分脱敏策略，
        # 便于不同租户或不同场景做临时差异化处理。
        for key in ("redaction_mode_text", "redaction_mode_face"):
            if key in job.options:
                if key == "redaction_mode_text":
                    for cat in (
                        "person_name",
                        "phone_number",
                        "email",
                        "id_card",
                        "bank_card",
                        "address",
                    ):
                        config[cat] = job.options[key]
                elif key == "redaction_mode_face":
                    config["face"] = job.options[key]

        return config

    def _generate_report(self, job: PictureJob, work_dir: Path) -> None:
        """Generate and persist the audit report JSON."""
        # 中文说明：报告对象汇总了任务的核心审计数据，
        # 是后续追责、复核、回归测试和外部系统集成的重要基础。
        report = PictureReport(
            job_id=job.job_id,
            route=job.route or RouteType.MIXED,
            decision=job.policy_result.decision
            if job.policy_result
            else DecisionType.PASS_RAW,
            findings=job.findings,
            moderation=job.moderation_result,
            redaction_operations=job.redaction_operations,
            provider_info=job.provider_versions,
            reason_codes=job.policy_result.reason_codes if job.policy_result else [],
            timestamps={
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat()
                if job.completed_at
                else "",
            },
            latency_ms=job.step_latencies,
        )

        report_path = work_dir / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # 中文说明：JSON 报告使用 UTF-8 编码写入，方便后续直接查看或供外部系统读取。
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # 中文说明：报告本地落盘后，再交给 storage backend 持久化。
        job.report_uri = self._storage.save(
            str(report_path), f"{job.job_id}/report.json"
        )
        logger.info("Report saved to %s", job.report_uri)
