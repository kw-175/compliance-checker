"""
FastAPI routes for the picture compliance engine.

Endpoints:
  POST   /v1/picture/jobs              - Create a compliance job
  GET    /v1/picture/jobs/{job_id}      - Get job status
  GET    /v1/picture/jobs/{job_id}/result   - Get job result
  GET    /v1/picture/jobs/{job_id}/findings - Get job findings
  GET    /v1/picture/jobs/{job_id}/report   - Get full audit report
  POST   /v1/picture/jobs/{job_id}/rerun   - Re-run a job
"""
# 中文说明：本文件集中定义图片合规服务的 REST 接口，并负责把外部请求转交给应用层执行。

from __future__ import annotations

import logging
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from picture.api.schemas import (
    ArtifactURIs,
    CreateJobRequest,
    CreateJobResponse,
    ErrorResponse,
    FindingResponse,
    FindingsListResponse,
    JobOptions,
    JobResultResponse,
    JobResultStats,
    JobStatusResponse,
    ManualRedactionRequest,
    ManualRedactionResponse,
)
from picture.application.services import build_redaction_operations, run_redaction
from picture.application.use_cases import _get_default_repo, create_orchestrator
from picture.domain.enums import FindingType, JobStatus, RedactionMode
from picture.domain.exceptions import JobNotFoundError
from picture.domain.models import BBox, PictureFinding, PictureJob, Polygon, RedactionOperation, RegionMask, SourceSpec
from picture.infra.config import get_settings

logger = logging.getLogger(__name__)
_CONTRACT_VERSION = "compliance-job.v1"
_RESULT_CONTRACT_VERSION = "compliance-result.v1"
_OPERATOR_CATALOG_VERSION = "image-compliance-operators.v1"

# 中文说明：所有 picture HTTP 路由都统一挂在这个前缀下，便于版本管理和模块隔离。
router = APIRouter(prefix="/v1/picture", tags=["picture-compliance"])


def _status_label(status: JobStatus | str) -> str:
    value = status.value if isinstance(status, JobStatus) else str(status)
    return {
        JobStatus.CREATED.value: "已创建",
        JobStatus.QUEUED.value: "排队中",
        JobStatus.PREPROCESSING.value: "预处理中",
        JobStatus.ROUTED.value: "路由判定中",
        JobStatus.DETECTING.value: "检测中",
        JobStatus.SEGMENTING.value: "区域定位中",
        JobStatus.REDACTING.value: "脱敏处理中",
        JobStatus.POLICY_EVALUATING.value: "策略评估中",
        JobStatus.DONE.value: "检测完成",
        JobStatus.DROPPED.value: "已按策略丢弃",
        JobStatus.FAILED.value: "检测失败",
    }.get(value, "处理中")


def _progress_for_status(status: JobStatus | str) -> int:
    value = status.value if isinstance(status, JobStatus) else str(status)
    return {
        JobStatus.CREATED.value: 5,
        JobStatus.QUEUED.value: 10,
        JobStatus.PREPROCESSING.value: 20,
        JobStatus.ROUTED.value: 30,
        JobStatus.DETECTING.value: 55,
        JobStatus.SEGMENTING.value: 70,
        JobStatus.REDACTING.value: 82,
        JobStatus.POLICY_EVALUATING.value: 92,
        JobStatus.DONE.value: 100,
        JobStatus.DROPPED.value: 100,
        JobStatus.FAILED.value: 0,
    }.get(value, 15)


def _operator_ids(options: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    operator_id = str(options.get("operator_id") or "").strip()
    if operator_id:
        ids.append(operator_id)
    for key in (
        "privacy_operator_ids",
        "content_safety_operator_ids",
        "visual_safety_operator_ids",
        "visual_sensitive_object_operator_ids",
    ):
        value = options.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).strip())
    return list(dict.fromkeys(ids))


def _effective_request(job: PictureJob) -> dict[str, Any]:
    requested = _operator_ids(job.options)
    return {
        "contract_version": job.contract_version or _CONTRACT_VERSION,
        "platform_task_id": job.platform_task_id,
        "idempotency_key": job.idempotency_key,
        "remote_task_id": job.job_id,
        "modality": "image",
        "operator_id": str(job.options.get("operator_id") or ""),
        "operator_catalog_version": str(job.options.get("operator_catalog_version") or _OPERATOR_CATALOG_VERSION),
        "requested_operator_ids": requested,
        "effective_operator_ids": requested,
        "profile": job.profile,
        "options": job.options,
    }


def _error_info(code: str, stage: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "code": code,
        "stage": stage,
        "message": message,
        "retryable": retryable,
    }


def _existing_job_for(repo: Any, platform_task_id: str, idempotency_key: str, tenant_id: str = "") -> PictureJob | None:
    if not platform_task_id and not idempotency_key:
        return None
    for job in repo.list_jobs(tenant_id=tenant_id or None, limit=1000):
        if idempotency_key and job.idempotency_key == idempotency_key:
            return job
        if platform_task_id and job.platform_task_id == platform_task_id:
            return job
    return None


def _create_job_response(job: PictureJob) -> CreateJobResponse:
    return CreateJobResponse(
        job_id=job.job_id,
        status=job.status.value,
        contract_version=job.contract_version,
        platform_task_id=job.platform_task_id,
        idempotency_key=job.idempotency_key,
        modality="image",
        status_label=_status_label(job.status),
        effective_request=job.effective_request or _effective_request(job),
        warnings=job.warnings,
    )


def _result_consistency(job: PictureJob) -> dict[str, Any]:
    artifact_keys = [
        key
        for key, value in {
            "original_uri": job.source.uri,
            "compliant_uri": job.compliant_image_uri,
            "overlay_uri": job.overlay_image_uri,
            "report_uri": job.report_uri,
            "annotation_package_uri": job.annotation_package_uri,
            "audit_package_uri": job.audit_package_uri,
        }.items()
        if value
    ]
    return {
        "modality": "image",
        "evidence_record_count": len(job.findings),
        "redaction_operation_count": len(job.redaction_operations),
        "review_required": bool(job.policy_result and job.policy_result.review_required),
        "review_task_count": 1 if job.policy_result and job.policy_result.review_required else 0,
        "artifact_keys": artifact_keys,
        "decision": job.policy_result.decision.value if job.policy_result else "",
    }


def _run_job_background(job: PictureJob) -> None:
    """Execute a job in the background."""
    repo = _get_default_repo()
    try:
        # 中文说明：后台任务重新构造 orchestrator，避免把请求阶段对象跨线程直接复用。
        settings = get_settings()
        orchestrator = create_orchestrator(settings, repository=repo)
        orchestrator.execute(job)
    except Exception as exc:
        # 中文说明：后台异常无法直接通过 HTTP 返回，因此必须打完整日志。
        logger.exception("Background job %s failed: %s", job.job_id, exc)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.error_info = _error_info("PICTURE_PIPELINE_FAILED", "picture_pipeline", str(exc), retryable=False)
        job.completed_at = datetime.now(timezone.utc)
        repo.save_job(job)


@router.post(
    "/jobs",
    response_model=CreateJobResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}},
    summary="Create a picture compliance job",
)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
) -> CreateJobResponse:
    """Create a new picture compliance processing job."""
    # 中文说明：当前默认仓储是内存实现，适合单进程服务与测试环境。
    repo = _get_default_repo()

    # 中文说明：API 层请求模型在这里转换成领域层真正使用的 PictureJob。
    job = PictureJob(
        tenant_id=request.tenant_id,
        source=SourceSpec(
            type=request.source.type,
            uri=request.source.uri,
            mime_type=request.source.mime_type,
        ),
        profile=request.profile,
        options=request.options.model_dump(),
    )
    job.platform_task_id = str(job.options.get("platform_task_id") or "")
    job.idempotency_key = str(job.options.get("idempotency_key") or "")
    job.contract_version = str(job.options.get("contract_version") or _CONTRACT_VERSION)
    job.effective_request = _effective_request(job)
    existing_job = _existing_job_for(repo, job.platform_task_id, job.idempotency_key, request.tenant_id)
    if existing_job is not None:
        return _create_job_response(existing_job)

    # 中文说明：先保存任务，再异步投递后台执行，保证状态查询立刻可见。
    repo.save_job(job)
    background_tasks.add_task(_run_job_background, job)

    logger.info("Created job %s for tenant %s", job.job_id, request.tenant_id)
    return _create_job_response(job)


@router.post(
    "/jobs/file",
    response_model=CreateJobResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}},
    summary="Create a picture compliance job from an uploaded file",
)
async def create_job_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: str = Form("default"),
    profile: str = Form("default_cn_enterprise"),
    options: str = Form("{}"),
    platform_task_id: str = Form(""),
    idempotency_key: str = Form(""),
    contract_version: str = Form(_CONTRACT_VERSION),
) -> CreateJobResponse:
    repo = _get_default_repo()
    try:
        options_payload = json.loads(options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"options must be JSON: {exc}")
    if not isinstance(options_payload, dict):
        raise HTTPException(status_code=400, detail="options must be a JSON object")

    job = PictureJob(
        tenant_id=tenant_id,
        profile=profile,
        options=JobOptions.model_validate(options_payload).model_dump(),
    )
    job.platform_task_id = platform_task_id or str(options_payload.get("platform_task_id") or "")
    job.idempotency_key = idempotency_key or str(options_payload.get("idempotency_key") or "")
    job.contract_version = contract_version or str(options_payload.get("contract_version") or _CONTRACT_VERSION)
    existing_job = _existing_job_for(repo, job.platform_task_id, job.idempotency_key, tenant_id)
    if existing_job is not None:
        return _create_job_response(existing_job)
    settings = get_settings()
    filename = Path(file.filename or "input.png").name
    upload_dir = settings.work_dir / "uploads" / job.job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / filename
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    job.source = SourceSpec(
        type="file",
        uri="local://" + str(input_path.resolve()),
        mime_type=file.content_type or _mime_type_for(input_path),
    )
    job.effective_request = _effective_request(job)

    repo.save_job(job)
    background_tasks.add_task(_run_job_background, job)

    logger.info("Created uploaded-file job %s for tenant %s", job.job_id, tenant_id)
    return _create_job_response(job)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get job status",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Query the current status of a picture compliance job."""
    repo = _get_default_repo()
    try:
        job = repo.get_job(job_id)
    except JobNotFoundError:
        # 中文说明：领域异常在 API 层被统一翻译成 404。
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # 中文说明：状态接口只返回任务进度视图，不返回完整审计结果。
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        contract_version=job.contract_version,
        platform_task_id=job.platform_task_id,
        idempotency_key=job.idempotency_key,
        modality="image",
        status_label=_status_label(job.status),
        stage=job.precheck.get("current_step") or job.status.value,
        progress=_progress_for_status(job.status),
        route=job.route.value if job.route else None,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
        error_info=job.error_info,
        effective_request=job.effective_request or _effective_request(job),
        warnings=job.warnings,
        current_step=job.precheck.get("current_step"),
        current_provider=job.precheck.get("current_provider"),
        current_step_started_at=job.precheck.get("current_step_started_at"),
    )


@router.get(
    "/jobs/{job_id}/result",
    response_model=JobResultResponse,
    responses={
        202: {"model": ErrorResponse, "description": "Job is still processing"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get job result",
)
async def get_job_result(job_id: str) -> JobResultResponse:
    """Get the final result of a completed picture compliance job."""
    repo = _get_default_repo()
    try:
        job = repo.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # 中文说明：任务仍在执行时返回 202，提示调用方稍后继续轮询。
    if job.status in (
        JobStatus.CREATED,
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
        JobStatus.ROUTED,
        JobStatus.DETECTING,
        JobStatus.SEGMENTING,
        JobStatus.REDACTING,
        JobStatus.POLICY_EVALUATING,
    ):
        raise HTTPException(status_code=202, detail="Job is still processing")

    # 中文说明：失败任务会透出 500 与任务错误消息，便于外部系统识别异常流程。
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "Job failed")

    decision = job.policy_result.decision.value if job.policy_result else "unknown"
    reason_codes = job.policy_result.reason_codes if job.policy_result else []

    # 中文说明：artifacts 告诉调用方产物位置，stats 则给出数量、时延和 provider 信息。
    return JobResultResponse(
        job_id=job.job_id,
        contract_version=_RESULT_CONTRACT_VERSION,
        platform_task_id=job.platform_task_id,
        idempotency_key=job.idempotency_key,
        modality="image",
        decision=decision,
        dataset_action=job.policy_result.dataset_action if job.policy_result else "",
        review_required=job.policy_result.review_required if job.policy_result else False,
        requires_restricted_dataset=job.policy_result.requires_restricted_dataset if job.policy_result else False,
        annotation_guidance_zh=job.policy_result.annotation_guidance_zh if job.policy_result else "",
        reason_codes=reason_codes,
        artifacts=ArtifactURIs(
            original_uri=job.source.uri,
            compliant_uri=job.compliant_image_uri,
            overlay_uri=job.overlay_image_uri,
            report_uri=job.report_uri,
        ),
        stats=JobResultStats(
            total_findings=len(job.findings),
            total_redactions=len(job.redaction_operations),
            latency_ms=job.step_latencies,
            provider_versions=job.provider_versions,
        ),
        effective_request=job.effective_request or _effective_request(job),
        result_consistency=_result_consistency(job),
        warnings=job.warnings,
        error_info=job.error_info,
    )


@router.get(
    "/jobs/{job_id}/artifact/{kind}",
    responses={404: {"model": ErrorResponse}},
    summary="Download a picture job artifact",
)
async def get_job_artifact(job_id: str, kind: str) -> FileResponse:
    repo = _get_default_repo()
    try:
        job = repo.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    normalized_kind = (kind or "").strip().lower()
    uri = {
        "original": job.source.uri,
        "compliant": job.compliant_image_uri,
        "overlay": job.overlay_image_uri,
        "report": job.report_uri,
    }.get(normalized_kind)
    path = _resolve_local_artifact_path(uri or "")
    if path is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{kind}' for job '{job_id}' not found")
    return FileResponse(path, media_type=_mime_type_for(path), filename=path.name)


@router.get(
    "/jobs/{job_id}/findings",
    response_model=FindingsListResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get job findings",
)
async def get_job_findings(job_id: str) -> FindingsListResponse:
    """Get detailed compliance findings for a job."""
    repo = _get_default_repo()
    try:
        job = repo.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    findings_response = []
    for f in job.findings:
        # 中文说明：领域层中的 RegionMask 会在这里显式展开成适合 JSON 输出的字典结构。
        region_dict = None
        metadata = dict(f.metadata or {})
        if f.region:
            region_dict = {
                "bbox": {
                    "x": f.region.bbox.x,
                    "y": f.region.bbox.y,
                    "w": f.region.bbox.w,
                    "h": f.region.bbox.h,
                },
                "confidence": f.region.confidence,
            }
            if f.region.polygon:
                # 中文说明：polygon 代表更细粒度区域信息，只有部分 provider 会返回。
                region_dict["polygon"] = f.region.polygon.points
            mask_path = f.region.mask_path or metadata.get("mask_path")
            if not mask_path and isinstance(metadata.get("evidence_regions"), list):
                mask_path = _first_evidence_value(metadata["evidence_regions"], "mask_path")
            if mask_path:
                # 中文说明：mask_path 是最精确的区域表达，前端应优先用 mask，其次 polygon，最后 bbox。
                region_dict["mask_path"] = mask_path
                region_dict["mask_uri"] = mask_path
            for key in ("polygons", "mask_area", "mask_area_ratio", "mask_bbox_fill_ratio", "mask_quality_score"):
                value = metadata.get(key)
                if value is None and isinstance(metadata.get("evidence_regions"), list):
                    value = _first_evidence_value(metadata["evidence_regions"], key)
                if value is not None:
                    region_dict[key] = value
        localization_status = str(metadata.get("localization_status") or "")
        boundary_status = str(metadata.get("boundary_status") or "")
        mask_quality_score = metadata.get("mask_quality_score")
        if mask_quality_score is None and isinstance(metadata.get("evidence_regions"), list):
            mask_quality_score = _first_evidence_value(metadata["evidence_regions"], "mask_quality_score")
        review_required = metadata.get("review_required")

        findings_response.append(
            FindingResponse(
                finding_id=f.finding_id,
                finding_type=f.finding_type.value,
                category=f.category,
                label=f.label,
                score=f.score,
                reason_code=f.reason_code,
                provider=f.provider,
                text_span=f.text_span,
                localization_status=localization_status or None,
                boundary_status=boundary_status or None,
                review_required=bool(review_required) if review_required is not None else None,
                mask_quality_score=float(mask_quality_score) if _is_number(mask_quality_score) else None,
                region=region_dict,
            )
        )

    return FindingsListResponse(
        job_id=job.job_id,
        total=len(findings_response),
        findings=findings_response,
    )


def _first_evidence_value(items: list[Any], key: str) -> Any:
    for item in items:
        if isinstance(item, dict) and item.get(key) is not None:
            return item.get(key)
    return None


def _resolve_local_artifact_path(uri: str) -> Path | None:
    if not uri:
        return None
    path_text = uri.removeprefix("local://")
    path = Path(path_text)
    if path.exists() and path.is_file():
        return path
    return None


def _mime_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    if suffix == ".json":
        return "application/json"
    return "image/png"


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


@router.get(
    "/jobs/{job_id}/report",
    responses={
        202: {"model": ErrorResponse, "description": "Job is still processing"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Get full picture compliance audit report",
)
async def get_job_report(job_id: str) -> dict[str, Any]:
    """Get the full audit report for a completed picture compliance job."""
    repo = _get_default_repo()
    try:
        job = repo.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status in (
        JobStatus.CREATED,
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
        JobStatus.ROUTED,
        JobStatus.DETECTING,
        JobStatus.SEGMENTING,
        JobStatus.REDACTING,
        JobStatus.POLICY_EVALUATING,
    ):
        raise HTTPException(status_code=202, detail="Job is still processing")
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "Job failed")

    return {
        "job_id": job.job_id,
        "contract_version": _RESULT_CONTRACT_VERSION,
        "platform_task_id": job.platform_task_id,
        "idempotency_key": job.idempotency_key,
        "modality": "image",
        "tenant_id": job.tenant_id,
        "status": job.status.value,
        "status_label": _status_label(job.status),
        "route": job.route.value if job.route else None,
        "source": job.source.model_dump(mode="json"),
        "profile": job.profile,
        "options": job.options,
        "effective_request": job.effective_request or _effective_request(job),
        "result_consistency": _result_consistency(job),
        "warnings": job.warnings,
        "precheck": job.precheck,
        "step_audits": job.step_audits,
        "findings": [finding.model_dump(mode="json") for finding in job.findings],
        "moderation": job.moderation_result.model_dump(mode="json") if job.moderation_result else None,
        "redaction_operations": [
            operation.model_dump(mode="json") for operation in job.redaction_operations
        ],
        "policy_snapshot": job.policy_result.model_dump(mode="json") if job.policy_result else {},
        "artifacts": {
            "original_uri": job.source.uri,
            "preprocessed_uri": job.asset.preprocessed_uri if job.asset else None,
            "compliant_uri": job.compliant_image_uri,
            "overlay_uri": job.overlay_image_uri,
            "report_uri": job.report_uri,
            "annotation_package_uri": job.annotation_package_uri,
            "audit_package_uri": job.audit_package_uri,
        },
        "provider_versions": job.provider_versions,
        "latency_ms": job.step_latencies,
        "trust_level": job.trust_level,
        "degrade_events": job.degrade_events,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "error_info": job.error_info,
        "error_detail": job.error_detail,
    }


@router.post(
    "/jobs/{job_id}/manual-redaction",
    response_model=ManualRedactionResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Apply manual picture review and regenerate redaction artifacts",
)
async def apply_manual_redaction(
    job_id: str,
    request: ManualRedactionRequest,
) -> ManualRedactionResponse:
    """Replace final findings with human-reviewed regions and regenerate artifacts."""
    repo = _get_default_repo()
    try:
        job = repo.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status in (
        JobStatus.CREATED,
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
        JobStatus.ROUTED,
        JobStatus.DETECTING,
        JobStatus.SEGMENTING,
        JobStatus.REDACTING,
        JobStatus.POLICY_EVALUATING,
    ):
        raise HTTPException(status_code=409, detail="Job is still processing")
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=409, detail=job.error or "Job failed")

    settings = get_settings()
    orchestrator = create_orchestrator(settings, repository=repo)
    work_dir = orchestrator._get_work_dir(job)
    source_path = _manual_source_path(job)
    manual_findings = [_manual_finding_to_domain(item) for item in request.findings]
    redaction_config = orchestrator._get_redaction_config(job)
    _apply_manual_redaction_modes(redaction_config, request.findings)
    operations = build_redaction_operations(manual_findings, redaction_config)

    compliant_path = str(work_dir / "manual_compliant.png")
    overlay_path = str(work_dir / "manual_overlay.png")
    start = datetime.now(timezone.utc)
    rendered_path, overlay_result = run_redaction(
        orchestrator._redactor,
        source_path,
        operations,
        compliant_path,
        overlay_path,
    )

    job.findings = manual_findings
    job.redaction_operations = operations
    job.compliant_image_uri = orchestrator._storage.save(rendered_path, f"{job.job_id}/manual_compliant.png")
    if overlay_result:
        job.overlay_image_uri = orchestrator._storage.save(overlay_result, f"{job.job_id}/manual_overlay.png")
    else:
        job.overlay_image_uri = None

    executed_steps = [item["step"] for item in job.step_audits if item.get("executed")]
    if "manual_region_review" not in executed_steps:
        executed_steps.append("manual_region_review")
    policy_context = {
        "ordinary_dataset_enabled": bool(job.options.get("ordinary_dataset_enabled", True)),
        "restricted_dataset_enabled": bool(job.options.get("restricted_dataset_enabled", False)),
        "restricted_use_case": str(job.options.get("restricted_use_case", "") or "").strip(),
        "authorized_sensitive_use": bool(job.options.get("authorized_sensitive_use", False)),
        "education_value_preserved": True,
        "ocr_executed": bool(job.ocr_result is not None),
        "executed_steps": executed_steps,
        "skipped_steps": [
            {"step": item["step"], "reason": item.get("skip_reason", "")}
            for item in job.step_audits
            if not item.get("executed")
        ],
    }
    job.policy_result = orchestrator._policy.evaluate(
        manual_findings,
        job.moderation_result,
        job.profile,
        context=policy_context,
    )
    job.step_audits.append(
        {
            "step": "manual_region_review",
            "executed": True,
            "skip_reason": "",
            "input_signals": {
                "finding_count": len(manual_findings),
                "redaction_count": len(operations),
                "reviewed_by": request.reviewed_by,
            },
        }
    )
    job.completed_at = datetime.now(timezone.utc)
    job.step_latencies["manual_redaction"] = (job.completed_at - start).total_seconds() * 1000
    orchestrator._generate_report(job, work_dir)
    repo.save_job(job)

    manual_review = {
        "reviewed": True,
        "reviewed_by": request.reviewed_by,
        "reviewed_at": job.completed_at.isoformat(),
        "review_note": request.review_note,
        "finding_count": len(manual_findings),
        "redaction_count": len(operations),
    }
    return ManualRedactionResponse(
        job_id=job.job_id,
        artifacts=ArtifactURIs(
            original_uri=job.source.uri,
            compliant_uri=job.compliant_image_uri,
            overlay_uri=job.overlay_image_uri,
            report_uri=job.report_uri,
        ),
        redaction_operations=[operation.model_dump(mode="json") for operation in operations],
        manual_review=manual_review,
        policy_snapshot=job.policy_result.model_dump(mode="json") if job.policy_result else {},
    )


def _manual_source_path(job: PictureJob) -> str:
    candidates = []
    if job.asset and job.asset.preprocessed_uri:
        candidates.append(job.asset.preprocessed_uri)
    candidates.append(job.source.uri)
    for uri in candidates:
        path = uri.replace("local://", "") if uri.startswith("local://") else uri
        if Path(path).exists():
            return path
    raise HTTPException(status_code=500, detail="Original image artifact is unavailable")


def _manual_finding_to_domain(item: Any) -> PictureFinding:
    try:
        finding_type = FindingType(item.finding_type)
    except ValueError:
        finding_type = FindingType.VISION_OBJECT
    bbox = item.region.bbox
    polygon = _manual_polygon_to_domain(item.region.polygon)
    mask_path = str(item.region.mask_path or "") or None
    return PictureFinding(
        finding_id=item.finding_id or f"manual_{uuid4().hex[:12]}",
        finding_type=finding_type,
        category=item.category or "manual_region",
        label=item.label or item.category or "manual_region",
        score=item.score,
        region=RegionMask(
            bbox=BBox(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h),
            polygon=polygon,
            mask_path=mask_path,
            confidence=item.region.confidence,
        ),
        text_span=item.text_span,
        reason_code=item.reason_code or "MANUAL_REVIEW",
        provider=item.provider or "human_review",
        explanation=item.explanation,
        metadata={**item.metadata, "manual_review": True},
    )


def _manual_polygon_to_domain(value: Any) -> Polygon | None:
    if value is None:
        return None
    raw_points = getattr(value, "points", value)
    if not isinstance(raw_points, list):
        return None
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if len(points) < 3:
        return None
    return Polygon(points=points)


def _manual_operation(item: Any, finding: PictureFinding) -> RedactionOperation:
    try:
        mode = RedactionMode(item.redaction_mode)
    except ValueError:
        mode = RedactionMode.BLACK_BOX
    return RedactionOperation(
        finding_id=finding.finding_id,
        region=finding.region,
        mode=mode,
        metadata={"source": "manual_review", "category": finding.category},
    )


def _apply_manual_redaction_modes(redaction_config: dict[str, str], findings: list[Any]) -> None:
    for item in findings:
        category = str(getattr(item, "category", "") or "manual_region").lower()
        mode = str(getattr(item, "redaction_mode", "") or "").strip()
        if category and mode:
            redaction_config[category] = mode


@router.post(
    "/jobs/{job_id}/rerun",
    response_model=CreateJobResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Re-run a job",
)
async def rerun_job(
    job_id: str,
    background_tasks: BackgroundTasks,
) -> CreateJobResponse:
    """Re-run an existing job with the same parameters."""
    repo = _get_default_repo()
    try:
        original = repo.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # 中文说明：rerun 不是重置原任务，而是复制参数重新新建一个任务。
    # 这样可以保留原任务的审计记录和所有产物，不会相互覆盖。
    new_job = PictureJob(
        tenant_id=original.tenant_id,
        source=original.source,
        profile=original.profile,
        options=original.options,
    )
    repo.save_job(new_job)
    background_tasks.add_task(_run_job_background, new_job)

    logger.info("Re-running job %s as new job %s", job_id, new_job.job_id)
    return CreateJobResponse(job_id=new_job.job_id, status=new_job.status.value)
