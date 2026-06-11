"""
FastAPI routes for the picture compliance engine.

Endpoints:
  POST   /v1/picture/jobs              - Create a compliance job
  GET    /v1/picture/jobs/{job_id}      - Get job status
  GET    /v1/picture/jobs/{job_id}/result   - Get job result
  GET    /v1/picture/jobs/{job_id}/findings - Get job findings
  POST   /v1/picture/jobs/{job_id}/rerun   - Re-run a job
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from picture.api.schemas import (
    ArtifactURIs,
    CreateJobRequest,
    CreateJobResponse,
    ErrorResponse,
    FindingResponse,
    FindingsListResponse,
    JobResultResponse,
    JobResultStats,
    JobStatusResponse,
)
from picture.application.use_cases import _get_default_repo, create_orchestrator
from picture.domain.enums import JobStatus
from picture.domain.exceptions import JobNotFoundError
from picture.domain.models import PictureJob, SourceSpec
from picture.infra.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/picture", tags=["picture-compliance"])


def _run_job_background(job: PictureJob) -> None:
    """Execute a job in the background."""
    try:
        settings = get_settings()
        repo = _get_default_repo()
        orchestrator = create_orchestrator(settings, repository=repo)
        orchestrator.execute(job)
    except Exception as exc:
        logger.exception("Background job %s failed: %s", job.job_id, exc)


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
    repo = _get_default_repo()

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

    repo.save_job(job)
    background_tasks.add_task(_run_job_background, job)

    logger.info("Created job %s for tenant %s", job.job_id, request.tenant_id)
    return CreateJobResponse(job_id=job.job_id, status=job.status.value)


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
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        route=job.route.value if job.route else None,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
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

    if job.status in (JobStatus.CREATED, JobStatus.QUEUED, JobStatus.PREPROCESSING,
                      JobStatus.ROUTED, JobStatus.DETECTING, JobStatus.SEGMENTING,
                      JobStatus.REDACTING, JobStatus.POLICY_EVALUATING):
        raise HTTPException(status_code=202, detail="Job is still processing")

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "Job failed")

    decision = job.policy_result.decision.value if job.policy_result else "unknown"
    reason_codes = job.policy_result.reason_codes if job.policy_result else []

    return JobResultResponse(
        job_id=job.job_id,
        decision=decision,
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
    )


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
        region_dict = None
        if f.region:
            region_dict = {
                "bbox": {"x": f.region.bbox.x, "y": f.region.bbox.y,
                         "w": f.region.bbox.w, "h": f.region.bbox.h},
                "confidence": f.region.confidence,
            }
            if f.region.polygon:
                region_dict["polygon"] = f.region.polygon.points

        findings_response.append(FindingResponse(
            finding_id=f.finding_id,
            finding_type=f.finding_type.value,
            category=f.category,
            label=f.label,
            score=f.score,
            reason_code=f.reason_code,
            provider=f.provider,
            text_span=f.text_span,
            region=region_dict,
        ))

    return FindingsListResponse(
        job_id=job.job_id,
        total=len(findings_response),
        findings=findings_response,
    )


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

    # Create a new job with the same parameters
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
