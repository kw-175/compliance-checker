"""Client for delegating video frames to the picture compliance API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from picture.domain.models import PictureJob


@dataclass(frozen=True)
class PictureApiConfig:
    base_url: str = "http://127.0.0.1:19012"
    submit_path: str = "/v1/picture/jobs"
    status_path: str = "/v1/picture/jobs/{job_id}"
    report_path: str = "/v1/picture/jobs/{job_id}/report"
    health_path: str = "/api/v1/health"
    timeout_seconds: int = 30
    task_timeout_seconds: int = 1800
    poll_interval_seconds: float = 1.0


class PictureComplianceApiClient:
    """Submit sampled frames to the picture service and return PictureJob models."""

    def __init__(self, config: PictureApiConfig | None = None):
        self.config = config or PictureApiConfig()

    def check_health(self) -> None:
        payload = self._request("GET", self._url(self.config.health_path))
        status = str(payload.get("status") or "").lower()
        service = str(payload.get("service") or "")
        if status not in {"healthy", "ok", "ready"}:
            raise RuntimeError(f"Picture compliance API is not healthy: {payload}")
        if service and service != "picture-compliance-checker":
            raise RuntimeError(f"Unexpected picture compliance API service: {service}")

    def run_frame(
        self,
        *,
        image_uri: str,
        tenant_id: str,
        profile: str,
        options: dict[str, Any],
    ) -> PictureJob:
        submit_payload = {
            "tenant_id": tenant_id,
            "source": {
                "type": "file",
                "uri": image_uri,
                "mime_type": "image/png",
            },
            "mode": "compliance_only",
            "profile": profile,
            "options": options,
        }
        created = self._request("POST", self._url(self.config.submit_path), submit_payload)
        job_id = str(created.get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"Picture compliance API did not return job_id: {created}")
        self._wait_for_completion(job_id)
        report = self._request("GET", self._url(self.config.report_path, job_id=job_id))
        return picture_job_from_api_report(report)

    def _wait_for_completion(self, job_id: str) -> None:
        deadline = time.monotonic() + max(1, self.config.task_timeout_seconds)
        while time.monotonic() < deadline:
            payload = self._request("GET", self._url(self.config.status_path, job_id=job_id))
            status = str(payload.get("status") or "").upper()
            if status in {"DONE", "DROPPED"}:
                return
            if status == "FAILED":
                raise RuntimeError(payload.get("error") or f"Picture compliance job failed: {job_id}")
            time.sleep(max(0.1, self.config.poll_interval_seconds))
        raise TimeoutError(f"Picture compliance job timed out: {job_id}")

    def _url(self, path: str, **values: str) -> str:
        resolved_path = path.format(**values)
        return self.config.base_url.rstrip("/") + "/" + resolved_path.lstrip("/")

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with urlopen(request, timeout=max(1, self.config.timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Picture compliance API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Picture compliance API request failed: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Picture compliance API returned non-JSON response: {raw[:200]!r}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Picture compliance API returned unexpected response: {data!r}")
        return data


def picture_job_from_api_report(report: dict[str, Any]) -> PictureJob:
    """Rehydrate the picture API full report into the domain model used by video."""
    if not isinstance(report, dict):
        raise TypeError("picture API report must be a dict")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    payload = {
        "job_id": report.get("job_id"),
        "tenant_id": report.get("tenant_id", ""),
        "status": report.get("status", "DONE"),
        "route": report.get("route"),
        "source": report.get("source") or {"uri": ""},
        "profile": report.get("profile", "default_cn_enterprise"),
        "options": report.get("options") if isinstance(report.get("options"), dict) else {},
        "precheck": report.get("precheck") if isinstance(report.get("precheck"), dict) else {},
        "step_audits": report.get("step_audits") if isinstance(report.get("step_audits"), list) else [],
        "findings": report.get("findings") if isinstance(report.get("findings"), list) else [],
        "moderation_result": report.get("moderation"),
        "redaction_operations": report.get("redaction_operations") if isinstance(report.get("redaction_operations"), list) else [],
        "policy_result": report.get("policy_snapshot") if isinstance(report.get("policy_snapshot"), dict) else None,
        "compliant_image_uri": artifacts.get("compliant_uri"),
        "overlay_image_uri": artifacts.get("overlay_uri"),
        "report_uri": artifacts.get("report_uri"),
        "annotation_package_uri": artifacts.get("annotation_package_uri"),
        "audit_package_uri": artifacts.get("audit_package_uri"),
        "provider_versions": report.get("provider_versions") if isinstance(report.get("provider_versions"), dict) else {},
        "step_latencies": report.get("latency_ms") if isinstance(report.get("latency_ms"), dict) else {},
        "degrade_events": report.get("degrade_events") if isinstance(report.get("degrade_events"), list) else [],
        "trust_level": report.get("trust_level", "full"),
        "created_at": report.get("created_at"),
        "updated_at": report.get("updated_at"),
        "completed_at": report.get("completed_at"),
        "error": report.get("error"),
        "error_detail": report.get("error_detail"),
    }
    return PictureJob.model_validate({key: value for key, value in payload.items() if value is not None})
