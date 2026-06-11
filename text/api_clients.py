from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from text.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class OpenAICompatibleAPIError(RuntimeError):
    """Raised when the OpenAI-compatible compliance API cannot return JSON."""


@dataclass(frozen=True)
class ProviderConfig:
    mode: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_chars: int
    max_tokens: int
    source_tool_prefix: str


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise OpenAICompatibleAPIError("A compliance model base URL is required for the text compliance pipeline")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _extract_json_from_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise OpenAICompatibleAPIError(f"API response is not JSON: {candidate[:200]}") from exc
        payload = json.loads(candidate[start : end + 1])

    if not isinstance(payload, dict):
        raise OpenAICompatibleAPIError("API JSON response must be an object")
    return payload


def resolve_provider_config(settings: Settings) -> ProviderConfig:
    requested_mode = str(getattr(settings, "compliance_provider_mode", "auto") or "auto").strip().lower()
    has_local = bool(settings.local_compliance_base_url.strip() and settings.local_compliance_model.strip())
    has_api = bool(settings.api_compliance_base_url.strip() and settings.api_compliance_model.strip())

    if requested_mode not in {"auto", "local", "api"}:
        raise OpenAICompatibleAPIError(
            "COMPLIANCE_COMPLIANCE_PROVIDER_MODE must be one of auto, local, or api"
        )

    if requested_mode == "local":
        if not has_local:
            raise OpenAICompatibleAPIError(
                "Local provider mode requires COMPLIANCE_LOCAL_COMPLIANCE_BASE_URL and "
                "COMPLIANCE_LOCAL_COMPLIANCE_MODEL"
            )
        return ProviderConfig(
            mode="local_model",
            base_url=settings.local_compliance_base_url,
            api_key=settings.local_compliance_api_key,
            model=settings.local_compliance_model,
            timeout_seconds=settings.local_compliance_timeout_seconds,
            temperature=settings.local_compliance_temperature,
            max_chars=settings.local_compliance_max_chars,
            max_tokens=settings.local_compliance_max_tokens,
            source_tool_prefix=settings.local_compliance_source_tool_prefix,
        )

    if requested_mode == "api":
        if not has_api:
            raise OpenAICompatibleAPIError(
                "API provider mode requires COMPLIANCE_API_COMPLIANCE_BASE_URL and "
                "COMPLIANCE_API_COMPLIANCE_MODEL"
            )
        return ProviderConfig(
            mode="api",
            base_url=settings.api_compliance_base_url,
            api_key=settings.api_compliance_api_key,
            model=settings.api_compliance_model,
            timeout_seconds=settings.api_compliance_timeout_seconds,
            temperature=settings.api_compliance_temperature,
            max_chars=settings.api_compliance_max_chars,
            max_tokens=settings.api_compliance_max_tokens,
            source_tool_prefix=settings.api_compliance_source_tool_prefix,
        )

    if has_local:
        return ProviderConfig(
            mode="local_model",
            base_url=settings.local_compliance_base_url,
            api_key=settings.local_compliance_api_key,
            model=settings.local_compliance_model,
            timeout_seconds=settings.local_compliance_timeout_seconds,
            temperature=settings.local_compliance_temperature,
            max_chars=settings.local_compliance_max_chars,
            max_tokens=settings.local_compliance_max_tokens,
            source_tool_prefix=settings.local_compliance_source_tool_prefix,
        )
    if has_api:
        return ProviderConfig(
            mode="api",
            base_url=settings.api_compliance_base_url,
            api_key=settings.api_compliance_api_key,
            model=settings.api_compliance_model,
            timeout_seconds=settings.api_compliance_timeout_seconds,
            temperature=settings.api_compliance_temperature,
            max_chars=settings.api_compliance_max_chars,
            max_tokens=settings.api_compliance_max_tokens,
            source_tool_prefix=settings.api_compliance_source_tool_prefix,
        )
    raise OpenAICompatibleAPIError(
        "No compliance model provider is configured. Set either the local provider "
        "(COMPLIANCE_LOCAL_COMPLIANCE_*) or the API provider (COMPLIANCE_API_COMPLIANCE_*)."
    )


class OpenAICompatibleComplianceClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = resolve_provider_config(self.settings)

    def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = _chat_completions_url(self.provider.base_url)
        headers = {"Content-Type": "application/json"}
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"

        request_body = {
            "model": self.provider.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task_name": task_name,
                            "payload": payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": self.provider.temperature,
            "max_tokens": self.provider.max_tokens,
            "response_format": {"type": "json_object"},
        }

        response = httpx.post(
            url,
            headers=headers,
            json=request_body,
            timeout=self.provider.timeout_seconds,
        )
        response.raise_for_status()
        envelope = response.json()
        if not isinstance(envelope, dict):
            raise OpenAICompatibleAPIError("API response envelope must be an object")

        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatibleAPIError("API response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatibleAPIError("API response missing message content")

        parsed = _extract_json_from_text(content)
        parsed.setdefault(
            "provider",
            {
                "task_name": task_name,
                "mode": self.provider.mode,
                "base_url": self.provider.base_url,
                "model": self.provider.model,
            },
        )
        logger.debug("OpenAI-compatible API task %s completed", task_name)
        return parsed
