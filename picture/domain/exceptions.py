"""
Domain-level exceptions for the picture compliance engine.

Exception hierarchy:
    PictureError
    ├── ConfigurationError     - missing config, bad profile
    ├── ProviderError          - provider init / execution failures
    │   ├── ProviderNotAvailableError
    │   └── ProviderTimeoutError
    ├── JobError               - job lifecycle errors
    │   ├── JobNotFoundError
    │   └── JobAlreadyCompletedError
    ├── StorageError           - file I/O or S3 errors
    ├── UnsupportedMediaError  - bad MIME type or format
    └── PolicyError            - policy evaluation failures
"""

from __future__ import annotations


class PictureError(Exception):
    """Base exception for the picture compliance engine."""

    def __init__(self, message: str = "", code: str = "PICTURE_ERROR") -> None:
        self.code = code
        super().__init__(message)


class ConfigurationError(PictureError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str = "Configuration error") -> None:
        super().__init__(message, code="CONFIG_ERROR")


class ProviderError(PictureError):
    """Raised when a provider fails during execution."""

    def __init__(self, provider: str, message: str = "") -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}", code="PROVIDER_ERROR")


class ProviderNotAvailableError(ProviderError):
    """Raised when a required provider is not installed or configured."""

    def __init__(self, provider: str) -> None:
        super().__init__(provider, f"Provider '{provider}' is not available. "
                         "Install required dependencies or provide configuration.")


class ProviderTimeoutError(ProviderError):
    """Raised when a provider times out."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(provider, f"Provider timed out after {timeout_seconds:.1f}s")


class JobError(PictureError):
    """Raised for job lifecycle errors."""

    def __init__(self, message: str = "Job error") -> None:
        super().__init__(message, code="JOB_ERROR")


class JobNotFoundError(JobError):
    """Raised when a job ID is not found."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job '{job_id}' not found")


class JobAlreadyCompletedError(JobError):
    """Raised when trying to modify a completed job."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job '{job_id}' is already completed")


class StorageError(PictureError):
    """Raised for storage I/O failures."""

    def __init__(self, message: str = "Storage error") -> None:
        super().__init__(message, code="STORAGE_ERROR")


class UnsupportedMediaError(PictureError):
    """Raised when image format or MIME type is not supported."""

    def __init__(self, mime_type: str) -> None:
        super().__init__(f"Unsupported media type: '{mime_type}'", code="UNSUPPORTED_MEDIA")


class PolicyError(PictureError):
    """Raised for policy evaluation failures."""

    def __init__(self, message: str = "Policy evaluation error") -> None:
        super().__init__(message, code="POLICY_ERROR")
