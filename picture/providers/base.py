"""
Abstract base interfaces for all picture compliance providers.

Every concrete provider must implement one of these interfaces.
Business logic depends on these abstractions, never on concrete implementations.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from picture.domain.enums import RedactionMode, RouteType
from picture.domain.models import (
    BBox,
    OCRLayoutResult,
    PictureFinding,
    PictureModerationResult,
    RedactionOperation,
    RegionMask,
)


class Router(abc.ABC):
    """Classifies an image into a route type."""

    @abc.abstractmethod
    def classify(self, image_path: str, metadata: dict[str, Any] | None = None) -> RouteType:
        """Return the route type for the given image."""
        ...


class Preprocessor(abc.ABC):
    """Preprocesses an image before analysis."""

    @abc.abstractmethod
    def preprocess(self, image_path: str, output_dir: str) -> str:
        """
        Preprocess the image and return the path to the preprocessed output.

        Includes: EXIF stripping, resizing, rotation correction, color space normalization.
        """
        ...

    def extract_pdf_pages(self, pdf_path: str, output_dir: str) -> list[str]:
        """Extract pages from a PDF as images. Default: not supported."""
        from picture.domain.exceptions import UnsupportedMediaError
        raise UnsupportedMediaError("application/pdf")


class OCRLayoutProvider(abc.ABC):
    """OCR and layout analysis provider."""

    @abc.abstractmethod
    def analyze(self, image_path: str) -> OCRLayoutResult:
        """Run OCR + layout analysis on the image and return structured results."""
        ...

    @property
    def name(self) -> str:
        """Provider name for audit logging."""
        return self.__class__.__name__


class PIIDetector(abc.ABC):
    """Text-based PII detection provider."""

    @abc.abstractmethod
    def detect(self, text: str, language: str = "zh") -> list[PictureFinding]:
        """Detect PII entities in the given text. Returns findings with text spans."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SafetyModerator(abc.ABC):
    """Image safety moderation provider."""

    @abc.abstractmethod
    def moderate(self, image_path: str) -> PictureModerationResult:
        """Run safety moderation on the image."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class VisionDetector(abc.ABC):
    """Vision-based object/region detection provider (faces, IDs, QR codes, etc.)."""

    @abc.abstractmethod
    def detect(self, image_path: str) -> list[PictureFinding]:
        """Detect sensitive visual regions in the image."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SegmentationProvider(abc.ABC):
    """Segmentation refinement provider (e.g. SAM 2)."""

    @abc.abstractmethod
    def refine(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        """Refine bounding boxes into tighter polygon/mask regions."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class Redactor(abc.ABC):
    """Image redaction renderer."""

    @abc.abstractmethod
    def redact(
        self,
        image_path: str,
        operations: list[RedactionOperation],
        output_path: str,
    ) -> str:
        """
        Apply redaction operations to the image and save the result.
        Returns the path to the redacted image.
        """
        ...

    def render_overlay(
        self,
        image_path: str,
        operations: list[RedactionOperation],
        output_path: str,
    ) -> str:
        """Render an overlay visualization image (optional capability)."""
        return self.redact(image_path, operations, output_path)


class StorageBackend(abc.ABC):
    """Abstraction for file storage (local FS, S3, MinIO, etc.)."""

    @abc.abstractmethod
    def save(self, local_path: str, key: str) -> str:
        """Save a local file to storage and return its URI."""
        ...

    @abc.abstractmethod
    def load(self, uri: str, local_path: str) -> str:
        """Load a file from storage to a local path. Returns the local path."""
        ...

    @abc.abstractmethod
    def exists(self, uri: str) -> bool:
        """Check if a URI exists in storage."""
        ...

    @abc.abstractmethod
    def get_uri(self, key: str) -> str:
        """Get the full URI for a storage key."""
        ...


class JobRepository(abc.ABC):
    """Persistence layer for PictureJob entities."""

    @abc.abstractmethod
    def save_job(self, job: Any) -> None:
        """Persist or update a job."""
        ...

    @abc.abstractmethod
    def get_job(self, job_id: str) -> Any:
        """Retrieve a job by ID."""
        ...

    @abc.abstractmethod
    def list_jobs(self, tenant_id: str | None = None, limit: int = 100) -> list[Any]:
        """List jobs, optionally filtered by tenant."""
        ...


class PolicyEngine(abc.ABC):
    """Configurable policy evaluation engine."""

    @abc.abstractmethod
    def evaluate(
        self,
        findings: list[PictureFinding],
        moderation: PictureModerationResult | None,
        profile: str,
    ) -> Any:
        """Evaluate findings against a policy profile and return a policy result."""
        ...
