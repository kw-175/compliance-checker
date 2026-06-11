"""
Use cases: high-level entry points for picture compliance operations.

Provides factory functions for creating fully-wired orchestrator instances
and convenience functions for common operations.
"""

from __future__ import annotations

import logging
from typing import Any

from picture.application.orchestrator import PictureComplianceOrchestrator
from picture.domain.models import PictureJob, SourceSpec
from picture.domain.policy import ConfigurablePolicyEngine
from picture.infra.config import PictureSettings, get_settings
from picture.infra.repository import InMemoryJobRepository
from picture.infra.storage import LocalFileStorageBackend
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

# Module-level singletons for the default repository and storage
_default_repo: InMemoryJobRepository | None = None
_default_storage: LocalFileStorageBackend | None = None


def _get_default_repo() -> InMemoryJobRepository:
    global _default_repo
    if _default_repo is None:
        _default_repo = InMemoryJobRepository()
    return _default_repo


def create_router(settings: PictureSettings | None = None) -> Router:
    """Create a router instance."""
    from picture.providers.router import HeuristicRouter
    return HeuristicRouter()


def create_preprocessor(settings: PictureSettings | None = None) -> Preprocessor:
    """Create a preprocessor instance."""
    from picture.providers.preprocess import DefaultPreprocessor
    return DefaultPreprocessor()


def create_ocr_provider(settings: PictureSettings | None = None) -> OCRLayoutProvider:
    """Create an OCR provider based on settings."""
    settings = settings or get_settings()
    provider_name = settings.ocr_provider.lower()

    if provider_name == "paddleocr":
        from picture.providers.ocr.paddleocr_vl import PaddleOCRVLProvider
        return PaddleOCRVLProvider(lang=settings.paddleocr_lang, use_gpu=settings.paddleocr_use_gpu)
    elif provider_name == "mineru":
        from picture.providers.ocr.mineru import MinerUProvider
        return MinerUProvider()
    elif provider_name == "surya":
        from picture.providers.ocr.surya import SuryaProvider
        return SuryaProvider()
    else:
        from picture.providers.ocr.mock import MockOCRLayoutProvider
        return MockOCRLayoutProvider()


def create_pii_detector(settings: PictureSettings | None = None) -> PIIDetector:
    """Create a PII detector based on settings."""
    settings = settings or get_settings()
    provider_name = settings.pii_provider.lower()

    if provider_name == "presidio":
        from picture.providers.pii.presidio import PresidioPIIDetector
        return PresidioPIIDetector(languages=settings.presidio_languages)
    else:
        from picture.providers.pii.mock import MockPIIDetector
        return MockPIIDetector()


def create_safety_moderator(settings: PictureSettings | None = None) -> SafetyModerator:
    """Create a safety moderator based on settings."""
    settings = settings or get_settings()
    provider_name = settings.safety_provider.lower()

    if provider_name == "shieldgemma2":
        from picture.providers.safety.shieldgemma2 import ShieldGemmaSafetyModerator
        return ShieldGemmaSafetyModerator(
            model_name=settings.shieldgemma_model,
            device=settings.shieldgemma_device,
        )
    else:
        from picture.providers.safety.mock import MockSafetyModerator
        return MockSafetyModerator()


def create_vision_detector(settings: PictureSettings | None = None) -> VisionDetector:
    """Create a vision detector based on settings."""
    settings = settings or get_settings()
    provider_name = settings.vision_provider.lower()

    if provider_name == "yolo26":
        from picture.providers.vision.yolo26 import YOLO26VisionDetector
        return YOLO26VisionDetector(
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.yolo_confidence,
        )
    elif provider_name == "grounding_dino":
        from picture.providers.vision.grounding_dino import GroundingDINOVisionDetector
        return GroundingDINOVisionDetector()
    else:
        from picture.providers.vision.mock import MockVisionDetector
        return MockVisionDetector()


def create_segmentation_provider(settings: PictureSettings | None = None) -> SegmentationProvider:
    """Create a segmentation provider based on settings."""
    settings = settings or get_settings()
    provider_name = settings.segmentation_provider.lower()

    if provider_name == "sam2":
        from picture.providers.segmentation.sam2 import SAM2SegmentationProvider
        return SAM2SegmentationProvider(model_id=settings.sam2_model_id)
    else:
        from picture.providers.segmentation.mock import MockSegmentationProvider
        return MockSegmentationProvider()


def create_redactor(settings: PictureSettings | None = None) -> Redactor:
    """Create a redactor instance."""
    from picture.providers.redaction.opencv_redactor import OpenCVRedactor
    return OpenCVRedactor()


def create_storage(settings: PictureSettings | None = None) -> StorageBackend:
    """Create a storage backend based on settings."""
    settings = settings or get_settings()
    if settings.storage_backend == "s3":
        from picture.infra.storage import S3StorageBackend
        return S3StorageBackend(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            endpoint_url=settings.s3_endpoint_url,
        )
    return LocalFileStorageBackend(settings.storage_base_path)


def create_orchestrator(
    settings: PictureSettings | None = None,
    repository: JobRepository | None = None,
) -> PictureComplianceOrchestrator:
    """
    Create a fully-wired orchestrator with all providers.

    This is the main factory function for creating the orchestrator.
    """
    settings = settings or get_settings()

    return PictureComplianceOrchestrator(
        router=create_router(settings),
        preprocessor=create_preprocessor(settings),
        ocr_provider=create_ocr_provider(settings),
        pii_detector=create_pii_detector(settings),
        safety_moderator=create_safety_moderator(settings),
        vision_detector=create_vision_detector(settings),
        segmentation_provider=create_segmentation_provider(settings),
        redactor=create_redactor(settings),
        policy_engine=ConfigurablePolicyEngine(settings.policy_config_dir),
        storage=create_storage(settings),
        repository=repository or _get_default_repo(),
        settings=settings,
    )


def process_image(
    image_path: str,
    tenant_id: str = "default",
    profile: str = "default_cn_enterprise",
    mime_type: str = "image/png",
    options: dict[str, Any] | None = None,
    settings: PictureSettings | None = None,
) -> PictureJob:
    """
    Convenience function: process a single image through the full pipeline.

    Returns the completed PictureJob with all results populated.
    """
    settings = settings or get_settings()
    orchestrator = create_orchestrator(settings)

    job = PictureJob(
        tenant_id=tenant_id,
        source=SourceSpec(uri=image_path, mime_type=mime_type),
        profile=profile,
        options=options or {},
    )

    return orchestrator.execute(job)
