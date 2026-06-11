"""
Use cases: high-level entry points for picture compliance operations.

Provides factory functions for creating fully-wired orchestrator instances
and convenience functions for common operations.
"""
# 中文说明：该文件是 application 层对外暴露的“组装入口”。
# 外部系统通常不需要手动创建每个 provider，而是直接调用这里的工厂函数。
from __future__ import annotations

import logging
import threading
from typing import Any

from picture.application.orchestrator import PictureComplianceOrchestrator
from picture.domain.models import PictureJob, SourceSpec
from picture.domain.policy import ConfigurablePolicyEngine
from picture.infra.config import PictureSettings, get_settings
from picture.infra.repository import InMemoryJobRepository
from picture.infra.storage import LocalFileStorageBackend
from picture.providers.openai_shared import OpenAIPictureAnalyzer
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

# 中文说明：默认仓储和默认存储后端做成模块级单例，
# 便于在简单脚本或本地调试场景下直接复用一套基础组件。
_default_repo: InMemoryJobRepository | None = None
_default_storage: LocalFileStorageBackend | None = None
_component_cache: dict[tuple[Any, ...], Any] = {}
_component_cache_lock = threading.Lock()


def _get_default_repo() -> InMemoryJobRepository:
    # 中文说明：懒加载默认内存仓储，避免模块导入时就立刻初始化所有对象。
    global _default_repo
    if _default_repo is None:
        _default_repo = InMemoryJobRepository()
    return _default_repo


def create_openai_analyzer(
    settings: PictureSettings | None = None,
) -> OpenAIPictureAnalyzer:
    """Create the shared OpenAI image analyzer used by multiple providers."""
    settings = settings or get_settings()
    return OpenAIPictureAnalyzer(settings)


def create_router(settings: PictureSettings | None = None) -> Router:
    """Create a router instance."""
    # 中文说明：路由器根据图像内容特征决定走 document / natural / mixed 哪条链路。
    from picture.providers.router import HeuristicRouter

    return HeuristicRouter()


def _settings_cache_key(settings: PictureSettings, component: str) -> tuple[Any, ...]:
    """Build a cache key for heavyweight providers that should live for the process."""
    return (
        component,
        settings.ocr_provider,
        settings.paddleocr_vl_api_url,
        settings.paddleocr_vl_api_timeout_seconds,
        settings.paddleocr_vl_api_file_type,
        settings.paddleocr_vl_api_use_layout_detection,
        settings.paddleocr_vl_api_use_chart_recognition,
        settings.paddleocr_vl_api_use_seal_recognition,
        settings.paddleocr_vl_api_prettify_markdown,
        settings.paddleocr_vl_api_visualize,
        str(settings.paddleocr_model_dir),
        settings.paddleocr_lang,
        settings.paddleocr_use_gpu,
        settings.paddleocr_device,
        settings.paddleocr_vl_task,
        settings.paddleocr_vl_backend,
        settings.paddleocr_vl_max_new_tokens,
        settings.paddleocr_vl_generation_timeout_seconds,
        settings.paddleocr_vl_qwen_fallback_enabled,
        settings.qwen_ocr_timeout_seconds,
        settings.qwen_ocr_max_tokens,
        settings.safety_provider,
        settings.vision_provider,
        settings.segmentation_provider,
        str(settings.sam3_model_dir),
        settings.sam3_device,
        settings.sam3_confidence,
        settings.sam3_api_url,
        settings.sam3_api_timeout_seconds,
        settings.yolo_model_path,
        settings.yolo_confidence,
        settings.yolo_device,
        settings.sam2_model_id,
        settings.sam2_device,
        settings.shieldgemma_model,
        settings.shieldgemma_device,
        settings.openai_timeout_seconds,
        settings.qwen35_vl_max_tokens,
        settings.qwen35_vl_image_max_side,
        settings.qwen35_vl_image_jpeg_quality,
    )


def _cached_component(key: tuple[Any, ...], factory: Any) -> Any:
    with _component_cache_lock:
        component = _component_cache.get(key)
        if component is None:
            component = factory()
            _component_cache[key] = component
        return component


def get_runtime_components(settings: PictureSettings | None = None) -> dict[str, Any]:
    """Return currently cached heavyweight components for runtime diagnostics."""
    settings = settings or get_settings()
    with _component_cache_lock:
        keys = [key for key in _component_cache if key and key[0] != "policy"]
    return {
        "cached_component_count": len(keys),
        "cached_components": [str(key[0]) for key in keys],
        "ocr_provider": settings.ocr_provider,
        "ocr_api_url": settings.paddleocr_vl_api_url,
        "ocr_api_timeout_seconds": settings.paddleocr_vl_api_timeout_seconds,
        "ocr_use_gpu": settings.paddleocr_use_gpu,
        "ocr_device": settings.paddleocr_device or ("gpu:0" if settings.paddleocr_use_gpu else "cpu"),
        "ocr_backend": settings.paddleocr_vl_backend,
        "ocr_task": settings.paddleocr_vl_task,
        "ocr_max_new_tokens": settings.paddleocr_vl_max_new_tokens,
        "ocr_generation_timeout_seconds": settings.paddleocr_vl_generation_timeout_seconds,
        "ocr_qwen_fallback_enabled": settings.paddleocr_vl_qwen_fallback_enabled,
        "qwen_ocr_timeout_seconds": settings.qwen_ocr_timeout_seconds,
        "qwen_ocr_max_tokens": settings.qwen_ocr_max_tokens,
        "safety_provider": settings.safety_provider,
        "vision_provider": settings.vision_provider,
        "segmentation_provider": settings.segmentation_provider,
        "sam3_device": settings.sam3_device,
        "sam3_model_dir": str(settings.sam3_model_dir),
        "sam3_api_url": settings.sam3_api_url,
        "sam3_api_timeout_seconds": settings.sam3_api_timeout_seconds,
    }


def warmup_runtime(settings: PictureSettings | None = None) -> dict[str, Any]:
    """Initialize heavyweight local providers once at service startup or on demand."""
    settings = settings or get_settings()
    orchestrator = create_orchestrator(settings)
    warmed: dict[str, Any] = {}
    for name, provider, attrs in (
        ("ocr", orchestrator._ocr, ("warmup", "_get_runtime", "_get_engine")),
        ("vision", orchestrator._vision, ("_get_predictor",)),
        ("segmentation", orchestrator._segmentation, ("_get_predictor",)),
        ("safety", orchestrator._safety, ("_get_provider",)),
    ):
        loader = None
        for attr in attrs:
            loader = getattr(provider, attr, None)
            if loader is not None:
                break
        if loader is None:
            warmed[name] = {"provider": provider.name, "warmed": False, "reason": "no loader"}
            continue
        try:
            loader()
            warmed[name] = {"provider": provider.name, "warmed": True}
        except Exception as exc:
            warmed[name] = {
                "provider": provider.name,
                "warmed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    warmed["runtime"] = get_runtime_components(settings)
    return warmed


def create_preprocessor(settings: PictureSettings | None = None) -> Preprocessor:
    """Create a preprocessor instance."""
    # 中文说明：当前默认使用统一预处理器，如后续接入多种预处理器，
    # 可以在这里基于 settings 做分支选择。
    from picture.providers.preprocess import DefaultPreprocessor

    return DefaultPreprocessor()


def create_ocr_provider(
    settings: PictureSettings | None = None,
    openai_analyzer: OpenAIPictureAnalyzer | None = None,
) -> OCRLayoutProvider:
    """Create an OCR provider based on settings."""
    settings = settings or get_settings()
    provider_name = settings.ocr_provider.lower()

    # 中文说明：根据配置动态选择 OCR provider，方便在 mock、轻量模型、重型模型之间切换。
    if provider_name in {"paddleocr_vl_api", "paddleocr_api", "paddlex_serving"}:
        from picture.providers.ocr.paddleocr_vl_api import PaddleOCRVLAPIProvider

        return _cached_component(
            _settings_cache_key(settings, "ocr:paddleocr_vl_api"),
            lambda: PaddleOCRVLAPIProvider(
                base_url=settings.paddleocr_vl_api_url,
                timeout_seconds=settings.paddleocr_vl_api_timeout_seconds,
                file_type=settings.paddleocr_vl_api_file_type,
                use_layout_detection=settings.paddleocr_vl_api_use_layout_detection,
                use_chart_recognition=settings.paddleocr_vl_api_use_chart_recognition,
                use_seal_recognition=settings.paddleocr_vl_api_use_seal_recognition,
                prettify_markdown=settings.paddleocr_vl_api_prettify_markdown,
                visualize=settings.paddleocr_vl_api_visualize,
            ),
        )
    if provider_name == "paddleocr":
        from picture.providers.ocr.paddleocr_vl import PaddleOCRVLProvider

        return _cached_component(
            _settings_cache_key(settings, "ocr:paddleocr"),
            lambda: PaddleOCRVLProvider(
                model_dir=str(settings.paddleocr_model_dir),
                lang=settings.paddleocr_lang,
                use_gpu=settings.paddleocr_use_gpu,
                device=settings.paddleocr_device or None,
                task=settings.paddleocr_vl_task,
                backend=settings.paddleocr_vl_backend,
                max_new_tokens=settings.paddleocr_vl_max_new_tokens,
                generation_timeout_seconds=settings.paddleocr_vl_generation_timeout_seconds,
                qwen_fallback_enabled=settings.paddleocr_vl_qwen_fallback_enabled,
                qwen_timeout_seconds=settings.qwen_ocr_timeout_seconds,
                qwen_max_tokens=settings.qwen_ocr_max_tokens,
                qwen_image_max_side=settings.qwen35_vl_image_max_side,
                qwen_image_jpeg_quality=settings.qwen35_vl_image_jpeg_quality,
            ),
        )
    elif provider_name == "mineru":
        from picture.providers.ocr.mineru import MinerUProvider

        return MinerUProvider()
    elif provider_name == "surya":
        from picture.providers.ocr.surya import SuryaProvider

        return SuryaProvider()
    elif provider_name == "openai_gpt52":
        from picture.providers.ocr.openai_gpt52 import OpenAIGPT52OCRLayoutProvider

        return OpenAIGPT52OCRLayoutProvider(
            analyzer=openai_analyzer or create_openai_analyzer(settings)
        )
    else:
        # 中文说明：默认回落到 mock，保证在未安装真实模型时项目也能跑通开发与测试链路。
        from picture.providers.ocr.mock import MockOCRLayoutProvider

        return MockOCRLayoutProvider()


def create_pii_detector(
    settings: PictureSettings | None = None,
    openai_analyzer: OpenAIPictureAnalyzer | None = None,
) -> PIIDetector:
    """Create a PII detector based on settings."""
    settings = settings or get_settings()
    provider_name = settings.pii_provider.lower()

    if provider_name == "text_compliance":
        from picture.providers.pii.mock import MockPIIDetector

        # 中文说明：OCR 文本合规默认在 orchestrator 中整体复用 text.api_pipeline。
        # 这里保留一个 mock PII provider 作为接口占位，避免图片侧重复启动 Qwen 或另建 PII 服务。
        return MockPIIDetector()
    elif provider_name == "presidio":
        from picture.providers.pii.presidio import PresidioPIIDetector

        return PresidioPIIDetector(languages=settings.presidio_languages)
    elif provider_name == "openai_gpt52":
        from picture.providers.pii.openai_gpt52 import OpenAIGPT52PIIDetector

        return OpenAIGPT52PIIDetector(
            analyzer=openai_analyzer or create_openai_analyzer(settings)
        )
    else:
        from picture.providers.pii.mock import MockPIIDetector

        return MockPIIDetector()


def create_safety_moderator(
    settings: PictureSettings | None = None,
    openai_analyzer: OpenAIPictureAnalyzer | None = None,
) -> SafetyModerator:
    """Create a safety moderator based on settings."""
    settings = settings or get_settings()
    provider_name = settings.safety_provider.lower()

    if provider_name == "shieldgemma2":
        from picture.providers.safety.shieldgemma2 import ShieldGemmaSafetyModerator

        return _cached_component(
            _settings_cache_key(settings, "safety:shieldgemma2"),
            lambda: ShieldGemmaSafetyModerator(
                model_name=settings.shieldgemma_model,
                device=settings.shieldgemma_device,
            ),
        )
    elif provider_name == "qwen35_vl":
        from picture.providers.safety.qwen35_vl import Qwen35VLSafetyModerator

        return _cached_component(
            _settings_cache_key(settings, "safety:qwen35_vl"),
            lambda: Qwen35VLSafetyModerator(
                timeout_seconds=settings.openai_timeout_seconds,
                max_tokens=settings.qwen35_vl_max_tokens,
                image_max_side=settings.qwen35_vl_image_max_side,
                image_jpeg_quality=settings.qwen35_vl_image_jpeg_quality,
            ),
        )
    elif provider_name in {"qwen_sam3_safety_fusion", "qwen35_sam3_safety_fusion"}:
        from picture.providers.safety.qwen_sam3_fusion import QwenSAM3SafetyFusionModerator

        return _cached_component(
            _settings_cache_key(settings, "safety:qwen_sam3_safety_fusion"),
            lambda: QwenSAM3SafetyFusionModerator(
                sam3_api_url=settings.sam3_api_url,
                sam3_timeout_seconds=settings.sam3_api_timeout_seconds,
                sam3_confidence=settings.sam3_confidence,
                qwen_timeout_seconds=settings.openai_timeout_seconds,
                qwen_max_tokens=settings.qwen35_vl_max_tokens,
                image_max_side=settings.qwen35_vl_image_max_side,
                image_jpeg_quality=settings.qwen35_vl_image_jpeg_quality,
            ),
        )
    elif provider_name == "openai_gpt52":
        from picture.providers.safety.openai_gpt52 import OpenAIGPT52SafetyModerator

        return OpenAIGPT52SafetyModerator(
            analyzer=openai_analyzer or create_openai_analyzer(settings)
        )
    else:
        from picture.providers.safety.mock import MockSafetyModerator

        return MockSafetyModerator()


def create_vision_detector(
    settings: PictureSettings | None = None,
    openai_analyzer: OpenAIPictureAnalyzer | None = None,
) -> VisionDetector:
    """Create a vision detector based on settings."""
    settings = settings or get_settings()
    provider_name = settings.vision_provider.lower()

    if provider_name == "yolo26":
        from picture.providers.vision.yolo26 import YOLO26VisionDetector

        return YOLO26VisionDetector(
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.yolo_confidence,
            device=settings.yolo_device,
        )
    elif provider_name == "sam3_api":
        from picture.providers.vision.sam3_api import SAM3APIVisionDetector

        return _cached_component(
            _settings_cache_key(settings, "vision:sam3_api"),
            lambda: SAM3APIVisionDetector(
                base_url=settings.sam3_api_url,
                confidence_threshold=settings.sam3_confidence,
                timeout_seconds=settings.sam3_api_timeout_seconds,
            ),
        )
    elif provider_name == "sam3":
        from picture.providers.vision.sam3 import SAM3SensitiveObjectDetector

        return _cached_component(
            _settings_cache_key(settings, "vision:sam3"),
            lambda: SAM3SensitiveObjectDetector(
                model_dir=str(settings.sam3_model_dir),
                confidence_threshold=settings.sam3_confidence,
                device=settings.sam3_device,
            ),
        )
    elif provider_name in {"qwen_sam3_api_fusion", "qwen35_sam3_api_fusion"}:
        from picture.providers.vision.qwen_sam3_fusion import QwenSAM3FusionVisionDetector
        from picture.providers.vision.sam3_api import SAM3APIVisionDetector

        return _cached_component(
            _settings_cache_key(settings, "vision:qwen_sam3_api_fusion"),
            lambda: QwenSAM3FusionVisionDetector(
                model_dir=str(settings.sam3_model_dir),
                confidence_threshold=settings.sam3_confidence,
                device=settings.sam3_device,
                semantic_threshold=settings.qwen_sam3_semantic_threshold,
                sam3_keep_without_qwen_threshold=settings.qwen_sam3_keep_without_qwen_threshold,
                qwen_timeout_seconds=settings.openai_timeout_seconds,
                qwen_max_tokens=max(settings.qwen35_vl_max_tokens, 768),
                image_max_side=settings.qwen35_vl_image_max_side,
                image_jpeg_quality=settings.qwen35_vl_image_jpeg_quality,
                sam3_detector=SAM3APIVisionDetector(
                    base_url=settings.sam3_api_url,
                    confidence_threshold=settings.sam3_confidence,
                    timeout_seconds=settings.sam3_api_timeout_seconds,
                ),
            ),
        )
    elif provider_name in {"qwen_sam3_fusion", "qwen35_sam3_fusion"}:
        from picture.providers.vision.qwen_sam3_fusion import QwenSAM3FusionVisionDetector

        return _cached_component(
            _settings_cache_key(settings, "vision:qwen_sam3_fusion"),
            lambda: QwenSAM3FusionVisionDetector(
                model_dir=str(settings.sam3_model_dir),
                confidence_threshold=settings.sam3_confidence,
                device=settings.sam3_device,
                semantic_threshold=settings.qwen_sam3_semantic_threshold,
                sam3_keep_without_qwen_threshold=settings.qwen_sam3_keep_without_qwen_threshold,
                qwen_timeout_seconds=settings.openai_timeout_seconds,
                qwen_max_tokens=max(settings.qwen35_vl_max_tokens, 768),
                image_max_side=settings.qwen35_vl_image_max_side,
                image_jpeg_quality=settings.qwen35_vl_image_jpeg_quality,
            ),
        )
    elif provider_name == "grounding_dino":
        from picture.providers.vision.grounding_dino import GroundingDINOVisionDetector

        return GroundingDINOVisionDetector()
    elif provider_name == "openai_gpt52":
        from picture.providers.vision.openai_gpt52 import OpenAIGPT52VisionDetector

        return OpenAIGPT52VisionDetector(
            analyzer=openai_analyzer or create_openai_analyzer(settings)
        )
    else:
        from picture.providers.vision.mock import MockVisionDetector

        return MockVisionDetector()


def create_segmentation_provider(
    settings: PictureSettings | None = None,
) -> SegmentationProvider:
    """Create a segmentation provider based on settings."""
    settings = settings or get_settings()
    provider_name = settings.segmentation_provider.lower()

    if provider_name == "sam2":
        from picture.providers.segmentation.sam2 import SAM2SegmentationProvider

        sam2_source = str(settings.sam2_model_dir)
        if not sam2_source or sam2_source == ".":
            sam2_source = settings.sam2_model_id
        return SAM2SegmentationProvider(
            model_id=sam2_source,
            device=settings.sam2_device,
        )
    elif provider_name == "sam3_api":
        from picture.providers.segmentation.sam3_api import SAM3APISegmentationProvider

        return _cached_component(
            _settings_cache_key(settings, "segmentation:sam3_api"),
            lambda: SAM3APISegmentationProvider(
                base_url=settings.sam3_api_url,
                timeout_seconds=settings.sam3_api_timeout_seconds,
                confidence_threshold=settings.sam3_confidence,
            ),
        )
    elif provider_name == "sam3":
        from picture.providers.segmentation.sam3 import SAM3SegmentationProvider

        return _cached_component(
            _settings_cache_key(settings, "segmentation:sam3"),
            lambda: SAM3SegmentationProvider(
                model_dir=str(settings.sam3_model_dir),
                device=settings.sam3_device,
            ),
        )
    else:
        from picture.providers.segmentation.mock import MockSegmentationProvider

        return MockSegmentationProvider()


def create_redactor(settings: PictureSettings | None = None) -> Redactor:
    """Create a redactor instance."""
    # 中文说明：当前只有 OpenCV 脱敏器；如果后续要支持 GPU 或更复杂的渲染器，
    # 可以继续在这里做工厂分发。
    from picture.providers.redaction.opencv_redactor import OpenCVRedactor

    return OpenCVRedactor()


def create_storage(settings: PictureSettings | None = None) -> StorageBackend:
    """Create a storage backend based on settings."""
    settings = settings or get_settings()

    # 中文说明：存储层决定最终产物和报告存到本地还是对象存储。
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
    openai_provider_selected = any(
        provider.lower() == "openai_gpt52"
        for provider in (
            settings.ocr_provider,
            settings.pii_provider,
            settings.safety_provider,
            settings.vision_provider,
        )
    )
    openai_analyzer = (
        create_openai_analyzer(settings) if openai_provider_selected else None
    )

    # 中文说明：这里把所有 provider、policy、storage、repository 一次性装配完成，
    # 让调用方拿到的 orchestrator 已经是可直接执行任务的完整实例。
    return PictureComplianceOrchestrator(
        router=create_router(settings),
        preprocessor=create_preprocessor(settings),
        ocr_provider=create_ocr_provider(settings, openai_analyzer=openai_analyzer),
        pii_detector=create_pii_detector(settings, openai_analyzer=openai_analyzer),
        safety_moderator=create_safety_moderator(
            settings, openai_analyzer=openai_analyzer
        ),
        vision_detector=create_vision_detector(
            settings, openai_analyzer=openai_analyzer
        ),
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

    # 中文说明：这里把最小输入参数封装成 PictureJob，
    # 适合脚本式调用或单图调试场景。
    job = PictureJob(
        tenant_id=tenant_id,
        source=SourceSpec(uri=image_path, mime_type=mime_type),
        profile=profile,
        options=options or {},
    )

    return orchestrator.execute(job)
