from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"


DEFAULT_OCR_MODEL_DIR = MODELS_ROOT / "paddleocr_vl" / "PaddleOCR-VL-1.5"
DEFAULT_QWEN35_MODEL_DIR = MODELS_ROOT / "Qwen" / "Qwen3.5-9B"
DEFAULT_SAM3_MODEL_DIR = MODELS_ROOT / "facebook" / "sam3"


def _file_status(path: Path, min_bytes: int = 1) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "ready": exists and size >= min_bytes,
    }


def _dir_status(path: Path, required_files: dict[str, int]) -> dict[str, Any]:
    files = {name: _file_status(path / name, min_size) for name, min_size in required_files.items()}
    return {
        "path": str(path),
        "exists": path.exists() and path.is_dir(),
        "files": files,
        "ready": path.exists() and path.is_dir() and all(item["ready"] for item in files.values()),
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _transformers_sam3_available() -> bool:
    try:
        import transformers

        return hasattr(transformers, "Sam3Model") and hasattr(transformers, "Sam3Processor")
    except Exception:
        return False


def _official_sam3_available() -> bool:
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        return Sam3Processor is not None and build_sam3_image_model is not None
    except Exception:
        return False


def _paddleocr_vl_pipeline_available() -> bool:
    try:
        from paddleocr import PaddleOCRVL

        return PaddleOCRVL is not None
    except Exception:
        return False


def _paddlex_ocr_extra_available() -> bool:
    required = [
        "bs4",
        "cssutils",
        "ftfy",
        "latex2mathml",
        "lxml",
        "openpyxl",
        "premailer",
        "sentencepiece",
        "sklearn",
        "tiktoken",
    ]
    return all(_module_available(name) for name in required)


def _endpoint_reachable(base_url: str) -> bool:
    if not base_url.strip():
        return False
    try:
        import httpx

        normalized = base_url.strip().rstrip("/")
        models_url = f"{normalized}/models" if normalized.endswith("/v1") else normalized
        response = httpx.get(models_url, timeout=2.0)
        return response.status_code < 500
    except Exception:
        return False


def check_picture_model_readiness(
    *,
    ocr_model_dir: str | Path = DEFAULT_OCR_MODEL_DIR,
    qwen35_model_dir: str | Path = DEFAULT_QWEN35_MODEL_DIR,
    sam3_model_dir: str | Path = DEFAULT_SAM3_MODEL_DIR,
) -> dict[str, Any]:
    """Return local model and runtime readiness for the documented picture pipeline."""
    ocr = _dir_status(
        Path(ocr_model_dir),
        {
            "config.json": 1,
            "model.safetensors": 1_000_000,
            "processing_paddleocr_vl.py": 1,
            "image_processing_paddleocr_vl.py": 1,
            "modeling_paddleocr_vl.py": 1,
            "processor_config.json": 1,
            "tokenizer.json": 1,
        },
    )
    qwen35 = _dir_status(
        Path(qwen35_model_dir),
        {
            "config.json": 1,
            "model.safetensors.index.json": 1,
            "preprocessor_config.json": 1,
            "tokenizer.json": 1,
        },
    )
    sam3 = _dir_status(
        Path(sam3_model_dir),
        {
            "config.json": 1,
            "processor_config.json": 1,
            "model.safetensors": 1_000_000,
            "sam3.pt": 1_000_000,
        },
    )
    try:
        from text.config.settings import get_settings as get_text_settings

        text_settings = get_text_settings()
        local_base_url = text_settings.local_compliance_base_url
        local_model = text_settings.local_compliance_model
    except Exception:
        local_base_url = ""
        local_model = ""
    qwen_endpoint = {
        "reused_from_text_compliance": True,
        "base_url_configured": bool(str(local_base_url).strip()),
        "model_configured": bool(str(local_model).strip()),
        "base_url": str(local_base_url),
        "model": str(local_model),
        "endpoint_reachable": _endpoint_reachable(str(local_base_url)),
        "note": "图片侧不单独启动 Qwen3.5-9B；视觉理解 provider 复用文本合规的 OpenAI-compatible endpoint。",
    }
    try:
        from picture.infra.config import get_settings as get_picture_settings

        picture_settings = get_picture_settings()
        ocr_provider = picture_settings.ocr_provider
        paddleocr_vl_api_url = picture_settings.paddleocr_vl_api_url
        vision_provider = picture_settings.vision_provider
        segmentation_provider = picture_settings.segmentation_provider
        sam3_api_url = picture_settings.sam3_api_url
    except Exception:
        ocr_provider = ""
        paddleocr_vl_api_url = ""
        vision_provider = ""
        segmentation_provider = ""
        sam3_api_url = ""
    paddleocr_vl_api = {
        "provider_configured": str(ocr_provider).strip().lower()
        in {"paddleocr_vl_api", "paddleocr_api", "paddlex_serving"},
        "base_url": str(paddleocr_vl_api_url),
        "endpoint_reachable": _endpoint_reachable(str(paddleocr_vl_api_url)),
        "expected_api": "/layout-parsing",
        "note": "图片侧 OCR 默认使用 PaddleX Serving 暴露的 PaddleOCR-VL 完整 pipeline API。",
    }
    sam3_api = {
        "provider_configured": str(vision_provider).strip().lower()
        in {"sam3_api", "qwen_sam3_api_fusion", "qwen35_sam3_api_fusion"}
        or str(segmentation_provider).strip().lower() == "sam3_api",
        "base_url": str(sam3_api_url),
        "endpoint_reachable": _endpoint_reachable(str(sam3_api_url)),
        "expected_api": "/v1/sam3/detect and /v1/sam3/refine",
        "note": "图片侧 SAM3 默认使用独立 FastAPI 服务；API 模式不要求 picture venv 安装 transformers Sam3Model。",
    }
    dependencies = {
        "paddle": _module_available("paddle"),
        "paddleocr": _module_available("paddleocr"),
        "paddleocr_vl_pipeline": _paddleocr_vl_pipeline_available(),
        "paddlex": _module_available("paddlex"),
        "paddlex_ocr_extra": _paddlex_ocr_extra_available(),
        "torch": _module_available("torch"),
        "transformers": _module_available("transformers"),
        "transformers_sam3": _transformers_sam3_available(),
        "official_sam3": _official_sam3_available(),
        "httpx": _module_available("httpx"),
    }
    sam3_runtime_ready = (
        sam3_api["provider_configured"]
        and sam3_api["endpoint_reachable"]
        or (sam3["ready"] and dependencies["torch"] and dependencies["transformers_sam3"])
        or (sam3["ready"] and dependencies["torch"] and dependencies["official_sam3"])
    )
    return {
        "ocr": ocr,
        "qwen35_vl": qwen35,
        "sam3": sam3,
        "qwen_endpoint": qwen_endpoint,
        "paddleocr_vl_api": paddleocr_vl_api,
        "sam3_api": sam3_api,
        "dependencies": dependencies,
        "ready_for_text_reuse": ocr["ready"] and qwen35["ready"],
        "ready_for_sam3_files": sam3["ready"],
        "ready_for_full_runtime": (
            (
                paddleocr_vl_api["provider_configured"]
                and paddleocr_vl_api["endpoint_reachable"]
                or (
                    ocr["ready"]
                    and dependencies["paddle"]
                    and dependencies["paddleocr"]
                    and dependencies["paddleocr_vl_pipeline"]
                    and dependencies["paddlex"]
                    and dependencies["paddlex_ocr_extra"]
                )
            )
            and qwen35["ready"]
            and sam3_runtime_ready
            and dependencies["httpx"]
            and qwen_endpoint["base_url_configured"]
            and qwen_endpoint["model_configured"]
            and qwen_endpoint["endpoint_reachable"]
        ),
    }
