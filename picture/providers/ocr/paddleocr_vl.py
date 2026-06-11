"""Local PaddleOCR-VL 1.5 OCR provider."""
from __future__ import annotations

import ast
from collections import Counter
import json
import logging
import multiprocessing as mp
import re
import threading
from pathlib import Path
from typing import Any

import httpx

from picture.domain.exceptions import ProviderNotAvailableError
from picture.domain.models import BBox, OCRLayoutResult, OCRTextBlock, Polygon
from picture.providers.base import OCRLayoutProvider
from picture.providers.safety.qwen35_vl import _extract_json, _image_data_url

logger = logging.getLogger(__name__)

_REQUIRED_MODEL_FILES = {
    "config.json": 1024,
    "model.safetensors": 100_000_000,
    "processing_paddleocr_vl.py": 1024,
    "image_processing_paddleocr_vl.py": 1024,
    "modeling_paddleocr_vl.py": 1024,
    "tokenizer.json": 1024,
}


class PaddleOCRVLProvider(OCRLayoutProvider):
    """PaddleOCR-VL 1.5 OCR + text spotting provider backed by local weights."""

    def __init__(
        self,
        model_dir: str = "/data/kw/compliance-checker/models/paddleocr_vl/PaddleOCR-VL-1.5",
        lang: str = "ch",
        use_gpu: bool = True,
        device: str | None = None,
        task: str = "spotting",
        backend: str = "transformers",
        max_new_tokens: int = 768,
        generation_timeout_seconds: float = 90.0,
        qwen_fallback_enabled: bool = True,
        qwen_timeout_seconds: float = 180.0,
        qwen_max_tokens: int = 4096,
        qwen_image_max_side: int = 1280,
        qwen_image_jpeg_quality: int = 85,
        **kwargs: Any,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._lang = lang
        self._use_gpu = use_gpu
        self._device = device
        self._task = task
        self._backend = backend.strip().lower() or "transformers"
        self._max_new_tokens = max_new_tokens
        self._generation_timeout_seconds = generation_timeout_seconds
        self._qwen_fallback_enabled = qwen_fallback_enabled
        self._qwen_timeout_seconds = qwen_timeout_seconds
        self._qwen_max_tokens = qwen_max_tokens
        self._qwen_image_max_side = qwen_image_max_side
        self._qwen_image_jpeg_quality = qwen_image_jpeg_quality
        self._kwargs = kwargs
        self._runtime: dict[str, Any] | None = None
        self._runtime_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-1.5(local)"

    def _resolve_paddle_device(self) -> str:
        requested = self._device or ("gpu:0" if self._use_gpu else "cpu")
        if requested.startswith("cuda"):
            requested = "gpu" + requested[4:]
        if requested == "gpu":
            requested = "gpu:0"
        if requested == "cpu" and self._use_gpu:
            raise ProviderNotAvailableError(
                "PaddleOCR-VL GPU mode requested, but resolved Paddle device is CPU"
            )
        return requested

    def _resolve_device(self, torch: Any) -> str:
        if self._device:
            requested = self._device
        else:
            requested = "cuda" if self._use_gpu else "cpu"
        if requested.startswith("gpu"):
            requested = "cuda" + requested[3:]
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ProviderNotAvailableError(
                f"PaddleOCR-VL requested device {requested!r}, but CUDA is not available"
            )
        if requested == "cpu" and self._use_gpu:
            raise ProviderNotAvailableError(
                "PaddleOCR-VL GPU mode requested, but resolved device is CPU"
            )
        return requested

    def _validate_model_dir(self) -> None:
        if not self._model_dir.is_dir():
            raise ProviderNotAvailableError(f"PaddleOCR-VL local model dir: {self._model_dir}")
        missing: list[str] = []
        for filename, min_size in _REQUIRED_MODEL_FILES.items():
            path = self._model_dir / filename
            if not path.is_file() or path.stat().st_size < min_size:
                missing.append(filename)
        if missing:
            raise ProviderNotAvailableError(
                f"PaddleOCR-VL local model files missing/invalid under {self._model_dir}: {', '.join(missing)}"
            )

    def _get_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            self._validate_model_dir()
            if self._backend not in {"auto", "transformers", "paddleocr_pipeline"}:
                raise ProviderNotAvailableError(
                    f"Unsupported PaddleOCR-VL backend: {self._backend}"
                )

            if self._backend in {"auto", "paddleocr_pipeline"}:
                try:
                    from paddleocr import PaddleOCRVL

                    device = self._resolve_paddle_device()
                    PaddleOCRVL
                    self._runtime = {
                        "backend": "paddleocr_pipeline_isolated",
                        "device": device,
                    }
                    return self._runtime
                except Exception as exc:
                    if self._backend == "paddleocr_pipeline":
                        raise ProviderNotAvailableError("PaddleOCR-VL official pipeline") from exc
                    logger.warning(
                        "PaddleOCR-VL official pipeline unavailable, falling back to transformers: %s: %s",
                        type(exc).__name__,
                        exc,
                    )

            try:
                import torch
                from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            except ImportError as exc:
                raise ProviderNotAvailableError("PaddleOCR-VL transformers runtime") from exc

            device = self._resolve_device(torch)
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            _patch_paddleocr_vl_rope_runtime(ROPE_INIT_FUNCTIONS)
            config = AutoConfig.from_pretrained(
                str(self._model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            _patch_paddleocr_vl_config(config)
            model = AutoModelForCausalLM.from_pretrained(
                str(self._model_dir),
                config=config,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(device)
            _patch_paddleocr_vl_generation(model, torch)
            model.eval()
            processor = AutoProcessor.from_pretrained(
                str(self._model_dir),
                local_files_only=True,
                trust_remote_code=True,
            )
            self._runtime = {
                "backend": "transformers",
                "torch": torch,
                "model": model,
                "processor": processor,
                "device": device,
            }
            return self._runtime

    def analyze(self, image_path: str) -> OCRLayoutResult:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ProviderNotAvailableError("Pillow for PaddleOCR-VL") from exc

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        requested_task = self._task.strip().lower() or "spotting"
        paddle_error = ""
        try:
            runtime = self._get_runtime()
            if runtime.get("backend") == "paddleocr_pipeline_isolated":
                result = self._analyze_with_paddle_pipeline(runtime, image_path, orig_w, orig_h, requested_task)
            else:
                result = self._analyze_with_transformers(runtime, image, orig_w, orig_h, requested_task)
        except Exception as exc:
            paddle_error = f"{type(exc).__name__}: {exc}"
            logger.exception("PaddleOCR-VL primary OCR failed; trying Qwen OCR fallback: %s", paddle_error)
            result = OCRLayoutResult(
                full_text="",
                text_blocks=[],
                engine_name=self.name,
                metadata={
                    "backend": self._backend,
                    "local_model_dir": str(self._model_dir),
                    "model_files_ready": self._model_dir.is_dir(),
                    "task": requested_task,
                    "valid_text": False,
                    "invalid_reason": "paddleocr_failed",
                    "paddle_error": paddle_error,
                },
            )

        if self._result_has_valid_text(result):
            return result
        if not self._qwen_fallback_enabled:
            return result
        try:
            fallback = self._analyze_with_qwen_fallback(
                image_path,
                orig_w,
                orig_h,
                requested_task,
                paddle_result=result,
                paddle_error=paddle_error,
            )
        except Exception as exc:
            logger.exception("Qwen3.5 OCR fallback failed: %s", exc)
            result.metadata = {
                **dict(result.metadata or {}),
                "qwen_fallback_attempted": True,
                "qwen_fallback_success": False,
                "qwen_fallback_error": f"{type(exc).__name__}: {exc}",
            }
            return result
        return fallback

    @staticmethod
    def _result_has_valid_text(result: OCRLayoutResult) -> bool:
        metadata = dict(result.metadata or {})
        valid_text = bool(metadata.get("valid_text", bool(result.full_text.strip())))
        return bool(valid_text and result.full_text and result.full_text.strip())

    def _analyze_with_paddle_pipeline(
        self,
        runtime: dict[str, Any],
        image_path: str,
        orig_w: int,
        orig_h: int,
        requested_task: str,
    ) -> OCRLayoutResult:
        prompt_labels = _official_prompt_labels(requested_task)
        for prompt_label in prompt_labels:
            logger.info(
                "PaddleOCR-VL official isolated pipeline scheduled: task=%s max_new_tokens=%d image=%s timeout=%.1fs",
                prompt_label,
                self._max_new_tokens,
                image_path,
                self._generation_timeout_seconds,
            )
        timeout = max(float(self._generation_timeout_seconds), 1.0) * max(len(prompt_labels), 1)
        raw_payloads = _run_paddle_pipeline_isolated(
            model_dir=str(self._model_dir),
            image_path=image_path,
            device=str(runtime["device"]),
            prompt_labels=prompt_labels,
            max_new_tokens=self._max_new_tokens,
            timeout_seconds=timeout,
        )
        blocks: list[OCRTextBlock] = []
        for payload in raw_payloads:
            blocks.extend(_parse_paddle_result_blocks(payload, orig_w, orig_h))

        blocks = _dedupe_blocks(blocks)
        full_text = "\n".join(block.text for block in blocks if block.text.strip())
        valid_text, invalid_reason = _validate_ocr_text(full_text)
        if not valid_text:
            full_text = ""
            blocks = []
        elif not blocks and full_text:
            blocks = [
                OCRTextBlock(
                    text=full_text,
                    bbox=BBox(x=0, y=0, w=float(orig_w), h=float(orig_h)),
                    confidence=0.62,
                )
            ]
        raw_output = json.dumps(_safe_json(raw_payloads), ensure_ascii=False, default=str)
        return OCRLayoutResult(
            full_text=full_text.strip(),
            text_blocks=blocks,
            engine_name=self.name,
            metadata={
                "backend": "paddleocr_pipeline",
                "isolation": "subprocess_hard_timeout",
                "local_model_dir": str(self._model_dir),
                "model_files_ready": True,
                "device": runtime["device"],
                "task": requested_task,
                "generation_passes": prompt_labels,
                "official_pipeline_timeout_seconds": timeout,
                "lang": self._lang,
                "valid_text": valid_text,
                "invalid_reason": invalid_reason,
                "bbox_source": "paddleocr_pipeline",
                "requires_manual_region_review": bool(valid_text and not blocks),
                "effective_text_preview": full_text[:1000],
                "raw_output": raw_output[:8000],
                "raw_output_truncated": len(raw_output) > 8000,
            },
        )

    def _analyze_with_qwen_fallback(
        self,
        image_path: str,
        orig_w: int,
        orig_h: int,
        requested_task: str,
        *,
        paddle_result: OCRLayoutResult,
        paddle_error: str = "",
    ) -> OCRLayoutResult:
        provider = _resolve_qwen_provider()
        url = _chat_completions_url(provider.base_url)
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        prompt = (
            "你是图片OCR引擎。请完整提取图片中所有可见文字，尤其是发票、账单、证件、表格、地址、电话、税号、"
            "账号、IBAN、金额、日期和商品明细。必须只返回JSON对象，不要Markdown。格式："
            "{\"full_text\":\"按阅读顺序合并的完整文本\","
            "\"text_blocks\":[{\"text\":\"单个文本块或单行文字\",\"bbox\":[x,y,w,h],\"confidence\":0.0至1.0,\"language\":\"zh/en/unknown\"}],"
            "\"notes\":\"简短说明\"}。"
            "bbox 使用原图像素坐标；如果不能精确定位，也要尽量给出近似区域。不要因为文字是英文或表格就返回空。"
        )
        payload = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": "你是严格的OCR模型，只负责从图片中提取文字和位置。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(
                                    image_path,
                                    max_side=self._qwen_image_max_side,
                                    jpeg_quality=self._qwen_image_jpeg_quality,
                                )
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self._qwen_max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(url, headers=headers, json=payload, timeout=self._qwen_timeout_seconds)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data = _extract_json(content)

        blocks = _qwen_blocks(data, orig_w, orig_h)
        full_text = str(data.get("full_text") or "").strip()
        if not full_text:
            full_text = "\n".join(block.text for block in blocks if block.text.strip()).strip()
        valid_text, invalid_reason = _validate_ocr_text(full_text)
        if not valid_text:
            blocks = []
            full_text = ""
        elif not blocks:
            blocks = [
                OCRTextBlock(
                    text=full_text,
                    bbox=BBox(x=0, y=0, w=float(orig_w), h=float(orig_h)),
                    confidence=0.55,
                    language="unknown",
                )
            ]
        deduped_blocks = _dedupe_blocks(blocks)
        region_review_required = _qwen_region_review_required(deduped_blocks, orig_w, orig_h)
        raw_output = json.dumps(_safe_json(data), ensure_ascii=False, default=str)
        return OCRLayoutResult(
            full_text=full_text,
            text_blocks=deduped_blocks,
            engine_name=f"{self.name}+Qwen3.5-OCR-Fallback",
            metadata={
                "backend": "qwen35_vl_fallback",
                "primary_backend": dict(paddle_result.metadata or {}).get("backend", self._backend),
                "local_model_dir": str(self._model_dir),
                "model_files_ready": self._model_dir.is_dir(),
                "device": dict(paddle_result.metadata or {}).get("device", ""),
                "task": requested_task,
                "generation_passes": list(dict(paddle_result.metadata or {}).get("generation_passes") or []),
                "valid_text": valid_text,
                "invalid_reason": invalid_reason,
                "qwen_fallback_attempted": True,
                "qwen_fallback_success": valid_text,
                "qwen_provider_model": provider.model,
                "qwen_provider_mode": getattr(provider, "mode", ""),
                "paddle_invalid_reason": dict(paddle_result.metadata or {}).get("invalid_reason", ""),
                "paddle_error": paddle_error or dict(paddle_result.metadata or {}).get("paddle_error", ""),
                "bbox_source": "qwen35_vl_json_or_full_page",
                "requires_manual_region_review": bool(valid_text and region_review_required),
                "region_review_reason": (
                    "Qwen3.5 OCR 兜底给出的文本框可能为近似区域，需要人工复核定位。"
                    if valid_text and region_review_required
                    else ""
                ),
                "qwen_bbox_count": len(deduped_blocks),
                "effective_text_preview": full_text[:1000],
                "raw_output": raw_output[:8000],
                "raw_output_truncated": len(raw_output) > 8000,
            },
        )

    def _analyze_with_transformers(
        self,
        runtime: dict[str, Any],
        image: Any,
        orig_w: int,
        orig_h: int,
        requested_task: str,
    ) -> OCRLayoutResult:
        pass_tasks = _ocr_pass_tasks(requested_task)
        generated_parts: list[tuple[str, str]] = []
        blocks: list[OCRTextBlock] = []
        for pass_task in pass_tasks:
            logger.info(
                "PaddleOCR-VL transformers generation started: task=%s max_new_tokens=%d timeout=%.1fs",
                pass_task,
                self._max_new_tokens,
                self._generation_timeout_seconds,
            )
            generated = self._generate(runtime, image, pass_task)
            logger.info(
                "PaddleOCR-VL transformers generation finished: task=%s chars=%d",
                pass_task,
                len(generated),
            )
            generated_parts.append((pass_task, generated))
            blocks.extend(_parse_text_blocks(generated, orig_w, orig_h))

        blocks = _dedupe_blocks(blocks)
        generated = "\n\n".join(f"[{name}]\n{text}" for name, text in generated_parts if text.strip())
        block_text = "\n".join(block.text for block in blocks if block.text.strip())
        plain_text = _best_plain_text_candidate(text for _, text in generated_parts)
        full_text = block_text or plain_text
        valid_text, invalid_reason = _validate_ocr_text(full_text)
        if not valid_text:
            full_text = ""
            blocks = []
        elif not blocks and full_text:
            blocks = [
                OCRTextBlock(
                    text=full_text,
                    bbox=BBox(x=0, y=0, w=float(orig_w), h=float(orig_h)),
                    confidence=0.62,
                )
            ]

        return OCRLayoutResult(
            full_text=full_text.strip(),
            text_blocks=blocks,
            engine_name=self.name,
            metadata={
                "local_model_dir": str(self._model_dir),
                "model_files_ready": True,
                "backend": "transformers",
                "device": runtime["device"],
                "task": requested_task,
                "generation_passes": pass_tasks,
                "lang": self._lang,
                "valid_text": valid_text,
                "invalid_reason": invalid_reason,
                "bbox_source": "model_or_parser" if block_text else ("fallback_full_page" if blocks else ""),
                "requires_manual_region_review": bool(valid_text and blocks and not block_text),
                "effective_text_preview": full_text[:1000],
                "raw_output": generated[:8000],
                "raw_output_truncated": len(generated) > 8000,
            },
        )

    def _generate(self, runtime: dict[str, Any], image: Any, task: str) -> str:
        torch = runtime["torch"]
        model = runtime["model"]
        processor = runtime["processor"]
        prompt = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }.get(task, "OCR:")
        max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        image_processor = processor.image_processor
        previous_max_pixels = getattr(image_processor, "max_pixels", None)
        previous_size = getattr(image_processor, "size", None)
        image_processor.max_pixels = max_pixels
        image_processor.size = {
            "min_pixels": getattr(image_processor, "min_pixels", 28 * 28 * 130),
            "max_pixels": max_pixels,
        }
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": getattr(image_processor, "min_pixels", 28 * 28 * 130),
                        "longest_edge": max_pixels,
                    }
                },
            ).to(model.device)
        finally:
            if previous_max_pixels is not None:
                image_processor.max_pixels = previous_max_pixels
            if previous_size is not None:
                image_processor.size = previous_size
        with torch.no_grad():
            generate_kwargs = {"max_new_tokens": self._max_new_tokens}
            if self._generation_timeout_seconds > 0:
                generate_kwargs["max_time"] = self._generation_timeout_seconds
            outputs = model.generate(**inputs, **generate_kwargs)
        token_start = inputs["input_ids"].shape[-1]
        return processor.decode(outputs[0][token_start:-1]).strip()


def _patch_paddleocr_vl_rope_runtime(rope_init_functions: dict[str, Any]) -> None:
    """Bridge PaddleOCR-VL local code to newer transformers RoPE init names."""
    proportional = rope_init_functions.get("proportional")
    if proportional is None:
        return
    rope_init_functions.setdefault("default", proportional)


def _patch_paddleocr_vl_config(config: Any) -> None:
    """Avoid transformers 5.x default-RoPE init path unsupported by this model code."""
    seen: set[int] = set()

    def patch_one(value: Any) -> None:
        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)

        rope_scaling = getattr(value, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            if rope_scaling.get("rope_type") == "default" or rope_scaling.get("type") == "default":
                rope_scaling["rope_type"] = "proportional"
                rope_scaling["type"] = "proportional"

        for child_name in ("text_config", "vision_config"):
            child = getattr(value, child_name, None)
            if child is not None:
                patch_one(child)

    patch_one(config)


def _patch_paddleocr_vl_generation(model: Any, torch: Any) -> None:
    """Keep PaddleOCR-VL remote-code generation compatible with transformers 5.x."""
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if prepare is None:
        return

    def prepare_inputs_for_generation(input_ids: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("cache_position") is None and input_ids is not None:
            kwargs["cache_position"] = torch.arange(input_ids.shape[-1], device=input_ids.device)
        return prepare(input_ids, *args, **kwargs)

    model.prepare_inputs_for_generation = prepare_inputs_for_generation


def _parse_text_blocks(text: str, width: int, height: int) -> list[OCRTextBlock]:
    blocks = _parse_json_like_blocks(text, width, height)
    if blocks:
        return blocks
    blocks = _parse_markup_blocks(text, width, height)
    if blocks:
        return blocks
    cleaned = _strip_markup(text)
    valid, _ = _validate_ocr_text(cleaned)
    if not valid:
        return []
    return [
        OCRTextBlock(
            text=cleaned,
            bbox=BBox(x=0, y=0, w=float(width), h=float(height)),
            confidence=0.65,
        )
    ]


def _parse_json_like_blocks(text: str, width: int, height: int) -> list[OCRTextBlock]:
    candidates = [text.strip()]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            try:
                payload = ast.literal_eval(candidate)
            except Exception:
                continue
        items = payload if isinstance(payload, list) else _find_items(payload)
        blocks = [_item_to_block(item, width, height) for item in items]
        blocks = [block for block in blocks if block is not None]
        if blocks:
            return blocks
    return []


def _find_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("blocks", "text_blocks", "ocr", "results", "items", "lines"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _item_to_block(item: Any, width: int, height: int) -> OCRTextBlock | None:
    if not isinstance(item, dict):
        return None
    value = item.get("text") or item.get("transcription") or item.get("content") or item.get("label")
    if value is None or not str(value).strip():
        return None
    valid, _ = _validate_ocr_text(str(value))
    if not valid:
        return None
    bbox_value = item.get("bbox") or item.get("box") or item.get("quad") or item.get("polygon") or item.get("points")
    bbox = _bbox_from_any(bbox_value, width, height)
    if bbox is None:
        bbox = BBox(x=0, y=0, w=float(width), h=float(height))
    score = item.get("confidence", item.get("score", 0.75))
    try:
        confidence = float(score)
    except Exception:
        confidence = 0.75
    return OCRTextBlock(text=str(value).strip(), bbox=bbox, confidence=confidence)


def _parse_markup_blocks(text: str, width: int, height: int) -> list[OCRTextBlock]:
    blocks: list[OCRTextBlock] = []
    patterns = [
        re.compile(r"(?P<txt>[^<\n]+)\s*<bbox>\s*(?P<box>[^<]+)\s*</bbox>", re.I),
        re.compile(r"<bbox>\s*(?P<box>[^<]+)\s*</bbox>\s*(?P<txt>[^<\n]+)", re.I),
        re.compile(r"(?P<txt>[^\[\]\n]+)\s*\[(?P<box>\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){3,7})\]"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            bbox = _bbox_from_any(match.group("box"), width, height)
            cleaned = _strip_markup(match.group("txt"))
            valid, _ = _validate_ocr_text(cleaned)
            if bbox is not None and cleaned and valid:
                blocks.append(OCRTextBlock(text=cleaned, bbox=bbox, confidence=0.75))
        if blocks:
            return blocks
    return []


def _bbox_from_any(value: Any, width: int, height: int) -> BBox | None:
    if value is None:
        return None
    if isinstance(value, str):
        nums = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    elif isinstance(value, (list, tuple)):
        flat: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flat.extend(float(v) for v in item[:2])
            elif isinstance(item, (int, float)):
                flat.append(float(item))
        nums = flat
    else:
        return None
    if len(nums) < 4:
        return None
    if len(nums) == 4:
        x1, y1, x2, y2 = nums
    else:
        xs = nums[0::2]
        ys = nums[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return BBox(
        x=max(0.0, min(float(width), x1)),
        y=max(0.0, min(float(height), y1)),
        w=max(0.0, min(float(width), x2) - max(0.0, min(float(width), x1))),
        h=max(0.0, min(float(height), y2) - max(0.0, min(float(height), y1))),
    )


def _strip_markup(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\[[\d\s,.;:-]{7,}\]", " ", cleaned)
    cleaned = re.sub(r"\\\(\s*\\\)", " ", cleaned)
    cleaned = re.sub(r"\\\[\s*\\\]", " ", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+(?:\{[^{}]*\})?", " ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _ocr_pass_tasks(requested_task: str) -> list[str]:
    ordered = ["ocr", requested_task]
    result: list[str] = []
    for task in ordered:
        task = task.strip().lower()
        if task and task not in result:
            result.append(task)
    return result


def _resolve_qwen_provider() -> Any:
    from text.api_clients import resolve_provider_config
    from text.config.settings import get_settings as get_text_settings

    return resolve_provider_config(get_text_settings())


def _run_paddle_pipeline_isolated(
    *,
    model_dir: str,
    image_path: str,
    device: str,
    prompt_labels: list[str],
    max_new_tokens: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_paddle_pipeline_worker,
        args=(queue, model_dir, image_path, device, prompt_labels, max_new_tokens),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(
            f"PaddleOCR-VL official pipeline timed out after {timeout_seconds:.1f}s "
            f"(prompts={prompt_labels})"
        )
    if process.exitcode not in (0, None):
        raise RuntimeError(f"PaddleOCR-VL official pipeline worker exited with code {process.exitcode}")
    if queue.empty():
        raise RuntimeError("PaddleOCR-VL official pipeline worker returned no result")
    message = queue.get()
    if not message.get("ok"):
        raise RuntimeError(str(message.get("error") or "PaddleOCR-VL official pipeline failed"))
    payloads = message.get("payloads") or []
    return [payload for payload in payloads if isinstance(payload, dict)]


def _paddle_pipeline_worker(
    queue: Any,
    model_dir: str,
    image_path: str,
    device: str,
    prompt_labels: list[str],
    max_new_tokens: int,
) -> None:
    try:
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            vl_rec_model_dir=model_dir,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=False,
            format_block_content=False,
            merge_layout_blocks=False,
            use_queues=False,
            device=device,
        )
        payloads: list[dict[str, Any]] = []
        for prompt_label in prompt_labels:
            results = pipeline.predict(
                image_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                use_ocr_for_image_block=False,
                layout_shape_mode="rect",
                prompt_label=prompt_label,
                min_pixels=28 * 28 * 130,
                max_pixels=2048 * 28 * 28 if prompt_label == "spotting" else 1280 * 28 * 28,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                use_queues=False,
            )
            for result in results:
                payloads.append(_safe_json(_paddle_result_to_dict(result)))
        queue.put({"ok": True, "payloads": payloads})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ProviderNotAvailableError("Qwen3.5 text-compliance endpoint")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _qwen_blocks(data: dict[str, Any], width: int, height: int) -> list[OCRTextBlock]:
    value = data.get("text_blocks") or data.get("blocks") or data.get("ocr") or data.get("lines") or []
    if isinstance(value, dict):
        value = _find_items(value)
    if not isinstance(value, list):
        return []
    blocks = [_item_to_block(item, width, height) for item in value]
    return [block for block in blocks if block is not None]


def _qwen_region_review_required(blocks: list[OCRTextBlock], width: int, height: int) -> bool:
    if not blocks:
        return True
    image_area = max(float(width * height), 1.0)
    if len(blocks) == 1:
        bbox = blocks[0].bbox
        if bbox.w * bbox.h >= image_area * 0.5:
            return True
    low_confidence_count = sum(1 for block in blocks if float(block.confidence) < 0.65)
    return low_confidence_count > max(len(blocks) // 2, 0)


def _official_prompt_labels(requested_task: str) -> list[str]:
    allowed = {"ocr", "formula", "table", "chart", "spotting", "seal"}
    ordered = ["ocr", requested_task]
    result: list[str] = []
    for task in ordered:
        task = task.strip().lower()
        if task in allowed and task not in result:
            result.append(task)
    return result or ["ocr"]


def _paddle_result_to_dict(result: Any) -> dict[str, Any]:
    payload = None
    json_value = getattr(result, "json", None)
    if isinstance(json_value, dict):
        payload = json_value.get("res", json_value)
    elif callable(json_value):
        try:
            value = json_value()
            payload = value.get("res", value) if isinstance(value, dict) else value
        except Exception:
            payload = None
    if payload is None and isinstance(result, dict):
        payload = result.get("res", result)
    if payload is None:
        try:
            payload = dict(result)
        except Exception:
            payload = {"raw": str(result)}
    return payload if isinstance(payload, dict) else {"raw": payload}


def _parse_paddle_result_blocks(payload: dict[str, Any], width: int, height: int) -> list[OCRTextBlock]:
    blocks: list[OCRTextBlock] = []
    for item in list(payload.get("parsing_res_list") or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("block_content") or "").strip()
        valid, _ = _validate_ocr_text(text)
        if not valid:
            continue
        bbox = _bbox_from_any(item.get("block_bbox"), width, height)
        if bbox is None:
            bbox = BBox(x=0, y=0, w=float(width), h=float(height))
        blocks.append(
            OCRTextBlock(
                text=text,
                bbox=bbox,
                polygon=_polygon_from_any(item.get("block_bbox")),
                confidence=0.82,
                metadata={"char_polys": item.get("char_polys") or item.get("char_boxes") or item.get("rec_char_polys") or []},
            )
        )

    spotting = payload.get("spotting_res")
    if isinstance(spotting, dict):
        texts = spotting.get("rec_texts") or []
        polygons = spotting.get("rec_polys") or spotting.get("rec_boxes") or []
        char_polys = spotting.get("char_polys") or spotting.get("char_boxes") or spotting.get("rec_char_polys") or []
        for idx, value in enumerate(texts):
            text = value[0] if isinstance(value, (list, tuple)) and value else value
            text = str(text or "").strip()
            valid, _ = _validate_ocr_text(text)
            if not valid:
                continue
            bbox_value = polygons[idx] if idx < len(polygons) else None
            bbox = _bbox_from_any(bbox_value, width, height)
            if bbox is None:
                bbox = BBox(x=0, y=0, w=float(width), h=float(height))
            blocks.append(
                OCRTextBlock(
                    text=text,
                    bbox=bbox,
                    polygon=_polygon_from_any(bbox_value),
                    confidence=0.82,
                    metadata={"char_polys": char_polys[idx] if idx < len(char_polys) else []},
                )
            )
    return blocks


def _polygon_from_any(value: Any) -> Polygon | None:
    if isinstance(value, dict):
        for key in ("points", "polygon", "poly", "bbox", "box"):
            polygon = _polygon_from_any(value.get(key))
            if polygon is not None:
                return polygon
        return None
    if not isinstance(value, list) or not value:
        return None
    points: list[tuple[float, float]] = []
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = [float(item) for item in value]
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        for point in value:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
    return Polygon(points=points) if len(points) >= 3 else None


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _safe_json(item)
            for key, item in value.items()
            if key not in {"image", "img", "output_img"}
        }
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _best_plain_text_candidate(candidates: Any) -> str:
    best_text = ""
    best_score = -1
    for candidate in candidates:
        cleaned = _strip_markup(str(candidate or ""))
        valid, _ = _validate_ocr_text(cleaned)
        if not valid:
            continue
        score = _ocr_text_quality_score(cleaned)
        if score > best_score:
            best_text = cleaned
            best_score = score
    return best_text


def _dedupe_blocks(blocks: list[OCRTextBlock]) -> list[OCRTextBlock]:
    deduped: list[OCRTextBlock] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for block in blocks:
        text = _normalize_ocr_text_for_validation(block.text)
        if not text:
            continue
        key = (
            text,
            int(round(block.bbox.x)),
            int(round(block.bbox.y)),
            int(round(block.bbox.w)),
            int(round(block.bbox.h)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)
    return deduped


def _validate_ocr_text(text: str) -> tuple[bool, str]:
    cleaned = _normalize_ocr_text_for_validation(text)
    if not cleaned:
        return False, "empty_text"

    lowered = cleaned.lower()
    if lowered in {"ocr", "[ocr]", "spotting", "[spotting]", "text", "none", "null"}:
        return False, "control_token_only"
    if any(phrase in lowered for phrase in _NO_TEXT_PHRASES):
        return False, "model_reported_no_text"
    if any(phrase in lowered for phrase in _KNOWN_LM_PLACEHOLDER_PHRASES):
        return False, "language_model_placeholder"
    if _looks_like_formula_hallucination(text, cleaned):
        return False, "formula_hallucination"

    content_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", cleaned)
    if len(content_chars) < 2:
        return False, "too_few_content_characters"
    if _looks_like_repetitive_numeric_hallucination(cleaned):
        return False, "repetitive_numeric_hallucination"
    if _looks_like_low_diversity_hallucination(cleaned, content_chars):
        return False, "low_diversity_hallucination"

    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff_.@:/#-]{1,}", cleaned)
    if len(tokens) >= 8:
        counts = Counter(token.lower() for token in tokens)
        most_common = counts.most_common(1)[0][1]
        if most_common / len(tokens) >= 0.55:
            return False, "repeated_token_hallucination"
        if len(counts) <= 2:
            return False, "low_diversity_repeated_text"

    return True, ""


def _normalize_ocr_text_for_validation(text: str) -> str:
    cleaned = _strip_markup(str(text or ""))
    cleaned = re.sub(r"^\s*\[[a-z_ -]{2,30}\]\s*", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _ocr_text_quality_score(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff_.@:/#-]{1,}", text)
    digits = len(re.findall(r"\d", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    alpha = len(re.findall(r"[A-Za-z]", text))
    line_like = len(re.findall(r"[,，:：#/@.-]", text))
    return len(tokens) * 3 + digits + cjk + alpha + line_like


def _looks_like_formula_hallucination(raw_text: str, cleaned: str) -> bool:
    raw_lower = str(raw_text or "").lower()
    cleaned_lower = cleaned.lower()
    compact = re.sub(r"[^a-z0-9]", "", raw_lower + cleaned_lower)
    compact_without_latex_words = compact.replace("text", "")
    if compact.count("c6h12o6") >= 2 or compact_without_latex_words.count("c6h12o6") >= 2:
        return True
    formula_hits = len(re.findall(r"\bc\s*6\b|\bh\s*12\b|\bo\s*6\b", cleaned_lower))
    unique_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", cleaned)
    }
    if formula_hits >= 6 and len(unique_tokens) <= 3:
        return True
    return False


def _looks_like_repetitive_numeric_hallucination(cleaned: str) -> bool:
    numeric_runs = re.findall(r"\d[\d.]{40,}", cleaned)
    for run in numeric_runs:
        digits = re.sub(r"\D", "", run)
        if len(digits) >= 40:
            counts = Counter(digits)
            if counts.most_common(1)[0][1] / len(digits) >= 0.82:
                return True
    return False


def _looks_like_low_diversity_hallucination(cleaned: str, content_chars: list[str]) -> bool:
    if len(content_chars) < 80:
        return False
    counts = Counter(char.lower() for char in content_chars)
    if counts.most_common(1)[0][1] / len(content_chars) >= 0.72:
        return True
    unique_words = {
        token.lower()
        for token in re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", cleaned)
    }
    digits = re.findall(r"\d", cleaned)
    if len(digits) >= 80 and len(unique_words) <= 8:
        digit_counts = Counter(digits)
        if digit_counts.most_common(1)[0][1] / len(digits) >= 0.70:
            return True
    return False


_NO_TEXT_PHRASES = {
    "no text",
    "no visible text",
    "unable to recognize",
    "cannot recognize",
    "can't recognize",
    "too blurry",
    "not readable",
    "unreadable",
    "无法识别",
    "没有文字",
    "未检测到文字",
    "无可识别文字",
}

_KNOWN_LM_PLACEHOLDER_PHRASES = {
    "the quick brown fox",
    "lorem ipsum",
    "sample text",
    "example text",
    "placeholder text",
}
