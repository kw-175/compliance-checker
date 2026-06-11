"""PaddleOCR-VL 1.5 provider backed by the official PaddleX Serving API."""
from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from picture.domain.exceptions import ProviderNotAvailableError
from picture.domain.models import BBox, LayoutRegion, OCRLayoutResult, OCRTextBlock, Polygon
from picture.providers.base import OCRLayoutProvider
from picture.providers.ocr.paddleocr_vl import _validate_ocr_text

logger = logging.getLogger(__name__)


class PaddleOCRVLAPIProvider(OCRLayoutProvider):
    """Call PaddleX Serving's PaddleOCR-VL ``/layout-parsing`` endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8217",
        timeout_seconds: float = 300.0,
        file_type: int = 1,
        use_layout_detection: bool = True,
        use_chart_recognition: bool = True,
        use_seal_recognition: bool = True,
        prettify_markdown: bool = True,
        visualize: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._file_type = file_type
        self._use_layout_detection = use_layout_detection
        self._use_chart_recognition = use_chart_recognition
        self._use_seal_recognition = use_seal_recognition
        self._prettify_markdown = prettify_markdown
        self._visualize = visualize

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-1.5(api)"

    def warmup(self) -> dict[str, Any]:
        """Check that the official PaddleOCR-VL serving endpoint is reachable."""
        try:
            with httpx.Client(timeout=min(self._timeout_seconds, 10.0)) as client:
                response = client.get(f"{self._base_url}/docs")
            return {"reachable": response.status_code < 500, "status_code": response.status_code}
        except Exception as exc:
            raise ProviderNotAvailableError(f"PaddleOCR-VL API at {self._base_url}") from exc

    def analyze(self, image_path: str) -> OCRLayoutResult:
        started = time.monotonic()
        image_file = Path(image_path)
        if not image_file.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        payload = {
            "file": base64.b64encode(image_file.read_bytes()).decode("ascii"),
            "fileType": self._file_type,
            "useLayoutDetection": self._use_layout_detection,
            "useChartRecognition": self._use_chart_recognition,
            "useSealRecognition": self._use_seal_recognition,
            "prettifyMarkdown": self._prettify_markdown,
            "visualize": self._visualize,
        }

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(f"{self._base_url}/layout-parsing", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise ProviderNotAvailableError(f"PaddleOCR-VL API at {self._base_url}") from exc

        error_code = data.get("errorCode")
        if error_code not in (None, 0):
            raise RuntimeError(
                f"PaddleOCR-VL API returned errorCode={error_code}: {data.get('errorMsg') or data.get('message')}"
            )

        spotting_instances: list[dict[str, Any]] = []
        spotting_error = ""
        try:
            spotting_payload = {
                **payload,
                "useLayoutDetection": False,
                "promptLabel": "spotting",
                "layoutShapeMode": "quad",
            }
            with httpx.Client(timeout=self._timeout_seconds) as client:
                spotting_response = client.post(f"{self._base_url}/layout-parsing", json=spotting_payload)
            spotting_response.raise_for_status()
            spotting_data = spotting_response.json()
            spotting_error_code = spotting_data.get("errorCode")
            if spotting_error_code not in (None, 0):
                spotting_error = str(spotting_data.get("errorMsg") or spotting_data.get("message") or spotting_error_code)
            else:
                spotting_instances = _extract_text_instances_from_response(spotting_data)
        except Exception as exc:
            spotting_error = f"{type(exc).__name__}: {exc}"

        result = _parse_layout_parsing_response(data)
        result.engine_name = self.name
        result.metadata = {
            **dict(result.metadata or {}),
            "backend": "paddlex_serving",
            "endpoint": f"{self._base_url}/layout-parsing",
            "model": "PaddleOCR-VL-1.5",
            "stages": ["layout_analysis", "vlm_recognition", "reading_order_merge"],
            "spotting_enabled": True,
            "spotting_endpoint": f"{self._base_url}/layout-parsing",
            "spotting_prompt_label": "spotting",
            "spotting_use_layout_detection": False,
            "spotting_layout_shape_mode": "quad",
            "spotting_text_instance_count": len(spotting_instances),
            "spotting_error": spotting_error,
            "text_instances": spotting_instances,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
        return result


def _parse_layout_parsing_response(data: dict[str, Any]) -> OCRLayoutResult:
    service_result = data.get("result") if isinstance(data.get("result"), dict) else {}
    pages = service_result.get("layoutParsingResults") or []
    if isinstance(pages, dict):
        pages = [pages]

    full_text_parts: list[str] = []
    placeholder_text_count = 0
    text_blocks: list[OCRTextBlock] = []
    layout_regions: list[LayoutRegion] = []
    raw_region_count = 0

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        markdown = page.get("markdown")
        if isinstance(markdown, dict):
            text = str(markdown.get("text") or "").strip()
            if text and not _is_image_placeholder_text(text):
                full_text_parts.append(text)
            elif text:
                placeholder_text_count += 1

        pruned = page.get("prunedResult")
        page_blocks, page_regions, count = _parse_pruned_result(pruned, page_index)
        text_blocks.extend(page_blocks)
        layout_regions.extend(page_regions)
        raw_region_count += count
    text_blocks = _dedupe_text_blocks(text_blocks)

    full_text = "\n\n".join(part for part in full_text_parts if part).strip()
    if not full_text and text_blocks:
        full_text = "\n".join(block.text for block in text_blocks if block.text).strip()
    valid_text, invalid_reason = _validate_ocr_text(full_text)
    if not text_blocks and placeholder_text_count > 0 and not full_text:
        valid_text = False
        invalid_reason = "image_placeholder_only"
    spatially_mappable_text = bool(text_blocks)

    return OCRLayoutResult(
        full_text=full_text,
        text_blocks=text_blocks,
        layout_regions=layout_regions,
        metadata={
            "page_count": len(pages),
            "raw_layout_region_count": raw_region_count,
            "valid_text": valid_text,
            "invalid_reason": invalid_reason,
            "spatially_mappable_text": spatially_mappable_text,
            "placeholder_text_count": placeholder_text_count,
            "effective_text_preview": full_text[:300],
        },
    )


def _is_image_placeholder_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    has_image_marker = "<img" in lowered or re.search(r"!\[[^\]]*\]\([^)]+\)", value) is not None
    if not has_image_marker:
        return False
    without_images = re.sub(r"<img\b[^>]*>", " ", value, flags=re.I)
    without_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", without_images)
    without_tags = re.sub(r"<[^>]+>", " ", without_images)
    without_paths = re.sub(r"\bimgs?/[-\w./]+(?:png|jpe?g|webp|bmp)\b", " ", without_tags, flags=re.I)
    content_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", without_paths)
    return len(content_chars) < 2


def _dedupe_text_blocks(blocks: list[OCRTextBlock]) -> list[OCRTextBlock]:
    deduped: list[OCRTextBlock] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for block in blocks:
        bbox = block.bbox
        key = (
            _first_chars(block.text, 80),
            round(float(bbox.x)),
            round(float(bbox.y)),
            round(float(bbox.w)),
            round(float(bbox.h)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)
    return deduped


def _first_chars(value: str, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _parse_pruned_result(pruned: Any, page_index: int) -> tuple[list[OCRTextBlock], list[LayoutRegion], int]:
    if not isinstance(pruned, dict):
        return [], [], 0

    candidates = []
    for key in ("parsing_res_list", "parsingResults", "blocks", "layoutBlocks"):
        value = pruned.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))

    layout_det = pruned.get("layout_det_res")
    if isinstance(layout_det, dict) and isinstance(layout_det.get("boxes"), list):
        candidates.extend(item for item in layout_det["boxes"] if isinstance(item, dict))

    text_blocks: list[OCRTextBlock] = []
    regions: list[LayoutRegion] = []
    for order, item in enumerate(candidates):
        text = _first_text(item, ("block_content", "text", "content", "rec_text", "html", "markdown"))
        bbox = _extract_bbox(item)
        if bbox is None:
            continue
        polygon = _extract_polygon(item)
        confidence = _first_float(item, ("score", "confidence", "rec_score"))
        region_type = _first_text(item, ("block_label", "label", "type", "region_type")) or "text"
        block = None
        if text and not _is_image_placeholder_text(text):
            block = OCRTextBlock(
                text=text,
                bbox=bbox,
                polygon=polygon,
                confidence=confidence,
                metadata={
                    "page_index": page_index,
                    "reading_order": order,
                    "region_type": str(region_type),
                    "char_polys": item.get("char_polys") or item.get("char_boxes") or item.get("rec_char_polys") or [],
                },
            )
            text_blocks.append(block)
        regions.append(
            LayoutRegion(
                region_type=str(region_type),
                bbox=bbox,
                text_blocks=[block] if block is not None else [],
            )
        )
        item.setdefault("_picture_provider_metadata", {})
        item["_picture_provider_metadata"].update({"page_index": page_index, "reading_order": order})
    spotting = pruned.get("spotting_res")
    if isinstance(spotting, dict):
        text_blocks.extend(_parse_spotting_blocks(spotting, page_index, len(text_blocks)))
    return text_blocks, regions, len(candidates)


def _parse_spotting_blocks(spotting: dict[str, Any], page_index: int, order_offset: int) -> list[OCRTextBlock]:
    texts = spotting.get("rec_texts") or []
    polygons = spotting.get("rec_polys") or spotting.get("rec_boxes") or []
    scores = spotting.get("rec_scores") or spotting.get("scores") or []
    char_polys = spotting.get("char_polys") or spotting.get("char_boxes") or spotting.get("rec_char_polys") or []
    blocks: list[OCRTextBlock] = []
    if not isinstance(texts, list):
        return blocks
    for index, value in enumerate(texts):
        text = value[0] if isinstance(value, (list, tuple)) and value else value
        text = str(text or "").strip()
        if not text:
            continue
        polygon_value = polygons[index] if isinstance(polygons, list) and index < len(polygons) else None
        bbox = _bbox_from_value(polygon_value)
        if bbox is None:
            continue
        blocks.append(
            OCRTextBlock(
                text=text,
                bbox=bbox,
                polygon=_polygon_from_value(polygon_value),
                confidence=_score_at(scores, index, default=0.82),
                metadata={
                    "page_index": page_index,
                    "reading_order": order_offset + index,
                    "region_type": "text_spotting",
                    "char_polys": char_polys[index] if isinstance(char_polys, list) and index < len(char_polys) else [],
                },
            )
        )
    return blocks


def _extract_text_instances_from_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    service_result = data.get("result") if isinstance(data.get("result"), dict) else {}
    pages = service_result.get("layoutParsingResults") or []
    if isinstance(pages, dict):
        pages = [pages]
    instances: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        pruned = page.get("prunedResult")
        if not isinstance(pruned, dict):
            continue
        instances.extend(_extract_text_instances_from_pruned(pruned, page_index, len(instances)))
    return instances


def _extract_text_instances_from_pruned(pruned: dict[str, Any], page_index: int, order_offset: int) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    spotting = pruned.get("spotting_res")
    if isinstance(spotting, dict):
        instances.extend(_spotting_instances(spotting, page_index, order_offset))
    if instances:
        return instances
    candidates: list[dict[str, Any]] = []
    for key in ("parsing_res_list", "parsingResults", "blocks", "layoutBlocks", "lines"):
        value = pruned.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    for order, item in enumerate(candidates, start=order_offset):
        text = _first_text(item, ("text", "rec_text", "block_content", "content", "html", "markdown"))
        bbox = _extract_bbox(item)
        if not text or bbox is None or _is_image_placeholder_text(text):
            continue
        polygon = _extract_polygon(item)
        instances.append(
            {
                "unit_id": f"spot_{page_index:04d}_{order:06d}",
                "text": text,
                "bbox": bbox.model_dump(mode="json"),
                "polygon": polygon.model_dump(mode="json") if polygon is not None else None,
                "confidence": _first_float(item, ("score", "confidence", "rec_score")),
                "source": "paddleocr_spotting_layout_item",
                "quality": "high" if polygon is not None else "medium",
                "page_index": page_index,
                "reading_order": order,
                "char_polys": item.get("char_polys") or item.get("char_boxes") or item.get("rec_char_polys") or [],
            }
        )
    return instances


def _spotting_instances(spotting: dict[str, Any], page_index: int, order_offset: int) -> list[dict[str, Any]]:
    texts = spotting.get("rec_texts") or []
    polygons = spotting.get("rec_polys") or spotting.get("rec_boxes") or []
    scores = spotting.get("rec_scores") or spotting.get("scores") or []
    char_polys = spotting.get("char_polys") or spotting.get("char_boxes") or spotting.get("rec_char_polys") or []
    instances: list[dict[str, Any]] = []
    if not isinstance(texts, list):
        return instances
    for index, value in enumerate(texts):
        text = value[0] if isinstance(value, (list, tuple)) and value else value
        text = str(text or "").strip()
        if not text:
            continue
        polygon_value = polygons[index] if isinstance(polygons, list) and index < len(polygons) else None
        bbox = _bbox_from_value(polygon_value)
        if bbox is None:
            continue
        polygon = _polygon_from_value(polygon_value)
        order = order_offset + index
        instances.append(
            {
                "unit_id": f"spot_{page_index:04d}_{order:06d}",
                "text": text,
                "bbox": bbox.model_dump(mode="json"),
                "polygon": polygon.model_dump(mode="json") if polygon is not None else None,
                "confidence": _score_at(scores, index, default=0.82),
                "source": "paddleocr_spotting_rec_poly",
                "quality": "high",
                "page_index": page_index,
                "reading_order": order,
                "char_polys": char_polys[index] if isinstance(char_polys, list) and index < len(char_polys) else [],
            }
        )
    return instances


def _score_at(values: Any, index: int, *, default: float) -> float:
    if isinstance(values, list) and index < len(values):
        try:
            return float(values[index])
        except (TypeError, ValueError):
            return default
    return default


def _first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_float(item: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = item.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _extract_bbox(item: dict[str, Any]) -> BBox | None:
    for key in ("block_bbox", "bbox", "box", "coordinate", "poly", "dt_polys"):
        bbox = _bbox_from_value(item.get(key))
        if bbox is not None:
            return bbox
    return None


def _extract_polygon(item: dict[str, Any]) -> Polygon | None:
    for key in ("block_bbox", "bbox", "box", "coordinate", "poly", "dt_polys", "points"):
        polygon = _polygon_from_value(item.get(key))
        if polygon is not None:
            return polygon
    return None


def _bbox_from_value(value: Any) -> BBox | None:
    if isinstance(value, dict):
        if {"x", "y", "w", "h"}.issubset(value):
            try:
                return BBox(x=float(value["x"]), y=float(value["y"]), w=float(value["w"]), h=float(value["h"]))
            except (TypeError, ValueError):
                return None
        for key in ("bbox", "box", "points"):
            bbox = _bbox_from_value(value.get(key))
            if bbox is not None:
                return bbox
        return None
    if not isinstance(value, list) or not value:
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = [float(item) for item in value]
        return BBox(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))
    points: list[tuple[float, float]] = []
    for point in value:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))
    return None


def _polygon_from_value(value: Any) -> Polygon | None:
    if isinstance(value, dict):
        for key in ("points", "polygon", "poly", "bbox", "box"):
            polygon = _polygon_from_value(value.get(key))
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
