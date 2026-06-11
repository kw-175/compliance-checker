from __future__ import annotations

import os
import logging
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="SAM3 API", version="0.1.0")
logger = logging.getLogger("sam3_api")

SAM3_MODEL_DIR = Path(os.getenv("SAM3_MODEL_DIR", "/data/kw/compliance-checker/models/facebook/sam3"))
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cuda")
SAM3_CONFIDENCE = float(os.getenv("SAM3_CONFIDENCE", "0.35"))
SAM3_USE_BF16_AUTOCAST = os.getenv("SAM3_USE_BF16_AUTOCAST", "1").strip().lower() not in {"0", "false", "no"}
SAM3_VIDEO_COMPILE = os.getenv("SAM3_VIDEO_COMPILE", "0").strip().lower() in {"1", "true", "yes"}
SAM3_VIDEO_OFFLOAD_TO_CPU = os.getenv("SAM3_VIDEO_OFFLOAD_TO_CPU", "0").strip().lower() in {"1", "true", "yes"}

_runtime: dict[str, Any] | None = None
_runtime_lock = threading.Lock()
_video_runtime: dict[str, Any] | None = None
_video_runtime_lock = threading.Lock()


class BBoxModel(BaseModel):
    x: float
    y: float
    w: float
    h: float


class PromptModel(BaseModel):
    category: str
    text: str
    threshold: float | None = None


class PointModel(BaseModel):
    x: float
    y: float
    label: bool = True


class PointPromptModel(BaseModel):
    category: str
    point: PointModel
    text: str = "visual"
    threshold: float | None = None
    box_size_ratio: float = Field(default=0.06, ge=0.001, le=1.0)
    box_size_pixels: float | None = Field(default=None, gt=0.0)


class DetectRequest(BaseModel):
    image_path: str
    prompts: list[PromptModel] = Field(default_factory=list)
    return_masks: bool = False
    return_polygons: bool = False


class PointDetectRequest(BaseModel):
    image_path: str
    point_prompts: list[PointPromptModel] = Field(default_factory=list)
    return_masks: bool = False
    return_polygons: bool = False


class RegionModel(BaseModel):
    bbox: BBoxModel
    confidence: float = 0.0
    frame_id: str = ""
    pts_ms: int | None = None
    mask_path: str = ""
    polygon: list[list[float]] | None = None


class RefineRequest(BaseModel):
    image_path: str
    regions: list[RegionModel] = Field(default_factory=list)
    threshold: float | None = None


class VideoFrameModel(BaseModel):
    frame_id: str
    frame_index: int = 0
    pts_ms: int = 0
    duration_ms: int = 0
    image_path: str = ""
    image_uri: str = ""


class VideoTrackSeedModel(BaseModel):
    risk_id: str
    track_id: str = ""
    category: str = ""
    target_type: str = ""
    confidence: float = 0.0
    seed_regions: list[RegionModel] = Field(default_factory=list)
    prompt: dict[str, Any] = Field(default_factory=dict)


class VideoTrackRequest(BaseModel):
    frames: list[VideoFrameModel] = Field(default_factory=list)
    tracks: list[VideoTrackSeedModel] = Field(default_factory=list)
    return_masks: bool = True
    return_polygons: bool = True
    output_prob_thresh: float | None = None
    max_frame_num_to_track: int | None = None
    offload_video_to_cpu: bool | None = None
    offload_state_to_cpu: bool = False


def _resolve_checkpoint() -> Path | None:
    for name in ("sam3.pt", "model.safetensors"):
        path = SAM3_MODEL_DIR / name
        if path.is_file():
            return path
    return None


def _get_runtime() -> dict[str, Any]:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        checkpoint = _resolve_checkpoint()
        if checkpoint is None:
            raise RuntimeError(f"SAM3 checkpoint not found under {SAM3_MODEL_DIR}")
        model = build_sam3_image_model(
            device=SAM3_DEVICE,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
        )
        processor = Sam3Processor(model, device=SAM3_DEVICE, confidence_threshold=SAM3_CONFIDENCE)
        _runtime = {
            "model": model,
            "processor": processor,
            "device": SAM3_DEVICE,
            "checkpoint_path": str(checkpoint),
        }
        return _runtime


def _get_video_runtime() -> dict[str, Any]:
    global _video_runtime
    if _video_runtime is not None:
        return _video_runtime
    with _video_runtime_lock:
        if _video_runtime is not None:
            return _video_runtime
        from sam3.model_builder import build_sam3_video_model

        checkpoint = _resolve_checkpoint()
        if checkpoint is None:
            raise RuntimeError(f"SAM3 checkpoint not found under {SAM3_MODEL_DIR}")
        model = build_sam3_video_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=SAM3_DEVICE,
            compile=SAM3_VIDEO_COMPILE,
        ).eval()
        _video_runtime = {
            "model": model,
            "device": SAM3_DEVICE,
            "checkpoint_path": str(checkpoint),
        }
        return _video_runtime


def _load_image(image_path: str) -> Any:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    from PIL import Image

    return Image.open(path).convert("RGB")


def _inference_context() -> Any:
    if SAM3_USE_BF16_AUTOCAST and str(SAM3_DEVICE).startswith("cuda"):
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _extract_boxes_scores(output: dict[str, Any]) -> tuple[list[Any], list[float]]:
    boxes = _to_list(output.get("boxes"))
    scores = _to_list(output.get("scores"))
    if not boxes and output.get("pred_boxes") is not None:
        boxes = _to_list(output.get("pred_boxes"))
    if not scores and output.get("pred_scores") is not None:
        scores = _to_list(output.get("pred_scores"))
    normalized_scores = []
    for score in scores:
        try:
            normalized_scores.append(float(score))
        except (TypeError, ValueError):
            normalized_scores.append(0.0)
    return boxes, normalized_scores


def _extract_masks(output: dict[str, Any]) -> list[Any]:
    masks = output.get("masks")
    if masks is None:
        masks = output.get("pred_masks")
    if masks is None:
        return []
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu()
    if hasattr(masks, "numpy"):
        masks = masks.numpy()
    try:
        import numpy as np

        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        return [masks[index] for index in range(masks.shape[0])]
    except Exception:
        return []


def _bbox_from_box(box: Any) -> BBoxModel | None:
    values = _to_list(box)
    if len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in values[:4]]
    except (TypeError, ValueError):
        return None
    return BBoxModel(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))


def _bbox_from_mask(mask: Any) -> BBoxModel | None:
    try:
        import numpy as np

        values = np.asarray(mask) > 0
        ys, xs = np.where(values)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        return BBoxModel(x=x1, y=y1, w=max(1.0, x2 - x1), h=max(1.0, y2 - y1))
    except Exception:
        return None


def _save_mask(mask: Any, image_path: str, prefix: str) -> str | None:
    try:
        import numpy as np
        from PIL import Image

        values = (np.asarray(mask) > 0).astype("uint8") * 255
        mask_dir = Path(image_path).parent / "sam3_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{prefix}_{uuid.uuid4().hex[:12]}.png"
        Image.fromarray(values, mode="L").save(mask_path)
        return str(mask_path)
    except Exception:
        logger.exception("Failed to save SAM3 mask for image_path=%s", image_path)
        return None


def _mask_polygons(mask: Any, max_points: int = 80, max_contours: int = 8) -> list[list[list[float]]] | None:
    try:
        import cv2
        import numpy as np

        values = (np.asarray(mask) > 0).astype("uint8") * 255
        contours, _ = cv2.findContours(values, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        polygons: list[list[list[float]]] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]:
            if cv2.contourArea(contour) <= 0:
                continue
            epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(approx) > max_points:
                step = max(1, len(approx) // max_points)
                approx = approx[::step][:max_points]
            points = [[float(x), float(y)] for x, y in approx]
            if points:
                polygons.append(points)
        return polygons or None
    except Exception:
        return None


def _mask_polygon(mask: Any, max_points: int = 80) -> list[list[float]] | None:
    polygons = _mask_polygons(mask, max_points=max_points, max_contours=1)
    return polygons[0] if polygons else None


def _mask_stats(mask: Any, bbox: BBoxModel | None = None) -> dict[str, float | int]:
    try:
        import numpy as np

        values = np.asarray(mask) > 0
        height, width = values.shape[:2]
        mask_area = int(values.sum())
        image_area = max(1, int(width) * int(height))
        stats: dict[str, float | int] = {
            "mask_area": mask_area,
            "mask_area_ratio": round(mask_area / image_area, 6),
        }
        if bbox is not None:
            bbox_area = max(1.0, float(bbox.w) * float(bbox.h))
            stats["mask_bbox_fill_ratio"] = round(mask_area / bbox_area, 6)
        return stats
    except Exception:
        return {}


def _mask_payload(
    mask: Any,
    image_path: str,
    prefix: str,
    *,
    return_masks: bool,
    return_polygons: bool,
    bbox: BBoxModel | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mask_path": None,
        "mask_rle": None,
        "polygon": None,
        "polygons": None,
    }
    if mask is None:
        return payload
    if return_masks:
        payload["mask_path"] = _save_mask(mask, image_path, prefix)
    if return_polygons:
        polygons = _mask_polygons(mask)
        payload["polygons"] = polygons
        payload["polygon"] = polygons[0] if polygons else None
    payload.update(_mask_stats(mask, bbox))
    return payload


def _normalized_cxcywh_box(bbox: BBoxModel, width: float, height: float) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height must be positive for SAM3 geometric prompts")
    cx = (bbox.x + bbox.w / 2.0) / width
    cy = (bbox.y + bbox.h / 2.0) / height
    w = bbox.w / width
    h = bbox.h / height
    return [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    ]


def _normalized_xywh_box(bbox: BBoxModel, width: float, height: float) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height must be positive for SAM3 video prompts")
    if 0.0 <= bbox.x <= 1.0 and 0.0 <= bbox.y <= 1.0 and 0.0 <= bbox.w <= 1.0 and 0.0 <= bbox.h <= 1.0:
        return [
            max(0.0, min(1.0, bbox.x)),
            max(0.0, min(1.0, bbox.y)),
            max(0.0, min(1.0, bbox.w)),
            max(0.0, min(1.0, bbox.h)),
        ]
    return [
        max(0.0, min(1.0, bbox.x / width)),
        max(0.0, min(1.0, bbox.y / height)),
        max(0.0, min(1.0, bbox.w / width)),
        max(0.0, min(1.0, bbox.h / height)),
    ]


def _absolute_bbox_from_normalized(box: Any, width: float, height: float) -> BBoxModel | None:
    values = _to_list(box)
    if len(values) < 4:
        return None
    try:
        x, y, w, h = [float(item) for item in values[:4]]
    except (TypeError, ValueError):
        return None
    if 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5 and 0.0 <= w <= 1.5 and 0.0 <= h <= 1.5:
        x *= width
        y *= height
        w *= width
        h *= height
    return BBoxModel(x=max(0.0, x), y=max(0.0, y), w=max(1.0, w), h=max(1.0, h))


def _video_frame_path(frame: VideoFrameModel) -> str:
    return frame.image_path or frame.image_uri


def _load_video_images(frames: list[VideoFrameModel]) -> list[Any]:
    from PIL import Image

    images = []
    for frame in frames:
        path = Path(_video_frame_path(frame))
        if not path.is_file():
            raise FileNotFoundError(f"Video frame does not exist: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def _seed_frame_position(track: VideoTrackSeedModel, frame_position_by_id: dict[str, int]) -> int:
    for region in track.seed_regions:
        if region.frame_id and region.frame_id in frame_position_by_id:
            return int(frame_position_by_id[region.frame_id])
    return 0


def _seed_region(track: VideoTrackSeedModel, frame_by_id: dict[str, VideoFrameModel]) -> RegionModel | None:
    if not track.seed_regions:
        return None
    valid = [region for region in track.seed_regions if region.frame_id and region.frame_id in frame_by_id]
    candidates = valid or list(track.seed_regions)
    return max(candidates, key=lambda region: float(region.confidence or 0.0))


def _video_masks(outputs: dict[str, Any]) -> list[Any]:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return []
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu()
    if hasattr(masks, "numpy"):
        masks = masks.numpy()
    try:
        import numpy as np

        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        return [masks[index] for index in range(masks.shape[0])]
    except Exception:
        return []


def _video_output_points(
    outputs: dict[str, Any],
    frame: VideoFrameModel,
    *,
    image_path: str,
    return_masks: bool,
    return_polygons: bool,
) -> list[dict[str, Any]]:
    boxes = _to_list(outputs.get("out_boxes_xywh"))
    obj_ids = _to_list(outputs.get("out_obj_ids"))
    scores = _to_list(outputs.get("out_probs")) or _to_list(outputs.get("out_tracker_probs"))
    masks = _video_masks(outputs)
    points: list[dict[str, Any]] = []
    from PIL import Image

    width, height = Image.open(image_path).size
    for index, box in enumerate(boxes):
        bbox = _absolute_bbox_from_normalized(box, width, height)
        if bbox is None:
            continue
        score = 0.0
        if index < len(scores):
            try:
                score = float(scores[index])
            except (TypeError, ValueError):
                score = 0.0
        mask = masks[index] if index < len(masks) else None
        mask_payload = _mask_payload(
            mask,
            image_path,
            "video_track",
            return_masks=return_masks,
            return_polygons=return_polygons,
            bbox=bbox,
        )
        points.append({
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "pts_ms": frame.pts_ms,
            "duration_ms": frame.duration_ms,
            "obj_id": int(obj_ids[index]) if index < len(obj_ids) else index,
            "bbox": bbox.model_dump(),
            "confidence": score,
            **mask_payload,
        })
    return points


def _merge_video_points(existing: dict[tuple[str, int], dict[str, Any]], points: list[dict[str, Any]]) -> None:
    for point in points:
        key = (str(point.get("frame_id") or ""), int(point.get("obj_id", 0) or 0))
        if key not in existing or float(point.get("confidence", 0.0) or 0.0) > float(existing[key].get("confidence", 0.0) or 0.0):
            existing[key] = point


def _select_seed_obj_id(points: list[dict[str, Any]], seed: RegionModel) -> int | None:
    if not points:
        return None
    seed_box = seed.bbox.model_dump()
    best = max(points, key=lambda point: _candidate_score(seed_box, point))
    if _candidate_score(seed_box, best) < 0.08:
        return None
    obj_id = best.get("obj_id")
    try:
        return int(obj_id)
    except (TypeError, ValueError):
        return None


def _filter_video_points_for_obj(points: list[dict[str, Any]], obj_id: int | None) -> list[dict[str, Any]]:
    if obj_id is None:
        return points
    filtered: list[dict[str, Any]] = []
    for point in points:
        try:
            point_obj_id = int(point.get("obj_id", -1))
        except (TypeError, ValueError):
            point_obj_id = -1
        if point_obj_id == obj_id:
            filtered.append(point)
    return filtered


def _coherent_video_points(points: list[dict[str, Any]], seed: RegionModel) -> tuple[list[dict[str, Any]], list[str]]:
    flags: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        frame_id = str(point.get("frame_id") or "")
        if not frame_id:
            continue
        grouped.setdefault(frame_id, []).append(point)
    if any(len(group) > 1 for group in grouped.values()):
        flags.append("ambiguous_sam3_points_same_frame")
    selected: list[dict[str, Any]] = []
    previous_box = seed.bbox.model_dump()
    for frame_id in sorted(grouped, key=lambda key: min(int(point.get("pts_ms", 0) or 0) for point in grouped[key])):
        candidates = grouped[frame_id]
        chosen = max(candidates, key=lambda point: _candidate_score(previous_box, point))
        if _candidate_score(previous_box, chosen) < 0.08 or not _bbox_transition_plausible(previous_box, chosen.get("bbox", {})):
            flags.append("unstable_sam3_track")
            continue
        selected.append(chosen)
        previous_box = chosen.get("bbox", previous_box)
    selected.sort(key=lambda item: (int(item.get("pts_ms", 0) or 0), str(item.get("frame_id", ""))))
    return selected, _dedupe_strings(flags)


def _candidate_score(reference_box: dict[str, float], point: dict[str, Any]) -> float:
    bbox = point.get("bbox") if isinstance(point.get("bbox"), dict) else {}
    iou = _bbox_iou(reference_box, bbox)
    center_score = max(0.0, 1.0 - _center_distance(reference_box, bbox) / max(1.0, _bbox_diagonal(reference_box) * 4.0))
    area_score = _area_similarity(reference_box, bbox)
    confidence = float(point.get("confidence", 0.0) or 0.0)
    return iou * 0.45 + center_score * 0.3 + area_score * 0.15 + confidence * 0.1


def _bbox_transition_plausible(left: dict[str, float], right: dict[str, float]) -> bool:
    if _bbox_iou(left, right) >= 0.02:
        return True
    if _area_similarity(left, right) < 0.18:
        return False
    return _center_distance(left, right) <= max(80.0, _bbox_diagonal(left) * 4.0)


def _bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    lx, ly, lw, lh = float(left.get("x", 0)), float(left.get("y", 0)), float(left.get("w", 0)), float(left.get("h", 0))
    rx, ry, rw, rh = float(right.get("x", 0)), float(right.get("y", 0)), float(right.get("w", 0)), float(right.get("h", 0))
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(left: dict[str, float], right: dict[str, float]) -> float:
    lx = float(left.get("x", 0)) + float(left.get("w", 0)) / 2.0
    ly = float(left.get("y", 0)) + float(left.get("h", 0)) / 2.0
    rx = float(right.get("x", 0)) + float(right.get("w", 0)) / 2.0
    ry = float(right.get("y", 0)) + float(right.get("h", 0)) / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _bbox_diagonal(bbox: dict[str, float]) -> float:
    return (float(bbox.get("w", 0)) ** 2 + float(bbox.get("h", 0)) ** 2) ** 0.5


def _area_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    left_area = max(1.0, float(left.get("w", 0)) * float(left.get("h", 0)))
    right_area = max(1.0, float(right.get("w", 0)) * float(right.get("h", 0)))
    return min(left_area, right_area) / max(left_area, right_area)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _point_anchor_box(point: PointModel, width: float, height: float, box_size_ratio: float, box_size_pixels: float | None) -> BBoxModel:
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height must be positive for SAM3 point prompts")
    size = float(box_size_pixels) if box_size_pixels is not None else min(width, height) * float(box_size_ratio)
    size = max(1.0, min(size, max(width, height)))
    half = size / 2.0
    x1 = max(0.0, min(float(point.x) - half, width - 1.0))
    y1 = max(0.0, min(float(point.y) - half, height - 1.0))
    x2 = max(x1 + 1.0, min(float(point.x) + half, width))
    y2 = max(y1 + 1.0, min(float(point.y) + half, height))
    return BBoxModel(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "service": "sam3-api"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    return {
        "status": "ready" if _runtime is not None else "not_warmed",
        "service": "sam3-api",
        "model_dir": str(SAM3_MODEL_DIR),
        "checkpoint": str(_resolve_checkpoint() or ""),
        "device": SAM3_DEVICE,
    }


@app.post("/warmup")
async def warmup() -> dict[str, Any]:
    started = time.monotonic()
    try:
        runtime = _get_runtime()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}")
    return {
        "status": "ready",
        "service": "sam3-api",
        "device": runtime["device"],
        "checkpoint": runtime["checkpoint_path"],
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
    }


@app.post("/v1/sam3/detect")
async def detect(request: DetectRequest) -> dict[str, Any]:
    started = time.monotonic()
    if not request.prompts:
        return {"model": "sam3", "detections": [], "metadata": {"reason": "no_prompts"}}
    try:
        runtime = _get_runtime()
        image = _load_image(request.image_path)
        processor = runtime["processor"]
        with _inference_context():
            state = processor.set_image(image)
        detections: list[dict[str, Any]] = []
        for prompt in request.prompts:
            threshold = SAM3_CONFIDENCE if prompt.threshold is None else float(prompt.threshold)
            processor.set_confidence_threshold(threshold, state=state)
            with _inference_context():
                output = processor.set_text_prompt(prompt=prompt.text, state=state)
            boxes, scores = _extract_boxes_scores(output)
            masks = _extract_masks(output)
            for index, (box, score) in enumerate(zip(boxes, scores)):
                if score < threshold:
                    continue
                mask = masks[index] if index < len(masks) else None
                mask_bbox = _bbox_from_mask(mask) if mask is not None else None
                bbox = mask_bbox or _bbox_from_box(box)
                if bbox is None:
                    continue
                mask_payload = _mask_payload(
                    mask,
                    request.image_path,
                    "detect",
                    return_masks=request.return_masks,
                    return_polygons=request.return_polygons,
                    bbox=bbox,
                )
                detections.append(
                    {
                        "category": prompt.category,
                        "prompt": prompt.text,
                        "score": score,
                        "bbox": bbox.model_dump(),
                        **mask_payload,
                    }
                )
            processor.reset_all_prompts(state)
    except Exception as exc:
        logger.exception("SAM3 detect failed: image_path=%s prompts=%s", request.image_path, request.prompts)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return {
        "model": "sam3",
        "detections": detections,
        "metadata": {
            "backend": "official_sam3",
            "device": runtime["device"],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        },
    }


@app.post("/v1/sam3/point-detect")
async def point_detect(request: PointDetectRequest) -> dict[str, Any]:
    started = time.monotonic()
    if not request.point_prompts:
        return {"model": "sam3", "detections": [], "metadata": {"reason": "no_point_prompts"}}
    try:
        runtime = _get_runtime()
        image = _load_image(request.image_path)
        width, height = image.size
        processor = runtime["processor"]
        with _inference_context():
            state = processor.set_image(image)
        detections: list[dict[str, Any]] = []
        grouped_prompts: dict[tuple[str, str, float], list[PointPromptModel]] = {}
        for prompt in request.point_prompts:
            threshold = SAM3_CONFIDENCE if prompt.threshold is None else float(prompt.threshold)
            key = (prompt.category, prompt.text.strip(), threshold)
            grouped_prompts.setdefault(key, []).append(prompt)
        for (category, text_prompt, threshold), prompts in grouped_prompts.items():
            processor.set_confidence_threshold(threshold, state=state)
            if text_prompt and text_prompt.lower() != "visual":
                with _inference_context():
                    processor.set_text_prompt(prompt=text_prompt, state=state)
            prompt_points: list[dict[str, Any]] = []
            anchor_boxes: list[dict[str, Any]] = []
            output: dict[str, Any] | None = None
            for prompt in prompts:
                anchor_bbox = _point_anchor_box(
                    prompt.point,
                    width,
                    height,
                    prompt.box_size_ratio,
                    prompt.box_size_pixels,
                )
                normalized_box = _normalized_cxcywh_box(anchor_bbox, width, height)
                prompt_points.append({"x": prompt.point.x, "y": prompt.point.y, "label": prompt.point.label})
                anchor_boxes.append(anchor_bbox.model_dump())
                with _inference_context():
                    output = processor.add_geometric_prompt(box=normalized_box, label=prompt.point.label, state=state)
            if output is None:
                processor.reset_all_prompts(state)
                continue
            boxes, scores = _extract_boxes_scores(output)
            masks = _extract_masks(output)
            for index, (box, score) in enumerate(zip(boxes, scores)):
                if score < threshold:
                    continue
                mask = masks[index] if index < len(masks) else None
                mask_bbox = _bbox_from_mask(mask) if mask is not None else None
                bbox = mask_bbox or _bbox_from_box(box)
                if bbox is None:
                    continue
                mask_payload = _mask_payload(
                    mask,
                    request.image_path,
                    "point",
                    return_masks=request.return_masks,
                    return_polygons=request.return_polygons,
                    bbox=bbox,
                )
                detections.append(
                    {
                        "category": category,
                        "prompt": text_prompt,
                        "prompt_type": "multi_point_box_proxy" if len(prompts) > 1 else ("positive_point" if prompts[0].point.label else "negative_point"),
                        "score": score,
                        "point": prompt_points[0] if prompt_points else None,
                        "point_prompts": prompt_points,
                        "point_anchor_bbox": anchor_boxes[0] if anchor_boxes else None,
                        "point_anchor_bboxes": anchor_boxes,
                        "bbox": bbox.model_dump(),
                        **mask_payload,
                    }
                )
            processor.reset_all_prompts(state)
    except Exception as exc:
        logger.exception("SAM3 point detect failed: image_path=%s point_prompts=%s", request.image_path, request.point_prompts)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return {
        "model": "sam3",
        "detections": detections,
        "metadata": {
            "backend": "official_sam3",
            "point_prompt_backend": "geometric_box_proxy",
            "device": runtime["device"],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        },
    }


@app.post("/v1/sam3/video-track")
async def video_track(request: VideoTrackRequest) -> dict[str, Any]:
    started = time.monotonic()
    if not request.frames:
        return {"model": "sam3_video", "tracks": [], "metadata": {"reason": "no_frames"}}
    if not request.tracks:
        return {"model": "sam3_video", "tracks": [], "metadata": {"reason": "no_tracks"}}
    try:
        runtime = _get_video_runtime()
        model = runtime["model"]
        images = _load_video_images(request.frames)
        frame_by_id = {frame.frame_id: frame for frame in request.frames}
        frame_position_by_id = {frame.frame_id: index for index, frame in enumerate(request.frames)}
        max_frame_num_to_track = request.max_frame_num_to_track or len(request.frames)
        offload_video_to_cpu = SAM3_VIDEO_OFFLOAD_TO_CPU if request.offload_video_to_cpu is None else bool(request.offload_video_to_cpu)
        tracks: list[dict[str, Any]] = []
        for track_index, track in enumerate(request.tracks):
            seed = _seed_region(track, frame_by_id)
            if seed is None:
                tracks.append({
                    "risk_id": track.risk_id,
                    "track_id": track.track_id or f"sam3_video_track_{track_index + 1}",
                    "points": [],
                    "metadata": {"reason": "no_seed_region"},
                })
                continue
            seed_frame_position = _seed_frame_position(track, frame_position_by_id)
            seed_frame_position = max(0, min(seed_frame_position, len(request.frames) - 1))
            seed_frame = request.frames[seed_frame_position]
            seed_image_path = _video_frame_path(seed_frame)
            width, height = images[seed_frame_position].size
            seed_box = _normalized_xywh_box(seed.bbox, width, height)
            with _inference_context():
                inference_state = model.init_state(
                    resource_path=images,
                    offload_video_to_cpu=offload_video_to_cpu,
                    offload_state_to_cpu=bool(request.offload_state_to_cpu),
                )
                _, prompted_outputs = model.add_prompt(
                    inference_state,
                    frame_idx=seed_frame_position,
                    boxes_xywh=[seed_box],
                    box_labels=[True],
                )

            points_by_key: dict[tuple[str, int], dict[str, Any]] = {}
            _merge_video_points(
                points_by_key,
                _video_output_points(
                    prompted_outputs,
                    seed_frame,
                    image_path=seed_image_path,
                    return_masks=request.return_masks,
                    return_polygons=request.return_polygons,
                ),
            )
            seed_obj_id = _select_seed_obj_id(list(points_by_key.values()), seed)
            with _inference_context():
                for frame_position, outputs in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=seed_frame_position,
                    max_frame_num_to_track=max_frame_num_to_track,
                    reverse=False,
                ):
                    if 0 <= frame_position < len(request.frames):
                        frame = request.frames[frame_position]
                        _merge_video_points(
                            points_by_key,
                            _filter_video_points_for_obj(
                                _video_output_points(
                                    outputs,
                                    frame,
                                    image_path=_video_frame_path(frame),
                                    return_masks=request.return_masks,
                                    return_polygons=request.return_polygons,
                                ),
                                seed_obj_id,
                            ),
                        )
                for frame_position, outputs in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=seed_frame_position,
                    max_frame_num_to_track=max_frame_num_to_track,
                    reverse=True,
                ):
                    if 0 <= frame_position < len(request.frames):
                        frame = request.frames[frame_position]
                        _merge_video_points(
                            points_by_key,
                            _filter_video_points_for_obj(
                                _video_output_points(
                                    outputs,
                                    frame,
                                    image_path=_video_frame_path(frame),
                                    return_masks=request.return_masks,
                                    return_polygons=request.return_polygons,
                                ),
                                seed_obj_id,
                            ),
                        )
            points, quality_flags = _coherent_video_points(list(points_by_key.values()), seed)
            tracks.append({
                "risk_id": track.risk_id,
                "track_id": track.track_id or f"sam3_video_track_{track_index + 1}",
                "category": track.category,
                "target_type": track.target_type,
                "points": points,
                "metadata": {
                    "seed_frame_id": seed_frame.frame_id,
                    "seed_frame_index": seed_frame.frame_index,
                    "seed_region_count": len(track.seed_regions),
                    "seed_obj_id": seed_obj_id,
                    "point_count": len(points),
                    "quality_flags": quality_flags,
                    "backend": "official_sam3_video",
                },
            })
        return {
            "model": "sam3_video",
            "tracks": tracks,
            "metadata": {
                "backend": "official_sam3_video",
                "device": runtime["device"],
                "checkpoint": runtime["checkpoint_path"],
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            },
        }
    except Exception as exc:
        logger.exception("SAM3 video track failed: frames=%s tracks=%s", len(request.frames), len(request.tracks))
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/v1/sam3/refine")
async def refine(request: RefineRequest) -> dict[str, Any]:
    started = time.monotonic()
    try:
        runtime = _get_runtime()
        image = _load_image(request.image_path)
        width, height = image.size
        processor = runtime["processor"]
        with _inference_context():
            state = processor.set_image(image)
        threshold = SAM3_CONFIDENCE if request.threshold is None else float(request.threshold)
        processor.set_confidence_threshold(threshold, state=state)
        refined: list[dict[str, Any]] = []
        for region in request.regions:
            bbox = region.bbox
            box = _normalized_cxcywh_box(bbox, width, height)
            with _inference_context():
                output = processor.add_geometric_prompt(box=box, label=True, state=state)
            boxes, scores = _extract_boxes_scores(output)
            masks = _extract_masks(output)
            if boxes:
                mask = masks[0] if masks else None
                new_bbox = _bbox_from_mask(mask) or _bbox_from_box(boxes[0])
                score = float(scores[0]) if scores else region.confidence
                if new_bbox is not None:
                    mask_payload = _mask_payload(
                        mask,
                        request.image_path,
                        "refine",
                        return_masks=mask is not None,
                        return_polygons=mask is not None,
                        bbox=new_bbox,
                    )
                    refined.append(
                        {
                            "bbox": new_bbox.model_dump(),
                            "confidence": max(region.confidence, score),
                            **mask_payload,
                        }
                    )
                    processor.reset_all_prompts(state)
                    continue
            refined.append(region.model_dump())
            processor.reset_all_prompts(state)
    except Exception as exc:
        logger.exception("SAM3 refine failed: image_path=%s regions=%s", request.image_path, request.regions)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return {
        "model": "sam3",
        "regions": refined,
        "metadata": {
            "backend": "official_sam3",
            "device": runtime["device"],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        },
    }
