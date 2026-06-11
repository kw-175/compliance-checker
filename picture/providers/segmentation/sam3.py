from __future__ import annotations

from pathlib import Path
from typing import Any

from picture.domain.exceptions import ProviderNotAvailableError
from picture.domain.models import RegionMask
from picture.providers.base import SegmentationProvider
from picture.providers.sam3_runtime import get_sam3_runtime


class SAM3SegmentationProvider(SegmentationProvider):
    """
    SAM3 prompted segmentation entrypoint.

    The ModelScope weights are local, but the runtime Python package still has
    to expose a SAM3 image predictor. If it is not installed, fail explicitly
    instead of silently falling back to mock segmentation.
    """

    def __init__(self, model_dir: str, device: str = "auto", **kwargs: Any) -> None:
        self._model_dir = Path(model_dir)
        self._device = device
        self._kwargs = kwargs
        self._predictor: Any = None

    @property
    def name(self) -> str:
        return "SAM3"

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        self._predictor = get_sam3_runtime(self._model_dir, self._device)
        return self._predictor

    def refine(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        predictor = self._get_predictor()
        try:
            from PIL import Image
        except ImportError as exc:
            raise ProviderNotAvailableError("Pillow for SAM3") from exc

        image = Image.open(image_path).convert("RGB")
        processor = predictor["processor"]
        model = predictor["model"]
        device = predictor["device"]
        torch = predictor["torch"]

        refined: list[RegionMask] = []
        for region in regions:
            bbox = region.bbox
            input_box = [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h]
            inputs = processor(
                images=image,
                input_boxes=[[input_box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            scores = results.get("scores", [])
            boxes = results.get("boxes", [])
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().tolist()
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().tolist()
            if boxes:
                x1, y1, x2, y2 = [float(v) for v in boxes[0]]
                from picture.domain.models import BBox

                refined_region = region.model_copy(
                    update={
                        "bbox": BBox(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1)),
                        "confidence": max(region.confidence, float(scores[0]) if scores else region.confidence),
                    }
                )
            else:
                refined_region = region
            refined.append(refined_region)
        return refined
