"""
SAM 2 segmentation provider skeleton.

Requires: segment-anything-2 (or sam2), torch
"""
# 中文说明：该 provider 用于把检测框进一步细化成更贴合目标边界的区域。
# 在脱敏场景里，这一步通常用于减少“框太大导致误伤”的问题。
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from picture.domain.models import RegionMask
from picture.providers.base import SegmentationProvider

logger = logging.getLogger(__name__)


class SAM2SegmentationProvider(SegmentationProvider):
    """SAM 2 based segmentation refinement provider."""

    def __init__(
        self,
        model_id: str = "",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._kwargs = kwargs
        self._predictor: Any = None

    def _get_predictor(self) -> Any:
        """Lazy initialization of SAM 2."""
        if self._predictor is None:
            try:
                import torch
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-untyped]
                from picture.domain.exceptions import ProviderNotAvailableError

                if self._device.startswith("cuda") and not torch.cuda.is_available():
                    raise ProviderNotAvailableError(f"SAM 2 requested device {self._device!r}, but CUDA is not available")
                if not str(self._model_id).strip():
                    raise ProviderNotAvailableError("SAM 2 local model path is required")
                model_path = Path(self._model_id)
                if not model_path.exists():
                    raise ProviderNotAvailableError(
                        f"SAM 2 local model path is required, got {self._model_id!r}"
                    )
                self._predictor = SAM2ImagePredictor.from_pretrained(str(model_path), local_files_only=True)
                model = getattr(self._predictor, "model", None)
                if model is not None and hasattr(model, "to"):
                    model.to(self._device if self._device != "auto" else "cuda")
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError

                raise ProviderNotAvailableError("SAM 2 (segment-anything-2)")
        return self._predictor

    @property
    def name(self) -> str:
        return "SAM2"

    def refine(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        """Refine regions using SAM 2 prompted segmentation."""
        predictor = self._get_predictor()

        try:
            from PIL import Image
            import numpy as np

            # 中文说明：SAM 通常接收 RGB 数组输入，因此这里先把图片转成 numpy 数组。
            image = np.array(Image.open(image_path).convert("RGB"))
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError

            raise ProviderNotAvailableError("Pillow (PIL)")

        predictor.set_image(image)

        refined: list[RegionMask] = []
        for region in regions:
            bbox = region.bbox
            input_box = np.array([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h])

            masks, scores, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=False,
            )

            # 中文说明：当前实现只取单框预测下的最高分 mask。
            best_mask = masks[0]
            best_score = float(scores[0])

            # 中文说明：为了保持模型输出结构简单，这里没有做精细轮廓提取，
            # 而是把 mask 的最小外接矩形转成 polygon。
            from picture.domain.models import Polygon

            ys, xs = np.where(best_mask)
            if len(xs) > 0:
                polygon = Polygon(
                    points=[
                        (float(xs.min()), float(ys.min())),
                        (float(xs.max()), float(ys.min())),
                        (float(xs.max()), float(ys.max())),
                        (float(xs.min()), float(ys.max())),
                    ]
                )
            else:
                # 中文说明：理论上如果 mask 为空，polygon 就无法构造，保留 None 即可。
                polygon = None

            refined.append(
                RegionMask(
                    bbox=bbox,
                    polygon=polygon,
                    confidence=best_score,
                )
            )

        return refined
