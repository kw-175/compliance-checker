"""
YOLO26 vision detection provider skeleton.

Requires: ultralytics
"""
# 中文说明：该 provider 把 YOLO 检测结果转换成 picture 模块统一的 finding 结构。
# 这样策略层和脱敏层都只需要面对统一接口，而不必理解具体检测框架。
from __future__ import annotations

import logging
from typing import Any

from picture.domain.enums import FindingType, VisionObjectType
from picture.domain.models import BBox, PictureFinding, RegionMask
from picture.providers.base import VisionDetector

logger = logging.getLogger(__name__)

# 中文说明：检测模型的类别名未必和项目内部类别完全一致，需要做一层映射。
_CLASS_MAP: dict[str, VisionObjectType] = {
    "person": VisionObjectType.FACE,  # will refine with face crop
    "face": VisionObjectType.FACE,
    "id_card": VisionObjectType.ID_CARD,
    "badge": VisionObjectType.BADGE,
    "signature": VisionObjectType.SIGNATURE,
    "stamp": VisionObjectType.STAMP,
    "qr_code": VisionObjectType.QR_CODE,
    "barcode": VisionObjectType.BARCODE,
    "license_plate": VisionObjectType.LICENSE_PLATE,
}


class YOLO26VisionDetector(VisionDetector):
    """YOLO26-based vision detection provider."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._device = device
        self._kwargs = kwargs
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy initialization of YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]

                self._model = YOLO(self._model_path)
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError

                raise ProviderNotAvailableError("YOLO26 (ultralytics)")
        return self._model

    @property
    def name(self) -> str:
        return "YOLO26"

    def detect(self, image_path: str) -> list[PictureFinding]:
        """Run YOLO26 detection on the image."""
        model = self._get_model()
        results = model.predict(image_path, conf=self._conf_threshold, verbose=False)

        findings: list[PictureFinding] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "unknown")
                obj_type = _CLASS_MAP.get(cls_name)

                # 中文说明：未映射到内部类别的检测结果直接忽略，
                # 避免把业务无关目标带入后续策略流程。
                if obj_type is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                findings.append(
                    PictureFinding(
                        finding_type=FindingType.VISION_OBJECT,
                        category=obj_type.value,
                        label=f"{obj_type.value} detected",
                        score=conf,
                        region=RegionMask(
                            bbox=BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                            confidence=conf,
                        ),
                        reason_code=f"VISION_{obj_type.value.upper()}",
                        provider=self.name,
                    )
                )

        return findings
