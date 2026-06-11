"""
ShieldGemma 2 safety moderation provider skeleton.

Requires: transformers, torch
"""
# 中文说明：该 provider 把 ShieldGemma 这类视觉安全审核模型输出转换成统一的审核结果。
# 上层策略只看 PictureModerationResult，而不关心底层到底是哪一个模型。
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from picture.domain.enums import SafetyCategory
from picture.domain.models import PictureModerationResult
from picture.providers.base import SafetyModerator

logger = logging.getLogger(__name__)

# 中文说明：不同模型的标签体系与项目内部的安全类别并不一致，
# 因此这里需要做一层标签映射。
_LABEL_MAP: dict[str, SafetyCategory] = {
    "sexually_explicit": SafetyCategory.EXPLICIT,
    "violence_gore": SafetyCategory.GRAPHIC_VIOLENCE,
    "hate": SafetyCategory.HATE_SYMBOL,
    "self_harm": SafetyCategory.SELF_HARM,
    "dangerous": SafetyCategory.DANGEROUS,
}


class ShieldGemmaSafetyModerator(SafetyModerator):
    """ShieldGemma 2 based image safety moderation provider."""

    def __init__(
        self,
        model_name: str = "",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._kwargs = kwargs
        self._pipeline: Any = None

    def _get_pipeline(self) -> Any:
        """Lazy initialization of the model pipeline."""
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline  # type: ignore[import-untyped]
                from picture.domain.exceptions import ProviderNotAvailableError

                device = self._resolve_device(torch)
                if not str(self._model_name).strip():
                    raise ProviderNotAvailableError("ShieldGemma 2 local model path is required")
                model_path = Path(self._model_name)
                if not model_path.exists():
                    raise ProviderNotAvailableError(
                        f"ShieldGemma 2 local model path is required, got {self._model_name!r}"
                    )
                self._pipeline = pipeline(
                    "image-classification",
                    model=str(model_path),
                    device=device,
                    local_files_only=True,
                )
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError

                raise ProviderNotAvailableError("ShieldGemma 2 (transformers + torch)")
        return self._pipeline

    def _resolve_device(self, torch: Any) -> int | str:
        if self._device == "auto":
            if torch.cuda.is_available():
                return 0
            from picture.domain.exceptions import ProviderNotAvailableError

            raise ProviderNotAvailableError("ShieldGemma 2 requires CUDA because GPU-first execution is configured")
        if str(self._device).startswith("cuda"):
            if not torch.cuda.is_available():
                from picture.domain.exceptions import ProviderNotAvailableError

                raise ProviderNotAvailableError(f"ShieldGemma 2 requested device {self._device!r}, but CUDA is not available")
            if ":" in str(self._device):
                return int(str(self._device).split(":", 1)[1])
            return 0
        return self._device

    @property
    def name(self) -> str:
        return "ShieldGemma2"

    def moderate(self, image_path: str) -> PictureModerationResult:
        """Run ShieldGemma 2 moderation on the image."""
        pipe = self._get_pipeline()
        results = pipe(image_path)

        categories: list[SafetyCategory] = []
        scores: dict[str, float] = {}
        reason_codes: list[str] = []

        for item in results:
            label = item["label"]
            score = item["score"]
            scores[label] = score

            # 中文说明：只有映射成功且分数超过阈值的标签才真正计入风险类别。
            cat = _LABEL_MAP.get(label)
            if cat and score > 0.5:
                categories.append(cat)
                reason_codes.append(f"SAFETY_{cat.value.upper()}")

        is_safe = len(categories) == 0

        return PictureModerationResult(
            is_safe=is_safe,
            categories=categories or [SafetyCategory.SAFE],
            scores=scores,
            reason_codes=reason_codes,
            provider=self.name,
        )
