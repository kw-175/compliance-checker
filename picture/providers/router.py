"""
Image router: classifies images into document / natural / mixed route types.

Uses heuristics based on image properties and optional OCR pre-scan.
"""
# 中文说明：路由器决定一张图该走哪条检测链路。
# 它不是为了绝对准确分类，而是为了把图片送入“更大概率合适”的处理流程。
from __future__ import annotations

import logging
from typing import Any

from picture.domain.enums import RouteType
from picture.providers.base import Router

logger = logging.getLogger(__name__)


class HeuristicRouter(Router):
    """
    Rule-based router that classifies images using aspect ratio,
    color distribution, and optional metadata hints.
    """

    def classify(
        self,
        image_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> RouteType:
        """Classify an image into a route type using heuristics."""
        metadata = metadata or {}

        # 中文说明：如果调用方显式传入 route_hint，则优先尊重该提示，
        # 这有助于业务侧在已知图片类型时跳过启发式误判。
        hint = metadata.get("route_hint", "auto")
        if hint != "auto":
            try:
                return RouteType(hint)
            except ValueError:
                logger.warning("Invalid route_hint '%s', falling back to auto", hint)

        # 中文说明：启发式逻辑尽量保持轻量，只依赖 Pillow 和可选 numpy，
        # 避免路由阶段本身变成高成本步骤。
        try:
            from PIL import Image

            img = Image.open(image_path)
            w, h = img.size
            aspect = w / max(h, 1)

            # 中文说明：极端长宽比的图片往往更像文档页、长截图或海报，
            # 比自然照片更接近 document/mixed 场景；这里偏保守归到 document。
            if aspect < 0.6 or aspect > 1.8:
                return RouteType.DOCUMENT

            # 中文说明：颜色标准差是一个粗糙但便宜的指标。
            # 纯文档图往往背景单一、色彩波动小；自然照片通常颜色更丰富。
            if img.mode in ("RGB", "RGBA"):
                import numpy as np

                arr = np.array(img.convert("RGB"))
                color_std = float(arr.std())
                if color_std < 40:
                    return RouteType.DOCUMENT
                elif color_std > 70:
                    return RouteType.NATURAL
                else:
                    # 中文说明：介于两者之间的图片，按 mixed 处理更稳妥，
                    # 因为 mixed 链路会同时照顾文字和视觉对象。
                    return RouteType.MIXED
        except ImportError:
            # 中文说明：依赖不完整时不让路由失败，直接走保守默认。
            pass
        except Exception as exc:
            logger.warning("Router heuristic failed: %s, defaulting to MIXED", exc)

        # 中文说明：默认回退到 mixed，宁可多做一步，也尽量避免漏掉文字或视觉敏感信息。
        return RouteType.MIXED
