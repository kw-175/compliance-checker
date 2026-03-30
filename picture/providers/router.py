"""
Image router: classifies images into document / natural / mixed route types.

Uses heuristics based on image properties and optional OCR pre-scan.
"""

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

    def classify(self, image_path: str, metadata: dict[str, Any] | None = None) -> RouteType:
        """Classify an image into a route type using heuristics."""
        metadata = metadata or {}

        # Honour explicit hint
        hint = metadata.get("route_hint", "auto")
        if hint != "auto":
            try:
                return RouteType(hint)
            except ValueError:
                logger.warning("Invalid route_hint '%s', falling back to auto", hint)

        # Try lightweight heuristic using PIL if available
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
            aspect = w / max(h, 1)

            # Tall narrow -> likely document
            if aspect < 0.6 or aspect > 1.8:
                return RouteType.DOCUMENT

            # Check color variance: document images tend to be less colorful
            if img.mode in ("RGB", "RGBA"):
                import numpy as np
                arr = np.array(img.convert("RGB"))
                color_std = float(arr.std())
                if color_std < 40:
                    return RouteType.DOCUMENT
                elif color_std > 70:
                    return RouteType.NATURAL
                else:
                    return RouteType.MIXED
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Router heuristic failed: %s, defaulting to MIXED", exc)

        return RouteType.MIXED
