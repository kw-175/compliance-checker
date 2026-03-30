"""
OpenCV/Pillow based image redaction renderer.

Supports: black_box, gaussian_blur, pixelate, solid_fill.
Falls back to Pillow if OpenCV is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from picture.domain.enums import RedactionMode
from picture.domain.models import RedactionOperation
from picture.providers.base import Redactor

logger = logging.getLogger(__name__)


class OpenCVRedactor(Redactor):
    """
    Redaction renderer using OpenCV (primary) with Pillow fallback.

    Renders redaction operations onto the image:
    - black_box: solid black rectangle
    - gaussian_blur: Gaussian blur over the region
    - pixelate: mosaic/pixelation effect
    - solid_fill: filled with specified color (default grey)
    """

    def __init__(self, fill_color: tuple[int, int, int] = (128, 128, 128)) -> None:
        self._fill_color = fill_color

    def redact(
        self,
        image_path: str,
        operations: list[RedactionOperation],
        output_path: str,
    ) -> str:
        """Apply redaction operations and save the result."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not operations:
            # No redactions needed, just copy
            import shutil
            shutil.copy2(image_path, output_path)
            return output_path

        # Try OpenCV first
        try:
            return self._redact_opencv(image_path, operations, output_path)
        except ImportError:
            logger.info("OpenCV not available, falling back to Pillow")
            return self._redact_pillow(image_path, operations, output_path)

    def _redact_opencv(
        self,
        image_path: str,
        operations: list[RedactionOperation],
        output_path: str,
    ) -> str:
        """Render redactions using OpenCV."""
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]

        for op in operations:
            if not op.applied:
                continue

            bbox = op.region.bbox
            x1 = max(0, int(bbox.x))
            y1 = max(0, int(bbox.y))
            x2 = min(w, int(bbox.x + bbox.w))
            y2 = min(h, int(bbox.y + bbox.h))

            if x2 <= x1 or y2 <= y1:
                continue

            if op.mode == RedactionMode.BLACK_BOX:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

            elif op.mode == RedactionMode.GAUSSIAN_BLUR:
                roi = img[y1:y2, x1:x2]
                ksize = max(51, ((min(x2 - x1, y2 - y1) // 2) | 1))
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
                img[y1:y2, x1:x2] = blurred

            elif op.mode == RedactionMode.PIXELATE:
                roi = img[y1:y2, x1:x2]
                rh, rw = roi.shape[:2]
                block_size = max(8, min(rw, rh) // 6)
                small = cv2.resize(roi, (max(1, rw // block_size), max(1, rh // block_size)),
                                   interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                img[y1:y2, x1:x2] = pixelated

            elif op.mode == RedactionMode.SOLID_FILL:
                # BGR format for OpenCV
                color = (self._fill_color[2], self._fill_color[1], self._fill_color[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        cv2.imwrite(output_path, img)
        logger.info("Redacted image saved to %s (%d operations)", output_path, len(operations))
        return output_path

    def _redact_pillow(
        self,
        image_path: str,
        operations: list[RedactionOperation],
        output_path: str,
    ) -> str:
        """Render redactions using Pillow (fallback)."""
        from PIL import Image, ImageDraw, ImageFilter

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for op in operations:
            if not op.applied:
                continue

            bbox = op.region.bbox
            x1 = max(0, int(bbox.x))
            y1 = max(0, int(bbox.y))
            x2 = min(w, int(bbox.x + bbox.w))
            y2 = min(h, int(bbox.y + bbox.h))

            if x2 <= x1 or y2 <= y1:
                continue

            if op.mode == RedactionMode.BLACK_BOX:
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

            elif op.mode == RedactionMode.GAUSSIAN_BLUR:
                roi = img.crop((x1, y1, x2, y2))
                blurred = roi.filter(ImageFilter.GaussianBlur(radius=30))
                img.paste(blurred, (x1, y1))
                draw = ImageDraw.Draw(img)  # refresh draw object

            elif op.mode == RedactionMode.PIXELATE:
                roi = img.crop((x1, y1, x2, y2))
                rw, rh = roi.size
                block_size = max(8, min(rw, rh) // 6)
                small = roi.resize((max(1, rw // block_size), max(1, rh // block_size)),
                                   Image.BILINEAR)
                pixelated = small.resize((rw, rh), Image.NEAREST)
                img.paste(pixelated, (x1, y1))
                draw = ImageDraw.Draw(img)

            elif op.mode == RedactionMode.SOLID_FILL:
                draw.rectangle([x1, y1, x2, y2], fill=self._fill_color)

        img.save(output_path)
        logger.info("Redacted image (Pillow) saved to %s (%d operations)", output_path, len(operations))
        return output_path

    def render_overlay(
        self,
        image_path: str,
        operations: list[RedactionOperation],
        output_path: str,
    ) -> str:
        """Render a semi-transparent overlay showing redacted regions."""
        try:
            from PIL import Image, ImageDraw

            img = Image.open(image_path).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Colors for different modes
            color_map = {
                RedactionMode.BLACK_BOX: (255, 0, 0, 80),
                RedactionMode.GAUSSIAN_BLUR: (0, 0, 255, 80),
                RedactionMode.PIXELATE: (0, 255, 0, 80),
                RedactionMode.SOLID_FILL: (255, 255, 0, 80),
            }

            for op in operations:
                if not op.applied:
                    continue
                bbox = op.region.bbox
                x1, y1 = int(bbox.x), int(bbox.y)
                x2, y2 = int(bbox.x + bbox.w), int(bbox.y + bbox.h)
                color = color_map.get(op.mode, (255, 0, 0, 80))
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (200,), width=2)

            result = Image.alpha_composite(img, overlay).convert("RGB")
            result.save(output_path)
            logger.info("Overlay image saved to %s", output_path)
            return output_path

        except ImportError:
            logger.warning("Pillow not available for overlay rendering")
            return self.redact(image_path, operations, output_path)
