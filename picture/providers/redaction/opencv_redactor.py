"""
OpenCV/Pillow based image redaction renderer.

Supports: black_box, gaussian_blur, pixelate, solid_fill.
Falls back to Pillow if OpenCV is unavailable.
"""
# 中文说明：该 redactor 负责把结构化的脱敏操作真正渲染到图像上。
# 它是 picture 模块里“把合规判断落地成可交付产物”的最后一步。
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
        # 中文说明：fill_color 主要给 solid_fill 模式使用。
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
            # 中文说明：没有脱敏动作时直接复制原图，避免无意义地重新编码图像。
            import shutil

            shutil.copy2(image_path, output_path)
            return output_path

        # 中文说明：优先走 OpenCV，因为它在大图处理和局部操作上通常更高效。
        try:
            return self._redact_opencv(image_path, operations, output_path)
        except ImportError:
            logger.info("OpenCV not available, falling back to Pillow")
            try:
                return self._redact_pillow(image_path, operations, output_path)
            except ImportError:
                logger.warning(
                    "Pillow not available, falling back to raw file copy for redaction output"
                )
                import shutil

                shutil.copy2(image_path, output_path)
                return output_path

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
            # 中文说明：允许某些 operation 被标记为未应用，用于保留审计记录但跳过执行。
            if not op.applied:
                continue

            mask = _load_mask_opencv(op.region.mask_path, w, h)
            if mask is None:
                mask = _polygon_mask_opencv(op.region.polygon, w, h)
            bbox = op.region.bbox
            x1 = max(0, int(bbox.x))
            y1 = max(0, int(bbox.y))
            x2 = min(w, int(bbox.x + bbox.w))
            y2 = min(h, int(bbox.y + bbox.h))

            # 中文说明：无效区域直接跳过，避免切片报错。
            if x2 <= x1 or y2 <= y1:
                continue

            if op.mode == RedactionMode.BLACK_BOX:
                # 中文说明：黑框模式是最保守的遮挡策略。
                if mask is not None:
                    img[mask > 0] = (0, 0, 0)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

            elif op.mode == RedactionMode.GAUSSIAN_BLUR:
                # 中文说明：模糊核必须是奇数，这里动态根据区域大小生成合适的核尺寸。
                roi = img[y1:y2, x1:x2]
                ksize = max(51, ((min(x2 - x1, y2 - y1) // 2) | 1))
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
                if mask is not None:
                    mask_roi = mask[y1:y2, x1:x2] > 0
                    roi[mask_roi] = blurred[mask_roi]
                    img[y1:y2, x1:x2] = roi
                else:
                    img[y1:y2, x1:x2] = blurred

            elif op.mode == RedactionMode.PIXELATE:
                # 中文说明：像素化通过先缩小再放大实现，放大时使用最近邻保持马赛克效果。
                roi = img[y1:y2, x1:x2]
                rh, rw = roi.shape[:2]
                block_size = max(8, min(rw, rh) // 6)
                small = cv2.resize(
                    roi,
                    (max(1, rw // block_size), max(1, rh // block_size)),
                    interpolation=cv2.INTER_LINEAR,
                )
                pixelated = cv2.resize(
                    small,
                    (rw, rh),
                    interpolation=cv2.INTER_NEAREST,
                )
                if mask is not None:
                    mask_roi = mask[y1:y2, x1:x2] > 0
                    roi[mask_roi] = pixelated[mask_roi]
                    img[y1:y2, x1:x2] = roi
                else:
                    img[y1:y2, x1:x2] = pixelated

            elif op.mode == RedactionMode.SOLID_FILL:
                # 中文说明：OpenCV 使用 BGR 顺序，因此这里需要把 RGB 颜色反转。
                color = (
                    self._fill_color[2],
                    self._fill_color[1],
                    self._fill_color[0],
                )
                if mask is not None:
                    img[mask > 0] = color
                else:
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

            mask = _load_mask_pillow(op.region.mask_path, w, h)
            if mask is None:
                mask = _polygon_mask_pillow(op.region.polygon, w, h)
            bbox = op.region.bbox
            x1 = max(0, int(bbox.x))
            y1 = max(0, int(bbox.y))
            x2 = min(w, int(bbox.x + bbox.w))
            y2 = min(h, int(bbox.y + bbox.h))

            if x2 <= x1 or y2 <= y1:
                continue

            if op.mode == RedactionMode.BLACK_BOX:
                if mask is not None:
                    fill = Image.new("RGB", img.size, (0, 0, 0))
                    img.paste(fill, (0, 0), mask)
                    draw = ImageDraw.Draw(img)
                else:
                    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

            elif op.mode == RedactionMode.GAUSSIAN_BLUR:
                roi = img.crop((x1, y1, x2, y2))
                blurred = roi.filter(ImageFilter.GaussianBlur(radius=30))
                if mask is not None:
                    img.paste(blurred, (x1, y1), mask.crop((x1, y1, x2, y2)))
                else:
                    img.paste(blurred, (x1, y1))

                # 中文说明：Pillow 在 paste 后原来的 draw 对象不会自动感知像素变化，
                # 因此这里重新创建，避免后续绘制状态不一致。
                draw = ImageDraw.Draw(img)

            elif op.mode == RedactionMode.PIXELATE:
                roi = img.crop((x1, y1, x2, y2))
                rw, rh = roi.size
                block_size = max(8, min(rw, rh) // 6)
                small = roi.resize(
                    (max(1, rw // block_size), max(1, rh // block_size)),
                    Image.BILINEAR,
                )
                pixelated = small.resize((rw, rh), Image.NEAREST)
                if mask is not None:
                    img.paste(pixelated, (x1, y1), mask.crop((x1, y1, x2, y2)))
                else:
                    img.paste(pixelated, (x1, y1))
                draw = ImageDraw.Draw(img)

            elif op.mode == RedactionMode.SOLID_FILL:
                if mask is not None:
                    fill = Image.new("RGB", img.size, self._fill_color)
                    img.paste(fill, (0, 0), mask)
                    draw = ImageDraw.Draw(img)
                else:
                    draw.rectangle([x1, y1, x2, y2], fill=self._fill_color)

        img.save(output_path)
        logger.info(
            "Redacted image (Pillow) saved to %s (%d operations)",
            output_path,
            len(operations),
        )
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

            # 中文说明：不同模式使用不同叠加颜色，便于调试时快速区分脱敏策略。
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
                mask = _load_mask_pillow(op.region.mask_path, img.width, img.height)
                if mask is not None:
                    colored = Image.new("RGBA", img.size, color)
                    overlay.alpha_composite(Image.composite(colored, Image.new("RGBA", img.size, (0, 0, 0, 0)), mask))
                elif op.region.polygon and op.region.polygon.points:
                    draw.polygon(op.region.polygon.points, fill=color, outline=color[:3] + (200,))
                else:
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        fill=color,
                        outline=color[:3] + (200,),
                        width=2,
                    )

            result = Image.alpha_composite(img, overlay).convert("RGB")
            result.save(output_path)
            logger.info("Overlay image saved to %s", output_path)
            return output_path

        except ImportError:
            # 中文说明：没有 Pillow 时就退化为普通脱敏输出，至少保证接口可用。
            logger.warning("Pillow not available for overlay rendering")
            return self.redact(image_path, operations, output_path)


def _load_mask_opencv(mask_path: str | None, width: int, height: int):
    if not mask_path:
        return None
    try:
        import cv2

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception:
        logger.warning("Failed to load redaction mask: %s", mask_path)
        return None


def _polygon_mask_opencv(polygon, width: int, height: int):
    points = _polygon_points(polygon)
    if len(points) < 3:
        return None
    try:
        import cv2
        import numpy as np

        mask = np.zeros((height, width), dtype=np.uint8)
        contour = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [contour], 255)
        return mask
    except Exception:
        logger.warning("Failed to build polygon redaction mask", exc_info=True)
        return None


def _load_mask_pillow(mask_path: str | None, width: int, height: int):
    if not mask_path:
        return None
    try:
        from PIL import Image

        mask = Image.open(mask_path).convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.NEAREST)
        return mask
    except Exception:
        logger.warning("Failed to load redaction mask: %s", mask_path)
        return None


def _polygon_mask_pillow(polygon, width: int, height: int):
    points = _polygon_points(polygon)
    if len(points) < 3:
        return None
    try:
        from PIL import Image, ImageDraw

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255)
        return mask
    except Exception:
        logger.warning("Failed to build polygon redaction mask", exc_info=True)
        return None


def _polygon_points(polygon) -> list[tuple[int, int]]:
    raw_points = getattr(polygon, "points", None)
    if not raw_points:
        return []
    points: list[tuple[int, int]] = []
    for point in raw_points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((int(round(float(point[0]))), int(round(float(point[1])))))
            except (TypeError, ValueError):
                continue
    return points
