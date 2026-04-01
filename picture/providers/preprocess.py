"""
Image preprocessor: EXIF stripping, resizing, rotation correction,
color space normalization, PDF page extraction.
"""
# 中文说明：预处理器是所有下游能力共享的入口。
# 它的目标不是做语义分析，而是把输入图像整理成更稳定、统一、可被后续模型消费的格式。
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from picture.providers.base import Preprocessor

logger = logging.getLogger(__name__)

# 中文说明：超过该边长的图片会被缩放，避免大图导致 OCR、检测、分割阶段显著变慢或爆内存。
MAX_DIM = 4096


class DefaultPreprocessor(Preprocessor):
    """
    Default preprocessor using Pillow.

    Operations:
    - Strip EXIF metadata
    - Auto-rotate based on EXIF orientation
    - Resize if any dimension > MAX_DIM
    - Convert to sRGB color space
    - Save as PNG
    """

    def preprocess(self, image_path: str, output_dir: str) -> str:
        """Preprocess the image and return path to the result."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        src = Path(image_path)
        out_path = out_dir / f"preprocessed_{src.stem}.png"

        try:
            from PIL import Image, ImageOps

            img = Image.open(image_path)

            # 中文说明：有些手机图片在文件像素上并未真正旋转，只在 EXIF 里记录了方向，
            # 因此先根据 EXIF 自动转正，避免后续 OCR 和检测方向错乱。
            img = ImageOps.exif_transpose(img)

            # 中文说明：通过新建图片并复制像素的方式剥离 EXIF，
            # 避免把原图的定位、拍摄时间等元数据带到结果图里。
            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))

            # 中文说明：如果不是 RGB/RGBA，则统一转成 RGB，
            # 降低后续模型或库在颜色模式上的兼容性问题。
            if clean.mode not in ("RGB", "RGBA"):
                clean = clean.convert("RGB")

            # 中文说明：对超大图做等比例缩放，保留长宽比不变。
            w, h = clean.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                clean = clean.resize((new_w, new_h), Image.LANCZOS)
                logger.info("Resized image from %dx%d to %dx%d", w, h, new_w, new_h)

            # 中文说明：统一保存成 PNG，避免不同源格式给下游造成额外差异。
            clean.save(str(out_path), format="PNG")
            logger.info("Preprocessed image saved to %s", out_path)
            return str(out_path)

        except ImportError:
            # 中文说明：如果当前环境没有 Pillow，就退化为原样复制，
            # 保证主流程仍可跑通，只是失去预处理增益。
            logger.warning("Pillow not available, copying image as-is")
            shutil.copy2(image_path, str(out_path))
            return str(out_path)

    def extract_pdf_pages(self, pdf_path: str, output_dir: str) -> list[str]:
        """Extract pages from a PDF as PNG images."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages: list[str] = []
            for i, page in enumerate(doc):
                # 中文说明：dpi=200 是一个兼顾清晰度和处理成本的折中值。
                pix = page.get_pixmap(dpi=200)
                out_path = out_dir / f"page_{i:04d}.png"
                pix.save(str(out_path))
                pages.append(str(out_path))
            doc.close()
            logger.info("Extracted %d pages from PDF", len(pages))
            return pages
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError

            raise ProviderNotAvailableError("PyMuPDF (fitz)")
