"""
Image preprocessor: EXIF stripping, resizing, rotation correction,
color space normalization, PDF page extraction.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from picture.providers.base import Preprocessor

logger = logging.getLogger(__name__)

# Maximum dimension for preprocessed images
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

            # Auto-rotate based on EXIF orientation
            img = ImageOps.exif_transpose(img)

            # Strip EXIF by copying pixel data only
            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))

            # Convert to RGB if needed
            if clean.mode not in ("RGB", "RGBA"):
                clean = clean.convert("RGB")

            # Resize if too large
            w, h = clean.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                clean = clean.resize((new_w, new_h), Image.LANCZOS)
                logger.info("Resized image from %dx%d to %dx%d", w, h, new_w, new_h)

            clean.save(str(out_path), format="PNG")
            logger.info("Preprocessed image saved to %s", out_path)
            return str(out_path)

        except ImportError:
            # Pillow not available – just copy the file
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
