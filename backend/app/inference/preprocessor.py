from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


class Preprocessor:
    """Image preprocessor for YOLO inference."""

    def __init__(self, target_size: tuple[int, int] = (640, 640)) -> None:
        self.target_size = target_size

    def decode_image(self, image_base64: str, image_format: str) -> Image.Image:
        """Decode base64 image string to PIL Image."""
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception:
            raise ValueError("Invalid base64 image data")
        if len(image_bytes) == 0:
            raise ValueError("Empty image data")
        fmt_map = {
            "png": "PNG",
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "bmp": "BMP",
            "webp": "WEBP",
        }
        pil_format = fmt_map.get(image_format.lower())
        if pil_format is None:
            raise ValueError(f"Unsupported image format: {image_format}")
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize image to target size."""
        return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def to_tensor(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to normalized NCHW float32 tensor."""
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, axis=0)

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to NCHW numpy array (float32, 0-1)."""
        return self.to_tensor(image)
