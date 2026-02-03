"""
Image utilities for the API.

Helpers for preprocessing images before sending to the GPU worker
(e.g. resize to 512x512 for detection).
"""
from __future__ import annotations

import io
from typing import Union

from PIL import Image


# Target size required by hybrid/full_inversion detection (DDIM inversion).
DETECTION_SIZE = (512, 512)


def resize_image_to_512(
    image: Union[bytes, Image.Image],
    *,
    output_format: str = "PNG",
) -> bytes:
    """
    Resize an image to 512x512 for detection.

    The GPU worker requires exactly 512x512 for hybrid/full_inversion
    detection modes. This helper resizes arbitrary input dimensions
    so callers can send any image size to the /detect endpoint.

    Args:
        image: Raw image bytes (JPEG/PNG/etc.) or a PIL Image.
        output_format: Format for output bytes ("PNG" or "JPEG"). Default PNG.

    Returns:
        Image bytes of the 512x512 image in the requested format.
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
    else:
        raise TypeError(f"Expected bytes or PIL Image, got {type(image)}")

    if image.size == DETECTION_SIZE:
        # Already correct size; still re-encode so output is consistent
        pass
    else:
        image = image.resize(DETECTION_SIZE, Image.LANCZOS)

    buf = io.BytesIO()
    if output_format.upper() == "JPEG":
        image.save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="PNG")
    return buf.getvalue()
