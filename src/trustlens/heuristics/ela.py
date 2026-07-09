"""Error Level Analysis (ELA).

Idea: re-save an image as JPEG at a known quality and diff it against the
original. JPEG is lossy, so a *single* consistent compression history produces a
low, uniform error surface. Regions that were pasted/edited (or that never had a
natural JPEG history, as with many GAN outputs) recompress differently and stand
out. We turn that error surface into a handful of scalar features.

Reference: Krawetz, "A Picture's Worth" (Hacker Factor), 2007.
"""
from __future__ import annotations

import io
from typing import Dict

import numpy as np
from PIL import Image, ImageChops


def error_level_image(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Return the ELA magnitude map as a float32 array in [0, 255].

    The map is the per-pixel max-channel absolute difference between the image
    and its own JPEG recompression.
    """
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(rgb, recompressed)
    arr = np.asarray(diff, dtype=np.float32)  # H x W x 3
    # Collapse channels to a single magnitude per pixel.
    return arr.max(axis=2)


def ela_features(img: Image.Image, quality: int = 90) -> Dict[str, float]:
    """Scalar ELA features.

    - ela_mean : average error level (uniform, low for clean single-history JPEGs)
    - ela_std  : spread of error; high when some regions differ from others
    - ela_p99  : 99th-percentile error, a robust "hot region" magnitude
    """
    mag = error_level_image(img, quality=quality)
    # Normalize to [0, 1] for scale stability across the pipeline.
    mag = mag / 255.0
    return {
        "ela_mean": float(mag.mean()),
        "ela_std": float(mag.std()),
        "ela_p99": float(np.percentile(mag, 99)),
    }
