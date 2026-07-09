"""Synthetic fixture generation.

Real datasets (140k faces, MIDV-500) are auth-gated and multi-GB, so tests and
CI must not depend on them. Instead we synthesize two visually-plausible classes
that exercise the *same signals* the real heuristics key on:

- **"real"**: broadband natural texture + fine grain, JPEG-compressed, WITH a
  consistent camera EXIF block.
- **"fake"**: smooth low-frequency content plus a periodic grid artifact (the
  upsampling checkerboard GANs leave), NO EXIF.

The point is not photorealism; it is that ELA / EXIF / FFT features separate the
classes, so the classifier and metrics have real signal to work with offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import piexif
except Exception:  # pragma: no cover - piexif is a hard dep, but stay defensive
    piexif = None


def _lowfreq_field(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Smooth field: upsample a small random grid to `size` (low-frequency)."""
    small = rng.random((cells, cells))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC
    )
    return np.asarray(img, dtype=np.float32) / 255.0


def real_array(seed: int = 0, size: int = 256) -> np.ndarray:
    """Natural-looking RGB uint8 array: smooth base + broadband grain."""
    rng = np.random.default_rng(seed)
    base = np.stack([_lowfreq_field(rng, size, 8) for _ in range(3)], axis=2)
    # Broadband fine texture (the hallmark of a real sensor capture).
    grain = rng.normal(0.0, 0.06, size=(size, size, 3))
    img = np.clip(base * 0.8 + 0.1 + grain, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def fake_array(seed: int = 0, size: int = 256) -> np.ndarray:
    """GAN-like RGB uint8 array: very smooth + periodic grid artifact."""
    rng = np.random.default_rng(seed + 10_000)
    base = np.stack([_lowfreq_field(rng, size, 4) for _ in range(3)], axis=2)
    base = base * 0.7 + 0.15

    # Periodic checkerboard/grid -> sharp peaks in the FFT spectrum.
    x = np.arange(size)
    freq = rng.integers(6, 12)  # cycles across the image
    grid = np.outer(np.sin(2 * np.pi * freq * x / size),
                    np.sin(2 * np.pi * freq * x / size))
    grid = (grid[:, :, None] * 0.05)

    # Minimal broadband noise (synthetic images lack real sensor grain).
    tiny = rng.normal(0.0, 0.01, size=(size, size, 3))
    img = np.clip(base + grid + tiny, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _camera_exif_bytes() -> bytes:
    zeroth = {
        piexif.ImageIFD.Make: b"Canon",
        piexif.ImageIFD.Model: b"Canon EOS 5D Mark IV",
        piexif.ImageIFD.DateTime: b"2024:06:01 09:30:00",
        piexif.ImageIFD.Software: b"",
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: b"2024:06:01 09:30:00",
        piexif.ExifIFD.ExposureTime: (1, 200),
        piexif.ExifIFD.FNumber: (28, 10),
        piexif.ExifIFD.ISOSpeedRatings: 100,
    }
    return piexif.dump({"0th": zeroth, "Exif": exif_ifd})


def save_real_image(path: str, seed: int = 0, size: int = 256) -> str:
    arr = real_array(seed=seed, size=size)
    img = Image.fromarray(arr, "RGB")
    kwargs = {"quality": 92}
    if piexif is not None:
        kwargs["exif"] = _camera_exif_bytes()
    img.save(path, "JPEG", **kwargs)
    return path


def save_fake_image(path: str, seed: int = 0, size: int = 256) -> str:
    arr = fake_array(seed=seed, size=size)
    img = Image.fromarray(arr, "RGB")
    # No EXIF: mimics a scrubbed/synthetic file.
    img.save(path, "JPEG", quality=92)
    return path


def generate_fixtures(
    out_dir: str, n_per_class: int = 20, size: int = 256, seed: int = 0
) -> List[Tuple[str, int]]:
    """Write n_per_class real + n_per_class fake JPEGs. Returns [(path, label)]."""
    out = Path(out_dir)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)

    manifest: List[Tuple[str, int]] = []
    for i in range(n_per_class):
        rp = str(out / "real" / f"real_{i:04d}.jpg")
        fp = str(out / "fake" / f"fake_{i:04d}.jpg")
        save_real_image(rp, seed=seed + i, size=size)
        save_fake_image(fp, seed=seed + i, size=size)
        manifest.append((rp, 0))
        manifest.append((fp, 1))
    return manifest
