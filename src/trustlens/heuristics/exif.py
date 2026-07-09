"""EXIF-consistency heuristics.

Authentic camera captures carry a rich, self-consistent EXIF block: a camera
make/model, a capture timestamp, exposure settings, and dimensions that match
the pixel data. Synthetic images (GAN/diffusion) and many manipulations either
strip EXIF entirely or leave tell-tale editor signatures ("Adobe Photoshop",
"GIMP") and mismatched dimensions. We convert those signals into a single
suspicion score in [0, 1] plus a presence flag.

Note: EXIF is trivially forgeable, so this is *supporting* evidence, never proof.
"""
from __future__ import annotations

from typing import Dict

from PIL import Image
from PIL.ExifTags import TAGS

# Software strings that indicate an editing/generation tool touched the file.
_EDITOR_SIGNATURES = (
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "stable diffusion", "midjourney", "dall", "firefly", "generated",
)


def _exif_dict(img: Image.Image) -> Dict[str, object]:
    raw = img.getexif()
    if not raw:
        return {}
    out: Dict[str, object] = {}
    for tag_id, value in raw.items():
        out[TAGS.get(tag_id, tag_id)] = value
    return out


def exif_features(img: Image.Image) -> Dict[str, float]:
    """Return {'exif_present', 'exif_suspicion'} with values in [0, 1]."""
    exif = _exif_dict(img)

    if not exif:
        # No EXIF at all: strong (but not conclusive) synthetic/scrubbed signal.
        return {"exif_present": 0.0, "exif_suspicion": 0.9}

    suspicion = 0.0

    # Missing core camera provenance fields.
    if not (exif.get("Make") or exif.get("Model")):
        suspicion += 0.35
    if not (exif.get("DateTimeOriginal") or exif.get("DateTime")):
        suspicion += 0.20

    # Editor / generator signature in the Software field.
    software = str(exif.get("Software", "")).lower()
    if any(sig in software for sig in _EDITOR_SIGNATURES):
        suspicion += 0.45

    # Dimension mismatch between EXIF and actual pixels.
    exif_w = exif.get("ExifImageWidth")
    exif_h = exif.get("ExifImageHeight")
    if exif_w and exif_h and (int(exif_w), int(exif_h)) != img.size:
        suspicion += 0.25

    return {"exif_present": 1.0, "exif_suspicion": float(min(suspicion, 1.0))}
