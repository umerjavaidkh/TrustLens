import numpy as np
from PIL import Image

from trustlens.heuristics.ela import ela_features, error_level_image


def test_error_level_image_shape_and_range(real_image):
    mag = error_level_image(real_image, quality=90)
    assert mag.ndim == 2
    assert mag.shape == real_image.size[::-1]  # (H, W)
    assert mag.min() >= 0.0
    assert mag.max() <= 255.0


def test_ela_features_keys_and_bounds(real_image):
    feats = ela_features(real_image, quality=90)
    assert set(feats) == {"ela_mean", "ela_std", "ela_p99"}
    for v in feats.values():
        assert 0.0 <= v <= 1.0
        assert np.isfinite(v)


def test_ela_low_on_recompression_idempotence():
    # A smooth image already saved at the ELA quality re-encodes with very low
    # error (JPEG is near-idempotent on content it already represents well).
    import io

    x = np.linspace(0, 255, 64)
    grad = np.tile(x, (64, 1))
    arr = np.stack([grad, grad, grad], axis=2).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=90)
    buf.seek(0)
    img = Image.open(buf)
    feats = ela_features(img, quality=90)
    assert feats["ela_mean"] < 0.01
