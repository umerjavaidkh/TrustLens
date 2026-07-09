import numpy as np

from trustlens.heuristics.fft import fft_features


def test_fft_features_keys_and_finite(real_image):
    feats = fft_features(real_image, size=192)
    assert set(feats) == {"fft_highfreq_ratio", "fft_spectral_slope", "fft_peakiness"}
    for v in feats.values():
        assert np.isfinite(v)


def test_highfreq_ratio_in_unit_interval(real_image, fake_image):
    for img in (real_image, fake_image):
        r = fft_features(img, size=192)["fft_highfreq_ratio"]
        assert 0.0 <= r <= 1.0


def test_grid_artifact_raises_peakiness(real_image, fake_image):
    # The synthetic fake carries a periodic grid -> more spectral peakiness
    # than the broadband-textured real image.
    real = fft_features(real_image, size=192)
    fake = fft_features(fake_image, size=192)
    assert fake["fft_peakiness"] > real["fft_peakiness"]
