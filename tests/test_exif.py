from trustlens.heuristics.exif import exif_features


def test_real_image_has_exif_low_suspicion(real_image):
    feats = exif_features(real_image)
    assert feats["exif_present"] == 1.0
    assert feats["exif_suspicion"] < 0.5


def test_fake_image_missing_exif_high_suspicion(fake_image):
    feats = exif_features(fake_image)
    assert feats["exif_present"] == 0.0
    assert feats["exif_suspicion"] >= 0.8


def test_suspicion_is_bounded(real_image, fake_image):
    for img in (real_image, fake_image):
        s = exif_features(img)["exif_suspicion"]
        assert 0.0 <= s <= 1.0
