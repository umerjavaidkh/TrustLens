import numpy as np
from PIL import Image

from trustlens.heuristics.classifier import (
    HeuristicClassifier,
    extract_features,
    features_to_vector,
    FEATURE_NAMES,
)
from trustlens.metrics import evaluate_at_fpr, threshold_for_fpr


def test_extract_features_shape(real_image):
    feats = extract_features(real_image)
    assert list(feats.keys()) == FEATURE_NAMES
    vec = features_to_vector(feats)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_default_classifier_probability_bounds(real_image, fake_image):
    clf = HeuristicClassifier.default()
    for img in (real_image, fake_image):
        p = clf.predict_proba(img)
        assert 0.0 <= p <= 1.0


def test_default_classifier_separates_classes(dataset):
    """Rule-based default should already rank fakes above reals on average."""
    clf = HeuristicClassifier.default()
    paths, labels = dataset
    probs = np.array([clf.score_path(p)["fake_probability"] for p in paths])
    labels = np.array(labels)
    assert probs[labels == 1].mean() > probs[labels == 0].mean()


def test_fitted_classifier_hits_target_fpr(dataset):
    """After calibration, recall at 1% FPR should be strong on the fixtures."""
    paths, labels = dataset
    X = np.array([features_to_vector(extract_features(Image.open(p))) for p in paths])
    y = np.array(labels)

    clf = HeuristicClassifier().fit(X, y)
    scores = clf.predict_proba_features(X)

    res = evaluate_at_fpr(y, scores, target_fpr=0.01)
    assert res.fpr <= 0.01 + 1e-9
    assert res.recall >= 0.8  # separable fixtures -> high recall at tight FPR

    clf.set_threshold(threshold_for_fpr(y, scores, target_fpr=0.01))
    assert 0.0 <= clf.threshold <= 1.0


def test_save_load_roundtrip(tmp_path, dataset):
    paths, labels = dataset
    X = np.array([features_to_vector(extract_features(Image.open(p))) for p in paths])
    clf = HeuristicClassifier().fit(X, labels)
    out = tmp_path / "model.joblib"
    clf.save(str(out))
    reloaded = HeuristicClassifier.load(str(out))
    np.testing.assert_allclose(
        reloaded.predict_proba_features(X), clf.predict_proba_features(X)
    )
