"""Tier-1 heuristic classifier.

Combines the ELA, EXIF and FFT scalar features into a single fake/fraud
probability. Two modes:

- **Rule-based default** (`HeuristicClassifier.default()`): hand-set weights in a
  logistic link. Works with zero training data — the true "ship the simple thing
  first" baseline.
- **Calibrated** (`.fit(X, y)`): a StandardScaler + LogisticRegression fitted on
  labelled features. Same feature set, learned weights.

Either way the output is a probability, thresholded at a chosen operating point
(default: the threshold that yields TARGET_FPR — set via `.set_threshold`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from ..config import settings
from .ela import ela_features
from .exif import exif_features
from .fft import fft_features

FEATURE_NAMES: List[str] = [
    "ela_mean",
    "ela_std",
    "ela_p99",
    "exif_present",
    "exif_suspicion",
    "fft_highfreq_ratio",
    "fft_spectral_slope",
    "fft_peakiness",
]

# Hand-tuned weights for the rule-based default (operating on raw features).
# Sign = direction of "more likely fake". Magnitudes are rough; `.fit` supersedes.
_DEFAULT_WEIGHTS = {
    "ela_mean": 1.5,
    "ela_std": 6.0,
    "ela_p99": 3.0,
    "exif_present": -1.2,
    "exif_suspicion": 2.5,
    "fft_highfreq_ratio": 4.0,
    "fft_spectral_slope": 0.6,   # less-negative slope -> flatter -> more synthetic
    "fft_peakiness": 0.15,
}
_DEFAULT_BIAS = -1.8


def load_image(path: str) -> Image.Image:
    return Image.open(path)


def extract_features(img: Image.Image) -> Dict[str, float]:
    """Run all Tier-1 extractors and return an ordered feature dict."""
    feats: Dict[str, float] = {}
    feats.update(ela_features(img, quality=settings.ela_quality))
    feats.update(exif_features(img))
    feats.update(fft_features(img, size=settings.work_size))
    return {k: feats[k] for k in FEATURE_NAMES}


def features_to_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _fit_logreg_numpy(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    epochs: int = 2000,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Balanced logistic regression via full-batch gradient descent (numpy only).

    Sample weights invert class frequency so the minority/expensive class (fakes)
    is not drowned out — mirrors sklearn's class_weight='balanced'.
    """
    n, d = X.shape
    y = y.astype(np.float64)
    # Balanced sample weights: n / (2 * n_c).
    n_pos = max(y.sum(), 1.0)
    n_neg = max(n - y.sum(), 1.0)
    w_pos, w_neg = n / (2.0 * n_pos), n / (2.0 * n_neg)
    sw = np.where(y == 1, w_pos, w_neg)

    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        err = (p - y) * sw
        grad_w = X.T @ err / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, float(b)


@dataclass
class HeuristicClassifier:
    """Feature -> fake-probability model with a tunable decision threshold."""

    threshold: float = 0.5
    _weights: np.ndarray = field(default=None, repr=False)
    _bias: float = 0.0
    _scaler_mean: np.ndarray = field(default=None, repr=False)
    _scaler_scale: np.ndarray = field(default=None, repr=False)
    fitted: bool = False

    # ---------------------------------------------------------------- factories
    @classmethod
    def default(cls) -> "HeuristicClassifier":
        w = np.array([_DEFAULT_WEIGHTS[k] for k in FEATURE_NAMES], dtype=np.float64)
        return cls(threshold=0.5, _weights=w, _bias=_DEFAULT_BIAS, fitted=False)

    # ---------------------------------------------------------------- inference
    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        """Fake probability from a (n, n_features) matrix."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self._scaler_mean is not None:
            X = (X - self._scaler_mean) / self._scaler_scale
        z = X @ self._weights + self._bias
        return _sigmoid(z)

    def predict_proba(self, img: Image.Image) -> float:
        feats = extract_features(img)
        return float(self.predict_proba_features(features_to_vector(feats))[0])

    def score_path(self, path: str) -> Dict[str, object]:
        """Full result for one image: probability, label and every feature."""
        img = load_image(path)
        feats = extract_features(img)
        prob = float(self.predict_proba_features(features_to_vector(feats))[0])
        return {
            "fake_probability": prob,
            "predicted_label": int(prob >= self.threshold),
            "threshold": self.threshold,
            "features": feats,
        }

    # ---------------------------------------------------------------- training
    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        use_sklearn: bool = True,
    ) -> "HeuristicClassifier":
        """Calibrate weights: standardize features, then fit balanced logistic
        regression. Uses scikit-learn if available, else a numpy fallback with
        identical semantics (balanced class weights honor the asymmetric cost of
        a missed fake).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        # Standardize (store params so inference reproduces the transform).
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Xs = (X - mean) / scale
        self._scaler_mean = mean
        self._scaler_scale = scale

        if use_sklearn:
            try:
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs, y)
                self._weights = lr.coef_.ravel().astype(np.float64)
                self._bias = float(lr.intercept_[0])
                self.fitted = True
                return self
            except Exception:  # noqa: BLE001 - fall back to numpy implementation
                pass

        self._weights, self._bias = _fit_logreg_numpy(Xs, y)
        self.fitted = True
        return self

    def set_threshold(self, threshold: float) -> "HeuristicClassifier":
        self.threshold = float(threshold)
        return self

    # ---------------------------------------------------------------- io
    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "HeuristicClassifier":
        import joblib

        return joblib.load(path)
