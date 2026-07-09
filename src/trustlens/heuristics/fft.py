"""FFT spectral analysis.

Natural photographs have a characteristic ~1/f power spectrum: energy falls off
smoothly with frequency and the spectrum has no periodic structure. GAN/diffusion
upsamplers leave two fingerprints:

1. Excess **high-frequency energy** (checkerboard from transposed convolutions).
2. **Periodic spectral peaks** — a regular grid of spikes in the 2-D spectrum.

We compute the log-magnitude spectrum, reduce it to an azimuthally-averaged 1-D
radial profile, and derive three scalars: the high-frequency energy ratio, the
slope of the radial profile (spectral roll-off), and a "peakiness" score that
captures periodic grid artifacts.

References: Durall et al. 2020 (upconvolution artifacts); Zhang et al. 2019
(GAN spectral fingerprints).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def _to_gray_array(img: Image.Image, size: int = 512) -> np.ndarray:
    g = img.convert("L")
    # Resize longest side to `size` keeping it square-ish for a stable spectrum.
    g = g.resize((size, size), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32) / 255.0


def _radial_profile(power: np.ndarray) -> np.ndarray:
    """Azimuthally-averaged power as a function of radius (pixels from center)."""
    h, w = power.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), weights=power.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def fft_features(img: Image.Image, size: int = 512) -> Dict[str, float]:
    """Return FFT-derived scalars.

    - fft_highfreq_ratio : energy above 0.5 * Nyquist / total energy
    - fft_spectral_slope : slope of log-power vs log-radius (more negative = smoother)
    - fft_peakiness      : excess kurtosis of the de-trended radial profile,
                           elevated by periodic grid spikes
    """
    arr = _to_gray_array(img, size=size)

    # Hann window to suppress edge leakage before the transform.
    win = np.hanning(arr.shape[0])[:, None] * np.hanning(arr.shape[1])[None, :]
    f = np.fft.fftshift(np.fft.fft2(arr * win))
    power = np.abs(f) ** 2

    prof = _radial_profile(power)
    prof = prof[: len(prof) // 2]  # keep up to Nyquist
    total = prof.sum() + 1e-12

    # High-frequency energy ratio.
    half = len(prof) // 2
    highfreq_ratio = float(prof[half:].sum() / total)

    # Spectral slope: log-log linear fit over the informative mid/high band.
    radii = np.arange(1, len(prof))
    logr = np.log(radii)
    logp = np.log(prof[1:] + 1e-12)
    slope = float(np.polyfit(logr, logp, 1)[0])

    # Peakiness: fit the smooth roll-off, look at the residual's heavy tail.
    trend = np.polyval(np.polyfit(logr, logp, 1), logr)
    resid = logp - trend
    r = resid - resid.mean()
    denom = (r.std() ** 4) + 1e-12
    kurtosis = float((r ** 4).mean() / denom - 3.0)

    return {
        "fft_highfreq_ratio": highfreq_ratio,
        "fft_spectral_slope": slope,
        "fft_peakiness": kurtosis,
    }
