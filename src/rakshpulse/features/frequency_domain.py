"""Spectral features of a PPG window: fundamental frequency, harmonic structure, entropy.

Relative harmonic amplitudes describe pulse shape independently of beat detection, so
they stay usable when fiducial points are unreliable.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

FREQ_FEATURES = ("f0", "h2_h1", "h3_h1", "spec_entropy", "pow_0p5_2", "pow_2_5", "pow_5_10")


def frequency_features(x: np.ndarray, fs: float) -> dict[str, float]:
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(8 * fs)), nfft=max(1024, len(x)))
    band = (f >= 0.5) & (f <= 10)
    out = dict.fromkeys(FREQ_FEATURES, np.nan)
    if not band.any() or p[band].sum() <= 0:
        return out
    fb, pb = f[band], p[band]
    hr_band = (fb >= 0.5) & (fb <= 3.7)
    f0 = fb[hr_band][np.argmax(pb[hr_band])]
    out["f0"] = float(f0)

    def amp_at(freq: float) -> float:
        m = (fb >= freq - 0.15) & (fb <= freq + 0.15)
        return float(np.sqrt(pb[m].max())) if m.any() else np.nan

    a1 = amp_at(f0)
    out["h2_h1"] = amp_at(2 * f0) / a1 if a1 > 0 else np.nan
    out["h3_h1"] = amp_at(3 * f0) / a1 if a1 > 0 else np.nan
    pn = pb / pb.sum()
    out["spec_entropy"] = float(-(pn * np.log(pn + 1e-20)).sum() / np.log(pn.size))
    tot = pb.sum()
    out["pow_0p5_2"] = float(pb[(fb < 2)].sum() / tot)
    out["pow_2_5"] = float(pb[(fb >= 2) & (fb < 5)].sum() / tot)
    out["pow_5_10"] = float(pb[(fb >= 5)].sum() / tot)
    return out
