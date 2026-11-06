"""Window-level time-domain features: heart rate, variability and statistics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..preprocessing.segmentation import detect_peaks

TIME_FEATURES = ("hr", "sdnn", "rmssd", "win_skew", "win_kurt")


def time_features(x: np.ndarray, fs: float) -> dict[str, float]:
    peaks = detect_peaks(x, fs)
    rr = np.diff(peaks) / fs
    f = dict.fromkeys(TIME_FEATURES, np.nan)
    if rr.size >= 2:
        f["hr"] = 60.0 / float(np.median(rr))
        f["sdnn"] = float(rr.std())
        f["rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    f["win_skew"] = float(stats.skew(x))
    f["win_kurt"] = float(stats.kurtosis(x))
    return f
