"""Fast sample skewness and excess kurtosis.

These match ``scipy.stats.skew(x)`` and ``scipy.stats.kurtosis(x)`` with their defaults (biased, Fisher), which
the feature pipeline used before 4.0. scipy's argument-handling wrapper cost about half of the per-window
feature time, because it is called once per beat. Constant input returns NaN, as scipy does.
"""

from __future__ import annotations

import numpy as np


def skew(x) -> float:
    x = np.asarray(x, float)
    d = x - x.mean()
    m2 = np.mean(d * d)
    if m2 <= 0 or not np.isfinite(m2):
        return float("nan")
    return float(np.mean(d**3) / m2**1.5)


def kurtosis(x) -> float:
    x = np.asarray(x, float)
    d = x - x.mean()
    m2 = np.mean(d * d)
    if m2 <= 0 or not np.isfinite(m2):
        return float("nan")
    return float(np.mean(d**4) / m2**2 - 3.0)
