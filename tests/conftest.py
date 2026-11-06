"""Shared fixtures.

The synthetic PPG generator exists **only to unit-test signal processing code paths**
(known heart rate, known shape). It is never used to train or evaluate BP models.
"""

from __future__ import annotations

import numpy as np
import pytest


def synthetic_ppg(
    fs: float = 100.0,
    seconds: float = 30.0,
    hr_bpm: float = 72.0,
    seed: int = 0,
    noise: float = 0.01,
    raw_counts: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    period = 60.0 / hr_bpm
    phase = (t % period) / period
    # systolic + diastolic Gaussian pulses (classic two-Gaussian PPG model)
    pulse = np.exp(-((phase - 0.18) ** 2) / (2 * 0.06**2)) + 0.45 * np.exp(-((phase - 0.48) ** 2) / (2 * 0.09**2))
    x = pulse + 0.05 * np.sin(2 * np.pi * 0.2 * t) + noise * rng.standard_normal(t.size)
    if raw_counts:  # MAX30102-like: large DC, absorption -> inverted pulse
        return 120000.0 - 800.0 * x
    return x


@pytest.fixture
def ppg_pair():
    fs = 100.0
    ir = synthetic_ppg(fs, 30, 72, seed=1)
    red = 0.8 * synthetic_ppg(fs, 30, 72, seed=2) + 10000
    return red, ir, fs
