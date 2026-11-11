"""Shared fixtures.

The synthetic PPG generator exists **only to unit-test signal processing code paths**
(known heart rate, known shape). It is never used to evaluate BP models; the only models fitted on it are
tiny test bundles that exercise the streaming / export code paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


@pytest.fixture
def stream_bundle():
    """Tiny ear ridge bundle trained on synthetic windows (shared by the 3.0 and 4.0 streaming tests)."""
    from rakshpulse.evaluation.auricle_study import add_normalised_shape, feature_set
    from rakshpulse.inference.model_loader import ModelBundle
    from rakshpulse.inference.realtime import site_window_features
    from rakshpulse.models.model_factory import make_model

    rows = []
    for hr in (55, 75, 95):
        for seed in range(2):
            rows += site_window_features(synthetic_ppg(100, 10, hr, seed=seed, raw_counts=False), 100, "ear")[0]
    X = add_normalised_shape(pd.DataFrame(rows), "ear")
    cols = feature_set("hr_shape", "ear", [])
    y = np.linspace(100, 140, len(X))
    s, _ = make_model("ridge")
    d, _ = make_model("ridge")
    return ModelBundle(
        s.fit(X[cols].to_numpy(float), y),
        d.fit(X[cols].to_numpy(float), y * 0.6),
        cols,
        ("ear",),
        False,
        {},
        "stream-test",
    )
