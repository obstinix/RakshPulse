"""Artefact handling.

The selected datasets have no accelerometer channel, so adaptive motion cancellation
(e.g. NLMS with an IMU reference) cannot be implemented or validated here. What the data
does support:

* replacing non-finite samples by linear interpolation (short gaps only);
* Hampel despiking (median ± k·MAD) for isolated spikes from ADC/transmission glitches;
* beat-level outlier rejection: beats whose shape deviates from the window template are
  dropped before features are averaged.

Windows that remain poor after this are rejected by the signal-quality gate. They are
never "repaired".
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def interpolate_nonfinite(x: np.ndarray, max_gap_fraction: float = 0.05) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.mean() > max_gap_fraction:
        raise ValueError(f"{bad.mean():.1%} of samples are non-finite (limit {max_gap_fraction:.0%})")
    idx = np.arange(x.size)
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def hampel(x: np.ndarray, half_window: int = 5, k: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    win = 2 * half_window + 1
    if x.size < win:
        return x.copy()
    med = signal.medfilt(x, win)
    mad = signal.medfilt(np.abs(x - med), win) * 1.4826
    out = x.copy()
    spikes = np.abs(x - med) > k * np.maximum(mad, 1e-12)
    out[spikes] = med[spikes]
    return out


def reject_outlier_beats(beats: list[np.ndarray], n: int = 64, min_corr: float = 0.8) -> list[np.ndarray]:
    if len(beats) < 3:
        return beats
    m = np.stack([signal.resample(b - b.mean(), n) for b in beats])
    tmpl = np.median(m, axis=0)
    keep = []
    for b, row in zip(beats, m, strict=True):
        if row.std() > 0 and np.corrcoef(row, tmpl)[0, 1] >= min_corr:
            keep.append(b)
    return keep
