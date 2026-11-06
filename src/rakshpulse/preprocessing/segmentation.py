"""Window and beat segmentation.

* Windows: 10 s with 50 % overlap by default. Ten seconds holds 8–30 beats at physiological
  heart rates, matching the PulseDB/MIMIC convention and the Foggia recording length.
  Overlap only multiplies correlated windows from one subject; subject-wise splitting
  (see evaluation.validation) keeps those windows from inflating test scores.
* Beats: systolic peaks are found with a refractory distance equal to the shortest
  plausible RR interval (220 bpm). Each beat runs between consecutive pulse feet
  (local minima before each peak).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

MAX_HR_BPM = 220.0
MIN_HR_BPM = 30.0


def sliding_windows(x: np.ndarray, fs: float, win_s: float = 10.0, step_s: float = 5.0) -> np.ndarray:
    """Return an array of shape (n_windows, ..., win) over the last axis."""
    x = np.asarray(x)
    win = int(round(win_s * fs))
    step = int(round(step_s * fs))
    n = x.shape[-1]
    if n < win:
        return np.empty((0, *x.shape[:-1], win))
    starts = range(0, n - win + 1, step)
    return np.stack([x[..., s : s + win] for s in starts])


def detect_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peak indices of a band-passed, upward-oriented PPG."""
    x = np.asarray(x, dtype=float)
    if x.size < fs:
        return np.array([], dtype=int)
    dist = max(1, int(fs * 60.0 / MAX_HR_BPM))
    prom = 0.3 * (np.percentile(x, 95) - np.percentile(x, 5))
    peaks, _ = signal.find_peaks(x, distance=dist, prominence=max(prom, 1e-12))
    edge = int(0.1 * fs)  # residual filter transients at the record edges
    return peaks[(peaks >= edge) & (peaks < x.size - edge)]


def detect_feet(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Pulse onset (foot) = minimum between consecutive peaks (before each peak)."""
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return peaks
    rr = int(np.median(np.diff(peaks))) if peaks.size > 1 else peaks[0]
    feet = []
    for i, p in enumerate(peaks):
        lo = peaks[i - 1] if i > 0 else max(0, p - rr)
        seg = x[lo : p + 1]
        feet.append(lo + int(np.argmin(seg)))
    return np.asarray(feet, dtype=int)


def split_beats(x: np.ndarray, fs: float) -> list[np.ndarray]:
    """Return a list of single beats (foot-to-next-foot) from a PPG window."""
    peaks = detect_peaks(x, fs)
    if peaks.size < 2:
        return []
    feet = detect_feet(x, peaks)
    beats = []
    min_len = int(fs * 60.0 / MAX_HR_BPM)
    max_len = int(fs * 60.0 / MIN_HR_BPM)
    for a, b in zip(feet[:-1], feet[1:], strict=False):
        if min_len <= b - a <= max_len:
            beats.append(x[a : b + 1])
    return beats
