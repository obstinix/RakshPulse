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


def estimate_period(x: np.ndarray, fs: float) -> float | None:
    """Dominant beat period (s) from the autocorrelation, searched over 30–220 bpm."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    if x.size < 2 * fs or x.std() == 0:
        return None
    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    ac /= ac[0]
    lo, hi = int(fs * 60 / MAX_HR_BPM), min(int(fs * 60 / MIN_HR_BPM), x.size // 2)
    if hi <= lo + 1:
        return None
    pk, props = signal.find_peaks(ac[lo:hi], height=0.2)
    if pk.size == 0:
        return None
    return (lo + pk[np.argmax(props["peak_heights"])]) / fs


def detect_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peak indices of a band-passed, upward-oriented PPG.

    The refractory distance adapts to the autocorrelation beat period (60 % of it). A fixed
    220-bpm distance lets a prominent diastolic wave count as an extra beat at low heart
    rates (e.g. 50 bpm in young, compliant arteries).
    """
    x = np.asarray(x, dtype=float)
    if x.size < fs:
        return np.array([], dtype=int)
    period = estimate_period(x, fs)
    dist = max(1, int(fs * 60.0 / MAX_HR_BPM))
    if period is not None:
        dist = max(dist, int(0.6 * period * fs))
    prom = 0.3 * (np.percentile(x, 95) - np.percentile(x, 5))
    peaks, _ = signal.find_peaks(x, distance=dist, prominence=max(prom, 1e-12))
    edge = int(0.1 * fs)  # residual filter transients at the record edges
    return peaks[(peaks >= edge) & (peaks < x.size - edge)]


def detect_feet(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Pulse onset (foot) = minimum between consecutive peaks (before each peak)."""
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return peaks
    rr = int(np.median(np.diff(peaks))) if peaks.size > 1 else int(peaks[0])
    dx = np.gradient(x)
    feet = []
    for i, p in enumerate(peaks):
        lo = int(peaks[i - 1]) if i > 0 else max(0, int(p) - rr)
        if p - lo < 2:
            feet.append(lo)
            continue
        # Intersecting-tangent foot: the tangent at the steepest upstroke point meets the
        # horizontal line through the preceding minimum. An argmin alone wanders inside a flat
        # diastolic floor and adds jitter to every timing feature.
        u = lo + int(np.argmax(dx[lo:p]))
        a = max(lo, u - int(0.3 * rr))
        floor = float(np.min(x[a : u + 1]))
        slope = float(dx[u])
        foot = u - (x[u] - floor) / slope if slope > 0 else a + int(np.argmin(x[a : u + 1]))
        feet.append(int(np.clip(round(foot), a, u)))
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
