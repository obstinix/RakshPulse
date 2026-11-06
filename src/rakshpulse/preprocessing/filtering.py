"""Band-pass filtering and resampling of PPG signals.

Parameter justification (see docs/DATA_PREPROCESSING.md for references):

* **Pass band 0.5–10 Hz.** Heart rate in adults spans roughly 30–220 bpm (0.5–3.7 Hz).
  The pulse morphology (systolic peak, dicrotic notch, diastolic wave) that BP models rely on
  is carried by the first few harmonics, which stay below about 10 Hz. Two of the selected
  datasets already apply hardware or software band limits at 0.5–12 Hz (PPG-BP) and
  0.5–10 Hz (Foggia). Using 0.5–10 Hz everywhere gives all datasets the same spectral support.
* **4th-order Butterworth, zero-phase (sosfiltfilt).** Zero phase keeps the timing features
  (rise time, notch position) free of phase distortion. Second-order sections are numerically
  stable at low normalised cut-offs.
* **Target rate 100 Hz.** The MAX30102 natively supports 100 sps, the Foggia dataset is
  100 Hz, and a 50 Hz Nyquist frequency is 5× the upper pass-band edge. Lower-rate data
  is never up-sampled to claim detail it does not contain.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

DEFAULT_BAND = (0.5, 10.0)
TARGET_FS = 100.0


def validate_sampling_rate(fs: float, min_fs: float = 2.5 * DEFAULT_BAND[1], max_fs: float = 10_000.0) -> float:
    """Check that ``fs`` is a finite positive number that supports the analysis band."""
    try:
        fs = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sampling rate must be numeric, got {fs!r}") from exc
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"sampling rate must be positive and finite, got {fs}")
    if fs < min_fs:
        raise ValueError(f"sampling rate {fs} Hz too low: need >= {min_fs} Hz to preserve the {DEFAULT_BAND} Hz band")
    if fs > max_fs:
        raise ValueError(f"sampling rate {fs} Hz is implausibly high (> {max_fs} Hz)")
    return fs


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float] = DEFAULT_BAND, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along the last axis."""
    fs = validate_sampling_rate(fs)
    lo, hi = band
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f"invalid band {band} for fs={fs}")
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    x = np.asarray(x, dtype=float)
    # Odd-reflection padding of up to 2 s suppresses edge transients on short records
    # (PPG-BP segments are only 2.1 s long, comparable to the 0.5 Hz high-pass period).
    padlen = min(int(2 * fs), x.shape[-1] - 1)
    return signal.sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=padlen)


def resample(x: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """Polyphase resampling (with built-in anti-aliasing) along the last axis."""
    fs_in = validate_sampling_rate(fs_in)
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=float)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(np.asarray(x, dtype=float), frac.numerator, frac.denominator, axis=-1)


def lowpass_dc(x: np.ndarray, fs: float, cutoff: float = 0.3) -> np.ndarray:
    """Estimate the slowly varying DC (baseline) component, used for perfusion/AC-DC ratios."""
    sos = signal.butter(2, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)
