"""Additional per-window features used by RakshPulse 3.0 experiments.

* ``pat_*``: ECG→PPG pulse arrival time (R-peak to PPG foot / maximum-upslope point), median and SD over
  the window. These need a simultaneous ECG (available in Auricle and VitalDB, **not** on the MAX30102), so they
  are only used in explicitly labelled ECG-assisted experiments.
* ``wav_e1``..``wav_e6``: relative energy of the PPG in Haar wavelet detail levels 1–6 (at 100 Hz: ~25–50 Hz down
  to ~0.8–1.6 Hz) plus the approximation. This is a dependency-free discrete wavelet decomposition of the
  processed (band-passed, z-scored) window.
* ``ppg_ac_dc``-style perfusion is unavailable for AC-coupled sources and is therefore not computed here.
"""

from __future__ import annotations

import numpy as np

from .ecg_aux import pat_features

PAT_COLUMNS = ("pat_foot_ms", "pat_slope_ms", "pat_foot_sd_ms")
WAVELET_COLUMNS = tuple(f"wav_e{i}" for i in range(1, 7)) + ("wav_a6",)


def haar_relative_energies(x: np.ndarray, levels: int = 6) -> dict[str, float]:
    """Relative energy per Haar DWT detail level (orthonormal, so energies sum to the signal energy)."""
    out = dict.fromkeys(WAVELET_COLUMNS, np.nan)
    a = np.asarray(x, dtype=float)
    if a.size < 2**levels or not np.isfinite(a).all():
        return out
    a = a[: (a.size // 2**levels) * 2**levels]
    energies = []
    for _ in range(levels):
        even, odd = a[0::2], a[1::2]
        energies.append(float(np.sum(((even - odd) / np.sqrt(2)) ** 2)))
        a = (even + odd) / np.sqrt(2)
    energies.append(float(np.sum(a**2)))
    total = sum(energies)
    if total <= 0:
        return out
    for i, e in enumerate(energies[:-1], start=1):
        out[f"wav_e{i}"] = e / total
    out["wav_a6"] = energies[-1] / total
    return out


def window_pat(
    ecg: np.ndarray | None, ppg_raw: np.ndarray, fs: float, range_ms: tuple[float, float] = (80.0, 500.0)
) -> dict[str, float]:
    """PAT features, or NaN when no usable ECG is available."""
    if ecg is None or not np.isfinite(ecg).all() or np.std(ecg) == 0:
        return dict.fromkeys(PAT_COLUMNS, np.nan)
    return pat_features(np.asarray(ecg, float), np.asarray(ppg_raw, float), fs, range_ms)
