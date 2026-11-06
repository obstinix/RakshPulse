"""Auxiliary ECG-assisted features (pulse arrival time, PAT). **Research only.**

The MAX30102 does not measure ECG, so these features can never be inputs to the
RakshPulse device model. They are computed only to check whether timing information that
PPG alone lacks would change the conclusions. Results are reported separately from all
PPG-only results.

PAT = time from the ECG R-peak to the PPG foot (and to the max-upslope point) of the
following pulse, restricted to 80–500 ms.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..preprocessing import filtering, normalization
from ..preprocessing.segmentation import detect_feet, detect_peaks

PAT_FEATURES = ("pat_foot_ms", "pat_slope_ms", "pat_foot_sd_ms")


def r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    sos = signal.butter(3, [5, 25], btype="bandpass", fs=fs, output="sos")
    e = signal.sosfiltfilt(sos, ecg - np.median(ecg))
    # QRS polarity can be either sign on lead I depending on electrode contact
    if np.abs(np.percentile(e, 0.5)) > np.abs(np.percentile(e, 99.5)):
        e = -e
    thr = 0.5 * np.percentile(e, 99)
    pk, _ = signal.find_peaks(e, height=thr, distance=int(0.27 * fs))
    return pk


def pat_features(ecg: np.ndarray, ppg_raw: np.ndarray, fs: float) -> dict[str, float]:
    out = dict.fromkeys(PAT_FEATURES, np.nan)
    if ecg.size < 4 * fs:
        return out
    rp = r_peaks(ecg, fs) / fs
    x = normalization.orient_ppg(filtering.bandpass(ppg_raw - np.median(ppg_raw), fs), fs)
    pk = detect_peaks(x, fs)
    if pk.size < 3 or rp.size < 3:
        return out
    feet = detect_feet(x, pk) / fs
    dx = np.gradient(x)
    slopes = np.array(
        [f + np.argmax(dx[int(f * fs) : p]) / fs if p > int(f * fs) else np.nan for f, p in zip(feet, pk, strict=True)]
    )
    pf, ps = [], []
    for f, s_ in zip(feet, slopes, strict=True):
        prior = rp[rp < f]
        if prior.size == 0:
            continue
        d = f - prior[-1]
        if 0.08 <= d <= 0.5:
            pf.append(d * 1000)
            if np.isfinite(s_):
                ps.append((s_ - prior[-1]) * 1000)
    if len(pf) >= 3:
        out["pat_foot_ms"] = float(np.median(pf))
        out["pat_foot_sd_ms"] = float(np.std(pf))
    if len(ps) >= 3:
        out["pat_slope_ms"] = float(np.median(ps))
    return out
