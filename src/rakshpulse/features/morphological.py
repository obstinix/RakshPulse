"""Beat-morphology features from PPG and its derivatives.

These are the pulse-wave-analysis features most often used in cuffless BP work
(Elgendi et al. 2019 npj Digit Med "The use of PPG for clinical BP measurement";
Kachuee et al. 2017 IEEE TBME; Charlton et al. 2022 "Assessing hemodynamics from the PPG").
All features are computed on a single amplitude-normalised beat (foot→next foot), so they
do not depend on sensor gain. That gain is not comparable across devices or body sites.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

WIDTH_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)

BEAT_FEATURES = (
    ["beat_T", "sys_T", "dia_T", "sys_ratio"]
    + [f"w{int(level * 100)}" for level in WIDTH_LEVELS]
    + [f"w{int(level * 100)}_sd_ratio" for level in WIDTH_LEVELS]
    + ["area_sys", "area_dia", "area_ratio", "max_slope", "t_max_slope", "beat_skew", "beat_kurt"]
    + ["apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "aging_index", "t_apg_b"]
)


def _norm_beat(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    # remove linear foot-to-foot trend (baseline wander inside the beat)
    trend = np.linspace(b[0], b[-1], b.size)
    b = b - trend
    rng = b.max() - b.min()
    return (b - b.min()) / rng if rng > 0 else b * 0


def _width(b: np.ndarray, pk: int, level: float, fs: float) -> tuple[float, float]:
    above = b >= level
    left = pk
    while left > 0 and above[left - 1]:
        left -= 1
    right = pk
    while right < b.size - 1 and above[right + 1]:
        right += 1
    sys_w = (pk - left) / fs
    dia_w = (right - pk) / fs
    return sys_w + dia_w, (dia_w / sys_w) if sys_w > 0 else np.nan


def _apg_points(apg: np.ndarray, pk: int) -> dict[str, float]:
    """Locate a,b,c,d,e waves of the second-derivative PPG (Takazawa 1998)."""
    out = {k: np.nan for k in ("a", "b", "c", "d", "e", "tb")}
    if pk < 3:
        return out
    search_end = min(apg.size, int(pk + 0.6 * (apg.size - pk)))
    ia = int(np.argmax(apg[: pk + 1]))
    out["a"] = apg[ia]
    seg = apg[ia:search_end]
    if seg.size < 4:
        return out
    # successive alternating extrema after a: b (min), c (max), d (min), e (max)
    d1 = np.diff(seg)
    ext = np.where(np.diff(np.sign(d1)) != 0)[0] + 1
    vals = [(i, seg[i]) for i in ext]
    names = ["b", "c", "d", "e"]
    want_min = True
    for i, v in vals:
        if not names:
            break
        is_min = d1[i - 1] < 0 <= d1[i] if i < d1.size else False
        if is_min == want_min:
            key = names.pop(0)
            out[key] = v
            if key == "b":
                out["tb"] = float(ia + i)
            want_min = not want_min
    return out


def beat_features(beat: np.ndarray, fs: float) -> dict[str, float]:
    b = _norm_beat(beat)
    n = b.size
    pk = int(np.argmax(b))
    f: dict[str, float] = {}
    f["beat_T"] = n / fs
    f["sys_T"] = pk / fs
    f["dia_T"] = (n - pk) / fs
    f["sys_ratio"] = pk / n
    for level in WIDTH_LEVELS:
        w, r = _width(b, pk, level, fs)
        f[f"w{int(level * 100)}"] = w
        f[f"w{int(level * 100)}_sd_ratio"] = r
    a_sys = np.trapezoid(b[: pk + 1]) / fs
    a_dia = np.trapezoid(b[pk:]) / fs
    f["area_sys"], f["area_dia"] = a_sys, a_dia
    f["area_ratio"] = a_dia / a_sys if a_sys > 0 else np.nan
    vpg = np.gradient(b) * fs
    f["max_slope"] = float(vpg.max())
    f["t_max_slope"] = float(np.argmax(vpg)) / fs
    f["beat_skew"] = float(stats.skew(b))
    f["beat_kurt"] = float(stats.kurtosis(b))
    apg = np.gradient(vpg) * fs
    p = _apg_points(apg, pk)
    a = p["a"]
    ok = np.isfinite(a) and a > 0
    for k in ("b", "c", "d", "e"):
        f[f"apg_{k}_a"] = p[k] / a if ok else np.nan
    f["aging_index"] = (p["b"] - p["c"] - p["d"] - p["e"]) / a if ok else np.nan
    f["t_apg_b"] = p["tb"] / fs if np.isfinite(p["tb"]) else np.nan
    return f


def aggregate_beats(beats: list[np.ndarray], fs: float) -> dict[str, float]:
    """Median (robust to residual bad beats) of per-beat features."""
    if not beats:
        return dict.fromkeys(BEAT_FEATURES, np.nan)
    rows = [beat_features(b, fs) for b in beats]
    return {
        k: float(np.nanmedian([r[k] for r in rows])) if any(np.isfinite(r[k]) for r in rows) else np.nan
        for k in BEAT_FEATURES
    }


def mean_beat(beats: list[np.ndarray], n: int = 100) -> np.ndarray:
    """Amplitude-normalised ensemble-average beat resampled to ``n`` points."""
    from scipy import signal

    if not beats:
        return np.full(n, np.nan)
    return np.mean([signal.resample(_norm_beat(b), n) for b in beats], axis=0)
