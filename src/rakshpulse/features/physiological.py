"""Pulse-contour features beyond ``morphological.py`` (RakshPulse 4.0).

``morphological.py`` already provides widths, systolic/diastolic times (rise and fall time), areas, slopes and
second-derivative (APG) ratios per beat. This module adds the contour landmarks that are usually discussed for
arterial pulse waves, computed on the **median beat template** of a window. Each beat is detrended
foot-to-foot, scaled to 0–1 and resampled to ``N_TEMPLATE`` points, so the features are amplitude- and
heart-rate-normalised.

* ``notch_present``, ``notch_t``, ``notch_h``: dicrotic notch, i.e. the first local minimum after the systolic
  peak, in the window 15–75 % of the beat. If there is no minimum (common in finger and ear PPG, where the notch
  is often only an inflection), the point of maximum curvature of the falling limb is used and
  ``notch_present = 0``. ``notch_t`` is the time fraction of the beat; ``notch_h`` is the height relative to the
  peak.
* ``dia_peak_h``, ``dia_peak_t``: highest point after the notch (relative height and time; reflection index).
* ``sys_dia_dt``: time from systolic to diastolic peak, as a beat fraction (stiffness-index analogue; it needs a
  height to become a stiffness index, which is not available here).
* ``ipa``: inflection-point area ratio, diastolic area after the notch / systolic area before it.
* ``upstroke_10_90``: 10 %→90 % rise time as a beat fraction (upstroke morphology).
* ``decay_tau``: exponential time constant of the diastolic run-off from the notch to the end of the beat, as a
  beat fraction (log-linear fit).
* ``beat_amp_cv``: beat-to-beat coefficient of variation of foot-to-peak amplitude (pulse amplitude variation).
* ``beat_template_corr``: median correlation of individual beats with the window template (morphology
  similarity / beat-to-beat stability).

Physiological plausibility is **not** evidence of predictive value; see RESEARCH_FINDINGS (4.0).
"""

from __future__ import annotations

import numpy as np

from ..preprocessing.segmentation import split_beats

N_TEMPLATE = 100
PHYSIO_FEATURES = [
    "notch_present",
    "notch_t",
    "notch_h",
    "dia_peak_h",
    "dia_peak_t",
    "sys_dia_dt",
    "ipa",
    "upstroke_10_90",
    "decay_tau",
    "beat_amp_cv",
    "beat_template_corr",
]


def _unit_beat(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float)
    b = b - np.linspace(b[0], b[-1], b.size)
    r = b.max() - b.min()
    b = (b - b.min()) / r if r > 0 else np.zeros_like(b)
    return np.interp(np.linspace(0, b.size - 1, N_TEMPLATE), np.arange(b.size), b)


def window_template(x: np.ndarray, fs: float) -> tuple[np.ndarray | None, dict[str, float]]:
    """Median unit-beat template of a window plus beat-to-beat statistics (``None`` if < 3 beats)."""
    beats = split_beats(np.asarray(x, float), fs)
    if len(beats) < 3:
        return None, {"beat_amp_cv": np.nan, "beat_template_corr": np.nan}
    amps = np.array([b.max() - b[0] for b in beats])
    units = np.vstack([_unit_beat(b) for b in beats])
    tpl = np.median(units, axis=0)
    corr = [np.corrcoef(u, tpl)[0, 1] for u in units if u.std() > 0]
    stats = {
        "beat_amp_cv": float(amps.std(ddof=1) / amps.mean()) if amps.mean() > 0 else np.nan,
        "beat_template_corr": float(np.median(corr)) if corr else np.nan,
    }
    return tpl, stats


def template_features(t: np.ndarray) -> dict[str, float]:
    """Contour landmarks of a unit template (length ``N_TEMPLATE``)."""
    n = t.size
    out = dict.fromkeys(PHYSIO_FEATURES[:-2], np.nan)
    pk = int(np.argmax(t))
    if pk < 3 or pk > 0.7 * n:
        return out
    lo, hi = max(pk + 2, int(0.15 * n)), int(0.75 * n)
    seg = t[lo:hi]
    if seg.size < 5:
        return out
    d1 = np.diff(seg)
    minima = np.flatnonzero((d1[:-1] < 0) & (d1[1:] >= 0)) + 1
    if minima.size:
        notch, present = lo + int(minima[0]), 1.0
    else:  # inflection: maximum of the second derivative on the falling limb
        notch, present = lo + int(np.argmax(np.diff(seg, 2))) + 1, 0.0
    after = t[notch:]
    dia = notch + int(np.argmax(after))
    rise = t[: pk + 1]
    i10 = int(np.argmax(rise >= 0.1))
    i90 = int(np.argmax(rise >= 0.9))
    tail = t[notch:]
    tail_t = np.arange(tail.size) / n
    ok = tail > 0.02
    tau = np.nan
    if ok.sum() >= 5:
        slope = np.polyfit(tail_t[ok], np.log(tail[ok]), 1)[0]
        tau = float(-1.0 / slope) if slope < 0 else np.nan
    a_sys = float(np.trapezoid(t[: notch + 1]))
    out.update(
        notch_present=present,
        notch_t=notch / n,
        notch_h=float(t[notch]),
        dia_peak_h=float(t[dia]),
        dia_peak_t=dia / n,
        sys_dia_dt=(dia - pk) / n,
        ipa=float(np.trapezoid(t[notch:])) / a_sys if a_sys > 0 else np.nan,
        upstroke_10_90=(i90 - i10) / n,
        decay_tau=tau,
    )
    return out


def physio_features(x: np.ndarray, fs: float) -> tuple[dict[str, float], np.ndarray | None]:
    """All PHYSIO_FEATURES of a processed window, plus its template (for morphology-drift analysis)."""
    tpl, beat_stats = window_template(x, fs)
    if tpl is None:
        return {**dict.fromkeys(PHYSIO_FEATURES, np.nan), **beat_stats}, None
    return {**template_features(tpl), **beat_stats}, tpl
