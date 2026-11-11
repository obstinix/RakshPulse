"""PPG signal-quality assessment.

Every index is cheap enough to run on a microcontroller. Thresholds come from the
literature where one exists, and are otherwise set conservatively and documented
in docs/DATA_PREPROCESSING.md:

* Skewness SQI (Elgendi 2016, Bioengineering 3:21) ranked best among 8 SQIs for
  telling excellent PPG from acceptable/unfit PPG. It is **reported but not used as a
  hard gate**: on the Guilin young-adult cohort, visually excellent pulses with a broad
  systolic plateau have negative skewness. A fixed threshold there rejected nearly all
  clean windows (checked manually against waveform plots).
* Relative power in the cardiac band (the RSQI used in the Foggia dataset paper):
  power in [0.5Â·HR, 1.5Â·HR]-ish band versus 0.5â€“10 Hz.
* Template correlation: mean Pearson correlation of each beat with the median
  beat (Orphanidou et al. 2015 used 0.86 for ECG/PPG usability).
* Physiological plausibility: heart rate within 30â€“220 bpm and a beat-interval
  coefficient of variation < 0.3 (higher values usually mean missed or extra beats).
* Saturation/flatline: fraction of samples at the min/max rail or with zero
  derivative (ADC clipping, sensor off skin).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from ..features import moments
from .segmentation import MAX_HR_BPM, MIN_HR_BPM, detect_peaks, split_beats


@dataclass
class SQIResult:
    skewness: float
    cardiac_power_ratio: float
    template_corr: float
    heart_rate_bpm: float
    rr_cv: float
    flat_fraction: float
    n_beats: int
    acceptable: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


THRESHOLDS = {
    "cardiac_power_ratio_min": 0.35,
    "template_corr_min": 0.86,
    "rr_cv_max": 0.30,
    "flat_fraction_max": 0.05,
    "min_beats": 4,
}


def _resample_beat(b: np.ndarray, n: int = 64) -> np.ndarray:
    return signal.resample(b, n) if b.size != n else b


def template_correlation(beats: list[np.ndarray]) -> float:
    if len(beats) < 2:
        return 0.0
    m = np.stack([_resample_beat(b - b.mean()) for b in beats])
    tmpl = np.median(m, axis=0)
    cs = [np.corrcoef(row, tmpl)[0, 1] if row.std() > 0 and tmpl.std() > 0 else 0.0 for row in m]
    return float(np.nanmean(cs))


def cardiac_power_ratio(x: np.ndarray, fs: float, hr_hz: float | None) -> float:
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    total = p[(f >= 0.5) & (f <= 10)].sum()
    if total <= 0:
        return 0.0
    if hr_hz is None or not np.isfinite(hr_hz):
        band = (f >= MIN_HR_BPM / 60) & (f <= MAX_HR_BPM / 60)
    else:
        band = (f >= 0.7 * hr_hz) & (f <= 1.3 * hr_hz)
        band |= (f >= 1.7 * hr_hz) & (f <= 2.3 * hr_hz)  # include first harmonic
    return float(p[band].sum() / total)


def flat_fraction(raw: np.ndarray, min_run: int = 5) -> float:
    """Fraction of samples inside runs of >= ``min_run`` identical consecutive values.

    Saturated or disconnected ADCs produce such runs. A clean 15–18-bit PPG almost never
    does. Must be applied to the *original* samples (before resampling/filtering).
    """
    raw = np.asarray(raw)
    if raw.size < 2:
        return 1.0
    same = np.concatenate([[False], raw[1:] == raw[:-1]])
    flat = 0
    run = 0
    for s_ in same:
        run = run + 1 if s_ else 0
        if run == min_run - 1:
            flat += min_run
        elif run >= min_run:
            flat += 1
    return float(flat / raw.size)


def assess(x: np.ndarray, fs: float, raw: np.ndarray | None = None) -> SQIResult:
    """Assess a band-passed, upward-oriented PPG window.

    ``raw`` (optional) is the unfiltered window, used for clipping/flatline detection.
    """
    x = np.asarray(x, dtype=float)
    reasons: list[str] = []
    if x.size < 2 * fs or not np.all(np.isfinite(x)):
        return SQIResult(np.nan, 0.0, 0.0, np.nan, np.nan, 1.0, 0, False, ["too_short_or_nonfinite"])
    peaks = detect_peaks(x, fs)
    beats = split_beats(x, fs)
    rr = np.diff(peaks) / fs if peaks.size > 1 else np.array([])
    hr = 60.0 / np.median(rr) if rr.size else np.nan
    rr_cv = float(rr.std() / rr.mean()) if rr.size > 1 else np.nan
    skew = float(moments.skew(x))
    cpr = cardiac_power_ratio(x, fs, hr / 60 if np.isfinite(hr) else None)
    tc = template_correlation(beats)
    ff = flat_fraction(raw) if raw is not None else 0.0

    t = THRESHOLDS
    if len(beats) < t["min_beats"]:
        reasons.append("too_few_beats")
    if not (np.isfinite(hr) and MIN_HR_BPM <= hr <= MAX_HR_BPM):
        reasons.append("implausible_heart_rate")
    if not np.isfinite(rr_cv) or rr_cv > t["rr_cv_max"]:
        reasons.append("irregular_beat_intervals")
    if cpr < t["cardiac_power_ratio_min"]:
        reasons.append("low_cardiac_power")
    if tc < t["template_corr_min"]:
        reasons.append("low_template_correlation")
    if ff > t["flat_fraction_max"]:
        reasons.append("clipping_or_flatline")
    return SQIResult(skew, cpr, tc, float(hr), rr_cv, ff, len(beats), not reasons, reasons)
