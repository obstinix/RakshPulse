"""Graded PPG quality score (RakshPulse 4.0), extending the binary gate in ``signal_quality.assess``.

Classes:
1. ``good``: passes every gate and every margin below;
2. ``degraded``: passes the existing acceptance gate but misses a margin (potentially usable);
3. ``unreliable``: fails the existing gate (as before 4.0: too few beats, implausible HR, irregular intervals,
   low cardiac power, low template correlation, clipping/flat runs) or is rail-clipped;
4. ``missing``: no usable sensor signal (non-finite samples, raw SD below ``flat_sd``, i.e. detached or
   disconnected sensor; or no detectable beats at all).

Components (all computed from the window itself; amplitude is relative to a reference amplitude supplied by the
caller, e.g. the running median of the stream, because absolute PPG amplitude depends on gain and placement):

==========================  ====================================================  ==========================
component                   meaning                                               degraded margin
==========================  ====================================================  ==========================
``template_corr``           beat-to-template similarity (motion, morphology)      < 0.95
``cardiac_power_ratio``     power at HR ± harmonic / 0.5–10 Hz power (SNR)          < 0.60
``rr_cv``                   inter-beat-interval CV (irregular pulse, missed beats)  > 0.15
``spike_frac``              fraction of samples with robust |z| > 5 (motion spikes) > 0.005
``baseline_ratio``          raw power < 0.5 Hz / power 0.5–10 Hz (unstable baseline) > 3.0
``hf_noise_ratio``          raw power 10 Hz–Nyquist / 0.5–10 Hz power (broadband      > 0.10
                            noise; ambient-light flicker would appear here only if
                            it aliases into this band; it is not separately
                            detectable without an ambient channel)
``rel_amplitude``           pulse amplitude / reference amplitude (low perfusion   < 0.30
                            or poor contact)
``plateau_frac``            samples within 0.5 % of the window's max or min in     > 0.02
                            runs ≥ 5 (saturation / clipping)
==========================  ====================================================  ==========================

The margins were pre-specified from the existing gate thresholds and common SQI practice. They were **not**
tuned on prediction errors; ``scripts/run_quality_study.py`` evaluates them against errors afterwards.

``score`` (0–100) is 100 × the geometric mean of per-component subscores in [0, 1], each 1 at or better than its
"good" margin and falling linearly to 0 at the gate (or at an equivalent bad level). Missing → 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal

from . import signal_quality as sq

CLASSES = ("good", "degraded", "unreliable", "missing")
MARGINS = {
    "template_corr_min": 0.95,
    "cardiac_power_ratio_min": 0.60,
    "rr_cv_max": 0.15,
    "spike_frac_max": 0.005,
    "baseline_ratio_max": 3.0,
    "hf_noise_ratio_max": 0.10,
    "rel_amplitude_min": 0.30,
    "plateau_frac_max": 0.02,
}
# levels at which a subscore reaches 0 (gate thresholds where one exists)
BAD = {
    "template_corr": sq.THRESHOLDS["template_corr_min"],
    "cardiac_power_ratio": sq.THRESHOLDS["cardiac_power_ratio_min"],
    "rr_cv": sq.THRESHOLDS["rr_cv_max"],
    "spike_frac": 0.03,
    "baseline_ratio": 20.0,
    "hf_noise_ratio": 0.5,
    "rel_amplitude": 0.05,
    "plateau_frac": 0.10,
}


@dataclass
class QualityResult:
    score: float
    quality_class: str
    components: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=min(len(x), int(4 * fs)))
    m = (f >= lo) & (f < hi)
    return float(p[m].sum()) if m.any() else 0.0


def plateau_fraction(raw: np.ndarray, rel_tol: float = 0.005, min_run: int = 5) -> float:
    """Fraction of samples in runs ≥ ``min_run`` lying within ``rel_tol`` of the window max or min."""
    r = np.asarray(raw, float)
    span = r.max() - r.min()
    if span <= 0:
        return 1.0
    at = (r >= r.max() - rel_tol * span) | (r <= r.min() + rel_tol * span)
    total, run = 0, 0
    for a in np.r_[at, False]:
        if a:
            run += 1
        else:
            total += run if run >= min_run else 0
            run = 0
    return total / r.size


def components(proc: np.ndarray, raw: np.ndarray, fs: float, ref_amplitude: float | None = None) -> dict:
    """All quality components of one window (``proc`` band-passed and oriented; ``raw`` unfiltered)."""
    s = sq.assess(proc, fs, raw=None)
    p = np.asarray(proc, float)
    r = np.asarray(raw, float)
    med = np.median(p)
    mad = 1.4826 * np.median(np.abs(p - med)) or (p.std() or 1.0)
    cardiac = _band_power(r, fs, 0.5, 10.0)
    nyq = fs / 2
    beats = sq.split_beats(r - np.median(r), fs) if r.std() > 0 else []
    amp = float(np.median([b.max() - b.min() for b in beats])) if beats else float(np.ptp(r))
    return {
        "template_corr": s.template_corr,
        "cardiac_power_ratio": s.cardiac_power_ratio,
        "rr_cv": s.rr_cv,
        "n_beats": s.n_beats,
        "heart_rate_bpm": s.heart_rate_bpm,
        "spike_frac": float(np.mean(np.abs(p - med) / mad > 5)),
        "baseline_ratio": _band_power(r, fs, 0.0, 0.5) / cardiac if cardiac > 0 else np.inf,
        "hf_noise_ratio": _band_power(r, fs, 10.0, nyq) / cardiac if cardiac > 0 and nyq > 10 else 0.0,
        "pulse_amplitude": amp,
        "rel_amplitude": amp / ref_amplitude if ref_amplitude else np.nan,
        "plateau_frac": plateau_fraction(r),
        "raw_sd": float(r.std()),
        "gate_reasons": s.reasons,
    }


def _sub(name: str, v: float) -> float:
    """Subscore in [0, 1]: 1 at the good margin, 0 at BAD, linear in between."""
    if not np.isfinite(v):
        return 1.0 if name == "rel_amplitude" else 0.0  # unknown reference amplitude: not penalised
    hi_good = name in ("template_corr", "cardiac_power_ratio", "rel_amplitude")
    good = MARGINS[f"{name}_{'min' if hi_good else 'max'}"]
    bad = BAD[name]
    t = (v - bad) / (good - bad)
    return float(np.clip(t, 0.0, 1.0))


def score_window(
    proc: np.ndarray,
    raw: np.ndarray,
    fs: float,
    ref_amplitude: float | None = None,
    flat_sd: float = 0.003,
    rail_pct: float | None = None,
    rail_max_pct: float = 1.0,
) -> QualityResult:
    """Grade one window. ``rail_pct`` (optional): % samples at the amplifier rail, if the device reports it."""
    raw = np.asarray(raw, float)
    if raw.size == 0 or not np.isfinite(raw).all() or not np.isfinite(proc).all():
        return QualityResult(0.0, "missing", {}, ["non_finite_samples"])
    if raw.std() < flat_sd:
        return QualityResult(0.0, "missing", {"raw_sd": float(raw.std())}, ["no_signal_flatline"])
    c = components(proc, raw, fs, ref_amplitude)
    return grade(c, rail_pct, rail_max_pct)


def grade(c: dict, rail_pct: float | None = None, rail_max_pct: float = 1.0) -> QualityResult:
    """Class and score from precomputed ``components`` (lets callers update ``rel_amplitude`` causally)."""
    if c.get("n_beats", 0) == 0:
        return QualityResult(0.0, "missing", c, ["no_beats_detected"])
    reasons = list(c["gate_reasons"])
    if rail_pct is not None and rail_pct > rail_max_pct:
        reasons.append("rail_clipping")
    names = list(BAD)
    subs = {n: _sub(n, c[n]) for n in names}
    score = 100.0 * float(np.exp(np.mean(np.log(np.clip(list(subs.values()), 1e-3, 1.0)))))
    if reasons:
        return QualityResult(min(score, 49.0), "unreliable", c, reasons)
    margin_fail = [n for n in names if subs[n] < 1.0]
    if margin_fail:
        return QualityResult(min(score, 89.0), "degraded", c, [f"margin:{n}" for n in margin_fail])
    return QualityResult(score, "good", c, [])
