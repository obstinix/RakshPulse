"""Synthetic red/IR PPG generator (RakshPulse 5.0). **Every output is simulated, not a measurement.**

The generator produces MAX30102-like raw ADC counts: an 18-bit full scale, a large DC level, and a pulsatile
component that *reduces* the count during systole (more blood absorbs more light). This lets the whole
acquisition → storage → processing → inference chain be exercised without hardware. It is meant for software
testing, tool validation and demonstrations. It must never be used as evidence about blood pressure:
``SIMULATED_FLAG`` marks the rows it creates, and the dataset store records ``sensor_type = "simulator"``.

Waveform model, per beat: a two-Gaussian systolic + reflected-wave pulse (a common phenomenological PPG model).
Shape parameters can drift slowly, so morphology changes can be simulated. Modelled effects:

* heart rate with slow drift and beat-to-beat variability (``hr_bpm``, ``hrv_sd_s``);
* respiration-induced baseline wander and amplitude modulation;
* AC amplitude as a fraction of DC per channel (perfusion), with red smaller than IR;
* white sensor noise, and optional 50/60 Hz ambient-light ripple;
* motion artefacts: bursts of large low-frequency transients at configurable times;
* saturation: counts are clamped to the ADC range (with a configurable DC so clipping can be forced);
* dropouts: whole missing samples (gaps in the sample counter), and timestamp jitter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SIMULATED_FLAG = "simulated"
ADC_MAX_18BIT = 2**18 - 1


@dataclass
class SimConfig:
    fs: float = 100.0
    hr_bpm: float = 72.0
    hr_drift_bpm: float = 4.0  # slow sinusoidal HR drift amplitude (period ``hr_drift_period_s``)
    hr_drift_period_s: float = 120.0
    hrv_sd_s: float = 0.02  # beat-to-beat interval jitter (s)
    resp_rate_hz: float = 0.25
    resp_baseline_frac: float = 0.002  # respiratory baseline wander, fraction of DC
    resp_amp_mod: float = 0.1  # respiratory modulation of the pulse amplitude (fraction)
    dc_ir: float = 120_000.0
    dc_red: float = 90_000.0
    perfusion_ir: float = 0.012  # AC/DC
    perfusion_red: float = 0.007
    reflection: float = 0.45  # relative height of the reflected (diastolic) wave
    reflection_delay: float = 0.30  # beat fraction between the systolic and the reflected wave
    noise_frac: float = 0.0004  # white noise SD, fraction of DC
    mains_hz: float | None = None  # e.g. 50 or 60: ambient-light ripple
    mains_frac: float = 0.0
    motion_times_s: tuple[float, ...] = ()  # centre times of motion bursts
    motion_duration_s: float = 3.0
    motion_frac: float = 0.03  # burst amplitude, fraction of DC
    adc_max: float = ADC_MAX_18BIT
    dropout_prob: float = 0.0  # probability that any sample is lost (creates counter gaps)
    timestamp_jitter_s: float = 0.0  # host-side arrival jitter (SD)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulatedRecording:
    t: np.ndarray  # nominal sample times (s), excluding dropped samples
    sample_index: np.ndarray  # device sample counter (gaps where samples were dropped)
    red: np.ndarray
    ir: np.ndarray
    beat_times: np.ndarray  # true systolic-onset times (ground truth for beat-detection tests)
    hr_true: np.ndarray  # instantaneous HR at each sample (bpm)
    config: SimConfig
    flags: dict = field(default_factory=lambda: {"source": SIMULATED_FLAG})


def _pulse(phase: np.ndarray, reflection: float, delay: float) -> np.ndarray:
    """Unit pulse over one beat (phase 0..1): systolic Gaussian + reflected wave + slow diastolic decay."""
    sys_ = np.exp(-((phase - 0.2) ** 2) / (2 * 0.085**2))
    ref = reflection * np.exp(-((phase - 0.2 - delay) ** 2) / (2 * 0.11**2))
    return sys_ + ref


def simulate(cfg: SimConfig, duration_s: float) -> SimulatedRecording:
    """Simulate ``duration_s`` seconds of red/IR raw counts at ``cfg.fs`` (deterministic for a given seed)."""
    if duration_s <= 0 or cfg.fs <= 0:
        raise ValueError("duration_s and fs must be positive")
    rng = np.random.default_rng(cfg.seed)
    n = int(round(duration_s * cfg.fs))
    t = np.arange(n) / cfg.fs
    # beat onsets from a drifting HR plus beat-to-beat jitter
    beats = [0.0]
    while beats[-1] < duration_s + 2:
        hr = cfg.hr_bpm + cfg.hr_drift_bpm * np.sin(2 * np.pi * beats[-1] / cfg.hr_drift_period_s)
        beats.append(beats[-1] + max(60.0 / max(hr, 20.0) + rng.normal(0, cfg.hrv_sd_s), 0.25))
    beats = np.asarray(beats)
    k = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, beats.size - 2)
    period = beats[k + 1] - beats[k]
    phase = (t - beats[k]) / period
    resp = np.sin(2 * np.pi * cfg.resp_rate_hz * t)
    pulse = _pulse(phase, cfg.reflection, cfg.reflection_delay) * (1 + cfg.resp_amp_mod * resp)
    pulse = pulse / max(pulse.max(), 1e-9)
    out = {}
    for ch, dc, perf in (("ir", cfg.dc_ir, cfg.perfusion_ir), ("red", cfg.dc_red, cfg.perfusion_red)):
        x = dc * (1 + cfg.resp_baseline_frac * np.sin(2 * np.pi * cfg.resp_rate_hz * t + 0.6))
        x = x - dc * perf * pulse  # absorption: counts fall in systole
        x = x + rng.normal(0, cfg.noise_frac * dc, n)
        if cfg.mains_hz:
            x = x + cfg.mains_frac * dc * np.sin(2 * np.pi * cfg.mains_hz * t)
        for tm in cfg.motion_times_s:
            w = np.abs(t - tm) < cfg.motion_duration_s / 2
            if w.any():
                burst = np.sin(2 * np.pi * 1.3 * (t[w] - tm)) + 0.6 * np.sin(2 * np.pi * 0.4 * (t[w] - tm) + 1)
                x[w] = x[w] + cfg.motion_frac * dc * burst * np.hanning(w.sum())
        out[ch] = np.clip(np.round(x), 0, cfg.adc_max)
    hr_true = 60.0 / period
    keep = rng.random(n) >= cfg.dropout_prob if cfg.dropout_prob > 0 else np.ones(n, bool)
    t_out = t[keep] + (rng.normal(0, cfg.timestamp_jitter_s, keep.sum()) if cfg.timestamp_jitter_s > 0 else 0.0)
    return SimulatedRecording(
        t=t_out,
        sample_index=np.flatnonzero(keep),
        red=out["red"][keep],
        ir=out["ir"][keep],
        beat_times=beats[beats < duration_s],
        hr_true=hr_true[keep],
        config=cfg,
    )
