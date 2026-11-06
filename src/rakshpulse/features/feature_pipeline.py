"""End-to-end window → feature-vector pipeline shared by training and inference.

The same function is used for dataset feature extraction and for live inference, so the
two cannot drift apart.

Channels are processed independently and their features are prefixed (``ir_``, ``red_``).
A missing channel is never synthesised from another one: its features are NaN, and the
models declare which channels they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import detrend as scipy_detrend

from ..preprocessing import artifact_removal, filtering, normalization, segmentation, signal_quality
from .frequency_domain import FREQ_FEATURES, frequency_features
from .morphological import BEAT_FEATURES, aggregate_beats
from .time_domain import TIME_FEATURES, time_features

CHANNELS = ("ir", "red")
PER_CHANNEL = tuple(BEAT_FEATURES) + tuple(TIME_FEATURES) + tuple(FREQ_FEATURES)
CROSS_FEATURES = ("red_ir_corr", "ratio_of_ratios")


def feature_names(channels: tuple[str, ...] = CHANNELS, cross: bool = True) -> list[str]:
    names = [f"{c}_{n}" for c in channels for n in PER_CHANNEL]
    if cross and len(channels) == 2:
        names += list(CROSS_FEATURES)
    return names


@dataclass
class ProcessedChannel:
    filtered: np.ndarray  # band-passed, oriented, z-scored, at TARGET_FS
    sqi: signal_quality.SQIResult
    ac_dc: float  # perfusion index estimate (NaN if DC unavailable)
    beats: list = field(default_factory=list)


def preprocess_channel(raw: np.ndarray, fs: float, has_dc: bool = True) -> ProcessedChannel:
    original = np.asarray(raw, dtype=float)
    raw = artifact_removal.interpolate_nonfinite(original)
    fs = filtering.validate_sampling_rate(fs)
    # Remove the DC level *before* resampling: polyphase resampling zero-pads at the edges,
    # and a large ADC offset (~4e5 counts on MAX3010x-class sensors) would otherwise create
    # a step transient that dominates the window.
    dc = float(np.median(raw))
    fs_t = filtering.TARGET_FS
    x = filtering.resample(scipy_detrend(raw - dc), fs, fs_t)
    x = artifact_removal.hampel(x)
    ac = filtering.bandpass(x, fs_t)
    ac_dc = np.nan
    if has_dc and abs(dc) > 0:
        ac_dc = float((np.percentile(ac, 95) - np.percentile(ac, 5)) / abs(dc))
    ac = normalization.orient_ppg(ac, fs_t)
    z = normalization.zscore(ac)
    # Clipping/flatline is judged on the original ADC samples (meaningless after z-scoring
    # or for datasets distributed already filtered).
    sqi = signal_quality.assess(z, fs_t, raw=original if has_dc else None)
    beats = artifact_removal.reject_outlier_beats(segmentation.split_beats(z, fs_t))
    return ProcessedChannel(z, sqi, ac_dc, beats)


def channel_features(pc: ProcessedChannel, fs: float = filtering.TARGET_FS) -> dict[str, float]:
    f = aggregate_beats(pc.beats, fs)
    f.update(time_features(pc.filtered, fs))
    f.update(frequency_features(pc.filtered, fs))
    return f


def extract_features(
    channels: dict[str, np.ndarray | None], fs: float, has_dc: bool = True
) -> tuple[dict[str, float], dict[str, ProcessedChannel]]:
    """Return (features, processed channels). Missing channels yield NaN features."""
    feats: dict[str, float] = {}
    processed: dict[str, ProcessedChannel] = {}
    for c in CHANNELS:
        sig = channels.get(c)
        if sig is None:
            feats.update({f"{c}_{n}": np.nan for n in PER_CHANNEL})
            continue
        pc = preprocess_channel(sig, fs, has_dc=has_dc)
        processed[c] = pc
        feats.update({f"{c}_{k}": v for k, v in channel_features(pc).items()})
    if "ir" in processed and "red" in processed:
        a, b = processed["ir"].filtered, processed["red"].filtered
        n = min(a.size, b.size)
        feats["red_ir_corr"] = float(np.corrcoef(a[:n], b[:n])[0, 1])
        ri, rr = processed["ir"].ac_dc, processed["red"].ac_dc
        feats["ratio_of_ratios"] = rr / ri if np.isfinite(ri) and ri > 0 and np.isfinite(rr) else np.nan
    else:
        feats.update(dict.fromkeys(CROSS_FEATURES, np.nan))
    return feats, processed
