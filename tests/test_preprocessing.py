import numpy as np
import pytest
from conftest import synthetic_ppg

from rakshpulse.preprocessing import artifact_removal, filtering, normalization, segmentation, signal_quality


def test_validate_sampling_rate_rejects_bad_values():
    for bad in (0, -5, np.nan, np.inf, "abc", 10):  # 10 Hz < 2.5 x 10 Hz band edge
        with pytest.raises(ValueError):
            filtering.validate_sampling_rate(bad)
    assert filtering.validate_sampling_rate(100) == 100.0


def test_bandpass_removes_out_of_band_components():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    inband = np.sin(2 * np.pi * 1.2 * t)
    x = inband + 3 * np.sin(2 * np.pi * 0.05 * t) + np.sin(2 * np.pi * 30 * t)
    y = filtering.bandpass(x, fs)
    core = slice(300, -300)
    assert np.corrcoef(y[core], inband[core])[0, 1] > 0.99


def test_resample_length():
    x = np.random.default_rng(0).standard_normal(2500)
    assert filtering.resample(x, 250, 100).size == 1000
    assert filtering.resample(x, 100, 100).size == 2500


def test_orientation_flips_inverted_raw_signal():
    x = synthetic_ppg(raw_counts=False)
    up = normalization.orient_ppg(filtering.bandpass(x, 100), 100)
    down = normalization.orient_ppg(filtering.bandpass(-x, 100), 100)
    assert np.corrcoef(up, down)[0, 1] > 0.99


def test_zscore_is_per_window():
    x = np.vstack([np.arange(10.0), 100 + 5 * np.arange(10.0)])
    z = normalization.zscore(x)
    assert np.allclose(z.mean(axis=1), 0) and np.allclose(z.std(axis=1), 1)


def test_sliding_windows_shape():
    x = np.zeros((2, 3000))
    w = segmentation.sliding_windows(x, 100, 10, 5)
    assert w.shape == (5, 2, 1000)
    assert segmentation.sliding_windows(np.zeros(500), 100, 10, 5).shape[0] == 0


@pytest.mark.parametrize("hr", [50, 72, 110, 160])
def test_peak_detection_recovers_heart_rate(hr):
    fs = 100.0
    x = normalization.zscore(filtering.bandpass(synthetic_ppg(fs, 20, hr, raw_counts=False), fs))
    pk = segmentation.detect_peaks(x, fs)
    est = 60 / np.median(np.diff(pk) / fs)
    assert abs(est - hr) < 3


def test_split_beats_returns_physiological_lengths():
    fs = 100.0
    x = normalization.zscore(filtering.bandpass(synthetic_ppg(fs, 20, 60, raw_counts=False), fs))
    beats = segmentation.split_beats(x, fs)
    assert len(beats) >= 15
    assert all(80 <= b.size <= 120 for b in beats)


def test_sqi_accepts_clean_and_rejects_noise():
    fs = 100.0
    clean = normalization.zscore(filtering.bandpass(synthetic_ppg(fs, 10, 70, raw_counts=False), fs))
    assert signal_quality.assess(clean, fs).acceptable
    noise = normalization.zscore(filtering.bandpass(np.random.default_rng(3).standard_normal(1000), fs))
    res = signal_quality.assess(noise, fs)
    assert not res.acceptable and res.reasons


def test_sqi_detects_flatline_saturation():
    raw = np.full(1000, 262143.0)  # 18-bit ADC rail
    raw[:200] = np.linspace(0, 1, 200)
    assert signal_quality.flat_fraction(raw) > 0.7
    assert signal_quality.flat_fraction(synthetic_ppg(seconds=10)) < 0.01


def test_sqi_rejects_too_short():
    assert not signal_quality.assess(np.zeros(50), 100).acceptable


def test_interpolate_nonfinite_and_limit():
    x = np.arange(100.0)
    x[10] = np.nan
    assert np.isclose(artifact_removal.interpolate_nonfinite(x)[10], 10)
    y = np.full(100, np.nan)
    y[:10] = 1
    with pytest.raises(ValueError):
        artifact_removal.interpolate_nonfinite(y)


def test_hampel_removes_spike():
    x = np.sin(np.linspace(0, 6, 200))
    x[100] += 50
    assert abs(artifact_removal.hampel(x)[100]) < 2
