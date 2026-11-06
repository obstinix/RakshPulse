import numpy as np
from conftest import synthetic_ppg

from rakshpulse.features.feature_pipeline import CROSS_FEATURES, PER_CHANNEL, extract_features, feature_names
from rakshpulse.features.morphological import BEAT_FEATURES, beat_features


def test_feature_names_cover_both_channels():
    names = feature_names()
    assert len(names) == 2 * len(PER_CHANNEL) + len(CROSS_FEATURES)
    assert len(set(names)) == len(names)


def test_extract_features_two_channels(ppg_pair):
    red, ir, fs = ppg_pair
    f, proc = extract_features({"red": red[:1000], "ir": ir[:1000]}, fs)
    assert set(feature_names()) <= set(f)
    assert proc["ir"].sqi.acceptable
    assert abs(f["ir_hr"] - 72) < 3
    assert f["red_ir_corr"] > 0.9
    assert np.isfinite(f["ratio_of_ratios"])


def test_missing_red_channel_is_not_synthesised(ppg_pair):
    _, ir, fs = ppg_pair
    f, proc = extract_features({"ir": ir[:1000], "red": None}, fs)
    assert "red" not in proc
    assert all(np.isnan(f[f"red_{n}"]) for n in PER_CHANNEL)
    assert np.isnan(f["red_ir_corr"]) and np.isnan(f["ratio_of_ratios"])
    assert np.isfinite(f["ir_hr"])


def test_ratio_of_ratios_unavailable_without_dc(ppg_pair):
    red, ir, fs = ppg_pair
    f, _ = extract_features({"red": red[:1000], "ir": ir[:1000]}, fs, has_dc=False)
    assert np.isnan(f["ratio_of_ratios"])


def test_sampling_rate_invariance():
    a = synthetic_ppg(100, 10, 70, seed=0)
    b = synthetic_ppg(250, 10, 70, seed=0)
    fa, _ = extract_features({"ir": a}, 100)
    fb, _ = extract_features({"ir": b}, 250)
    for k in ("ir_hr", "ir_sys_ratio", "ir_w50"):
        assert abs(fa[k] - fb[k]) / abs(fa[k]) < 0.1


def test_beat_features_sane_values():
    t = np.linspace(0, 1, 100)
    beat = np.exp(-((t - 0.2) ** 2) / 0.005) + 0.4 * np.exp(-((t - 0.5) ** 2) / 0.01)
    f = beat_features(beat, 100)
    assert set(BEAT_FEATURES) <= set(f)
    assert 0.1 < f["sys_ratio"] < 0.35
    assert f["w50"] > f["w75"] > f["w90"]
