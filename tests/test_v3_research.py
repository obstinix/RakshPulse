"""Tests for RakshPulse 3.0 research components (features, beyond-reference evaluation, registry)."""

import numpy as np
import pandas as pd
import pytest

from rakshpulse.evaluation.beyond_reference import evaluate, make_pairs
from rakshpulse.features.advanced import WAVELET_COLUMNS, haar_relative_energies
from rakshpulse.features.ecg_aux import pat_features


def test_haar_energies_partition_and_localise():
    rng = np.random.default_rng(0)
    e = haar_relative_energies(rng.standard_normal(1024))
    assert set(e) == set(WAVELET_COLUMNS) and sum(e.values()) == pytest.approx(1.0)
    assert e["wav_e1"] > 0.3  # white noise: most energy at the finest scale
    slow = haar_relative_energies(np.sin(2 * np.pi * np.arange(1024) / 128))  # period 128 samples
    assert max(slow, key=slow.get) in ("wav_e6", "wav_a6", "wav_e5")
    assert np.isnan(haar_relative_energies(np.ones(10))["wav_e1"])


def _ecg_ppg(fs=500, seconds=20, rr=0.75, lag=0.65):
    t = np.arange(int(fs * seconds)) / fs
    ecg = np.zeros_like(t)
    beats = np.arange(0.2, seconds, rr)
    for b in beats:
        ecg += np.exp(-((t - b) ** 2) / (2 * 0.008**2))
    ppg = np.zeros_like(t)
    for b in beats:
        ph = t - (b + lag)
        ppg += np.where(ph > 0, (ph / 0.12) * np.exp(1 - ph / 0.12), 0.0)
    return ecg, ppg, fs


def test_pat_recovers_lag_longer_than_rr_when_range_allows():
    ecg, ppg, fs = _ecg_ppg(lag=0.65, rr=0.75)
    default = pat_features(ecg, ppg, fs)  # 80-500 ms window: cannot see a 650 ms lag
    wide = pat_features(ecg, ppg, fs, range_ms=(300, 1000))
    assert np.isnan(default["pat_foot_ms"])
    assert wide["pat_foot_ms"] == pytest.approx(650, abs=25)
    ecg, ppg, fs = _ecg_ppg(lag=0.22, rr=0.8)
    assert pat_features(ecg, ppg, fs)["pat_foot_ms"] == pytest.approx(220, abs=25)


def test_ear_finger_ptt_measures_known_delay():
    from rakshpulse.data.auricle_advanced import ear_finger_ptt

    _, ear, fs = _ecg_ppg(fs=1000, lag=0.20)
    _, finger, _ = _ecg_ppg(fs=1000, lag=0.245)
    ptt, sd = ear_finger_ptt(ear, finger, fs)
    assert ptt == pytest.approx(45, abs=5) and sd < 5


def _toy(n_subj=10, seed=0, informative=True):
    """BP drifts within each session; feature x tracks the drift (if informative) or is pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        offset = 100 + 5 * s
        phase = rng.uniform(0, 2 * np.pi)  # subject-specific drift: no shared time pattern to learn
        for i in range(120):
            drift = 15 * np.sin(i / 15 + phase)
            rows.append(
                {
                    "subject_id": f"s{s}",
                    "session": 1,
                    "t_start_s": 10.0 * i,
                    "activity": 0,
                    "y": offset + drift + rng.normal(0, 1),
                    "x": (drift / 15 if informative else 0) + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def _refs(d, every):
    r = d[d.t_start_s % every == 0]
    return pd.DataFrame(
        {
            "subject_id": r.subject_id,
            "session": r.session,
            "t_ref_s": r.t_start_s,
            "ref_reported": r.y,
            "ref_truth": r.y,
            "ref_x": r.x,
        }
    )


def test_make_pairs_uses_only_past_references():
    d = _toy(3)
    pairs = make_pairs(d, _refs(d, 300), ["x"], "y", min_gap_s=10)
    assert (pairs.gap_s >= 10).all() and (pairs.gap_s < 310).all()
    # historical mean never includes readings after the window
    s0 = pairs[pairs.subject_id == "s0"].sort_values("t_s")
    first_ref = d[(d.subject_id == "s0") & (d.t_start_s == 0)].y.iloc[0]
    assert s0.historical_mean.iloc[0] == pytest.approx(first_ref)


def test_delta_model_beats_last_reference_only_when_signal_is_informative():
    for informative, expect_better in ((True, True), (False, False)):
        d = _toy(10, informative=informative)
        pairs = make_pairs(d, _refs(d, 300), ["x"], "y")
        m, p, _ = evaluate(pairs, {"x": ["x"]}, d.groupby("subject_id").y.mean(), n_boot=300)
        row = p.set_index("comparison").loc["delta_x vs mean_change"]
        assert (row.significant and row.mae_diff < 0) == expect_better
    assert set(m.method) >= {"last_reference", "historical_mean", "population_mean", "mean_change", "delta_x"}


def test_registry_records_provenance(tmp_path, monkeypatch):
    from rakshpulse.experiments import registry

    monkeypatch.setattr(registry, "RUNS", tmp_path / "runs")
    data = tmp_path / "d.csv"
    data.write_text("a,b\n1,2\n")
    run = registry.ExperimentRun("unit", [data], {"model": "ridge"}, seed=7)
    run.splits = {"s1": 0, "s2": 1}
    run.add_metrics([{"method": "m", "mae": 1.5}])
    out = run.finish()
    cfg = pd.read_csv(out / "config.csv").set_index("key").value
    assert cfg["seed"] == "7" and cfg["config:model"] == "ridge" and len(cfg[f"dataset:{data}"]) == 64
    assert {"python", "numpy", "scikit-learn", "git_commit"} <= set(cfg.index)
    assert pd.read_csv(out / "metrics.csv").mae.iloc[0] == 1.5 and len(pd.read_csv(out / "splits.csv")) == 2
    registry.ExperimentRun("unit2", [data], {}).finish()
    assert len(pd.read_csv(tmp_path / "runs" / "runs.csv")) == 2  # appended, not overwritten


# ------------------------------------------------------------------------------------- streaming
@pytest.fixture
def stream_bundle(tmp_path):
    from conftest import synthetic_ppg

    from rakshpulse.evaluation.auricle_study import add_normalised_shape, feature_set
    from rakshpulse.inference.model_loader import ModelBundle
    from rakshpulse.inference.realtime import site_window_features
    from rakshpulse.models.model_factory import make_model

    rows = []
    for hr in (55, 75, 95):
        for seed in range(2):
            rows += site_window_features(synthetic_ppg(100, 10, hr, seed=seed, raw_counts=False), 100, "ear")[0]
    X = add_normalised_shape(pd.DataFrame(rows), "ear")
    cols = feature_set("hr_shape", "ear", [])
    y = np.linspace(100, 140, len(X))
    s, _ = make_model("ridge")
    d, _ = make_model("ridge")
    return ModelBundle(
        s.fit(X[cols].to_numpy(float), y),
        d.fit(X[cols].to_numpy(float), y * 0.6),
        cols,
        ("ear",),
        False,
        {},
        "stream-test",
    )


def test_streaming_windows_quality_and_aggregation(stream_bundle, tmp_path):
    import json

    from conftest import synthetic_ppg

    from rakshpulse.inference.streaming import StreamingEstimator
    from rakshpulse.personalization import PersonalizationEngine, RefPoint

    eng = PersonalizationEngine("u")
    eng.calibrate([RefPoint(0.0, 110.0, 70.0, 120.0, 78.0)])
    s = StreamingEstimator(stream_bundle, fs=100, engine=eng, log_path=tmp_path / "log.jsonl", lost_after=2)
    x = synthetic_ppg(100, 40, 72, raw_counts=False)
    events = []
    for i in range(0, x.size, 37):  # odd chunk size: windows must still fall every 5 s
        events += s.push(x[i : i + 37], timestamp=min(i + 37, x.size) / 100)
    assert [round(e.timestamp, 2) for e in events] == [5.0 * k for k in range(1, 9)]
    assert events[0].status == "warming_up" and all(e.accepted for e in events[1:])
    assert events[-1].estimate_type == "personalised" and events[-1].n_windows_aggregated >= 5
    lost = s.push(np.zeros(1500), timestamp=55.0)
    assert lost[-1].status == "signal_lost" and "no_signal_flatline" in lost[-1].rejection_reasons
    assert lost[-1].sbp is not None  # last good aggregate is still reported, flagged by status
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == len(events) + len(lost) and json.loads(lines[1])["accepted"]
    lat = s.latency_summary()
    assert lat["windows"] > 0 and lat["feature_ms_p50"] > 0


def test_streaming_rejects_bad_configuration_and_time_travel(stream_bundle):
    from rakshpulse.inference.predictor import InputError
    from rakshpulse.inference.streaming import StreamingEstimator

    with pytest.raises(InputError):
        StreamingEstimator(stream_bundle, fs=5)
    with pytest.raises(InputError):
        StreamingEstimator(stream_bundle, fs=100, buffer_s=5)
    s = StreamingEstimator(stream_bundle, fs=100)
    s.push(np.zeros(100), timestamp=10.0)
    with pytest.raises(InputError):
        s.push(np.zeros(100), timestamp=5.0)


def _load_script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sequence_architectures_share_the_training_loop():
    import torch

    from rakshpulse.models.cnn import CNNConfig, CNNRegressor
    from rakshpulse.models.sequence import ARCHITECTURES

    assert set(ARCHITECTURES) == {"cnn", "gru", "cnn_lstm", "tcn", "transformer"}
    for build in ARCHITECTURES.values():
        assert tuple(build(1, 16)(torch.zeros(3, 1, 1000)).shape) == (3, 2)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 1, 1000)).astype("float32")
    y = np.c_[rng.normal(120, 10, 40), rng.normal(75, 8, 40)]
    p = CNNRegressor(1, CNNConfig(epochs=2, patience=1), arch="gru").fit(X, y, np.repeat(np.arange(10), 4)).predict(X)
    assert p.shape == (40, 2) and np.isfinite(p).all()
    with pytest.raises(KeyError):
        CNNRegressor(1, CNNConfig(epochs=1), arch="nope").fit(X, y, np.repeat(np.arange(10), 4))


def test_calibration_methods_shrink_as_specified():
    import pandas as pd

    m = _load_script("run_calibration_study")
    cal = pd.DataFrame({"pred": [100.0, 110.0, 120.0], "y": [130.0, 150.0, 170.0]})  # offset +40, slope 2
    ev = pd.DataFrame({"pred": [110.0]})
    out = m.calibrate(cal, ev, 3, tau=3.0)
    assert out["reference_mean"][0] == 150.0 and out["offset"][0] == 150.0
    assert out["bayes_offset"][0] == pytest.approx(110 + 40 * 3 / 6)  # k / (k + tau) = 0.5
    ev2 = pd.DataFrame({"pred": [130.0]})
    o2 = m.calibrate(cal, ev2, 3, tau=3.0)
    assert o2["linear_free"][0] == pytest.approx(190.0)  # slope 2 extrapolated
    assert 170.0 < o2["linear_shrunk"][0] < 190.0  # slope shrunk towards 1
    assert set(m.calibrate(cal, ev, 0, 3.0)) == {"population"}
    # no-PPG control: cuff mean shrunk towards the training mean with its own tau
    calm, evm = cal.assign(pmean=120.0), ev.assign(pmean=120.0)
    o3 = m.calibrate(calm, evm, 3, tau=3.0, tau0=1.0)
    assert o3["bayes_reference"][0] == pytest.approx(120 + 30 * 3 / 4)
    assert set(m.calibrate(calm, evm, 0, 3.0, 1.0)) == {"population", "training_mean"}
    g = pd.DataFrame({"t_start_s": [0.0, 10.0, 70.0, 100.0, 140.0]}, index=[5, 6, 7, 8, 9])
    assert list(m.pick_calibration(g, 3).index) == [5, 7, 9]  # >= 60 s apart


def test_icc31_extremes():
    m = _load_script("run_feature_analysis")
    a = np.arange(10.0)
    assert m.icc31(a, a + 5) == pytest.approx(1.0)  # consistency ICC ignores a constant day shift
    rng = np.random.default_rng(0)
    assert m.icc31(rng.normal(size=200), rng.normal(size=200)) < 0.2
    assert np.isnan(m.icc31(a[:3], a[:3]))
