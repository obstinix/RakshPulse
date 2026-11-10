import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_ppg

from rakshpulse.personalization import (
    BaselineTracker,
    EngineConfig,
    PersonalizationEngine,
    RefPoint,
    reference_category,
)

# ------------------------------------------------------------------------- personalisation engine


def test_offset_calibration_from_single_reference():
    e = PersonalizationEngine("u", EngineConfig(method="offset"))
    assert e.status(0)["recalibration_recommended"] and not e.apply(110, 70)["calibrated"]
    e.calibrate([RefPoint(0, 110, 70, 125, 80)])
    out = e.apply(100, 65, 5)
    assert out["calibrated"] and out["sbp"] == pytest.approx(115) and out["dbp"] == pytest.approx(75)
    assert e.state.method_used == {"sbp": "offset", "dbp": "offset"}


def test_linear_needs_enough_spread_and_is_shrunk_towards_unit_slope():
    e = PersonalizationEngine("u", EngineConfig(method="linear", min_points_linear=4, ridge_lambda=1.0))
    e.calibrate([RefPoint(0, 120, 70, 130, 80), RefPoint(1, 121, 70, 131, 80)])
    assert e.state.method_used["sbp"] == "offset"  # too few / too clustered -> offset
    pts = [RefPoint(t, p, 70, 10 + 1.5 * p, 80) for t, p in enumerate(range(100, 160, 10), start=2)]
    for p in pts:
        e.update(p)
    assert e.state.method_used["sbp"] == "linear"
    assert 1.0 < e.state.sbp.b < 1.5  # true slope 1.5, shrunk towards 1


def test_update_records_prequential_residual_and_rollback():
    e = PersonalizationEngine("u")
    e.calibrate([RefPoint(0, 110, 70, 120, 78)])
    rec = e.update(RefPoint(60, 110, 70, 130, 82))
    assert rec["residual_sbp"] == pytest.approx(10.0)  # error of the pre-update calibration
    v_after = e.state.version
    e.rollback()
    assert e.state.version == v_after - 1 and e.state.n_points == 1
    assert e.apply(110, 70, 61)["sbp"] == pytest.approx(120)
    with pytest.raises(ValueError):
        e.rollback(5)


def test_no_future_leakage_and_input_validation():
    e = PersonalizationEngine("u")
    e.calibrate([RefPoint(100, 110, 70, 120, 78)])
    with pytest.raises(ValueError):
        e.update(RefPoint(50, 110, 70, 120, 78))  # back-dated reference
    with pytest.raises(ValueError):
        e.apply(110, 70, timestamp=10)  # prediction before the calibration it would use
    for bad in (RefPoint(200, 110, 70, 70, 90), RefPoint(200, np.nan, 70, 120, 80), RefPoint(200, 110, 70, 400, 80)):
        with pytest.raises(ValueError):
            e.update(bad)
    with pytest.raises(ValueError):
        e.calibrate([])


def test_unstable_update_flagged_and_drift_triggers_recommendation():
    cfg = EngineConfig(max_offset_jump=10, residual_alarm=8, residual_window=2, half_life_s=1.0)
    e = PersonalizationEngine("u", cfg)
    e.calibrate([RefPoint(0, 110, 70, 120, 78)])
    e.update(RefPoint(100, 110, 70, 150, 80))  # 30 mmHg jump with fast forgetting
    s = e.status(101)
    assert e.state.unstable and s["recalibration_recommended"]
    e.update(RefPoint(200, 110, 70, 151, 80))
    assert any("residual" in r or "unstable" in r for r in e.status(201)["reasons"])
    assert e.status(10 * 86400)["recalibration_recommended"]  # stale calibration


def test_engine_state_roundtrip(tmp_path):
    e = PersonalizationEngine("u", EngineConfig(method="linear"))
    e.calibrate([RefPoint(0, 110, 70, 120, 78)])
    e.update(RefPoint(10, 115, 72, 126, 80))
    e2 = PersonalizationEngine.load(e.save(tmp_path / "u.json"))
    assert e2.state == e.state and e2.apply(118, 71, 11) == e.apply(118, 71, 11)
    assert len(e2.history) == len(e.history) and e2.config == e.config


def test_engines_are_isolated_per_user():
    a, b = PersonalizationEngine("a"), PersonalizationEngine("b")
    a.calibrate([RefPoint(0, 110, 70, 140, 90)])
    b.calibrate([RefPoint(0, 110, 70, 110, 70)])
    assert a.apply(110, 70)["sbp"] != b.apply(110, 70)["sbp"]


# ----------------------------------------------------------------------------------- baseline


def test_baseline_stats_rolling_trend_and_deviation(tmp_path):
    b = BaselineTracker("u", rolling_windows=(3, 5))
    for i in range(6):
        b.add(i * 86400, 120 + i, 80)
    s = b.stats()
    assert s["n"] == 6 and s["sbp_median"] == pytest.approx(122.5)
    assert s["sbp_rolling3_mean"] == pytest.approx(124) and s["sbp_trend_per_day"] == pytest.approx(1.0)
    assert np.isnan(BaselineTracker("v").stats().get("sbp_mean", np.nan))
    d = b.deviation(130, 80)
    assert d["sbp_minus_baseline"] == pytest.approx(7.5) and np.isfinite(d["sbp_z"])
    assert b.stats(upto=86400)["n"] == 2  # time-restricted view (no future readings)
    assert BaselineTracker.load(b.save(tmp_path / "b.json")).stats() == s
    assert not BaselineTracker("w").deviation(120, 80)["available"]


def test_baseline_rejects_invalid_and_backdated():
    b = BaselineTracker("u")
    b.add(10, 120, 80)
    for args in ((5, 120, 80), (20, 80, 90), (20, np.nan, 80)):
        with pytest.raises(ValueError):
            b.add(*args)


def test_reference_category_is_a_lookup_only():
    assert reference_category(118, 76) == "normal range"
    assert reference_category(125, 78) == "elevated range"
    assert reference_category(132, 70) == "stage 1 range"
    assert reference_category(120, 92) == "stage 2 range"


# ----------------------------------------------------------------------------------- realtime


@pytest.fixture
def ear_bundle(tmp_path):
    from rakshpulse.evaluation.auricle_study import add_normalised_shape, feature_set
    from rakshpulse.inference.model_loader import ModelBundle
    from rakshpulse.inference.realtime import site_window_features
    from rakshpulse.models.model_factory import make_model

    rows = []
    for hr in (55, 65, 75, 85, 95):
        for seed in range(3):
            r, _ = site_window_features(synthetic_ppg(100, 10, hr, seed=seed, raw_counts=False), 100, "ear")
            rows += r
    X = add_normalised_shape(pd.DataFrame(rows), "ear")
    cols = feature_set("hr_shape", "ear", [])
    y = np.linspace(100, 140, len(X))
    s, _ = make_model("ridge")
    dd, _ = make_model("ridge")
    b = ModelBundle(
        s.fit(X[cols].to_numpy(float), y),
        dd.fit(X[cols].to_numpy(float), y * 0.6),
        cols,
        ("ear",),
        False,
        {"sbp": {"mae": 14.0}, "dbp": {"mae": 9.0}},
        "test-ear",
    )
    return b.save(tmp_path / "ear.joblib")


def test_realtime_population_and_personalised(ear_bundle):
    from rakshpulse.inference.realtime import RealtimeEstimator

    est = RealtimeEstimator.from_file(ear_bundle)
    ppg = synthetic_ppg(100, 30, 70, seed=4, raw_counts=False)
    r = est.estimate(ear_ppg=ppg, sampling_rate=100, finger_ppg=ppg, timestamp=100)
    assert r.accepted and r.estimate_type == "population" and r.recalibration_recommended
    assert r.signal_quality["finger"]["used_by_model"] is False
    e = PersonalizationEngine("u")
    e.calibrate([RefPoint(50, r.population_sbp, r.population_dbp, r.population_sbp + 7, r.population_dbp + 3)])
    bt = BaselineTracker("u")
    bt.add(50, 130, 85)
    r2 = est.estimate(ear_ppg=ppg, sampling_rate=100, timestamp=100, engine=e, baseline=bt)
    assert r2.estimate_type == "personalised" and r2.sbp == pytest.approx(r.population_sbp + 7)
    assert r2.baseline_deviation["available"] and r2.calibration["calibrated"]
    assert "not a clinical" in r2.uncertainty["note"]


def test_realtime_rejects_bad_inputs(ear_bundle):
    from rakshpulse.inference.predictor import InputError
    from rakshpulse.inference.realtime import RealtimeEstimator

    est = RealtimeEstimator.from_file(ear_bundle)
    noise = np.random.default_rng(0).standard_normal(3000)
    r = est.estimate(ear_ppg=noise, sampling_rate=100)
    assert not r.accepted and r.sbp is None and r.rejection_reasons == ["insufficient_signal_quality"]
    assert est.estimate(finger_ppg=noise, sampling_rate=100).rejection_reasons == ["missing_required_signal:ear"]
    assert est.estimate(ear_ppg=noise[:500], sampling_rate=100).rejection_reasons[0].startswith("too_short")
    with pytest.raises(InputError):
        est.estimate(ear_ppg=noise, sampling_rate=None)
    with pytest.raises(InputError):
        est.estimate(ear_ppg=np.zeros((2, 2000)), sampling_rate=100)
