"""Complete ML workflow: dataset preparation, training protocol, export, hardware inference API, pipeline script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rakshpulse.acquisition.simulator import SimConfig, simulate
from rakshpulse.inference.api import BPEstimator, StreamingBPEstimator
from rakshpulse.inference.contract import build_feature_vector, inference_mode, validate_bundle
from rakshpulse.inference.model_loader import load_bundle
from rakshpulse.ml import DatasetPaths, PipelineConfig, export_bundle, make_smoke_dataset, run_training
from rakshpulse.ml.dataset import build_ear_dataset, candidate_features, validate_auricle
from rakshpulse.ml.training import QuantileClipper, metrics, split_subjects

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "models" / "rakshpulse_ear_bp.joblib"


@pytest.fixture(scope="module")
def smoke():
    d = make_smoke_dataset(n_subjects=8, windows_per_session=6, seed=1)
    r = run_training(
        d,
        cfg=PipelineConfig(cv_folds=3, n_bootstrap=100, models=("ridge", "random_forest"), n_jobs=1),
        log=lambda *_: None,
    )
    return d, r


@pytest.fixture(scope="module")
def signal():
    rec = simulate(SimConfig(fs=100.0, seed=5), 30.0)
    return rec.ir, rec.red


# ------------------------------------------------------------------------------------------------ dataset
def test_smoke_dataset_is_labelled_synthetic():
    d = make_smoke_dataset(n_subjects=3, windows_per_session=2)
    assert d.synthetic.all() and d.subject_id.str.startswith("synthetic_").all()
    assert set(candidate_features()) <= set(d.columns)
    assert (d.sbp > d.dbp).all()


def test_committed_auricle_tables_pass_integrity_checks():
    v = validate_auricle(DatasetPaths(ROOT))
    assert (v.status == "PASS").all(), v[v.status != "PASS"]


def test_ear_dataset_cleaning_rules_hold():
    d, log = build_ear_dataset(DatasetPaths(ROOT), write=False)
    assert log.windows.is_monotonic_decreasing and log.windows.iloc[-1] == len(d)
    assert d.sbp.between(60, 250).all() and d.dbp.between(30, 150).all()
    assert (d.sbp - d.dbp >= 10).all()
    assert d[["ear_hr", "ear_beat_T"]].notna().all().all()
    assert set(d.session) <= {1, 2} and not d.synthetic.any()
    assert d.groupby("subject_id").size().min() >= 20
    # fit windows do not overlap (10 s windows starting 10 s apart)
    fit = d[d.fit_window]
    assert ((fit.t_start_s / 5).round() % 2 == 0).all()


# ------------------------------------------------------------------------------------------------ training protocol
def test_split_is_by_subject_and_deterministic(smoke):
    d, _ = smoke
    a, b = split_subjects(d, 0.25, 0), split_subjects(d, 0.25, 0)
    assert a == b and not set(a[0]) & set(a[1]) and set(a[0]) | set(a[1]) == set(d.subject_id)


def test_training_result_has_baseline_selection_and_no_leakage(smoke):
    d, r = smoke
    assert not set(r["train_subjects"]) & set(r["test_subjects"])
    comp = r["comparison"]
    assert set(comp.model) == {"mean", "ridge", "random_forest"} and set(comp.target) == {"sbp", "dbp"}
    for col in ("cv_mae", "test_mae", "test_rmse", "test_r2", "test_bias", "test_sd_error", "test_loa_low"):
        assert comp[col].notna().all()
    assert set(r["selection"].model) <= {"ridge", "random_forest"}  # the baseline is never "selected"
    assert set(r["test_predictions"].subject_id) == set(r["test_subjects"])
    assert set(r["features"]) <= set(candidate_features())


def test_metrics_and_clipper():
    m = metrics([100, 110, 120], [102, 108, 120], ["a", "a", "b"])
    assert m["mae"] == pytest.approx(4 / 3) and m["bias"] == pytest.approx(0.0) and m["n_subjects"] == 2
    X = np.array([[0.0], [1.0], [2.0], [1000.0], [np.nan]])
    c = QuantileClipper(0.0, 0.75).fit(X)
    out = c.transform(X)
    assert out[3, 0] <= c.hi_[0] and np.isnan(out[4, 0])


def test_export_bundle_contract(smoke, tmp_path):
    _, r = smoke
    b = export_bundle(r, "test", synthetic=True)
    c = validate_bundle(load_bundle(b.save(tmp_path / "m.joblib")))
    assert c["validated_for"] == [] and inference_mode(c, "max30102") == "demonstration"
    assert any("SYNTHETIC" in x for x in c["limitations"])
    assert all(f["source"].startswith("ir_") for f in c["features"])


# ------------------------------------------------------------------------------------------------ shipped model
def test_shipped_model_contract_is_honest():
    c = validate_bundle(load_bundle(SHIPPED))
    assert c["validated_for"] == []
    assert inference_mode(c, "max30102") == "demonstration"
    assert inference_mode(c, "auricle_earlobe") == "experimental"
    assert c["uncertainty"]["valid_for"] == ["auricle_earlobe"]


def test_shipped_model_reproduces_its_reported_test_metrics():
    b = load_bundle(SHIPPED)
    c = b.contract
    d = build_ear_dataset(DatasetPaths(ROOT), write=False)[0]
    _, test_s = split_subjects(d, 0.2, 0)
    test = d[d.subject_id.isin(test_s)]
    for t in ("sbp", "dbp"):
        rows = [
            build_feature_vector(c, {f["source"]: r[f["name"]] for f in c["features"]}, {})[0]
            for _, r in test.head(300).iterrows()
        ]
        p = getattr(b, t).predict(np.vstack(rows))
        assert np.isfinite(p).all()
    X = test[b.features].to_numpy(float)
    for t in ("sbp", "dbp"):
        mae = np.mean(np.abs(getattr(b, t).predict(X) - test[t].to_numpy()))
        assert mae == pytest.approx(c["performance"][f"{t}_mae"], abs=1e-6)


# ------------------------------------------------------------------------------------------------ inference API
def test_api_returns_structured_demonstration_result(signal):
    ir, red = signal
    r = BPEstimator(SHIPPED).estimate(ir, 100.0, red=red, metadata={"device": "sim"})
    d = r.to_dict()
    assert r.status == "ok" and 40 < r.sbp < 250 and 20 < r.dbp < 160
    assert r.inference_mode == "demonstration" and r.sbp_interval_90 is None
    assert any("DEMONSTRATION" in w for w in r.warnings)
    assert d["model_version"].startswith("rakshpulse-ear-bp") and d["metadata"] == {"device": "sim"}
    assert d["signal_quality"]["windows_total"] >= 4


@pytest.mark.parametrize(
    "ir,fs,red,expected",
    [
        (np.zeros((10, 100)), 100.0, None, "one-dimensional"),
        (np.r_[np.ones(2000), np.inf], 100.0, None, "infinite"),
        (np.full(2000, np.nan), 100.0, None, "missing"),
        (np.ones(500), 100.0, None, "at least 10 s"),
        (np.ones(2000), 0.0, None, "positive"),
        (np.ones(2000), 100.0, np.ones(1999), "same length"),
        (["a"] * 2000, 100.0, None, "numbers"),
    ],
)
def test_api_rejects_invalid_input(ir, fs, red, expected):
    r = BPEstimator(SHIPPED).estimate(ir, fs, red=red)
    assert r.status == "invalid_input" and expected in r.message and r.sbp is None


def test_api_poor_signal_returns_no_bp():
    r = BPEstimator(SHIPPED).estimate(np.full(3000, 120000.0) + np.random.default_rng(0).normal(0, 0.5, 3000), 100.0)
    assert r.status == "poor_signal" and r.sbp is None and r.dbp is None


def test_api_unsupported_rate_is_reported():
    rec = simulate(SimConfig(fs=40.0, seed=2), 30.0)
    r = BPEstimator(SHIPPED).estimate(rec.ir, 40.0)
    assert r.status in {"incompatible_model", "invalid_input"} and r.sbp is None


def test_streaming_buffer(signal):
    ir, red = signal
    s = StreamingBPEstimator(BPEstimator(SHIPPED), fs=100.0, buffer_s=20)
    for k in range(0, ir.size, 32):
        s.push(ir[k : k + 32], red[k : k + 32])
    assert s.buffered_s == pytest.approx(20.0) and s.samples_seen == ir.size
    assert s.estimate().status == "ok"
    with pytest.raises(ValueError):
        s.push(ir[:10])  # red missing
    with pytest.raises(ValueError):
        StreamingBPEstimator(BPEstimator(SHIPPED), fs=100.0, buffer_s=5)


# ------------------------------------------------------------------------------------------------ one command
def test_pipeline_smoke_test_runs_end_to_end():
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_complete_pipeline.py"), "--smoke-test", "--n-jobs", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "SMOKE TEST PASSED" in p.stdout and "not model validation" in p.stdout
    report = (ROOT / "reports" / "smoke_test" / "smoke_test_report.md").read_text(encoding="utf-8")
    assert "SMOKE TEST ON SYNTHETIC DATA" in report


# ------------------------------------------------------------------------------------------------ 5.2 additions
def test_nested_cv_and_calibration_study_on_smoke_data(smoke):
    from rakshpulse.ml.evaluation import calibration_study, nested_cv

    d, r = smoke
    n = nested_cv(
        d,
        candidate_features(),
        PipelineConfig(cv_folds=2, n_bootstrap=50, models=("ridge",), n_jobs=1),
        outer_folds=2,
        log=lambda *_: None,
    )
    assert set(n["predictions"].subject_id) == set(d.subject_id)  # every subject predicted out of fold
    assert set(n["summary"].model) == {"nested-selected", "mean"}
    for _, row in n["choices"].iterrows():  # chosen without the outer-fold subjects
        assert row.model == "ridge"
    c = calibration_study(
        r["data"],
        r["train_subjects"],
        r["test_subjects"],
        r["models"],
        r["features"],
        PipelineConfig(cv_folds=2, n_bootstrap=50, n_jobs=1),
        gap_s=10.0,
    )
    assert {"population", "carry_forward", "offset", "delta_ridge"} <= set(c["summary"].method)
    assert set(c["predictions"].subject_id) <= set(r["test_subjects"])


def test_api_calibration_heart_rate_and_status(signal):
    ir, red = signal
    est = BPEstimator(SHIPPED, sensor="simulator")
    assert est.supports_calibration
    r0 = est.estimate(ir, 100.0, red=red)
    assert r0.calibration_status == "uncalibrated" and r0.input_validation == "passed"
    assert r0.heart_rate_bpm is not None and 40 < r0.heart_rate_bpm < 150
    assert not est.calibrate(300, 80, ir, 100.0)["ok"]  # implausible cuff reading
    assert est.calibrate(125, 82, ir, 100.0, red=red, timestamp=1000.0)["ok"]
    r1 = est.estimate(ir, 100.0, red=red, timestamp=1060.0)
    assert r1.calibration_status.startswith("calibrated") and r1.population_sbp == r0.sbp
    assert abs(r1.sbp - 125) < 15 and abs(r1.dbp - 82) < 15  # same signal: close to the cuff reading
    r2 = est.estimate(ir, 100.0, red=red, timestamp=1000.0 + 10_000)
    assert r2.calibration_status == "expired" and r2.sbp == r2.population_sbp


def test_cli_estimate_writes_json_and_dashboard(tmp_path, capsys):
    import json

    from rakshpulse.cli import main

    code = main(
        [
            "estimate",
            "--source",
            "sim",
            "--seconds",
            "20",
            "--json",
            str(tmp_path / "r.json"),
            "--plot",
            str(tmp_path / "d.png"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "DEMONSTRATION MODE" in out and "heart rate" in out
    r = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert (
        r["status"] == "ok" and r["sensor_status"]["source"] == "simulator" and r["inference_mode"] == "demonstration"
    )
    assert (tmp_path / "d.png").stat().st_size > 10_000


def test_dashboard_banner_never_hides_demonstration_mode():
    from rakshpulse.inference.api import BPEstimate
    from rakshpulse.inference.dashboard import banner

    text, _ = banner(BPEstimate("ok", inference_mode="demonstration", sensor_status={"source": "serial"}))
    assert "LIVE SENSOR" in text and "NOT A BLOOD-PRESSURE MEASUREMENT" in text
    text, _ = banner(BPEstimate("poor_signal", sensor_status={"source": "simulator"}))
    assert "SIMULATED" in text and "NO ESTIMATE" in text
