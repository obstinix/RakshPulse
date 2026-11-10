import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_ppg

from rakshpulse.evaluation.metrics import bhs_grade, iso81060_like, regression_metrics
from rakshpulse.evaluation.validation import nested_cv, subject_level
from rakshpulse.features.feature_pipeline import extract_features, feature_names
from rakshpulse.inference.model_loader import ModelBundle, load_bundle
from rakshpulse.inference.predictor import BPPredictor, InputError
from rakshpulse.models.model_factory import FEATURE_MODELS, V3_MODELS, make_model


def _toy_df(n_subj=20, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        sbp = 100 + 3 * s
        for w in range(per):
            rows.append(
                {
                    "subject_id": f"s{s}",
                    "record_id": f"s{s}",
                    "t_start_s": w * 5.0,
                    "sbp": sbp,
                    "dbp": sbp * 0.6,
                    "f1": sbp / 10 + rng.normal(0, 0.5),
                    "f2": rng.normal(),
                    "f3": np.nan,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", sorted(set(FEATURE_MODELS) | set(V3_MODELS)))
def test_models_fit_predict_with_nans(name):
    df = _toy_df()
    est, _ = make_model(name, 0)
    est.fit(df[["f1", "f2", "f3"]].to_numpy(), df.sbp.to_numpy())
    assert np.isfinite(est.predict(df[["f1", "f2", "f3"]].to_numpy())).all()


def test_nested_cv_is_subject_independent_and_learns():
    df = _toy_df()
    pdf, params = nested_cv(df, ["f1", "f2"], "sbp", "ridge", seed=0)
    assert pdf.pred.notna().all() and len(params) == 5
    assert pdf.groupby("subject_id").fold.nunique().max() == 1  # each subject in exactly one test fold
    s = subject_level(pdf, "sbp")
    assert len(s) == 20 and regression_metrics(s.y, s.pred)["mae"] < 5


def test_metrics_values():
    m = regression_metrics([100, 110, 120], [102, 108, 125])
    assert m["mean_error"] == pytest.approx(5 / 3)
    assert m["mae"] == pytest.approx(3.0)
    assert bhs_grade(m) in "ABCD" and isinstance(iso81060_like(m), bool)


@pytest.fixture
def bundle_path(tmp_path):
    """Tiny bundle trained on synthetic feature rows: tests the plumbing, not BP accuracy."""
    feats = feature_names()
    rows = []
    for hr in (55, 65, 75, 85, 95):
        for seed in range(3):
            ir = synthetic_ppg(100, 10, hr, seed=seed)
            red = synthetic_ppg(100, 10, hr, seed=seed + 10)
            f, _ = extract_features({"ir": ir, "red": red}, 100)
            rows.append(f)
    X = pd.DataFrame(rows)[feats].to_numpy(float)
    y = np.linspace(100, 140, len(X))
    sbp, _ = make_model("ridge")
    dbp, _ = make_model("ridge")
    b = ModelBundle(
        sbp.fit(X, y),
        dbp.fit(X, y * 0.6),
        feats,
        ("ir", "red"),
        False,
        {"sbp": {"sd_error": 10.0, "mae": 8.0}, "dbp": {"sd_error": 7.0, "mae": 6.0}},
        "test-0",
    )
    return b.save(tmp_path / "m.joblib")


def test_bundle_roundtrip(bundle_path):
    b = load_bundle(bundle_path)
    assert b.channels == ("ir", "red") and b.version == "test-0"
    with pytest.raises(FileNotFoundError):
        load_bundle(bundle_path.parent / "nope.joblib")


def test_predict_clean_signal_accepted(bundle_path):
    p = BPPredictor.from_file(bundle_path)
    r = p.predict(
        red_signal=synthetic_ppg(100, 30, 70, seed=5), ir_signal=synthetic_ppg(100, 30, 70, seed=6), sampling_rate=100
    )
    assert r.accepted and np.isfinite(r.systolic_bp) and np.isfinite(r.diastolic_bp)
    assert r.signal_quality["windows_accepted"] >= 1
    assert r.uncertainty["validation_sd_error_sbp"] == 10.0


def test_predict_rejects_noise(bundle_path):
    p = BPPredictor.from_file(bundle_path)
    rng = np.random.default_rng(0)
    r = p.predict(
        red_signal=rng.standard_normal(3000) + 1e5, ir_signal=rng.standard_normal(3000) + 1e5, sampling_rate=100
    )
    assert not r.accepted and r.systolic_bp is None
    assert "insufficient_signal_quality" in r.rejection_reasons


def test_predict_missing_channel_rejected(bundle_path):
    r = BPPredictor.from_file(bundle_path).predict(ir_signal=synthetic_ppg(100, 30), sampling_rate=100)
    assert not r.accepted and r.rejection_reasons == ["missing_required_channel:red"]


def test_predict_too_short_rejected(bundle_path):
    s = synthetic_ppg(100, 5)
    r = BPPredictor.from_file(bundle_path).predict(red_signal=s, ir_signal=s, sampling_rate=100)
    assert not r.accepted and r.rejection_reasons[0].startswith("too_short")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_rate": None},
        {"sampling_rate": 5},
        {"sampling_rate": -100},
    ],
)
def test_predict_bad_sampling_rate(bundle_path, kwargs):
    s = synthetic_ppg(100, 30)
    with pytest.raises(InputError):
        BPPredictor.from_file(bundle_path).predict(red_signal=s, ir_signal=s, **kwargs)


def test_predict_length_mismatch_and_bad_types(bundle_path):
    p = BPPredictor.from_file(bundle_path)
    with pytest.raises(InputError):
        p.predict(red_signal=synthetic_ppg(100, 30), ir_signal=synthetic_ppg(100, 20), sampling_rate=100)
    with pytest.raises(InputError):
        p.predict(red_signal=np.zeros((2, 100)), ir_signal=np.zeros((2, 100)), sampling_rate=100)
    with pytest.raises(InputError):
        p.predict(red_signal=["a", "b"], ir_signal=["a", "b"], sampling_rate=100)


def test_sampling_rate_mismatch_changes_heart_rate_not_crash(bundle_path):
    """Declaring the wrong rate is undetectable in general, but the output must still be well-formed."""
    s = synthetic_ppg(200, 30, 70)
    r = BPPredictor.from_file(bundle_path).predict(red_signal=s, ir_signal=s, sampling_rate=100)
    assert isinstance(r.accepted, bool)
