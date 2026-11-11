"""RakshPulse 4.0: statistics, normalisation leakage, physiological features, registry, quality, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------------------------------ statistics


def test_multiple_comparison_corrections_match_hand_calculation():
    from rakshpulse.evaluation.statistics import benjamini_hochberg, holm

    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.04, 0.03, 0.5]), [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert np.isnan(holm([np.nan, 0.2])[0]) and holm([np.nan, 0.2])[1] == pytest.approx(0.2)
    assert (holm(np.full(5, 0.3)) <= 1).all()


def test_paired_test_is_subject_level_and_exact_for_small_n():
    from rakshpulse.evaluation.statistics import paired_test

    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = paired_test(a, a + 1.0, n_boot=200)
    assert r["mean_diff"] == pytest.approx(-1.0) and r["n_units"] == 6
    assert r["p_perm"] == pytest.approx(2 / 64)  # exact sign-flip: only all-negative and all-positive as extreme
    null = paired_test(a, a, n_boot=200)
    assert null["mean_diff"] == 0 and null["p_perm"] == pytest.approx(1.0)


def test_bland_altman_repeated_recovers_variance_components():
    from rakshpulse.evaluation.statistics import bland_altman_repeated

    rng = np.random.default_rng(1)
    rows = []
    for s in range(60):
        off = rng.normal(2.0, 4.0)
        for _ in range(40):
            rows.append((s, 100.0, 100.0 + off + rng.normal(0, 3.0)))
    t = pd.DataFrame(rows, columns=["subject_id", "y", "yhat"])
    r = bland_altman_repeated(t, n_boot=100)
    assert r["bias"] == pytest.approx(2.0, abs=1.5)
    assert r["sd_within_subject"] == pytest.approx(3.0, abs=0.3)
    assert r["sd_between_subject"] == pytest.approx(4.0, abs=1.2)
    assert r["loa_low"] < r["bias"] < r["loa_high"] and r["n_subjects"] == 60


def test_icc31():
    from rakshpulse.evaluation.statistics import icc31

    a = np.arange(10.0)
    assert icc31(a, a + 3) == pytest.approx(1.0)
    assert np.isnan(icc31(a[:4], a[:4]))


# ------------------------------------------------------------------------------------ normalisation


def _subject_days(seed: int = 0):
    rng = np.random.default_rng(seed)
    d1 = pd.DataFrame({"t_start_s": np.arange(0, 600, 5.0), "f": rng.normal(1.0, 0.2, 120)})
    d2 = pd.DataFrame({"t_start_s": np.arange(0, 600, 5.0), "f": rng.normal(3.0, 0.4, 120)})
    return d1, d2, d1.iloc[:3]


@pytest.mark.parametrize("method", ["raw", "session", "subject", "robust", "adaptive", "domain"])
def test_normalisation_is_causal_and_label_free(method):
    """Changing day-2 data AFTER the warm-up must not change how earlier day-2 windows are normalised."""
    from rakshpulse.evaluation.domain_shift import normalise

    d1, d2, ref = _subject_days()
    zr, z2 = normalise(method, d1, d2, ref, ["f"], warmup_s=180)
    late = d2.t_start_s >= 400
    d2b = d2.copy()
    d2b.loc[late, "f"] += 50.0  # future data: must not leak into earlier windows
    zr_b, z2_b = normalise(method, d1, d2b, ref, ["f"], warmup_s=180)
    assert np.allclose(zr, zr_b)
    early = (~late).to_numpy()
    assert np.allclose(z2[early], z2_b[early])
    assert "sbp_mmHg" not in d1 and np.isfinite(z2).all()


def test_session_and_domain_normalisation_remove_a_feature_offset():
    from rakshpulse.evaluation.domain_shift import normalise

    d1, d2, ref = _subject_days()
    zr, z2 = normalise("session", d1, d2, ref, ["f"])
    assert abs(np.median(z2) - 0) < 0.15 and abs(zr[0]) < 0.3  # both days centred on their own warm-up
    zr, z2 = normalise("domain", d1, d2, ref, ["f"])
    assert np.median(z2) == pytest.approx(np.median(d1.f), abs=0.15)  # day 2 mapped onto day 1's scale
    with pytest.raises(KeyError):
        normalise("nope", d1, d2, ref, ["f"])


def test_shift_diagnostics_detect_an_injected_cross_day_shift():
    from rakshpulse.evaluation.domain_shift import shift_diagnostics

    rng = np.random.default_rng(0)
    rows = []
    for s in range(12):
        for sess, mu in ((1, 0.0), (2, 2.0)):
            for i in range(30):
                rows.append(
                    {
                        "subject_id": s,
                        "session": sess,
                        "activity": 0,
                        "t_start_s": i * 5.0,
                        "shifted": rng.normal(mu, 1),
                        "stable": rng.normal(0, 1),
                        "sbp_mmHg": 120.0,
                    }
                )
    feats, subj = shift_diagnostics(pd.DataFrame(rows), ["shifted", "stable"])
    f = feats.set_index("feature")
    assert f.loc["shifted", "cross_shift_z_median"] > 1.2 > f.loc["stable", "cross_shift_z_median"]
    assert f.loc["shifted", "shift_excess_p"] < 0.01 and len(subj) == 12


# --------------------------------------------------------------------------------- physiological


def _pulse(n: int = 100, notch: bool = True) -> np.ndarray:
    t = np.linspace(0, 1, n)
    sys_ = np.exp(-((t - 0.18) ** 2) / 0.004)
    dia = 0.55 * np.exp(-((t - 0.5) ** 2) / 0.006) if notch else 0.0
    return sys_ + dia + 0.05 * (1 - t)


def test_template_features_find_the_dicrotic_notch():
    from rakshpulse.features.physiological import template_features

    f = template_features(_pulse())
    assert f["notch_present"] == 1.0 and 0.25 < f["notch_t"] < 0.45
    assert 0.4 < f["dia_peak_h"] < 0.7 and f["dia_peak_t"] == pytest.approx(0.5, abs=0.05)
    assert f["sys_dia_dt"] == pytest.approx(0.32, abs=0.05) and f["upstroke_10_90"] > 0
    g = template_features(_pulse(notch=False) / _pulse(notch=False).max())
    assert g["notch_present"] == 0.0  # monotone fall: inflection fallback


def test_physio_features_on_a_synthetic_window():
    from conftest import synthetic_ppg

    from rakshpulse.features.physiological import PHYSIO_FEATURES, physio_features

    f, tpl = physio_features(synthetic_ppg(100.0, 10, 72, raw_counts=False), 100.0)
    assert set(f) == set(PHYSIO_FEATURES) and tpl is not None and tpl.shape == (100,)
    assert f["beat_template_corr"] > 0.9 and 0 <= f["beat_amp_cv"] < 0.5
    f0, t0 = physio_features(np.zeros(1000), 100.0)
    assert t0 is None and all(np.isnan(v) for v in f0.values())


# ------------------------------------------------------------------------------------------ registry


def test_registry_records_status_phases_and_keeps_old_rows(tmp_path, monkeypatch):
    from rakshpulse.experiments import registry

    monkeypatch.setattr(registry, "RUNS", tmp_path)
    data = tmp_path / "d.csv"
    data.write_text("a\n1\n")
    pd.DataFrame([{"run_id": "old-1", "name": "old", "seed": 0}]).to_csv(tmp_path / "runs.csv", index=False)
    run = registry.ExperimentRun(
        "v4test",
        [data],
        {"k": 1},
        seed=3,
        dataset_versions={"d": "v1"},
        feature_config={"set": "x"},
        model_config={"m": "ridge"},
    )
    with run.phase("train"):
        pass
    with run.phase("evaluate"):
        pass
    out = run.finish(status="completed")
    cfg = pd.read_csv(out / "config.csv").set_index("key").value
    assert cfg["status"] == "completed" and cfg["dataset_version:d"] == "v1" and cfg["model_config:m"] == "ridge"
    assert "phase_duration_s:train" in cfg and cfg["git_dirty_code"] in ("True", "False")
    idx = pd.read_csv(tmp_path / "runs.csv")
    assert list(idx.run_id) == ["old-1", run.run_id] and "status" in idx and "train_s" in idx


# ------------------------------------------------------------------------------------------- quality


def test_quality_score_grades_good_degraded_unreliable_and_missing():
    from conftest import synthetic_ppg

    from rakshpulse.features.feature_pipeline import preprocess_channel
    from rakshpulse.preprocessing.quality_score import score_window

    fs = 100.0
    clean = synthetic_ppg(fs, 10, 72, seed=0, noise=0.005, raw_counts=False)
    good = score_window(preprocess_channel(clean, fs, has_dc=False).filtered, clean, fs, flat_sd=1e-6)
    assert good.quality_class == "good" and good.score > 90
    rng = np.random.default_rng(3)
    noisy = clean + rng.normal(0, 0.6, clean.size)
    bad = score_window(preprocess_channel(noisy, fs, has_dc=False).filtered, noisy, fs, flat_sd=1e-6)
    assert bad.quality_class in ("degraded", "unreliable") and bad.score < good.score
    flat = np.full(1000, 0.5)
    assert score_window(flat, flat, fs).quality_class == "missing"
    nan = clean.copy()
    nan[10] = np.nan
    assert score_window(nan, nan, fs).quality_class == "missing"
    railed = score_window(preprocess_channel(clean, fs, has_dc=False).filtered, clean, fs, flat_sd=1e-6, rail_pct=5.0)
    assert railed.quality_class == "unreliable" and "rail_clipping" in railed.reasons


def test_quality_low_relative_amplitude_is_degraded():
    from conftest import synthetic_ppg

    from rakshpulse.features.feature_pipeline import preprocess_channel
    from rakshpulse.preprocessing.quality_score import components, grade

    x = synthetic_ppg(100.0, 10, 72, seed=0, noise=0.002, raw_counts=False)
    c = components(preprocess_channel(x, 100.0, has_dc=False).filtered, x, 100.0, ref_amplitude=None)
    assert grade(c).quality_class == "good"
    c["rel_amplitude"] = 0.1  # pulse a tenth of the stream's usual amplitude: low perfusion / poor contact
    q = grade(c)
    assert q.quality_class == "degraded" and "margin:rel_amplitude" in q.reasons


def test_fast_moments_match_scipy():
    from scipy import stats

    from rakshpulse.features import moments

    x = np.random.default_rng(0).gamma(2.0, size=500)
    assert moments.skew(x) == pytest.approx(stats.skew(x), rel=1e-12)
    assert moments.kurtosis(x) == pytest.approx(stats.kurtosis(x), rel=1e-12)
    assert np.isnan(moments.skew(np.ones(5))) and np.isnan(moments.kurtosis(np.ones(5)))


# ------------------------------------------------------------------------------ hierarchical calibration


def _hier_data(seed=0, n=40, sb=6.0, sc=4.0, sr=3.0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        b = rng.normal(0, sb)
        for sess in (1, 2):
            c = rng.normal(0, sc)
            for i in range(60):
                rows.append({"subject_id": s, "session": sess, "t_start_s": i * 10.0, "r": b + c + rng.normal(0, sr)})
    return pd.DataFrame(rows)


def test_hierarchical_params_recover_variance_components():
    from rakshpulse.personalization.hierarchical import fit_params

    p = fit_params(_hier_data(n=200))
    assert p.sigma_b2 == pytest.approx(36, rel=0.35)
    assert p.sigma_c2 == pytest.approx(16, rel=0.35)
    assert p.sigma_w2 == pytest.approx(9, rel=0.15) and p.kappa < 0.01  # no drift simulated


def test_hierarchical_posterior_shrinks_and_transfers_across_days():
    from rakshpulse.personalization.hierarchical import HierarchicalCalibrator, HierParams

    hc = HierarchicalCalibrator(HierParams(sigma_b2=36, sigma_c2=16, sigma_r2=9, sigma_w2=9, kappa=0.0))
    m0, v0 = hc.posterior(np.array([]), np.array([], bool))
    assert m0 == 0 and v0 == pytest.approx(52)
    m1, v1 = hc.posterior(np.array([10.0]), np.array([True]))
    assert m1 == pytest.approx(10 * 52 / 61) and v1 < v0  # one current reading: shrunk towards 0
    mp, vp = hc.posterior(np.array([10.0, 10.0, 10.0]), np.array([False, False, False]))
    assert 0 < mp < 10 * 36 / 52 + 1e-9 and vp > 16  # earlier-day readings inform b only
    out = hc.predict(np.array([120.0, 120.0]), np.array([0.0, np.nan]), np.array([10.0]), np.array([True]))
    assert (out.hi > out.lo).all() and out.sd.iloc[1] >= out.sd.iloc[0]


def test_interval_metrics_coverage_and_winkler():
    from rakshpulse.personalization.hierarchical import interval_metrics

    y = np.array([0.0, 1.0, 2.0, 10.0])
    r = interval_metrics(y, np.full(4, -1.0), np.full(4, 3.0), 0.9, np.array([1, 1, 2, 2]))
    assert r["coverage"] == pytest.approx(0.75) and r["mean_width"] == pytest.approx(4.0)
    assert r["winkler"] > 4.0 and r["n_subjects"] == 2


# ----------------------------------------------------------------------------------- export / streaming


def test_linear_export_matches_sklearn_and_c_header(stream_bundle_v4):
    from rakshpulse.inference.export import export_linear, predict_exported, to_c_header

    b, X = stream_bundle_v4
    params = export_linear(b)
    np.testing.assert_allclose(predict_exported(params, X, "sbp"), b.sbp.predict(X), atol=1e-9)
    np.testing.assert_allclose(predict_exported(params, X, "dbp", "float32"), b.dbp.predict(X), atol=1e-3)
    Xn = X.copy()
    Xn[0, 0] = np.nan  # missing value -> imputed exactly as the pipeline does
    np.testing.assert_allclose(predict_exported(params, Xn, "sbp")[0], b.sbp.predict(Xn)[0], atol=1e-9)
    h = to_c_header(params)
    assert "rakshpulse_bp_predict" in h and h.count("static const float") == 6


@pytest.fixture
def stream_bundle_v4():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from rakshpulse.inference.model_loader import ModelBundle

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))

    def pipe(y):
        return Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ).fit(X, y)

    b = ModelBundle(
        pipe(X @ [3, -2, 1, 0] + 120),
        pipe(X @ [1, 1, 0, 2] + 75),
        [f"f{i}" for i in range(4)],
        ("ear",),
        False,
        {},
        "export-test",
    )
    return b, X


def test_ring_buffer_matches_a_shifting_buffer():
    from rakshpulse.inference.streaming import RingBuffer

    rng = np.random.default_rng(0)
    ref = np.zeros(50)
    rb = RingBuffer(50)
    for n in (7, 13, 50, 3, 61, 1, 29):
        x = rng.normal(size=n)
        ref = np.concatenate([ref, x])[-50:]
        rb.write(x)
        np.testing.assert_array_equal(rb.last(20), ref[-20:])
    assert rb.filled == 50


def test_sensor_replay_drives_the_estimator_with_quality(stream_bundle):
    from conftest import synthetic_ppg

    from rakshpulse.inference.sensor import ArraySource, PPGSource, run_stream
    from rakshpulse.inference.streaming import StreamingEstimator

    src = ArraySource(synthetic_ppg(100.0, 30, 72, raw_counts=False), 100.0, chunk=32)
    assert isinstance(src, PPGSource)
    est = StreamingEstimator(stream_bundle, fs=100.0, grade_quality=True)
    ev = list(run_stream(src, est))
    assert len(ev) == 6 and ev[-1].accepted and ev[-1].quality_class in ("good", "degraded")
    with pytest.raises(ValueError):
        list(run_stream(ArraySource(np.zeros(10), 50.0), est))
