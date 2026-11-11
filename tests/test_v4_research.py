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
