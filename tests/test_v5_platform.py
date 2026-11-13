"""RakshPulse 5.0 platform tests: acquisition, storage, processing, calibration, splits, models, inference, CLI.

Synthetic signals here test *software behaviour* (timing, gaps, quality decisions, leakage rules). They are never
used as evidence about blood pressure.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from rakshpulse.acquisition.ingest import Recorder, TimestampSync
from rakshpulse.acquisition.simulator import ADC_MAX_18BIT, SimConfig, simulate
from rakshpulse.acquisition.sources import (
    LineFormat,
    MAX30102Config,
    ReplaySource,
    SampleBlock,
    SimulatedSensor,
    TextStreamSource,
)
from rakshpulse.datastore.association import associate_references
from rakshpulse.datastore.store import DatasetStore, SessionMetadata
from rakshpulse.preprocessing.pipeline import PipelineConfig, process_recording


def _raw(rec) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": rec.t, "sample_index": rec.sample_index, "red": rec.red, "infrared": rec.ir})


def _sim_session(store, sid="S1", ses="D1", seconds=120, **kw):
    cfg = SimConfig(seed=kw.pop("seed", 0), **kw)
    src = SimulatedSensor(cfg, seconds)
    with store.create_session(SessionMetadata(sid, ses, cfg.fs, sensor_type="simulator", placement="finger")) as w:
        rec = Recorder(src, [w.write_raw])
        rec.run(duration_s=seconds)
        w.add_gaps(rec.sync.gap_table())
    return src.rec


# ------------------------------------------------------------------------------------------- simulator
def test_simulator_is_deterministic_realistic_and_flagged():
    a, b = simulate(SimConfig(seed=4), 60), simulate(SimConfig(seed=4), 60)
    np.testing.assert_array_equal(a.ir, b.ir)
    assert a.flags["source"] == "simulated"
    assert 65 <= len(a.beat_times) <= 80  # 72 bpm ± drift over 60 s
    assert a.ir.max() <= ADC_MAX_18BIT and a.ir.min() >= 0
    ac_ir = np.ptp(a.ir) / np.median(a.ir)
    ac_red = np.ptp(a.red) / np.median(a.red)
    assert ac_red < ac_ir  # red perfusion amplitude below IR, as configured
    sat = simulate(SimConfig(dc_ir=ADC_MAX_18BIT, seed=1), 10)
    # DC at the rail: the systolic dip stays below it, the rest is clamped (never exceeds the 18-bit range)
    assert sat.ir.max() == ADC_MAX_18BIT and (sat.ir == ADC_MAX_18BIT).mean() > 0.1
    drop = simulate(SimConfig(dropout_prob=0.01, seed=2), 30)
    assert drop.ir.size < 3000 and np.any(np.diff(drop.sample_index) > 1)
    with pytest.raises(ValueError):
        simulate(SimConfig(), 0)


# ------------------------------------------------------------------------------------------- sources
def test_max30102_config_validation():
    c = MAX30102Config()
    assert c.validate() == [] and c.adc_bits == 18 and c.output_rate_hz == 100
    assert MAX30102Config(sample_rate_hz=800, pulse_width_us=411).validate()  # warning, not an error
    assert MAX30102Config(sample_rate_hz=400, sample_averaging=4).output_rate_hz == 100
    for bad in (
        {"sample_rate_hz": 123},
        {"pulse_width_us": 100},
        {"adc_range_na": 1000},
        {"led_current_ir_ma": 60},
        {"sample_averaging": 3},
        {"mode": "hr"},
    ):
        with pytest.raises(ValueError):
            MAX30102Config(**bad).validate()


def test_line_format_and_text_stream_count_bad_lines():
    fmt = LineFormat(fields=("device_time_us", "red", "ir", "temperature_c"))
    assert fmt.parse("# comment") is None and fmt.parse("1,2") is None and fmt.parse("a,b,c,d") is None
    v = fmt.parse("1000,95000,118000,31.5")
    assert v == {"device_time_us": 1000.0, "red": 95000.0, "ir": 118000.0, "temperature_c": 31.5}
    text = "\n".join(["# start"] + [f"{i * 10000},{95000 + i},{118000 + i},31.0" for i in range(10)] + ["garbage"])
    src = TextStreamSource(io.StringIO(text), 100.0, fmt, block=4)
    blocks = list(src.blocks())
    assert sum(len(b) for b in blocks) == 10 and src.bad_lines == 1
    assert blocks[0].device_time_s is not None and blocks[0].temperature_c is not None


# ------------------------------------------------------------------------------------------- timestamping
def _block(idx, host=0.0):
    idx = np.asarray(idx)
    return SampleBlock(red=np.ones(idx.size), ir=np.ones(idx.size), host_time_s=host, sample_index=idx)


def test_timestamp_sync_counter_gaps_and_wraparound():
    s = TimestampSync(100.0)
    a = s.process(_block([0, 1, 2]))
    b = s.process(_block([5, 6]))  # 3 and 4 missing
    assert list(a.timestamp) == [0.0, 0.01, 0.02] and list(b.sample_index) == [5, 6]
    assert s.n_missing == 2 and s.gaps[0].first_missing_index == 3 and s.gaps[0].detected_by == "counter"
    with pytest.raises(ValueError):
        s.process(_block([4]))  # backwards
    w = TimestampSync(100.0, counter_modulus=8)
    w.process(_block([5, 6, 7]))
    out = w.process(_block([0, 1, 3]))  # wrapped; 2 missing
    assert list(out.sample_index) == [3, 4, 6] and w.n_missing == 1


def test_timestamp_sync_device_time_and_host_time_gaps():
    s = TimestampSync(100.0)
    blk = SampleBlock(np.ones(3), np.ones(3), 0.0, device_time_s=np.array([10.00, 10.01, 10.05]))
    out = s.process(blk)
    assert out.timestamp.iloc[-1] == pytest.approx(0.05) and s.n_missing == 3
    h = TimestampSync(100.0, gap_tolerance_s=0.2)
    h.process(SampleBlock(np.ones(10), np.ones(10), host_time_s=0.09))
    out = h.process(SampleBlock(np.ones(10), np.ones(10), host_time_s=1.19))  # ~1 s late
    assert h.gaps and h.gaps[0].detected_by == "host_time" and out.sample_index.iloc[0] > 10


def test_recorder_limits_stop_and_drift():
    src = SimulatedSensor(SimConfig(), 30)
    got = []
    rec = Recorder(src, [got.append])
    s = rec.run(max_samples=1000)
    assert sum(len(x) for x in got) == 1000 and s["samples"] >= 1000 and abs(s["drift_ppm"]) < 1
    src2 = SimulatedSensor(SimConfig(), 30)
    rec2 = Recorder(src2, [lambda rows: rec2.stop()])
    assert rec2.run()["stopped_by"] == "stop()" and rec2.state == "stopped"


# ------------------------------------------------------------------------------------------- dataset store
def test_store_never_overwrites_and_detects_tampering(tmp_path):
    st = DatasetStore(tmp_path)
    _sim_session(st, seconds=60, dropout_prob=0.001)
    with pytest.raises(FileExistsError):
        st.create_session(SessionMetadata("S1", "D1", 100.0))
    v = st.validate()
    assert (v.status == "PASS").all(), v[v.status != "PASS"]
    assert (
        st.sessions().simulated.iloc[0]
        and len(st.raw("S1", "D1"))
        == len(st.raw("S1", "D1", chunksize=1000).__next__()) * 0
        + 6000
        - pd.read_csv(tmp_path / "sessions/S1/D1/gaps.csv").n_missing.sum()
    )
    raw = tmp_path / "sessions/S1/D1/raw_ppg.csv"
    raw.write_text(raw.read_text().replace(",S1,", ",S2,", 1))
    v = st.validate().set_index("check")
    assert v.loc["checksum:raw_ppg.csv", "status"] == "FAIL"


def test_store_references_derived_and_import_export(tmp_path):
    st = DatasetStore(tmp_path / "ds")
    _sim_session(st, seconds=60)
    with pytest.raises(ValueError):
        st.append_reference("S1", "D1", 30, 300, 80)  # implausible
    st.append_reference("S1", "D1", 30, 125, 80, 70, "cuff-X")
    assert len(st.references("S1", "D1")) == 1 and (st.validate().status == "PASS").all()
    p = st.write_derived("S1", "D1", "windows", pd.DataFrame({"a": [1]}))
    assert p.exists()
    with pytest.raises(FileExistsError):
        st.write_derived("S1", "D1", "windows", pd.DataFrame({"a": [2]}))
    with pytest.raises(ValueError):
        st.write_derived("S1", "D1", "raw_ppg", pd.DataFrame({"a": [2]}))
    with pytest.raises(ValueError):
        st.session_dir("../x", "D1")
    dst = st.export_session("S1", "D1", tmp_path / "export")
    assert (dst / "raw_ppg.csv").exists() and (dst / "manifest_rows.csv").exists()
    st2 = DatasetStore(tmp_path / "ds2")
    st2.import_session(
        SessionMetadata("S9", "D1", 100.0, sensor_type="simulator"), dst / "raw_ppg.csv", dst / "reference_bp.csv"
    )
    assert len(st2.raw("S9", "D1")) == len(st.raw("S1", "D1")) and (st2.validate().status == "PASS").all()


# ------------------------------------------------------------------------------------------- association
def test_reference_association_never_labels_unrelated_windows():
    w = pd.DataFrame(
        {
            "subject_id": "A",
            "session_id": ["D1"] * 6 + ["D2"] * 2,
            "t_start_s": [0, 10, 20, 30, 40, 50, 0, 10],
            "t_end_s": [10, 20, 30, 40, 50, 60, 10, 20],
            "quality_class": ["good"] * 5 + ["unreliable"] + ["good"] * 2,
        }
    )
    r = pd.DataFrame(
        {
            "subject_id": "A",
            "session_id": "D1",
            "timestamp": [12.0, 16.0, 200.0, 55.0],
            "sbp": [120, 130, 140, 150],
            "dbp": [80, 85, 90, 95],
        }
    )
    lw, lr = associate_references(w, r, tolerance_s=0.0, require_usable=True)
    assert lw.loc[1, "ref_sbp"] == 130  # both readings fall in window 1: the one nearer its centre (16 s) wins
    assert lw.loc[lw.session_id == "D2", "ref_sbp"].isna().all()  # other session never labelled
    st = lr.set_index("timestamp").status
    assert st[200.0] == "no_window_within_tolerance" and st[55.0] == "no_usable_window_within_tolerance"
    assert st[12.0] == "superseded_by_closer_reading" and lw.ref_sbp.notna().sum() == 1


# ------------------------------------------------------------------------------------------- pipeline
def test_pipeline_chunked_equals_whole_and_handles_gaps_and_jitter():
    rec = simulate(SimConfig(seed=5, dropout_prob=0.0005), 90)
    raw = _raw(rec)
    raw = raw[(raw.timestamp < 40) | (raw.timestamp > 43)]  # long gap
    a, sa = process_recording(raw, PipelineConfig(), "S", "D")
    b, sb = process_recording((raw.iloc[i : i + 777] for i in range(0, len(raw), 777)), PipelineConfig(), "S", "D")
    assert len(a) == len(b) and np.allclose(a.select_dtypes("number").fillna(-9), b.select_dtypes("number").fillna(-9))
    assert sa.equals(sb) and len(sa) == 2  # split at the long gap only; short gaps filled
    assert not ((a.t_start_s < 43) & (a.t_end_s > 40)).any()  # no window spans the gap
    assert a.reasons.str.contains("filled_").any()
    jit = raw.copy()
    jit["timestamp"] = jit.timestamp + np.random.default_rng(0).uniform(-0.003, 0.003, len(jit))  # host jitter
    c, _ = process_recording(jit, PipelineConfig(), "S", "D")
    assert len(c) == len(a) and c.reasons.str.contains("resampled_irregular").any()
    empty, _ = process_recording(raw.iloc[:300], PipelineConfig(), "S", "D")  # too short for one window
    assert empty.empty and "status" in empty.columns


def test_pipeline_quality_classes_and_heart_rate():
    rec = simulate(SimConfig(seed=6, hr_bpm=80, hr_drift_bpm=0), 90)
    raw = _raw(rec)
    raw.loc[raw.timestamp.between(30, 45), ["red", "infrared"]] = ADC_MAX_18BIT  # saturation
    raw.loc[raw.timestamp.between(55, 70), ["red", "infrared"]] = 60000.0  # detached: flat
    w, _ = process_recording(raw, PipelineConfig(), "S", "D")
    clean = w[w.t_end_s <= 30]
    assert (clean.status == "usable").all() and np.allclose(clean.ir_hr, 80, atol=3)
    sat = w[(w.t_start_s >= 30) & (w.t_end_s <= 45)]
    assert (sat.status.isin(["unusable", "missing"])).all() and sat.reasons.str.len().gt(0).all()
    flat = w[(w.t_start_s >= 55) & (w.t_end_s <= 70)]
    assert (flat.status == "missing").all() and flat.reasons.str.contains("flatline").all()
    motion = simulate(SimConfig(seed=7, motion_times_s=(20,), motion_frac=0.08), 40)
    m, _ = process_recording(_raw(motion), PipelineConfig(), "S", "D")
    assert (m[(m.t_start_s < 21) & (m.t_end_s > 19)].status != "usable").any()


# ------------------------------------------------------------------------------------------- calibration
def _calib_frame(n_sub=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        off = rng.normal(0, 8)
        for ses in ("D1", "D2"):
            day = rng.normal(0, 4)
            for i in range(60):
                t = i * 20.0
                truth = 120 + off + day + 5 * np.sin(t / 300)
                rows.append(
                    {
                        "subject_id": f"P{s}",
                        "session_id": ses,
                        "t": t,
                        "y": truth + rng.normal(0, 2),
                        "pred": 120 + 0.8 * (truth - 120 - off - day) + rng.normal(0, 3),
                        "pmean": 120.0,
                    }
                )
    return pd.DataFrame(rows)


def test_calibration_split_never_evaluates_readings():
    from rakshpulse.personalization.strategies import CalibrationPolicy, pick_readings, split_readings

    f = _calib_frame()
    pol = CalibrationPolicy(min_spacing_s=60, exclude_s=25)
    g = f[f.subject_id == "P0"].sort_values(["session_id", "t"])
    r = pick_readings(g[g.session_id == "D1"], 3, pol)
    assert list(r.t) == [0.0, 60.0, 120.0]
    ev = split_readings(g, r, pol)
    assert not set(ev.index) & set(r.index)
    d1 = ev[ev.session_id == "D1"]
    assert d1.t.min() > 120 + 0 and (np.abs(d1.t.to_numpy()[:, None] - r.t.to_numpy()[None]) > 25).all()
    assert (ev.session_id == "D2").any()  # later sessions remain available for cross-session tests


def test_calibration_policy_expiry_and_strategies():
    from rakshpulse.personalization.strategies import STRATEGIES, CalibrationPolicy, calibration_status, evaluate

    pol = CalibrationPolicy(expiry_s=600, max_halfwidth_mmhg=10)
    r = pd.DataFrame({"t": [0.0], "session_id": ["D1"], "y": [120.0]})
    assert calibration_status(r, 100, "D1", pol) == (True, "valid")
    assert calibration_status(r, 700, "D1", pol)[1] == "expired_age"
    assert calibration_status(r, 100, "D2", pol)[1] == "too_few_readings_in_session"
    assert calibration_status(r, 100, "D1", pol, halfwidth=12)[1] == "expired_uncertainty"
    f = _calib_frame()
    folds = {f"P{i}": i % 3 for i in range(12)}
    scen = {
        "within": {"k": 2, "reading_session": "D1", "eval_sessions": ("D1",)},
        "cross": {"k": 2, "reading_session": "D1", "eval_sessions": ("D2",)},
    }
    P, fit = evaluate(f, folds, list(STRATEGIES), CalibrationPolicy(), scen)
    mae = P.assign(ae=(P.yhat - P.y).abs()).groupby(["scenario", "strategy"]).ae.mean()
    assert mae["within", "cuff_hierarchical"] < mae["within", "population"]  # the cuff carries the offset
    assert (fit.groupby("strategy").fit_mae_at_readings.mean()["cuff_mean"] < 3) and set(fit.strategy) == set(
        STRATEGIES
    )
    assert not P[P.scenario == "cross"].calibration_valid.any()  # day-1 readings never validate day 2


# ------------------------------------------------------------------------------------------- splits / adaptation
def test_splits_are_leakage_checked():
    from rakshpulse.evaluation.splits import SplitSpec, check_no_leakage, make_splits

    f = _calib_frame()
    for spec in (
        SplitSpec("subject", 3),
        SplitSpec("cross_day", 3, train_sessions=("D1",)),
        SplitSpec("session", 3, train_sessions=("D1",)),
        SplitSpec("session", train_sessions=("D1",), same_subjects=True),
    ):
        for tr, te in make_splits(f, spec):
            a, b = f.iloc[tr], f.iloc[te]
            if spec.kind != "session" or not spec.same_subjects:
                assert not set(a.subject_id) & set(b.subject_id)
            if spec.kind != "subject":
                assert set(a.session_id) == {"D1"} and set(b.session_id) == {"D2"}
    with pytest.raises(AssertionError):
        check_no_leakage(f, np.arange(10), np.arange(5, 15), SplitSpec("subject"))
    with pytest.raises(ValueError):
        make_splits(f.iloc[:60], SplitSpec("subject", 5))


def test_domain_adaptation_transforms():
    from rakshpulse.models.domain_adaptation import CORAL, SessionStandardizer

    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (50, 3)), rng.normal(5, 2, (50, 3))]
    g = np.repeat(["a", "b"], 50)
    z = SessionStandardizer().transform(X, g)
    assert np.allclose(z[:50].mean(0), 0, atol=1e-9) and np.allclose(z[50:].std(0), 1, atol=1e-9)
    src = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 2000)
    tgt = rng.multivariate_normal([3, -1], [[4, 0], [0, 0.25]], 2000)
    al = CORAL(eps=1e-6).fit(src).transform(tgt)
    assert np.allclose(np.cov(al.T), np.cov(src.T), atol=0.1) and np.allclose(al.mean(0), src.mean(0), atol=0.1)


# ------------------------------------------------------------------------------------------- models + inference
@pytest.fixture(scope="module")
def sim_table(tmp_path_factory):
    """A small simulated feature table with associated synthetic cuff readings (software test only)."""
    from rakshpulse.cli import main

    root = tmp_path_factory.mktemp("sim")
    assert (
        main(
            [
                "simulate",
                "--out",
                str(root / "ds"),
                "--subjects",
                "5",
                "--sessions",
                "2",
                "--duration",
                "120",
                "--reference-every",
                "30",
                "--seed",
                "3",
            ]
        )
        == 0
    )
    assert main(["preprocess", "--dataset", str(root / "ds")]) == 0
    assert main(["features", "--dataset", str(root / "ds"), "--out", str(root / "table.csv")]) == 0
    return root, pd.read_csv(root / "table.csv", dtype={"subject_id": str, "session_id": str})


def test_model_training_versioning_and_oof(sim_table):
    from rakshpulse.evaluation.splits import SplitSpec
    from rakshpulse.models.registry import TrainConfig, load_model, save_model, train_model

    root, t = sim_table
    assert t.simulated.all() and t.y_sbp.notna().any()
    b, cv, oof = train_model(t, TrainConfig(model="ridge", split=SplitSpec("subject", 5), hyperparams={"alpha": 10}))
    assert len(cv) == 10 and b.validation["conformal_90_halfwidth_sbp"] > 0 and b.version.startswith("rakshpulse-")
    assert oof.p_sbp.notna().all()  # every eligible window predicted exactly once, out of fold
    path = save_model(b, root / "models")
    with pytest.raises(FileExistsError):
        save_model(b, root / "models")
    again = load_model(root / "models")
    X = t.dropna(subset=["y_sbp"]).reindex(columns=b.features).to_numpy(float)[:5]
    np.testing.assert_allclose(again.sbp.predict(X), b.sbp.predict(X))
    assert path.exists() and pd.read_csv(root / "models" / "registry.csv").version.iloc[-1] == b.version
    b2, cv2, _ = train_model(
        t, TrainConfig(model="random_forest", split=SplitSpec("cross_day", 5, train_sessions=("D1",)))
    )
    assert set(cv2.target) == {"sbp", "dbp"}


def test_inference_engine_end_to_end(sim_table):
    from rakshpulse.inference.engine import EngineConfig, InferenceEngine
    from rakshpulse.models.registry import TrainConfig, train_model
    from rakshpulse.personalization.hierarchical import HierParams
    from rakshpulse.personalization.strategies import CalibrationPolicy

    _, t = sim_table
    b, _, _ = train_model(t, TrainConfig(hyperparams={"alpha": 10}))
    hp = {k: HierParams(100, 0, 10, 20, 0.01) for k in ("sbp", "dbp")}
    eng = InferenceEngine(
        b, EngineConfig(fs=100.0, sensor="simulator", hop_s=5, policy=CalibrationPolicy(expiry_s=40)), hier_params=hp
    )
    rec = simulate(SimConfig(seed=11), 120)
    raw = _raw(rec)
    out = []
    for i in range(0, len(raw), 32):
        out += eng.push(raw.iloc[i : i + 32])
    assert out[0].status == "warming_up" and out[-1].status == "usable"
    p = eng.predictions()
    pop = p[p.estimate_type == "population"].iloc[0]
    assert pop.sbp_hi - pop.sbp_lo == pytest.approx(2 * b.validation["conformal_90_halfwidth_sbp"])
    assert eng.add_reference(30.0, 130.0, 85.0)["accepted"]
    more = []
    for i in range(0, 1500, 32):
        more += eng.push(
            pd.DataFrame(
                {
                    "timestamp": rec.t[-1] + (np.arange(i, min(i + 32, 1500)) + 1) / 100,
                    "sample_index": rec.sample_index[-1] + np.arange(i, min(i + 32, 1500)) + 1,
                    "red": rec.red[: min(32, 1500 - i)],
                    "infrared": rec.ir[: min(32, 1500 - i)],
                }
            )
        )
    assert {x.calibration_reason for x in more} >= {"expired_age"}  # the reading is more than 40 s old
    with pytest.raises(ValueError):
        eng.push(raw.iloc[:5])  # time went backwards
    gap = raw.iloc[:1200].copy()
    gap.loc[gap.index >= 600, "sample_index"] += 50  # 50 missing samples inside the next windows
    e2 = InferenceEngine(b, EngineConfig(fs=100.0, sensor="simulator"))
    st = [x.status for x in e2.push(gap)]
    assert "missing" in st and e2.predictions().reasons.str.contains("gap_in_window").any()


def test_inference_engine_reports_errors_and_maps_prefixes(sim_table):
    from rakshpulse.inference.engine import EngineConfig, InferenceEngine
    from rakshpulse.models.registry import TrainConfig, train_model

    _, t = sim_table
    b, _, _ = train_model(t, TrainConfig(hyperparams={"alpha": 10}))

    class Broken:
        def predict(self, X):
            raise RuntimeError("model failure")

    b.sbp = Broken()
    e = InferenceEngine(b, EngineConfig(fs=100.0, sensor="simulator"))
    rec = simulate(SimConfig(seed=12), 30)
    p = e.push(_raw(rec))
    assert p[-1].status == "error" and "model failure" in p[-1].reasons
    ear = type(b)(b.dbp, b.dbp, [c.replace("ir_", "ear_", 1) for c in b.features], ("ear",), False, {}, "ear-test")
    e3 = InferenceEngine(ear, EngineConfig(fs=100.0, sensor="simulator", feature_prefix_map={"ir_": "ear_"}))
    assert np.isfinite(e3.push(_raw(rec))[-1].sbp_window)
    with pytest.raises(ValueError):
        InferenceEngine(b, EngineConfig(fs=100.0, sensor="simulator", hop_s=0))


def test_replay_source_round_trip(tmp_path):
    st = DatasetStore(tmp_path)
    _sim_session(st, seconds=30)
    raw = st.raw("S1", "D1")
    rows = []
    rec = Recorder(ReplaySource(st.session_dir("S1", "D1") / "raw_ppg.csv", 100.0), [rows.append])
    rec.run()
    back = pd.concat(rows)
    np.testing.assert_allclose(back.infrared.to_numpy(), raw.infrared.to_numpy())
    np.testing.assert_allclose(back.timestamp.to_numpy(), raw.timestamp.to_numpy(), atol=1e-9)


# ------------------------------------------------------------------------------------------- CLI
def test_cli_doctor_errors_and_simulated_guard(tmp_path, sim_table, capsys):
    from rakshpulse.cli import main

    root, _ = sim_table
    assert main(["doctor"]) == 0
    assert main(["inspect", "--dataset", str(tmp_path / "nothing")]) == 2
    assert main(["train", "--table", str(root / "table.csv"), "--models-dir", str(tmp_path / "m")]) == 2
    assert "simulated" in capsys.readouterr().err
    assert main(["inspect", "--dataset", str(root / "ds"), "--validate"]) == 0
    assert main(["preprocess", "--dataset", str(root / "ds")]) == 2  # refuses to overwrite derived files
