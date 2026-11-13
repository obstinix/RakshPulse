"""RakshPulse 5.1: model contract, safe inference, serial acquisition, sampling rates and the hardware handoff.

Signals are simulated: these tests check software behaviour, never blood-pressure accuracy.
"""

from __future__ import annotations

import io
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rakshpulse.acquisition.ingest import Recorder, TimestampSync
from rakshpulse.acquisition.simulator import SimConfig, simulate
from rakshpulse.acquisition.sources import (
    AcquisitionError,
    LineFormat,
    SampleBlock,
    SerialSource,
    TextStreamSource,
    check_serial,
    parse_banner,
)
from rakshpulse.inference.contract import (
    IncompatibleModelError,
    InvalidInputError,
    build_feature_vector,
    check_compatibility,
    legacy_contract,
    validate_bundle,
)
from rakshpulse.inference.engine import EngineConfig, InferenceEngine, predict_windows
from rakshpulse.inference.model_loader import ModelBundle, load_bundle
from rakshpulse.preprocessing.pipeline import PipelineConfig, process_recording

ROOT = Path(__file__).resolve().parents[1]
EAR_V2 = ROOT / "models" / "auricle_ear_ridge_v2.joblib"
HW = ROOT / "hardware"


def _raw(rec) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": rec.t, "sample_index": rec.sample_index, "red": rec.red, "infrared": rec.ir})


def _feed(engine, raw, chunk=32):
    out = []
    for i in range(0, len(raw), chunk):
        out += engine.push(raw.iloc[i : i + chunk])
    return out


@pytest.fixture(scope="module")
def ear():
    return load_bundle(EAR_V2)


@pytest.fixture(scope="module")
def demo_bundle():
    """A tiny ridge bundle that needs demographics (legacy contract: demographics are mandatory)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats = ["ir_hr", "ir_sys_ratio", "demo_age", "demo_sex", "demo_height_cm", "demo_weight_kg"]
    rng = np.random.default_rng(0)
    X = np.c_[
        rng.normal(70, 8, 200),
        rng.normal(0.3, 0.05, 200),
        rng.normal(40, 10, 200),
        rng.integers(0, 2, 200),
        rng.normal(170, 8, 200),
        rng.normal(70, 10, 200),
    ]

    def pipe(y):
        return Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ).fit(X, y)

    return ModelBundle(pipe(X[:, 2] * 0.5 + 100), pipe(X[:, 2] * 0.2 + 60), feats, ("ir",), True, {}, "demo-test")


# ------------------------------------------------------------------------------------------- contract
def test_shipped_model_contract_is_valid_and_honest(ear):
    c = validate_bundle(ear)
    assert c["model_id"] == "auricle-ear-ridge" and not c["validated_for"] and not c["experimental_for"]
    assert "max30102" in c["demonstration_for"] and c["uncertainty"]["valid_for"] == ["auricle_earlobe"]
    assert [f["name"] for f in c["features"]] == ear.features
    assert all(f["source"].startswith("ir_") for f in c["features"])
    assert sum(not f["required"] for f in c["features"]) == 2  # only the two features ever missing in training
    m = check_compatibility(c, "max30102", 100.0)
    assert m["mode"] == "demonstration" and m["uncertainty"] == {} and "not transferable" in m["uncertainty_note"]
    assert check_compatibility(c, "auricle_earlobe", 100.0)["uncertainty"]["sbp"] > 0


def test_compatibility_rejects_unsupported_inputs(ear):
    c = validate_bundle(ear)
    for kwargs, msg in (
        ({"sensor": "ecg_patch", "fs": 100.0}, "does not support sensor"),
        ({"sensor": "max30102", "fs": 20.0}, "outside the model's supported range"),
        ({"sensor": "max30102", "fs": 100.0, "processing": {"window_s": 8.0}}, "preprocessing mismatch"),
        ({"sensor": "max30102", "fs": 100.0, "channels": ("red",)}, "needs channels"),
    ):
        with pytest.raises(IncompatibleModelError, match=msg):
            check_compatibility(c, **kwargs)


def test_bundle_validation_rejects_tampered_bundles(ear):
    import copy

    b = copy.deepcopy(ear)
    b.contract["features"] = list(reversed(b.contract["features"]))
    with pytest.raises(IncompatibleModelError, match="does not match bundle.features"):
        validate_bundle(b)
    b2 = copy.deepcopy(ear)
    b2.features = b2.features[:-1]
    b2.contract["features"] = b2.contract["features"][:-1]
    with pytest.raises(IncompatibleModelError, match="expects 28 inputs"):
        validate_bundle(b2)
    b3 = copy.deepcopy(ear)
    b3.contract["contract_version"] = 99
    with pytest.raises(IncompatibleModelError, match="newer than this software"):
        validate_bundle(b3)


def test_feature_vector_requires_required_inputs_and_never_passes_nan_or_inf(ear):
    c = validate_bundle(ear)
    feats = {f["source"]: 1.0 for f in c["features"]}
    X, imputed = build_feature_vector(c, feats, {})
    assert X.shape == (1, 28) and np.isfinite(X).all() and imputed == []
    req = next(f for f in c["features"] if f["required"])
    with pytest.raises(InvalidInputError, match="required features missing"):
        build_feature_vector(c, {**feats, req["source"]: np.nan}, {})
    with pytest.raises(InvalidInputError, match="infinite"):
        build_feature_vector(c, {**feats, req["source"]: np.inf}, {})
    with pytest.raises(InvalidInputError, match="required features missing"):
        build_feature_vector(c, {k: v for k, v in feats.items() if k != req["source"]}, {})
    opt = [f for f in c["features"] if not f["required"]]
    X, imputed = build_feature_vector(c, {**feats, opt[0]["source"]: np.nan}, {})
    assert imputed == [opt[0]["name"]] and X[0, c["features"].index(opt[0])] == pytest.approx(opt[0]["fill"])
    two_missing = {**feats, opt[0]["source"]: np.nan, opt[1]["source"]: np.nan}
    assert len(build_feature_vector(c, two_missing, {})[1]) == 2  # 2/28 = 7 % ≤ the 10 % cap
    with pytest.raises(InvalidInputError, match="features missing"):
        build_feature_vector({**c, "max_imputed_fraction": 0.05}, two_missing, {})


def test_demographics_are_mandatory_and_validated(demo_bundle):
    c = legacy_contract(demo_bundle)
    assert c["demographics"] == ["age", "sex", "height_cm", "weight_kg"]
    with pytest.raises(IncompatibleModelError, match="requires demographic inputs"):
        check_compatibility(c, "simulator", 100.0)
    good = {"age": 40, "sex": 1, "height_cm": 170, "weight_kg": 70}
    assert check_compatibility(c, "simulator", 100.0, demographics=good)["demographics"]["age"] == 40.0
    for bad in ({**good, "age": 300}, {**good, "sex": "male"}, {**good, "weight_kg": float("nan")}):
        with pytest.raises(IncompatibleModelError):
            check_compatibility(c, "simulator", 100.0, demographics=bad)
    with pytest.raises(IncompatibleModelError, match="requires demographic"):
        InferenceEngine(demo_bundle, EngineConfig(fs=100.0, sensor="simulator"))
    eng = InferenceEngine(demo_bundle, EngineConfig(fs=100.0, sensor="simulator", demographics=good))
    p = _feed(eng, _raw(simulate(SimConfig(seed=3), 30)))
    assert p[-1].status == "usable" and p[-1].inference_mode == "demonstration" and np.isfinite(p[-1].sbp)


# ------------------------------------------------------------------------------------------- engine
def test_engine_modes_uncertainty_and_explicit_fs(ear):
    with pytest.raises(TypeError):
        EngineConfig()  # the sampling rate is never assumed
    raw = _raw(simulate(SimConfig(seed=4), 40))
    sim = InferenceEngine(ear, EngineConfig(fs=100.0, sensor="simulator"))
    p = sim.predictions() if not _feed(sim, raw) else sim.predictions()
    done = p[p.estimate_type == "population"]
    assert len(done) and (p.inference_mode == "demonstration").all()
    assert done.sbp_lo.isna().all() and "not transferable" in sim.uncertainty_note  # no fake interval
    au = InferenceEngine(ear, EngineConfig(fs=100.0, sensor="auricle_earlobe"))
    _feed(au, raw)
    q = au.predictions().query("estimate_type == 'population'")
    hw = ear.contract["uncertainty"]["halfwidth_90"]["sbp"]
    assert np.allclose(q.sbp_hi - q.sbp_lo, 2 * hw)
    with pytest.raises(IncompatibleModelError):
        InferenceEngine(ear, EngineConfig(fs=100.0, sensor="unknown_sensor"))


def test_engine_rejects_windows_without_required_features(ear):
    import copy

    b = copy.deepcopy(ear)
    b.contract["features"][0]["source"] = "ir_feature_the_pipeline_never_produces"
    eng = InferenceEngine(b, EngineConfig(fs=100.0, sensor="simulator"))
    p = eng.predictions() if not _feed(eng, _raw(simulate(SimConfig(seed=5), 30))) else eng.predictions()
    judged = p[p.status != "warming_up"]
    assert (judged.status == "invalid_input").all() and judged.sbp.isna().all()
    assert judged.reasons.str.contains("required features missing").all()


def test_engine_reference_readings_are_validated(ear):
    eng = InferenceEngine(ear, EngineConfig(fs=100.0, sensor="simulator", session_id="S1"))
    _feed(eng, _raw(simulate(SimConfig(seed=6), 40)))
    assert eng.add_reference(30.0, 300.0, 80.0)["reason"].startswith("implausible")
    assert eng.add_reference(30.0, 80.0, 90.0)["reason"].startswith("implausible")  # SBP <= DBP
    assert "belongs to session" in eng.add_reference(30.0, 120.0, 80.0, session_id="S2")["reason"]
    assert "no accepted window" in eng.add_reference(500.0, 120.0, 80.0)["reason"]
    assert eng.add_reference(30.0, 120.0, 80.0)["accepted"] and len(eng.readings) == 1


def test_predict_windows_matches_streaming_feature_path(ear, tmp_path):
    raw = _raw(simulate(SimConfig(seed=8), 60))
    w, _ = process_recording(raw, PipelineConfig(fs=100.0), "S", "D")
    pred, info = predict_windows(ear, w, "simulator", 100.0)
    assert info["mode"] == "demonstration" and len(pred) == len(w)
    assert pred[pred.status.isin(["usable", "degraded"])].sbp.notna().all()
    assert pred[~pred.status.isin(["usable", "degraded", "invalid_input"])].sbp.isna().all()


# ------------------------------------------------------------------------------------------- acquisition
class FakePort:
    """Minimal pyserial stand-in: readline() returns queued byte lines, then b'' (timeouts)."""

    def __init__(self, lines, fail_after=None):
        self.lines = [ln if isinstance(ln, bytes) else (ln + "\n").encode() for ln in lines]
        self.fail_after, self.reads, self.closed = fail_after, 0, False

    def readline(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise OSError("device disconnected")
        if self.lines:
            return self.lines.pop(0)
        time.sleep(0.01)
        return b""

    def close(self):
        self.closed = True


def _stream_lines(path=HW / "examples" / "serial_stream_valid.txt"):
    return path.read_text(encoding="ascii").splitlines()


def test_example_streams_match_the_parser():
    good = TextStreamSource(io.StringIO("\n".join(_stream_lines())), 100.0)
    n = sum(len(b) for b in good.blocks())
    assert n == 500 and good.bad_lines == 0 and good.device_info["fs_hz"] == "100"
    bad = TextStreamSource(io.StringIO((HW / "examples" / "serial_stream_malformed.txt").read_text()), 100.0)
    rows = pd.concat([TimestampSync(100.0).process(b) for b in [bb for bb in bad.blocks()]])
    assert bad.bad_lines == 6 and np.isfinite(rows[["red", "infrared"]].to_numpy()).all()
    assert parse_banner("# hello") == {} and parse_banner("# rakshpulse fs_hz=50")["fs_hz"] == "50"
    assert LineFormat(("sample_index", "red", "infrared")).parse("1,2,3") == {
        "sample_index": 1.0,
        "red": 2.0,
        "ir": 3.0,
    }


def test_firmware_example_matches_the_serial_contract():
    ino = (HW / "firmware" / "rakshpulse_max30102" / "rakshpulse_max30102.ino").read_text()
    assert '"# rakshpulse fw=1 fs_hz="' in ino and "SAMPLE_RATE / SAMPLE_AVERAGE" in ino
    data_prints = re.findall(r"Serial\.print(?:ln)?\(([^)]*)\);", ino.split("void loop()")[1])
    assert data_prints == ["sampleIndex", "','", "red", "','", "ir"]  # → "sample_index,red,ir\n"
    assert "REG_OVF_COUNTER" in ino and "sampleIndex += lost" in ino  # lost samples advance the counter
    for line in re.findall(r'Serial\.println\("([^"]*)"\)', ino):
        assert line.startswith("#")  # every status message is a comment for the parser


def test_serial_source_with_fake_port_handles_banner_timeout_and_reconnect():
    lines = _stream_lines()[:201]
    src = SerialSource("FAKE", 115200, 100.0, port_factory=lambda p, b, t: FakePort(lines), no_data_timeout_s=0.3)
    got = []
    with pytest.raises(AcquisitionError, match="no valid samples"):
        for blk in src.blocks():
            got.append(len(blk))
    assert sum(got) >= 192 and src.device_info["fs_hz"] == "100"
    ports = iter([FakePort(lines[:50], fail_after=50), FakePort(lines[50:120])])
    rec = SerialSource(
        "FAKE", 115200, 100.0, port_factory=lambda p, b, t: next(ports), reconnect_wait_s=0.0, no_data_timeout_s=0.2
    )
    n = 0
    with pytest.raises(AcquisitionError):
        for blk in rec.blocks():
            n += len(blk)
    assert rec.reconnects == 1 and n >= 100
    with pytest.raises(AcquisitionError, match="could not be reopened"):
        calls = []

        def factory(p, b, t):
            calls.append(p)
            if len(calls) == 1:
                return FakePort([], fail_after=0)
            raise AcquisitionError("port gone")

        list(
            SerialSource(
                "FAKE", 115200, 100.0, port_factory=factory, reconnect_wait_s=0.0, reconnect_attempts=2
            ).blocks()
        )
    with pytest.raises(AcquisitionError):
        SerialSource("FAKE", 0, 100.0, port_factory=lambda *a: FakePort([]))


def test_check_serial_reports_rate_and_banner_problems():
    lines = _stream_lines()
    ok = check_serial(
        SerialSource("FAKE", 115200, 100.0, port_factory=lambda *a: FakePort(lines), no_data_timeout_s=0.2), seconds=0.5
    )
    assert ok["samples"] == 500 and ok["declared_fs_hz"] == 100.0 and ok["malformed_lines"] == 0
    wrong = check_serial(
        SerialSource("FAKE", 115200, 50.0, port_factory=lambda *a: FakePort(lines), no_data_timeout_s=0.2), seconds=0.5
    )
    assert not wrong["ok"] and any("declares fs_hz=100" in p for p in wrong["problems"])
    dead = check_serial(
        SerialSource("FAKE", 115200, 100.0, port_factory=lambda *a: FakePort([]), no_data_timeout_s=0.2), seconds=0.5
    )
    assert dead["samples"] == 0 and not dead["ok"]
    flat = check_serial(
        SerialSource(
            "FAKE",
            115200,
            100.0,
            port_factory=lambda *a: FakePort([f"{i},5000,5000" for i in range(50)]),
            no_data_timeout_s=0.2,
        ),
        seconds=0.5,
    )
    assert any("constant" in p for p in flat["problems"])


def test_real_pyserial_loopback_port():
    serial = pytest.importorskip("serial")
    port = serial.serial_for_url("loop://", timeout=0.1)
    payload = "\n".join(_stream_lines()[:101]) + "\n"
    port.write(payload.encode())
    src = SerialSource("loop://", 115200, 100.0, port_factory=lambda *a: port, no_data_timeout_s=0.3)
    rows = []
    with pytest.raises(AcquisitionError):
        Recorder(src, [rows.append]).run()
    got = pd.concat(rows)
    assert len(got) == 100 and got.sample_index.tolist() == list(range(100))


def test_duplicates_and_device_reboot_in_the_counter():
    s = TimestampSync(100.0)

    def blk(i):
        return SampleBlock(np.ones(len(i)), np.ones(len(i)), 0.0, sample_index=np.array(i))

    s.process(blk(list(range(300))))
    out = s.process(blk([300, 300, 301, 0, 1, 2]))
    assert out.sample_index.tolist() == [300, 301, 402, 403, 404]
    assert s.n_duplicates == 1 and s.n_resets == 1 and s.gaps[-1].detected_by == "counter_reset"
    s.process(blk([3, 4]))  # the restarted counter continues normally
    with pytest.raises(ValueError, match="went backwards"):
        s.process(blk([2]))  # a small backwards step is corruption, not a reboot


# ------------------------------------------------------------------------------------------- sampling rates
@pytest.mark.parametrize("fs", [50.0, 200.0, 400.0])
def test_pipeline_handles_other_sampling_rates(fs):
    rec = simulate(SimConfig(fs=fs, hr_bpm=75, hr_drift_bpm=0, seed=9), 40)
    w, _ = process_recording(_raw(rec), PipelineConfig(fs=fs), "S", "D")
    assert len(w) == 7 and (w.status == "usable").all() and np.allclose(w.ir_hr, 75, atol=3)


def test_pipeline_refuses_rate_mismatch_bad_columns_and_bad_config():
    raw = _raw(simulate(SimConfig(fs=200.0, seed=10), 30))
    with pytest.raises(ValueError, match="timestamps imply about 200"):
        process_recording(raw, PipelineConfig(fs=100.0), "S", "D")
    with pytest.raises(ValueError, match="lacks columns"):
        process_recording(raw.drop(columns=["red"]), PipelineConfig(fs=200.0), "S", "D")
    for cfg in (
        PipelineConfig(fs=10.0),
        PipelineConfig(fs=100.0, window_s=4.0),
        PipelineConfig(fs=100.0, band=(0.5, 60)),
    ):
        with pytest.raises(ValueError):
            process_recording(raw, cfg, "S", "D")


def test_pipeline_non_finite_values_are_recorded_or_rejected():
    raw = _raw(simulate(SimConfig(seed=12), 30))
    few = raw.copy()
    few.loc[[150, 151, 152], "infrared"] = np.nan
    w, _ = process_recording(few, PipelineConfig(), "S", "D")
    assert w.reasons.str.contains("interpolated_3_non_finite_ir").any() and (w.status == "usable").all()
    many = raw.copy()
    many.loc[100:400, "red"] = np.inf
    w2, _ = process_recording(many, PipelineConfig(), "S", "D")
    assert (w2.loc[w2.t_start_s < 4, "status"] == "missing").all()
    assert w2.reasons.str.contains("non_finite_samples_red").any()


# ------------------------------------------------------------------------------------------- CLI end to end
def test_cli_models_record_replay_and_analyze(tmp_path, capsys):
    from rakshpulse.cli import main

    ds = tmp_path / "ds"
    shutil.copytree(HW / "examples" / "example_dataset", ds)
    assert main(["models"]) == 0
    assert main(["analyze", "--dataset", str(ds), "--subject", "SIM001", "--session", "D1"]) == 0
    out = capsys.readouterr().out
    assert "mode=demonstration" in out and "NOT blood-pressure measurements" in out
    pfile = ds / "sessions/SIM001/D1/derived/predictions_auricle-ear-ridge.csv"
    first = pd.read_csv(pfile)
    assert len(first) and first.inference_mode.eq("demonstration").all()
    assert main(["analyze", "--dataset", str(ds), "--subject", "SIM001", "--session", "D1"]) == 2  # no overwrite
    assert (
        main(
            [
                "analyze",
                "--dataset",
                str(ds),
                "--subject",
                "SIM001",
                "--session",
                "D1",
                "--overwrite",
                "--calibrate",
                "1",
            ]
        )
        == 0
    )
    second = pd.read_csv(pfile)
    assert np.allclose(second.sbp, first.sbp, equal_nan=True)  # repeated runs give identical predictions
    assert (second.role == "calibration").sum() >= 1 and second.sbp_calibrated.notna().any()
    raw = ds / "sessions/SIM001/D1/raw_ppg.csv"
    assert (
        main(
            [
                "record",
                "--source",
                "replay",
                "--input",
                str(raw),
                "--out",
                str(tmp_path / "ds2"),
                "--subject",
                "R1",
                "--session",
                "D1",
            ]
        )
        == 2
    )  # --fs is never assumed
    assert (
        main(
            [
                "record",
                "--source",
                "replay",
                "--input",
                str(raw),
                "--fs",
                "100",
                "--sensor",
                "simulator",
                "--out",
                str(tmp_path / "ds2"),
                "--subject",
                "R1",
                "--session",
                "D1",
            ]
        )
        == 0
    )
    assert main(["inspect", "--dataset", str(tmp_path / "ds2"), "--validate"]) == 0
    assert main(["analyze", "--dataset", str(ds), "--subject", "NOPE", "--session", "D1"]) == 2


def test_cli_rejects_short_recordings_and_incompatible_models(tmp_path, demo_bundle):
    from rakshpulse.cli import main
    from rakshpulse.datastore.store import DatasetStore, SessionMetadata

    st = DatasetStore(tmp_path / "short")
    with st.create_session(SessionMetadata("S", "D1", 100.0, sensor_type="simulator")) as w:
        w.write_raw(_raw(simulate(SimConfig(seed=13), 6)))
    assert main(["analyze", "--dataset", str(tmp_path / "short"), "--subject", "S", "--session", "D1"]) == 1
    mp = demo_bundle.save(tmp_path / "demo.joblib")
    assert main(["infer", "--source", "sim", "--duration", "20", "--model", str(mp)]) == 4
    assert (
        main(
            [
                "infer",
                "--source",
                "sim",
                "--duration",
                "20",
                "--model",
                str(mp),
                "--demographics",
                "age=40,sex=1,height_cm=170,weight_kg=70",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "infer",
                "--source",
                "sim",
                "--duration",
                "20",
                "--model",
                str(mp),
                "--demographics",
                "age=400,sex=1,height_cm=170,weight_kg=70",
            ]
        )
        == 4
    )
