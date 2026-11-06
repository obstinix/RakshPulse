import http.server
import threading
import zipfile
from functools import partial

import numpy as np
import pandas as pd
import pytest

from rakshpulse.evaluation.calibration import calibrated_errors, within_session_r

wfdb = pytest.importorskip("wfdb")


def _session(rid, n=40, slope=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 5.0
    act = np.where((t > 80) & (t < 140), 2, 0)
    bp = 120 + np.where(act == 2, 25.0, 0.0)
    pred = 100 + slope * (bp - 120) + noise * rng.standard_normal(n)  # offset of -20 mmHg
    return pd.DataFrame({"record_id": rid, "t_start_s": t, "activity": act, "sbp": bp, "pred": pred})


def test_perfect_tracker_beats_carry_forward_after_calibration():
    d = pd.concat([_session("a"), _session("b")])
    r = calibrated_errors(d, "sbp", k_cal=3)
    assert r["n_sessions"] == 2
    assert r["calibrated_model_mae"] < 1e-9  # offset removed, changes tracked exactly
    assert r["carry_forward_mae"] > 5 and r["stressor_carry_forward_mae"] == pytest.approx(25)
    assert r["delta_tracking_r"] == pytest.approx(1.0)


def test_non_tracking_model_equals_carry_forward():
    d = _session("a", slope=0.0)
    r = calibrated_errors(d, "sbp", k_cal=3)
    assert r["calibrated_model_mae"] == pytest.approx(r["carry_forward_mae"])


def test_calibration_windows_excluded_from_evaluation():
    d = _session("a", n=40)
    r = calibrated_errors(d, "sbp", k_cal=3, gap_s=10)
    assert r["n_windows"] == 40 - 3 - 2  # 3 calibration windows + 2 overlapping within the gap


def test_within_session_r():
    assert within_session_r(_session("a"), "sbp") == pytest.approx(1.0)


def test_read_auricle_synthetic_record(tmp_path, monkeypatch):
    from rakshpulse.data import dataset_loader as dl

    root = tmp_path / "auricle"
    rec = root / "data" / "HC_7_1"
    rec.mkdir(parents=True)
    fs, n = 1000, 30_000
    t = np.arange(n) / fs
    sig = np.column_stack(
        [
            np.sin(2 * np.pi * 1.2 * t),  # PPG_Ear
            -6 + 0.5 * np.sin(2 * np.pi * 1.2 * t),  # PPG_Finger
            np.full(n, 121.0),
            np.full(n, 75.0),  # Systolic_BP, Diastolic_BP
            np.r_[np.zeros(5000), np.ones(n - 5000)],  # CNAP_Annotations (1 = valid)
            np.zeros(n),  # Activity_Annotations
        ]
    )
    wfdb.wrsamp(
        "HC_7_1",
        fs=fs,
        units=["mV"] * 6,
        sig_name=dl.AURICLE_CHANNELS,
        p_signal=sig,
        fmt=["32"] * 6,
        write_dir=str(rec),
    )
    (root / "RECORDS").write_text("HC_7_1/HC_7_1\n")
    (root / "demographics.csv").write_text(
        "ID,Cohort,Participant,Age,Gender,Height (cm),Weight (kg)\nHC_7,HC,7,22,male,181,60\n"
    )
    assert dl.auricle_records(root) == ["HC_7_1"]
    r = dl.read_auricle("HC_7_1", root)
    assert r.subject_id == "auricle_HC_7" and r.cohort == "HC" and r.fs == 1000
    assert r.cnap_valid[:5000].sum() == 0 and r.cnap_valid[5000:].all()
    assert np.allclose(r.sbp, 121, atol=0.01) and r.demographics["age"] == 22


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), int(b) if b else len(data) - 1
            body = data[a : b + 1]
            self.send_response(206)
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        import io

        return io.BytesIO(body if self.command == "GET" else b"")


def _serve(directory):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(directory)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _local_project(port):
    from dataclasses import dataclass

    from rakshpulse.data.physionet import PhysioNetProject

    @dataclass(frozen=True)
    class Local(PhysioNetProject):
        @property
        def base(self):
            return f"http://127.0.0.1:{port}/files/"

        @property
        def zip(self):
            return f"http://127.0.0.1:{port}/p.zip"

    return Local("t", "t", "1", "t", "doi", "lic", "cite", 1)


def test_interrupted_download_resumes_from_part(tmp_path):
    import hashlib

    from rakshpulse.data.physionet import fetch_file

    (tmp_path / "srv" / "files" / "data").mkdir(parents=True)
    payload = np.random.default_rng(1).bytes(200_000)
    (tmp_path / "srv" / "files" / "data" / "x.dat").write_bytes(payload)
    dest = tmp_path / "dest"
    (dest / "data").mkdir(parents=True)
    (dest / "data" / "x.dat.part").write_bytes(payload[:77_777])  # simulated interruption
    srv = _serve(tmp_path / "srv")
    try:
        res = fetch_file(_local_project(srv.server_port), "data/x.dat", hashlib.sha256(payload).hexdigest(), dest)
    finally:
        srv.shutdown()
    assert res["status"] == "downloaded"
    assert (dest / "data" / "x.dat").read_bytes() == payload
    assert not (dest / "data" / "x.dat.part").exists()


def test_checksum_mismatch_is_rejected_and_part_removed(tmp_path):
    from rakshpulse.data.physionet import fetch_file

    (tmp_path / "srv" / "files").mkdir(parents=True)
    (tmp_path / "srv" / "files" / "y.dat").write_bytes(b"tampered content")
    srv = _serve(tmp_path / "srv")
    try:
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            fetch_file(_local_project(srv.server_port), "y.dat", "0" * 64, tmp_path / "d")
    finally:
        srv.shutdown()
    assert not (tmp_path / "d" / "y.dat").exists() and not (tmp_path / "d" / "y.dat.part").exists()


def test_zip_member_fetch_verifies_sha256(tmp_path):
    import hashlib

    from rakshpulse.data.physionet import fetch_member_from_zip

    payload = np.random.default_rng(2).bytes(150_000)
    (tmp_path / "srv").mkdir()
    with zipfile.ZipFile(tmp_path / "srv" / "p.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("proj-1/data/R_1_1/R_1_1.dat", payload)
    srv = _serve(tmp_path / "srv")
    try:
        proj = _local_project(srv.server_port)
        good = fetch_member_from_zip(proj, "data/R_1_1/R_1_1.dat", hashlib.sha256(payload).hexdigest(), tmp_path / "d")
        assert good["status"].startswith("downloaded")
        again = fetch_member_from_zip(proj, "data/R_1_1/R_1_1.dat", hashlib.sha256(payload).hexdigest(), tmp_path / "d")
        assert again["status"] == "present"  # verified files are never re-downloaded
        with pytest.raises(RuntimeError):
            fetch_member_from_zip(proj, "data/R_1_1/R_1_1.dat", "f" * 64, tmp_path / "e")
    finally:
        srv.shutdown()


def test_verify_project_flags_corrupt_missing_and_mask(tmp_path, monkeypatch):
    import hashlib
    import json

    from rakshpulse.data import physionet as pn

    root = tmp_path / "raw" / "t"
    good = {}
    for rid, chans in (
        ("HC_1_1", ["PPG_Ear", "PPG_Finger", "Systolic_BP", "Diastolic_BP"]),
        ("HC_2_1", ["PPG_Ear", "PPG_Finger", "Systolic_BP", "Diastolic_BP", "CNAP_Annotations"]),
    ):
        d = root / "data" / rid
        d.mkdir(parents=True)
        wfdb.wrsamp(
            rid,
            fs=1000,
            units=["mV"] * len(chans),
            sig_name=chans,
            p_signal=np.zeros((100, len(chans))),
            fmt=["16"] * len(chans),
            write_dir=str(d),
        )
        for ext in (".hea", ".dat"):
            good[f"data/{rid}/{rid}{ext}"] = hashlib.sha256((d / f"{rid}{ext}").read_bytes()).hexdigest()
    good["data/HC_3_1/HC_3_1.hea"] = "a" * 64  # listed but never downloaded
    good["data/HC_3_1/HC_3_1.dat"] = "b" * 64
    (root / "SHA256SUMS.txt").write_text("\n".join(f"{h} {r}" for r, h in good.items()) + "\n")
    (root / "RECORDS").write_text("HC_1_1/HC_1_1\nHC_2_1/HC_2_1\nHC_3_1/HC_3_1\n")
    (tmp_path / "metadata").mkdir()
    monkeypatch.setattr(pn, "DATA_ROOT", tmp_path)
    monkeypatch.setitem(pn.PROJECTS, "t", pn.PhysioNetProject("t", "t", "1", "t", "d", "l", "c", 1))
    (root / "data" / "HC_2_1" / "HC_2_1.dat").write_bytes(b"corrupted")  # break one file
    s = pn.verify_project("t")
    by = {r["record"]: r for r in s["records"]}
    assert by["HC_1_1"]["status"] == "ok" and by["HC_1_1"]["trusted_labels"] is False  # no CNAP mask
    assert by["HC_2_1"]["status"] == "corrupt"
    assert by["HC_3_1"]["status"] == "missing"
    assert json.loads((tmp_path / "metadata" / "t_manifest.json").read_text())["trusted_records"] == 0


def test_http_range_file_reads_zip_member(tmp_path):
    from rakshpulse.data.physionet import HttpRangeFile

    payload = np.random.default_rng(0).bytes(300_000)
    with zipfile.ZipFile(tmp_path / "p.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("proj-1.0.0/data/X_1_1/X_1_1.dat", payload)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(tmp_path)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        f = HttpRangeFile(f"http://127.0.0.1:{srv.server_port}/p.zip", block=8192)
        with zipfile.ZipFile(f) as zf:
            assert zf.read("proj-1.0.0/data/X_1_1/X_1_1.dat") == payload
    finally:
        srv.shutdown()


# ------------------------------------------------------------------ cross-day study module
def _study_frame(n_subj=8, seed=0):
    """Two sessions per subject. BP = subject offset + 0.8*stressor effect; 'x' tracks the stressor."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        off = 100 + 4 * s
        for sess in (1, 2):
            for w in range(40):
                act = 2 if 15 <= w < 25 else 0
                eff = 20.0 if act else 0.0
                rows.append(
                    {
                        "subject_id": f"s{s}",
                        "session": sess,
                        "t_start_s": 5.0 * w,
                        "activity": act,
                        "sbp": off + eff + rng.normal(0, 1),
                        "x": eff / 20 + rng.normal(0, 0.05),
                        "hr": 70 + rng.normal(0, 3),
                    }
                )
    return pd.DataFrame(rows)


def test_subject_folds_are_disjoint_and_balanced():
    from rakshpulse.evaluation.auricle_study import subject_folds

    d = _study_frame(10)
    f = subject_folds(d.subject_id, d.sbp, n=5)
    assert len(f) == 10 and sorted(pd.Series(f).value_counts()) == [2] * 5


def test_population_predictions_no_leakage_and_learn():
    from rakshpulse.evaluation.auricle_study import population_predictions, subject_folds

    d = _study_frame(10)
    fold = subject_folds(d.subject_id, d.sbp, n=5)
    pred = population_predictions(d[d.session == 1].reset_index(drop=True), d, ["x"], "sbp", "ridge", fold)
    assert np.isfinite(pred).all()
    within = (
        pd.DataFrame({"s": d.subject_id, "y": d.sbp, "p": pred})
        .groupby("s")
        .apply(lambda g: np.corrcoef(g.y, g.p)[0, 1], include_groups=False)
    )
    assert within.median() > 0.9  # tracks within-subject changes


def test_calibration_modes_and_personal_models():
    from rakshpulse.evaluation.auricle_study import (
        calibrate,
        calibration_drift,
        paired_subject_diff,
        personal_model_predictions,
        point_metrics,
    )

    d = _study_frame(6)
    d["pred"] = 110 + 20 * d["x"]  # population model with wrong absolute level but right tracking
    t = calibrate(d, "sbp", k_cal=3).table
    assert set(t["mode"]) == {"carry_forward", "offset", "uncalibrated", "linear"}
    assert set(t.scope) == {"within_day", "cross_day", "within_day_s2"}
    cd = t[t.scope == "cross_day"]
    off, carry = cd[cd["mode"] == "offset"], cd[cd["mode"] == "carry_forward"]
    assert point_metrics(off)["mae"] < point_metrics(carry)["mae"]
    diff, lo, hi = paired_subject_diff(off, carry)
    assert hi < 0  # offset calibration significantly better than carry-forward
    assert abs(calibration_drift(off)["drift_mean"]) < 2
    pm = personal_model_predictions(d, ["x", "hr"], "sbp")
    assert pm.subject_id.nunique() == 6 and point_metrics(pm)["mae"] < 5


def test_subject_bootstrap_ci_contains_estimate():
    from rakshpulse.evaluation.auricle_study import subject_bootstrap

    t = pd.DataFrame(
        {
            "subject_id": np.repeat(list("abcdefgh"), 5),
            "y": np.arange(40.0),
            "yhat": np.arange(40.0) + np.repeat(np.arange(8.0), 5),
        }
    )
    est, lo, hi = subject_bootstrap(t, lambda x: (x.yhat - x.y).mean(), n_boot=500)
    assert lo <= est <= hi and est == pytest.approx(3.5)


def test_shape_features_are_hr_invariant_by_construction():
    from rakshpulse.evaluation.auricle_study import SHAPE_FEATURES, add_normalised_shape

    assert not any(f in SHAPE_FEATURES for f in ("hr", "beat_T", "f0", "sdnn", "w50"))
    df = pd.DataFrame(
        {f"e_{k}": [0.5, 1.0] for k in ("w10", "w25", "w50", "w75", "w90", "t_max_slope", "t_apg_b", "max_slope")}
        | {"e_beat_T": [0.5, 1.0]}
    )
    add_normalised_shape(df, "e")
    assert np.allclose(df["e_w50_n"], 1.0)  # duration-scaled widths become identical


def test_rail_gate_rejects_clipped_windows_only():
    from rakshpulse.data.build_auricle import apply_rail_gate

    df = pd.DataFrame(
        {
            "ear_sat_fraction": [0.0, 0.005, 0.2],
            "finger_sat_fraction": [0.02, 0.0, 0.0],
            "ear_sqi_ok": [True, True, True],
            "finger_sqi_ok": [True, True, False],
            "ear_sqi_reasons": ["", "", ""],
            "finger_sqi_reasons": ["", "", "too_few_beats"],
        }
    )
    out = apply_rail_gate(df.copy())
    assert out.ear_sqi_ok.tolist() == [True, True, False]
    assert out.finger_sqi_ok.tolist() == [False, True, False]
    assert out.finger_sqi_reasons[0] == "clipped_at_rail" and out.finger_sqi_reasons[2] == "too_few_beats"
    assert apply_rail_gate(out.copy()).finger_sqi_reasons[0] == "clipped_at_rail"  # idempotent
