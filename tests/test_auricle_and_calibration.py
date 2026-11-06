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


def test_cross_session_calibration():
    from rakshpulse.evaluation.calibration import cross_session_errors

    s1 = _session("a_1").assign(subject_id="a", session=1)
    s2 = _session("a_2").assign(subject_id="a", session=2, sbp=lambda d: d.sbp + 10, pred=lambda d: d.pred + 10)
    r = cross_session_errors(pd.concat([s1, s2]), "sbp", k_cal=3)
    assert r["n_subjects"] == 1
    assert r["calibrated_model_mae"] < 1e-9  # model tracks the +10 day-to-day shift
    assert r["carry_forward_mae"] > 10
