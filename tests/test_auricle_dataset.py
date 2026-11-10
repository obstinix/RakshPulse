"""End-to-end test of the standardised Auricle dataset on a tiny synthetic WFDB tree."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_ppg

wfdb = pytest.importorskip("wfdb")

CHANNELS = [
    "PPG_Ear",
    "PPG_Finger",
    "Systolic_BP",
    "Diastolic_BP",
    "Mean_BP",
    "CNAP_Annotations",
    "Activity_Annotations",
]


def _write_record(root, rid, seconds=60, mask_from_s=12, with_mask=True, clip_finger=False, seed=0):
    fs = 1000
    n = seconds * fs
    t = np.arange(n) / fs
    ear = synthetic_ppg(fs, seconds, 70, seed=seed, raw_counts=False)
    finger = -6 + 0.5 * synthetic_ppg(fs, seconds, 70, seed=seed + 1, raw_counts=False)
    if clip_finger:
        finger[: n // 2] = -10.0  # amplifier rail
    sbp = 115 + 10 * (t > 30)
    dbp = np.full(n, 72.0)
    mbp = (sbp + 2 * dbp) / 3
    mask = (t >= mask_from_s).astype(float)
    act = np.where((t > 30) & (t < 45), 2.0, 0.0)
    cols = [ear, finger, sbp, dbp, mbp] + ([mask] if with_mask else []) + [act]
    names = CHANNELS if with_mask else [c for c in CHANNELS if c != "CNAP_Annotations"]
    d = root / "data" / rid
    d.mkdir(parents=True)
    wfdb.wrsamp(
        rid,
        fs=fs,
        units=["mV"] * len(names),
        sig_name=names,
        p_signal=np.column_stack(cols),
        fmt=["32"] * len(names),
        write_dir=str(d),
    )


@pytest.fixture
def tiny_auricle(tmp_path, monkeypatch):
    from rakshpulse.data import auricle_dataset as ds

    root = tmp_path / "raw" / "auricle"
    _write_record(root, "HC_1_1", seed=0)
    _write_record(root, "HC_1_2", seed=2, clip_finger=True)
    _write_record(root, "HC_2_1", seed=4, with_mask=False)
    (root / "RECORDS").write_text("HC_1_1/HC_1_1\nHC_1_2/HC_1_2\nHC_2_1/HC_2_1\n")
    (root / "demographics.csv").write_text(
        "ID,Cohort,Participant,Age,Gender,Height (cm),Weight (kg)\n"
        "HC_1,HC,1,24,female,165,60\nHC_2,HC,2,30,male,180,80\n"
    )
    (tmp_path / "metadata").mkdir()
    man = {"records": [{"record": r, "status": "ok"} for r in ("HC_1_1", "HC_1_2", "HC_2_1")]}
    (tmp_path / "metadata" / "auricle_manifest.json").write_text(json.dumps(man))
    out = tmp_path / "auricle"
    monkeypatch.setattr(ds, "AURICLE_DIR", root)
    monkeypatch.setattr(ds, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "OUT_DIR", out)
    monkeypatch.setattr(ds, "WAVE_DIR", out / "waveforms")
    ds.build(jobs=1)
    return ds, out, root


def test_build_writes_populated_tables(tiny_auricle):
    ds, out, _ = tiny_auricle
    for name in ("recordings.csv", "windows.csv", "features.csv", "feature_schema.csv", "subjects.csv", "summary.csv"):
        df = pd.read_csv(out / name)
        assert len(df) > 0, name
    rec = pd.read_csv(out / "recordings.csv")
    assert set(rec.record_id) == {"HC_1_1", "HC_1_2", "HC_2_1"} and (rec.n_windows == 11).all()
    sub = pd.read_csv(out / "subjects.csv").set_index("subject_id")
    assert sub.loc["auricle_HC_1", "sessions_available"] == "1;2"
    assert "no session 2" in sub.loc["auricle_HC_2", "exclusion_reasons"]


def test_labels_masks_and_clipping(tiny_auricle):
    _, out, _ = tiny_auricle
    w = pd.read_csv(out / "windows.csv")
    a = w[w.record_id == "HC_1_1"]
    # mask starts at 12 s: windows starting before 12 s are unlabelled
    assert (~a[a.t_start_s < 12].label_valid).all() and a[a.t_start_s < 12].sbp_mmHg.isna().all()
    assert a[a.t_start_s >= 15].label_valid.all()
    assert set(a.label_status) <= {"ok", "cnap_unreliable"}
    nm = w[w.record_id == "HC_2_1"]
    assert (~nm.label_valid).all() and (nm.label_status == "no_cnap_mask").all()
    c = w[(w.record_id == "HC_1_2") & (w.t_end_s <= 30)]
    assert (c.finger_rail_pct > 1).all() and (~c.finger_sqi_ok).all()
    assert c.finger_sqi_reasons.str.contains("clipped_at_rail").all()
    elig = w[w.ear_calibration_eligible]
    assert (elig.activity_code == 0).all() and elig.label_valid.all() and elig.ear_sqi_ok.all()


def test_features_traceable_and_free_of_bp_inputs(tiny_auricle):
    ds, out, _ = tiny_auricle
    w = pd.read_csv(out / "windows.csv")
    f = pd.read_csv(out / "features.csv")
    assert list(f.window_id) == list(w.window_id) or set(f.window_id) == set(w.window_id)
    fcols = [c for c in f.columns if c not in ("window_id", "subject_id", "record_id", "session")]
    assert set(fcols) == set(pd.read_csv(out / "feature_schema.csv").column)
    assert not [c for c in fcols if "bp" in c.lower() or "cnap" in c.lower()]
    ok = f.merge(w[["window_id", "ear_sqi_ok"]], on="window_id")
    assert ok[ok.ear_sqi_ok].ear_hr.between(60, 80).all()  # synthetic 70 bpm


def test_waveforms_reference_original_samples(tiny_auricle):
    ds, out, root = tiny_auricle
    w = pd.read_csv(out / "windows.csv")
    for sess, g in w.groupby("session"):
        z = np.load(out / f"waveforms/auricle_waveforms_session{sess}.npz")
        assert z["raw_ear"].shape == (len(g), ds.WAVE_LEN) and z["raw_ear"].dtype == np.float32
        assert (z["window_id"][g.waveform_index.to_numpy()] == g.window_id.to_numpy()).all()
        meta = json.loads(str(z["metadata"]))
        assert meta["fs_hz"] == 100 and "mV" in meta["arrays"]["raw_ear"]
        row = g.iloc[5]
        orig = wfdb.rdrecord(
            str(root / "data" / row.record_id / row.record_id),
            channel_names=["PPG_Ear"],
            sampfrom=int(row.sample_start),
            sampto=int(row.sample_end),
        ).p_signal[:, 0]
        assert np.corrcoef(orig.reshape(-1, 10).mean(1)[50:-50], z["raw_ear"][row.waveform_index][50:-50])[0, 1] > 0.99


def test_validate_passes_and_detects_tampering(tiny_auricle):
    ds, out, root = tiny_auricle
    rep = ds.validate(check_raw=True, sample=3)
    assert (rep.status == "PASS").all(), rep[rep.status != "PASS"]
    w = pd.read_csv(out / "windows.csv")
    w.loc[w.index[w.label_valid][0], "sbp_mmHg"] = 400  # corrupt a label
    w.to_csv(out / "windows.csv", index=False)
    rep = ds.validate(check_raw=False)
    assert rep.set_index("check").loc["bp_labels_physiological", "status"] == "FAIL"


def test_build_is_reproducible(tiny_auricle):
    ds, out, _ = tiny_auricle
    h1 = hashlib.sha256((out / "features.csv").read_bytes()).hexdigest()
    ds.build(jobs=1)
    assert hashlib.sha256((out / "features.csv").read_bytes()).hexdigest() == h1


def test_release_chunking_and_verified_reassembly(tmp_path, monkeypatch):
    import http.server
    import threading
    from functools import partial

    from rakshpulse.data import release

    payload = np.random.default_rng(0).bytes(250_000)
    src = tmp_path / "big.npz"
    src.write_bytes(payload)
    monkeypatch.setattr(release, "STAGING", tmp_path / "staging")
    parts = release._split(src, 100_000)
    assert len(parts) == 3 and b"".join(p.read_bytes() for p in parts) == payload
    srv_root = tmp_path / "srv" / "o" / "r" / "releases" / "download" / "tag"
    srv_root.mkdir(parents=True)
    rows = []
    for i, p in enumerate(parts):
        (srv_root / p.name).write_bytes(p.read_bytes())
        rows.append(
            {
                "collection": "waveforms",
                "release_tag": "tag",
                "asset_name": p.name,
                "original_file": "out/big.npz",
                "part": i,
                "n_parts": 3,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "original_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    man = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "srv"))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(release, "BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setattr(release, "REPO", "o/r")
    try:
        got = release.fetch("waveforms", dest_root=tmp_path / "dest", manifest=man)
        assert got[0].read_bytes() == payload
        rows[1]["sha256"] = "0" * 64
        pd.DataFrame(rows).to_csv(man, index=False)
        (tmp_path / "dest" / "out" / "big.npz").unlink()
        with pytest.raises(RuntimeError):
            release.fetch("waveforms", dest_root=tmp_path / "dest", manifest=man)
    finally:
        srv.shutdown()
