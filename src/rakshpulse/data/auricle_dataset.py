"""Standardised, parameterised Auricle dataset (CSV tables + NPZ waveforms).

Layout (``data/auricle/``)::

    recordings.csv        one row per WFDB record (channels, duration, checksums, quality)
    windows.csv           one row per 10 s window: ids, timing, labels, flags, waveform reference
    features.csv          one row per window: PPG features for ear and finger (no BP-derived inputs)
    feature_schema.csv    feature name, site, group, unit, definition
    subjects.csv          one row per subject: demographics, sessions, counts, inclusion status
    summary.csv           dataset-level statistics
    validation_report.csv output of ``python -m rakshpulse.data validate``
    waveforms/auricle_waveforms_session{1,2}.npz   (published as GitHub Release assets)

Every window has a ``window_id``. ``features.csv`` rows and the NPZ arrays are keyed by it
(``waveform_file`` + ``waveform_index`` in ``windows.csv``), so each feature row traces to its waveform
and from there, by ``record_id`` + ``sample_start``/``sample_end``, to the original WFDB samples.

Nothing is fabricated: values that cannot be derived reliably are left empty and explained by a status or
reason column (e.g. BP labels outside CNAP-valid periods, features of windows with too few beats).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from ..features.feature_pipeline import PER_CHANNEL, extract_features
from ..preprocessing import filtering
from ..preprocessing.segmentation import detect_feet, detect_peaks
from .dataset_loader import AURICLE_ACTIVITIES, AURICLE_DIR, auricle_records, load_auricle_demographics
from .download import DATA_ROOT

OUT_DIR = DATA_ROOT / "auricle"
WAVE_DIR = OUT_DIR / "waveforms"
WIN_S, STEP_S = 10.0, 5.0
TARGET_FS = filtering.TARGET_FS  # 100 Hz
WAVE_LEN = int(WIN_S * TARGET_FS)
RAIL_V = 9.99  # BIOPAC amplifier rail (±10 V input range)
MAX_RAIL_FRACTION = 0.01
SBP_RANGE, DBP_RANGE = (60.0, 250.0), (30.0, 150.0)
K_CAL_REST = 3  # matches the cross-day study protocol
SITES = ("ear", "finger")
WFDB_CHANNEL = {"ear": "PPG_Ear", "finger": "PPG_Finger"}
READ_CHANNELS = [
    "PPG_Ear",
    "PPG_Finger",
    "Systolic_BP",
    "Diastolic_BP",
    "Mean_BP",
    "CNAP_Annotations",
    "Activity_Annotations",
]
PREPROCESSING = {
    "window_s": WIN_S,
    "step_s": STEP_S,
    "waveform_fs_hz": TARGET_FS,
    "raw_array": "record median removed, polyphase anti-aliased resampling 1000 -> 100 Hz, median added back; mV",
    "processed_array": "per-window: median/linear detrend, resample to 100 Hz, Hampel despike, 0.5-10 Hz zero-phase "
    "4th-order Butterworth, upward orientation, z-score (dimensionless)",
    "rail_threshold_v": RAIL_V,
    "max_rail_fraction": MAX_RAIL_FRACTION,
    "label": "median of beat-level CNAP Systolic_BP / Diastolic_BP / Mean_BP within the window, only if the CNAP "
    "reliability mask is 1 for the whole window and SBP in [60,250], DBP in [30,150], SBP > DBP",
}

# ------------------------------------------------------------------------------- feature schema
_UNITS = {
    "beat_T": ("s", "morphology", "median beat duration (foot to next foot)"),
    "sys_T": ("s", "morphology", "systolic rise time: foot to systolic peak"),
    "dia_T": ("s", "morphology", "systolic peak to next foot"),
    "sys_ratio": ("1", "morphology", "sys_T / beat_T"),
    "area_sys": ("s", "morphology", "area under normalised beat before the peak"),
    "area_dia": ("s", "morphology", "area under normalised beat after the peak"),
    "area_ratio": ("1", "morphology", "area_dia / area_sys"),
    "max_slope": ("1/s", "morphology", "maximum upstroke slope of the amplitude-normalised beat"),
    "t_max_slope": ("s", "morphology", "time from foot to maximum upstroke slope"),
    "beat_skew": ("1", "morphology", "skewness of the normalised beat"),
    "beat_kurt": ("1", "morphology", "excess kurtosis of the normalised beat"),
    "apg_b_a": ("1", "morphology", "second-derivative b/a wave ratio (Takazawa)"),
    "apg_c_a": ("1", "morphology", "second-derivative c/a ratio"),
    "apg_d_a": ("1", "morphology", "second-derivative d/a ratio"),
    "apg_e_a": ("1", "morphology", "second-derivative e/a ratio (diastolic-shoulder related)"),
    "aging_index": ("1", "morphology", "(b - c - d - e) / a"),
    "t_apg_b": ("s", "morphology", "time from beat start to APG b wave"),
    "hr": ("bpm", "physiological", "heart rate from median inter-beat interval"),
    "sdnn": ("s", "physiological", "SD of inter-beat intervals in the window"),
    "rmssd": ("s", "physiological", "RMS of successive inter-beat interval differences"),
    "win_skew": ("1", "time", "skewness of the processed window"),
    "win_kurt": ("1", "time", "excess kurtosis of the processed window"),
    "f0": ("Hz", "frequency", "dominant (pulse) frequency in 0.5-3.7 Hz"),
    "h2_h1": ("1", "frequency", "2nd / 1st harmonic amplitude"),
    "h3_h1": ("1", "frequency", "3rd / 1st harmonic amplitude"),
    "spec_entropy": ("1", "frequency", "normalised spectral entropy in 0.5-10 Hz"),
    "pow_0p5_2": ("1", "frequency", "fraction of 0.5-10 Hz power in 0.5-2 Hz"),
    "pow_2_5": ("1", "frequency", "fraction of 0.5-10 Hz power in 2-5 Hz"),
    "pow_5_10": ("1", "frequency", "fraction of 0.5-10 Hz power in 5-10 Hz"),
    "raw_mean": ("mV", "time", "mean of the raw window (amplifier output, includes offset)"),
    "raw_std": ("mV", "time", "standard deviation of the raw window"),
    "raw_var": ("mV^2", "time", "variance of the raw window"),
    "raw_rms": ("mV", "time", "RMS of the mean-removed raw window"),
    "raw_peak": ("mV", "time", "maximum absolute deviation from the window mean"),
    "raw_p2p": ("mV", "time", "peak-to-peak range of the raw window"),
    "raw_skew": ("1", "time", "skewness of the raw window"),
    "raw_kurt": ("1", "time", "excess kurtosis of the raw window"),
    "pulse_amp": ("mV", "morphology", "median foot-to-peak amplitude of band-passed beats (gain-dependent)"),
    "ibi_mean": ("ms", "physiological", "mean inter-beat interval"),
    "ibi_sd": ("ms", "physiological", "SD of inter-beat intervals"),
}
for _w in (10, 25, 50, 75, 90):
    _UNITS[f"w{_w}"] = ("s", "morphology", f"pulse width at {_w} % of beat amplitude")
    _UNITS[f"w{_w}_sd_ratio"] = ("1", "morphology", f"diastolic/systolic part of the width at {_w} %")
EXTRA_FEATURES = (
    "raw_mean",
    "raw_std",
    "raw_var",
    "raw_rms",
    "raw_peak",
    "raw_p2p",
    "raw_skew",
    "raw_kurt",
    "pulse_amp",
    "ibi_mean",
    "ibi_sd",
)
SITE_FEATURES = tuple(PER_CHANNEL) + EXTRA_FEATURES


def feature_schema() -> pd.DataFrame:
    rows = []
    for site in SITES:
        for f in SITE_FEATURES:
            unit, group, desc = _UNITS[f]
            rows.append(
                {"column": f"{site}_{f}", "site": site, "feature": f, "group": group, "unit": unit, "definition": desc}
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- per-record work
@dataclass
class RecordResult:
    recording: dict
    windows: pd.DataFrame
    features: pd.DataFrame
    raw: dict  # site -> (n, WAVE_LEN) float32, mV
    proc: dict  # site -> (n, WAVE_LEN) float32, dimensionless


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _extra_features(raw_win: np.ndarray, proc: np.ndarray, fs_proc: float) -> dict:
    """Raw-signal statistics (mV) plus pulse amplitude and inter-beat interval statistics."""
    x = raw_win.astype(float)
    xc = x - x.mean()
    out = {
        "raw_mean": float(x.mean()),
        "raw_std": float(x.std()),
        "raw_var": float(x.var()),
        "raw_rms": float(np.sqrt(np.mean(xc**2))),
        "raw_peak": float(np.abs(xc).max()),
        "raw_p2p": float(np.ptp(x)),
        "raw_skew": float(stats.skew(x)),
        "raw_kurt": float(stats.kurtosis(x)),
        "pulse_amp": np.nan,
        "ibi_mean": np.nan,
        "ibi_sd": np.nan,
    }
    # band-passed signal in mV (same chain as the processed array, without z-scoring)
    ac = filtering.bandpass(filtering.resample(x - np.median(x), 1000.0, fs_proc), fs_proc)
    pk = detect_peaks(proc, fs_proc)
    if pk.size >= 3:
        ibi = np.diff(pk) / fs_proc * 1000.0
        out["ibi_mean"], out["ibi_sd"] = float(ibi.mean()), float(ibi.std())
        feet = detect_feet(proc, pk)
        amps = [abs(ac[p] - ac[f]) for f, p in zip(feet, pk, strict=True) if 0 <= f < p < ac.size]
        if amps:
            out["pulse_amp"] = float(np.median(amps))
    return out


def process_record(rid: str, demo: dict, manifest: dict) -> RecordResult:
    import wfdb

    path = AURICLE_DIR / "data" / rid / rid
    hdr = wfdb.rdheader(str(path))
    present = list(hdr.sig_name)
    r = wfdb.rdrecord(str(path), channel_names=[c for c in READ_CHANNELS if c in present])
    x = {n: r.p_signal[:, i] for i, n in enumerate(r.sig_name)}
    fs = float(r.fs)
    cohort, num, sess = rid.split("_")
    sid = f"auricle_{cohort}_{num}"
    has_mask = "CNAP_Annotations" in x
    has_act = "Activity_Annotations" in x
    mask = np.round(x["CNAP_Annotations"]).astype(int) == 1 if has_mask else np.zeros(r.sig_len, bool)
    act = np.round(x["Activity_Annotations"]).astype(int) if has_act else np.full(r.sig_len, -1)
    win, step = int(WIN_S * fs), int(STEP_S * fs)
    starts = list(range(0, r.sig_len - win + 1, step))
    # record-level raw arrays at 100 Hz (offset-safe resampling)
    raw100 = {}
    for site in SITES:
        s = x[WFDB_CHANNEL[site]]
        med = float(np.median(s))
        raw100[site] = (filtering.resample(s - med, fs, TARGET_FS) + med).astype(np.float32)
    rows, feats = [], []
    raw_arr = {s: np.zeros((len(starts), WAVE_LEN), np.float32) for s in SITES}
    proc_arr = {s: np.full((len(starts), WAVE_LEN), np.nan, np.float32) for s in SITES}
    for i, a in enumerate(starts):
        b = a + win
        wid = f"{rid}_w{i:04d}"
        a100 = int(round(a * TARGET_FS / fs))
        for site in SITES:
            seg = raw100[site][a100 : a100 + WAVE_LEN]
            raw_arr[site][i, : seg.size] = seg
        cnap_ok = bool(mask[a:b].all()) if has_mask else False
        sbp, dbp, mbp = (
            float(np.median(x[c][a:b])) if c in x else np.nan for c in ("Systolic_BP", "Diastolic_BP", "Mean_BP")
        )
        plausible = SBP_RANGE[0] <= sbp <= SBP_RANGE[1] and DBP_RANGE[0] <= dbp <= DBP_RANGE[1] and sbp > dbp
        label_valid = cnap_ok and plausible
        a_vals = act[a:b]
        a_code = int(np.bincount(a_vals[a_vals >= 0]).argmax()) if (a_vals >= 0).any() else -1
        row = {
            "window_id": wid,
            "subject_id": sid,
            "record_id": rid,
            "cohort": cohort,
            "session": int(sess),
            "t_start_s": a / fs,
            "t_end_s": b / fs,
            "sample_start": a,
            "sample_end": b,
            "source_fs_hz": fs,
            "waveform_file": f"waveforms/auricle_waveforms_session{sess}.npz",
            "waveform_index": None,  # filled when the session NPZ is assembled
            "activity_code": a_code,
            "activity_name": AURICLE_ACTIVITIES.get(a_code, "unlabelled") if a_code >= 0 else "unlabelled",
            "activity_pure": bool(a_code >= 0 and (a_vals == a_code).all()),
            "cnap_mask_available": has_mask,
            "cnap_valid": cnap_ok,
            "label_valid": label_valid,
            "label_status": "ok"
            if label_valid
            else ("no_cnap_mask" if not has_mask else ("cnap_unreliable" if not cnap_ok else "implausible_bp")),
            "sbp_mmHg": round(sbp, 2) if label_valid else np.nan,
            "dbp_mmHg": round(dbp, 2) if label_valid else np.nan,
            "map_mmHg": round(mbp, 2) if label_valid and np.isfinite(mbp) else np.nan,
            "sbp_range_in_window_mmHg": round(float(np.ptp(x["Systolic_BP"][a:b])), 2) if label_valid else np.nan,
        }
        frow = {"window_id": wid}
        for site in SITES:
            sig = x[WFDB_CHANNEL[site]][a:b]
            rail = float(np.mean(np.abs(sig) >= RAIL_V))
            fts, proc = extract_features({"ir": sig}, fs, has_dc=False)
            pc = proc["ir"]
            proc_arr[site][i, : pc.filtered.size] = pc.filtered[:WAVE_LEN]
            sqi_ok = bool(pc.sqi.acceptable and rail <= MAX_RAIL_FRACTION)
            reasons = list(pc.sqi.reasons) + (["clipped_at_rail"] if rail > MAX_RAIL_FRACTION else [])
            row.update(
                {
                    f"{site}_sqi_ok": sqi_ok,
                    f"{site}_sqi_reasons": "|".join(reasons),
                    f"{site}_template_corr": round(pc.sqi.template_corr, 4),
                    f"{site}_cardiac_power_ratio": round(pc.sqi.cardiac_power_ratio, 4),
                    f"{site}_n_beats": pc.sqi.n_beats,
                    f"{site}_rail_pct": round(100 * rail, 3),
                    f"{site}_hr_bpm": round(fts["ir_hr"], 2) if np.isfinite(fts["ir_hr"]) else np.nan,
                    f"{site}_calibration_eligible": bool(label_valid and sqi_ok and a_code == 0),
                }
            )
            extra = _extra_features(sig, pc.filtered, TARGET_FS)
            n_ok = sum(np.isfinite(fts[f"ir_{k}"]) for k in PER_CHANNEL)
            row[f"{site}_feature_status"] = "complete" if n_ok == len(PER_CHANNEL) else "partial" if n_ok else "failed"
            frow.update({f"{site}_{k}": fts[f"ir_{k}"] for k in PER_CHANNEL})
            frow.update({f"{site}_{k}": v for k, v in extra.items()})
        rows.append(row)
        feats.append(frow)
    d = AURICLE_DIR / "data" / rid
    rec = {
        "record_id": rid,
        "subject_id": sid,
        "cohort": cohort,
        "session": int(sess),
        "source_fs_hz": fs,
        "n_samples": int(r.sig_len),
        "duration_s": round(r.sig_len / fs, 3),
        "n_channels": len(present),
        "channels": ";".join(present),
        "units": ";".join(f"{n}:{u}" for n, u in zip(hdr.sig_name, hdr.units, strict=True)),
        "cnap_mask_available": has_mask,
        "activity_annotations_available": has_act,
        "cnap_valid_fraction": round(float(mask.mean()), 4) if has_mask else np.nan,
        "ear_rail_fraction": round(float(np.mean(np.abs(x["PPG_Ear"]) >= RAIL_V)), 5),
        "finger_rail_fraction": round(float(np.mean(np.abs(x["PPG_Finger"]) >= RAIL_V)), 5),
        "n_windows": len(starts),
        "n_label_valid_windows": int(sum(rw["label_valid"] for rw in rows)),
        "sha256_dat": _sha256(d / f"{rid}.dat"),
        "sha256_hea": _sha256(d / f"{rid}.hea"),
        "sha256_verified_against_manifest": manifest.get(rid, {}).get("status") == "ok",
        "recording_datetime": "unavailable (de-identified by PhysioNet)",
    }
    return RecordResult(rec, pd.DataFrame(rows), pd.DataFrame(feats), raw_arr, proc_arr)


# ------------------------------------------------------------------------------------ assembly
def build(jobs: int = 4, cohorts: tuple[str, ...] = ("HC",)) -> dict:
    """Process all verified records of ``cohorts`` and write recordings/windows/features/waveforms."""
    t0 = time.time()
    man = json.loads((DATA_ROOT / "metadata" / "auricle_manifest.json").read_text())
    manifest = {r["record"]: r for r in man["records"]}
    recs = [r for r in auricle_records(AURICLE_DIR, cohorts=cohorts) if manifest.get(r, {}).get("status") == "ok"]
    demo = load_auricle_demographics(AURICLE_DIR)
    results = Parallel(n_jobs=jobs, batch_size=1)(delayed(process_record)(r, demo, manifest) for r in recs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAVE_DIR.mkdir(parents=True, exist_ok=True)
    windows, features, recordings = [], [], []
    for sess in sorted({rr.recording["session"] for rr in results}):
        part = [rr for rr in results if rr.recording["session"] == sess]
        ids, arrays = [], {f"{k}_{s}": [] for k in ("raw", "proc") for s in SITES}
        offset = 0
        for rr in part:
            w = rr.windows.copy()
            w["waveform_index"] = np.arange(offset, offset + len(w))
            offset += len(w)
            windows.append(w)
            features.append(rr.features)
            recordings.append(rr.recording)
            ids.extend(w.window_id)
            for s in SITES:
                arrays[f"raw_{s}"].append(rr.raw[s])
                arrays[f"proc_{s}"].append(rr.proc[s])
        meta = {
            "dataset": "Auricle v1.0.0 (PhysioNet, ODC-By 1.0), healthy cohort",
            "session": sess,
            "fs_hz": TARGET_FS,
            "samples_per_window": WAVE_LEN,
            "arrays": {
                "raw_ear": "earlobe PPG, mV (BIOPAC amplifier output), 100 Hz",
                "raw_finger": "finger PPG, mV, 100 Hz",
                "proc_ear": "earlobe PPG, processed, dimensionless z-score, 100 Hz (NaN if unavailable)",
                "proc_finger": "finger PPG, processed, dimensionless z-score, 100 Hz",
                "window_id": "row key; join with windows.csv / features.csv",
            },
            "preprocessing": PREPROCESSING,
        }
        np.savez_compressed(
            WAVE_DIR / f"auricle_waveforms_session{sess}.npz",
            window_id=np.asarray(ids),
            **{k: np.concatenate(v) for k, v in arrays.items()},
            metadata=np.asarray(json.dumps(meta)),
        )
    win = pd.concat(windows, ignore_index=True)
    feat = pd.concat(features, ignore_index=True)
    feat = win[["window_id", "subject_id", "record_id", "session"]].merge(feat, on="window_id")
    rec = pd.DataFrame(recordings).sort_values(["session", "record_id"])
    win.to_csv(OUT_DIR / "windows.csv", index=False)
    feat.to_csv(OUT_DIR / "features.csv", index=False, float_format="%.6g")
    rec.to_csv(OUT_DIR / "recordings.csv", index=False)
    feature_schema().to_csv(OUT_DIR / "feature_schema.csv", index=False)
    subjects(win, rec, demo).to_csv(OUT_DIR / "subjects.csv", index=False)
    summ = summary()
    print(f"built {len(rec)} recordings, {len(win)} windows in {time.time() - t0:.0f}s")
    return summ


def subjects(win: pd.DataFrame, rec: pd.DataFrame, demo: dict) -> pd.DataFrame:
    rows = []
    for sid, g in win.groupby("subject_id"):
        key = sid.replace("auricle_", "")
        d = demo.get(key, {})
        sess = sorted(g.session.unique())
        r = rec[rec.subject_id == sid]
        cal1 = {s: int(((g.session == 1) & g[f"{s}_calibration_eligible"]).sum()) for s in SITES}
        reasons = []
        if not r.cnap_mask_available.all():
            reasons.append("session(s) without CNAP mask: " + ";".join(r.record_id[~r.cnap_mask_available]))
        if 2 not in sess:
            reasons.append("no session 2")
        if 1 not in sess or g[g.session == 1].label_valid.sum() == 0:
            reasons.append("no labelled session-1 windows")
        for s in SITES:
            if cal1[s] < K_CAL_REST:
                reasons.append(f"{s}: <{K_CAL_REST} calibration-eligible session-1 rest windows")
        rows.append(
            {
                "subject_id": sid,
                "cohort": key.split("_")[0],
                "age_years": d.get("age", np.nan),
                "sex": {1.0: "male", 0.0: "female"}.get(d.get("sex"), "unknown"),
                "height_cm": d.get("height_cm", np.nan),
                "weight_kg": d.get("weight_kg", np.nan),
                "sessions_available": ";".join(map(str, sess)),
                "recording_ids": ";".join(sorted(r.record_id)),
                "recording_dates": "unavailable (de-identified)",
                "sites_available": "ear;finger",
                "n_windows": len(g),
                "n_label_valid_windows": int(g.label_valid.sum()),
                "n_windows_ear_sqi_ok": int(g.ear_sqi_ok.sum()),
                "n_windows_finger_sqi_ok": int(g.finger_sqi_ok.sum()),
                "n_calibration_eligible_s1_ear": cal1["ear"],
                "n_calibration_eligible_s1_finger": cal1["finger"],
                "cross_day_eligible_ear": bool(2 in sess and cal1["ear"] >= K_CAL_REST and r.cnap_mask_available.all()),
                "cross_day_eligible_finger": bool(
                    2 in sess and cal1["finger"] >= K_CAL_REST and r.cnap_mask_available.all()
                ),
                "exclusion_reasons": " | ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def summary() -> pd.DataFrame:
    win = pd.read_csv(OUT_DIR / "windows.csv")
    rec = pd.read_csv(OUT_DIR / "recordings.csv")
    sub = pd.read_csv(OUT_DIR / "subjects.csv")
    feat_cols = [
        c
        for c in pd.read_csv(OUT_DIR / "features.csv", nrows=1).columns
        if c not in ("window_id", "subject_id", "record_id", "session")
    ]
    lv = win[win.label_valid]
    items = [
        ("subjects", sub.shape[0], ""),
        ("subjects_with_both_sessions", int((sub.sessions_available == "1;2").sum()), ""),
        ("recordings", rec.shape[0], ""),
        ("recordings_session1", int((rec.session == 1).sum()), ""),
        ("recordings_session2", int((rec.session == 2).sum()), ""),
        ("total_recording_hours", round(rec.duration_s.sum() / 3600, 2), "h"),
        ("windows_total", len(win), ""),
        ("windows_label_valid", len(lv), ""),
        ("windows_label_valid_session1", int((lv.session == 1).sum()), ""),
        ("windows_label_valid_session2", int((lv.session == 2).sum()), ""),
        ("windows_ear_sqi_ok", int(win.ear_sqi_ok.sum()), ""),
        ("windows_finger_sqi_ok", int(win.finger_sqi_ok.sum()), ""),
        ("windows_label_valid_and_ear_sqi_ok", int((win.label_valid & win.ear_sqi_ok).sum()), ""),
        ("windows_label_valid_and_finger_sqi_ok", int((win.label_valid & win.finger_sqi_ok).sum()), ""),
        ("windows_finger_rail_clipped", int((win.finger_rail_pct > 100 * MAX_RAIL_FRACTION).sum()), ""),
        ("features_per_window", len(feat_cols), ""),
        ("features_per_site", len(SITE_FEATURES), ""),
        ("sbp_mean", round(lv.sbp_mmHg.mean(), 2), "mmHg"),
        ("sbp_sd", round(lv.sbp_mmHg.std(), 2), "mmHg"),
        ("sbp_min", round(lv.sbp_mmHg.min(), 2), "mmHg"),
        ("sbp_max", round(lv.sbp_mmHg.max(), 2), "mmHg"),
        ("dbp_mean", round(lv.dbp_mmHg.mean(), 2), "mmHg"),
        ("dbp_sd", round(lv.dbp_mmHg.std(), 2), "mmHg"),
        ("dbp_min", round(lv.dbp_mmHg.min(), 2), "mmHg"),
        ("dbp_max", round(lv.dbp_mmHg.max(), 2), "mmHg"),
        ("age_median", round(sub.age_years.median(), 1), "years"),
        ("age_min", sub.age_years.min(), "years"),
        ("age_max", sub.age_years.max(), "years"),
        ("female_fraction", round((sub.sex == "female").mean(), 3), ""),
    ]
    for name in sorted(lv.activity_name.unique()):
        items.append((f"label_valid_windows_activity_{name}", int((lv.activity_name == name).sum()), ""))
    for f in sorted(p for p in OUT_DIR.rglob("*") if p.is_file() and p.suffix in (".csv", ".npz")):
        if f.name != "summary.csv":
            items.append((f"file_size_{f.relative_to(OUT_DIR).as_posix()}", round(f.stat().st_size / 1e6, 2), "MB"))
    df = pd.DataFrame(items, columns=["statistic", "value", "unit"])
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    return df


# --------------------------------------------------------------------------------- validation
def validate(check_raw: bool = True, sample: int = 25, seed: int = 0) -> pd.DataFrame:
    """Validate the exported dataset. Returns (and writes) a check table; ``status`` FAIL entries are errors."""
    import wfdb

    res = []

    def add(check, ok, detail, level="FAIL"):
        res.append({"check": check, "status": "PASS" if ok else level, "detail": str(detail)})

    win = pd.read_csv(OUT_DIR / "windows.csv")
    feat = pd.read_csv(OUT_DIR / "features.csv")
    rec = pd.read_csv(OUT_DIR / "recordings.csv")
    sub = pd.read_csv(OUT_DIR / "subjects.csv")
    schema = pd.read_csv(OUT_DIR / "feature_schema.csv")
    man = {
        r["record"]: r for r in json.loads((DATA_ROOT / "metadata" / "auricle_manifest.json").read_text())["records"]
    }

    add(
        "recordings_sha256_match_manifest",
        rec.sha256_verified_against_manifest.all(),
        f"{int(rec.sha256_verified_against_manifest.sum())}/{len(rec)} verified",
    )
    if check_raw:
        bad = [
            r
            for r in rec.record_id
            if _sha256(AURICLE_DIR / "data" / r / f"{r}.dat") != rec.set_index("record_id").sha256_dat[r]
        ]
        add("raw_files_unchanged_since_build", not bad, bad or "all .dat hashes match recordings.csv")
    add("sampling_frequency_1000hz", (rec.source_fs_hz == 1000).all(), rec.source_fs_hz.unique().tolist())
    req = ["PPG_Ear", "PPG_Finger", "Systolic_BP", "Diastolic_BP"]
    miss = {r.record_id: [c for c in req if c not in r.channels.split(";")] for r in rec.itertuples()}
    miss = {k: v for k, v in miss.items() if v}
    add("required_channels_present", not miss, miss or "PPG_Ear, PPG_Finger, Systolic_BP, Diastolic_BP in all")
    no_mask = rec.record_id[~rec.cnap_mask_available].tolist()
    add(
        "cnap_mask_missing_records_flagged",
        (win[win.record_id.isin(no_mask)].label_valid == False).all(),  # noqa: E712
        f"records without mask: {no_mask}; none of their windows carry labels",
    )
    no_act = rec.record_id[~rec.activity_annotations_available].tolist()
    add(
        "activity_annotations_missing_flagged",
        (win[win.record_id.isin(no_act)].activity_code == -1).all(),
        f"records without activity channel: {no_act}",
    )
    lv = win[win.label_valid]
    add(
        "bp_labels_physiological",
        lv.sbp_mmHg.between(*SBP_RANGE).all()
        and lv.dbp_mmHg.between(*DBP_RANGE).all()
        and (lv.sbp_mmHg > lv.dbp_mmHg).all(),
        f"{len(lv)} labelled windows",
    )
    add(
        "labels_only_where_cnap_valid",
        (lv.cnap_valid).all() and win[~win.label_valid].sbp_mmHg.isna().all(),
        "labels blank outside CNAP-valid, plausible windows",
    )
    add(
        "window_ids_unique",
        win.window_id.is_unique and not win.duplicated(["record_id", "sample_start"]).any(),
        f"{len(win)} windows",
    )
    add(
        "window_geometry",
        ((win.sample_end - win.sample_start) == WIN_S * win.source_fs_hz).all()
        and (win.t_end_s - win.t_start_s == WIN_S).all(),
        "all windows 10 s",
    )
    within = win.merge(rec[["record_id", "n_samples"]], on="record_id")
    add("windows_inside_recordings", (within.sample_end <= within.n_samples).all(), "")
    sid_ok = all(r.subject_id == f"auricle_{r.record_id.rsplit('_', 1)[0]}" for r in win.itertuples())
    sess_ok = all(int(r.record_id.rsplit("_", 1)[1]) == r.session for r in win.itertuples())
    add("subject_session_consistency", sid_ok and sess_ok, "subject_id and session match record_id")
    add("subjects_table_complete", set(sub.subject_id) == set(win.subject_id), f"{len(sub)} subjects")
    clipped = (win.ear_rail_pct > 100 * MAX_RAIL_FRACTION) & win.ear_sqi_ok | (
        (win.finger_rail_pct > 100 * MAX_RAIL_FRACTION) & win.finger_sqi_ok
    )
    add(
        "clipped_windows_rejected",
        not clipped.any(),
        f"finger windows clipped > {100 * MAX_RAIL_FRACTION:.0f} %: {int((win.finger_rail_pct > 1).sum())}; "
        f"ear: {int((win.ear_rail_pct > 1).sum())}",
    )
    add(
        "features_one_row_per_window",
        len(feat) == len(win) and set(feat.window_id) == set(win.window_id),
        f"{len(feat)} feature rows",
    )
    fcols = [c for c in feat.columns if c not in ("window_id", "subject_id", "record_id", "session")]
    add("feature_schema_complete", set(fcols) == set(schema.column), f"{len(fcols)} features documented")
    leak = [c for c in fcols if any(t in c.lower() for t in ("sbp", "dbp", "map", "bp_", "cnap"))]
    add("no_bp_derived_features", not leak, leak or "no BP/CNAP-derived inputs")
    # waveform references
    ok_wave, n_checked, wave_err = True, 0, []
    rng = np.random.default_rng(seed)
    for sess, g in win.groupby("session"):
        z = np.load(OUT_DIR / f"waveforms/auricle_waveforms_session{sess}.npz")
        ids = z["window_id"]
        if len(ids) != len(g) or not (ids[g.waveform_index.to_numpy()] == g.window_id.to_numpy()).all():
            ok_wave = False
            wave_err.append(f"session {sess}: index mismatch")
        for k in ("raw_ear", "raw_finger", "proc_ear", "proc_finger"):
            if z[k].shape != (len(ids), WAVE_LEN):
                ok_wave = False
                wave_err.append(f"session {sess}: {k} shape {z[k].shape}")
        if check_raw:
            for row in g.sample(min(sample, len(g)), random_state=int(rng.integers(1e9))).itertuples():
                rr = wfdb.rdrecord(
                    str(AURICLE_DIR / "data" / row.record_id / row.record_id),
                    channel_names=["PPG_Ear"],
                    sampfrom=row.sample_start,
                    sampto=row.sample_end,
                )
                orig = rr.p_signal[:, 0]
                ref = z["raw_ear"][row.waveform_index]
                # Re-derive the stored 100 Hz array with the same polyphase anti-aliasing resampler and require
                # an exact match away from the window edges (record- vs window-level filtering differs only there).
                med = float(np.median(orig))
                redo = filtering.resample(orig - med, row.source_fs_hz, TARGET_FS) + med
                core = slice(50, WAVE_LEN - 50)
                err = float(np.abs(redo[core] - ref[core]).max())
                n_checked += 1
                if err > 1e-3 * (np.ptp(orig) + 1e-6) + 1e-5:
                    ok_wave = False
                    wave_err.append(f"{row.window_id}: max |diff| {err:.2e} mV")
    add("waveform_references_match_wfdb", ok_wave, wave_err or f"{n_checked} random windows matched raw WFDB")
    hc2 = [r for r in man if r.startswith("HC_") and r.endswith("_2")]
    add("healthy_session2_complete", set(hc2) <= set(rec.record_id), f"{len(set(hc2) & set(rec.record_id))}/{len(hc2)}")
    out = pd.DataFrame(res)
    out.to_csv(OUT_DIR / "validation_report.csv", index=False)
    return out
