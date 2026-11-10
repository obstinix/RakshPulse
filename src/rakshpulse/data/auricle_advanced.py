"""RakshPulse 3.0 supplementary Auricle features (``data/auricle/features_advanced.csv``).

The 2.0 tables (``features.csv``, ``feature_schema.csv``) are left untouched as the comparison baseline. This
table adds, per ``window_id``:

* ``ear_pat_*`` / ``finger_pat_*``: ECG (lead from the BIOPAC ECG channel) to PPG pulse arrival time.
  **ECG-assisted**: the MAX30102 has no ECG.
* ``ear_finger_ptt_ms``, ``ear_finger_ptt_sd_ms``: ear-foot → finger-foot pulse transit time, ECG-free. This needs
  two optical sites, which is a hardware option, not the current single-sensor design.
* ``ear_finger_corr``: correlation of the processed ear and finger windows (same window, same time base).
* ``ear_wav_*`` / ``finger_wav_*``: Haar wavelet relative energies of the processed waveform.

Schema: ``data/auricle/feature_schema_advanced.csv``. Command: ``python -m rakshpulse.data export-advanced``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..features.advanced import PAT_COLUMNS, WAVELET_COLUMNS, haar_relative_energies, window_pat
from ..preprocessing import filtering, normalization
from ..preprocessing.segmentation import detect_feet, detect_peaks
from . import auricle_dataset as ds

SCHEMA = [
    ("{site}_pat_foot_ms", "ms", "ecg_timing", "median ECG R-peak to PPG foot interval (80-500 ms beats only)"),
    ("{site}_pat_slope_ms", "ms", "ecg_timing", "median ECG R-peak to PPG maximum-upslope interval"),
    ("{site}_pat_foot_sd_ms", "ms", "ecg_timing", "SD of beat-wise R-to-foot intervals"),
] + [(f"{{site}}_{c}", "1", "wavelet", f"Haar relative energy, {c.split('_')[1]}") for c in WAVELET_COLUMNS]
PAIR_SCHEMA = [
    ("ear_finger_ptt_ms", "ms", "cross_site", "median ear-foot to next finger-foot delay (0-300 ms), ECG-free"),
    ("ear_finger_ptt_sd_ms", "ms", "cross_site", "SD of beat-wise ear-to-finger delays"),
    ("ear_finger_corr", "1", "cross_site", "Pearson correlation of processed ear and finger windows"),
]


def advanced_schema() -> pd.DataFrame:
    rows = [
        {"column": c.format(site=s), "site": s, "group": g, "unit": u, "definition": d}
        for s in ("ear", "finger")
        for c, u, g, d in SCHEMA
    ]
    rows += [{"column": c, "site": "ear+finger", "group": g, "unit": u, "definition": d} for c, u, g, d in PAIR_SCHEMA]
    return pd.DataFrame(rows)


def _feet_times(x: np.ndarray, fs: float) -> np.ndarray:
    z = normalization.orient_ppg(filtering.bandpass(x - np.median(x), fs), fs)
    pk = detect_peaks(z, fs)
    return detect_feet(z, pk) / fs if pk.size >= 3 else np.array([])


def ear_finger_ptt(ear: np.ndarray, finger: np.ndarray, fs: float) -> tuple[float, float]:
    fe, ff = _feet_times(ear, fs), _feet_times(finger, fs)
    if fe.size < 3 or ff.size < 3:
        return np.nan, np.nan
    d = []
    for t in fe:
        nxt = ff[(ff >= t) & (ff <= t + 0.3)]
        if nxt.size:
            d.append((nxt[0] - t) * 1000)
    return (float(np.median(d)), float(np.std(d))) if len(d) >= 3 else (np.nan, np.nan)


def process_record(rid: str, windows: pd.DataFrame, proc: dict[str, np.ndarray]) -> pd.DataFrame:
    import wfdb

    r = wfdb.rdrecord(str(ds.AURICLE_DIR / "data" / rid / rid), channel_names=["ECG", "PPG_Ear", "PPG_Finger"])
    x = {n: r.p_signal[:, i] for i, n in enumerate(r.sig_name)}
    fs = float(r.fs)
    rows = []
    for w in windows.itertuples():
        a, b = int(w.sample_start), int(w.sample_end)
        row = {"window_id": w.window_id}
        for site, ch in (("ear", "PPG_Ear"), ("finger", "PPG_Finger")):
            ok = bool(getattr(w, f"{site}_sqi_ok"))
            pat = window_pat(x["ECG"][a:b], x[ch][a:b], fs) if ok else dict.fromkeys(PAT_COLUMNS, np.nan)
            row.update({f"{site}_{k}": v for k, v in pat.items()})
            p = proc[site][int(w.waveform_index)]
            wav = haar_relative_energies(p) if ok and np.isfinite(p).all() else dict.fromkeys(WAVELET_COLUMNS, np.nan)
            row.update({f"{site}_{k}": v for k, v in wav.items()})
        both = bool(w.ear_sqi_ok and w.finger_sqi_ok)
        ptt, ptt_sd = ear_finger_ptt(x["PPG_Ear"][a:b], x["PPG_Finger"][a:b], fs) if both else (np.nan, np.nan)
        pe, pf = proc["ear"][int(w.waveform_index)], proc["finger"][int(w.waveform_index)]
        corr = float(np.corrcoef(pe, pf)[0, 1]) if both and np.isfinite(pe).all() and np.isfinite(pf).all() else np.nan
        row.update(ear_finger_ptt_ms=ptt, ear_finger_ptt_sd_ms=ptt_sd, ear_finger_corr=corr)
        rows.append(row)
    return pd.DataFrame(rows)


def build_advanced(jobs: int = 4) -> pd.DataFrame:
    t0 = time.time()
    win = pd.read_csv(ds.OUT_DIR / "windows.csv", low_memory=False)
    out = []
    for sess, g in win.groupby("session"):
        z = np.load(ds.OUT_DIR / f"waveforms/auricle_waveforms_session{sess}.npz")
        proc = {"ear": z["proc_ear"], "finger": z["proc_finger"]}
        out += Parallel(n_jobs=jobs, batch_size=1)(
            delayed(process_record)(rid, gg, proc) for rid, gg in g.groupby("record_id")
        )
    adv = win[["window_id", "subject_id", "record_id", "session"]].merge(pd.concat(out), on="window_id")
    adv.to_csv(ds.OUT_DIR / "features_advanced.csv", index=False, float_format="%.6g")
    advanced_schema().to_csv(ds.OUT_DIR / "feature_schema_advanced.csv", index=False)
    print(f"features_advanced.csv: {len(adv)} rows x {adv.shape[1] - 4} features ({time.time() - t0:.0f}s)")
    return adv
