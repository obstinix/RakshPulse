"""Build processed, model-ready tables from the extracted datasets.

Outputs (in data/processed/):
* ``<dataset>_features.csv``: one row per window with subject id, labels, demographics,
  SQI fields and all PPG features.
* ``<dataset>_windows.npz``: band-passed, z-scored waveforms at 100 Hz, shape
  (n, 2, 1000), channel order [ir, red], aligned row-for-row with the CSV. Missing
  channels are NaN.

Usage::

    python -m rakshpulse.data.build guilin_dualwave foggia_multiwave ppg_bp
    python -m rakshpulse.data.build guilin_unlabelled   # post-exercise states, SQI only
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..features.feature_pipeline import extract_features
from ..preprocessing.filtering import TARGET_FS
from ..preprocessing.segmentation import sliding_windows
from .dataset_loader import Record, iter_foggia, iter_guilin, iter_ppg_bp
from .download import DATA_ROOT

PROCESSED = DATA_ROOT / "processed"
WIN_S = 10.0
STEP_S = 5.0
SETTLE_S = 10.0  # discard the first 10 s of each long recording (sensor/AGC settling transient)
WAVE_LEN = int(WIN_S * TARGET_FS)


def _process_record(rec: Record) -> list[dict]:
    chans = [c for c in ("ir", "red") if c in rec.channels]
    n = min(rec.channels[c].size for c in chans)
    duration = n / rec.fs
    rows = []
    if duration >= WIN_S + SETTLE_S:  # long recordings -> sliding windows
        start = int(SETTLE_S * rec.fs)
        stack = np.stack([rec.channels[c][start:n] for c in chans])
        windows = sliding_windows(stack, rec.fs, WIN_S, STEP_S)
        offsets = [SETTLE_S + i * STEP_S for i in range(len(windows))]
    else:  # short recordings -> whole record is a single window
        windows = np.stack([rec.channels[c][:n] for c in chans])[None]
        offsets = [0.0]
    for w, off in zip(windows, offsets, strict=True):
        sig = {c: w[i] for i, c in enumerate(chans)}
        feats, proc = extract_features(sig, rec.fs, has_dc=rec.has_dc)
        wave = np.full((2, WAVE_LEN), np.nan, dtype=np.float32)
        for i, c in enumerate(("ir", "red")):
            if c in proc:
                z = proc[c].filtered[:WAVE_LEN]
                wave[i, : z.size] = z
        row = {
            "dataset": rec.dataset,
            "subject_id": rec.subject_id,
            "record_id": rec.record_id,
            "state": rec.state,
            "t_start_s": off,
            "sbp": rec.sbp if rec.sbp is not None else np.nan,
            "dbp": rec.dbp if rec.dbp is not None else np.nan,
            **{f"demo_{k}": v for k, v in rec.demographics.items()},
        }
        for c in ("ir", "red"):
            if c in proc:
                s = proc[c].sqi
                row[f"{c}_sqi_ok"] = s.acceptable
                row[f"{c}_sqi_reasons"] = "|".join(s.reasons)
                row[f"{c}_sqi_template_corr"] = s.template_corr
                row[f"{c}_sqi_cardiac_power"] = s.cardiac_power_ratio
                row[f"{c}_sqi_skewness"] = s.skewness
                row[f"{c}_n_beats"] = s.n_beats
                row[f"{c}_perfusion_ac_dc"] = proc[c].ac_dc
            else:
                row[f"{c}_sqi_ok"] = False
                row[f"{c}_sqi_reasons"] = "channel_missing"
        row.update(feats)
        row["_wave"] = wave
        rows.append(row)
    return rows


def build(name: str, n_jobs: int = -1) -> pd.DataFrame:
    sources = {
        "guilin_dualwave": lambda: iter_guilin(states=("1",)),
        "guilin_unlabelled": lambda: iter_guilin(states=("2", "3")),
        "foggia_multiwave": iter_foggia,
        "ppg_bp": iter_ppg_bp,
    }
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_process_record)(r) for r in sources[name]())
    rows = [r for rs in results for r in rs]
    waves = np.stack([r.pop("_wave") for r in rows]) if rows else np.empty((0, 2, WAVE_LEN))
    df = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / f"{name}_features.csv", index=False)
    np.savez_compressed(PROCESSED / f"{name}_windows.npz", waves=waves)
    summary = {
        "dataset": name,
        "n_windows": len(df),
        "n_subjects": int(df["subject_id"].nunique()) if len(df) else 0,
        "ir_sqi_ok_fraction": float(df["ir_sqi_ok"].mean()) if len(df) else None,
        "red_sqi_ok_fraction": float(df["red_sqi_ok"].mean()) if len(df) else None,
        "seconds": round(time.time() - t0, 1),
    }
    (PROCESSED / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    for d in a.datasets:
        build(d, a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
