"""Window-level feature tables for Auricle (paired finger + earlobe PPG, CNAP BP labels).

Per 10 s window (5 s step) of each record:
* features are extracted **independently** from the finger and the earlobe PPG with the same
  pipeline as the other datasets. Each is a single unspecified-wavelength Nonin channel, run
  through the single-channel path and renamed ``finger_*`` / ``ear_*``. They are not
  treated as red/IR;
* the BP label is the median beat-level CNAP SBP/DBP inside the window;
* a window is labelled only if the CNAP reliability mask is 1 for the **whole** window and
  the label is physiological (SBP 60–250, DBP 30–150, SBP > DBP). The CNAP auto-calibration
  gaps (~5 s every ~35 s) and the uncalibrated start of every session are excluded.

Results are cached per record in data/interim/auricle/ so the build can run incrementally
while the download is still in progress.

Usage: ``python -m rakshpulse.data.build_auricle [--jobs 6]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..features.feature_pipeline import PER_CHANNEL, extract_features
from .build import PROCESSED
from .dataset_loader import MissingChannelError, auricle_records, load_auricle_demographics, read_auricle
from .download import DATA_ROOT

INTERIM = DATA_ROOT / "interim" / "auricle"
WIN_S, STEP_S = 10.0, 5.0
SBP_OK, DBP_OK = (60.0, 250.0), (30.0, 150.0)


def _site_features(sig: np.ndarray, fs: float, site: str) -> dict:
    feats, proc = extract_features({"ir": sig}, fs, has_dc=False)
    out = {f"{site}_{n}": feats[f"ir_{n}"] for n in PER_CHANNEL}
    s = proc["ir"].sqi
    out[f"{site}_sqi_ok"] = s.acceptable
    out[f"{site}_sqi_reasons"] = "|".join(s.reasons)
    out[f"{site}_sqi_template_corr"] = s.template_corr
    out[f"{site}_sat_fraction"] = float(np.mean(np.abs(sig) >= 9.99))  # ±10 V amplifier rail
    return out


def process_record(rid: str, demo: dict) -> pd.DataFrame:
    cache = INTERIM / f"{rid}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    try:
        rec = read_auricle(rid, demo=demo)
    except MissingChannelError as exc:
        print(f"[skip] {exc}")
        return pd.DataFrame()
    fs = rec.fs
    win, step = int(WIN_S * fs), int(STEP_S * fs)
    rows = []
    for a in range(0, rec.ear.size - win + 1, step):
        b = a + win
        if not rec.cnap_valid[a:b].all():
            continue
        sbp, dbp = float(np.median(rec.sbp[a:b])), float(np.median(rec.dbp[a:b]))
        if not (SBP_OK[0] <= sbp <= SBP_OK[1] and DBP_OK[0] <= dbp <= DBP_OK[1] and sbp > dbp):
            continue
        acts = rec.activity[a:b]
        row = {
            "dataset": "auricle",
            "subject_id": rec.subject_id,
            "record_id": rid,
            "cohort": rec.cohort,
            "session": int(rid.split("_")[2]),
            "t_start_s": a / fs,
            "sbp": sbp,
            "dbp": dbp,
            "sbp_range_in_window": float(np.ptp(rec.sbp[a:b])),
            "activity": int(np.bincount(acts[acts >= 0]).argmax()) if (acts >= 0).any() else -1,
            **{f"demo_{k}": v for k, v in rec.demographics.items()},
        }
        row.update(_site_features(rec.ear[a:b], fs, "ear"))
        row.update(_site_features(rec.finger[a:b], fs, "finger"))
        rows.append(row)
    df = pd.DataFrame(rows)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def build(jobs: int = 6) -> pd.DataFrame:
    t0 = time.time()
    demo = load_auricle_demographics()
    recs = auricle_records()
    dfs = Parallel(n_jobs=jobs)(delayed(process_record)(r, demo) for r in recs)
    df = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "auricle_features.csv", index=False)
    print(
        f"auricle: {len(recs)} records, {df.subject_id.nunique()} subjects, {len(df)} labelled windows, "
        f"ear SQI {df.ear_sqi_ok.mean():.3f}, finger SQI {df.finger_sqi_ok.mean():.3f} ({time.time() - t0:.0f}s)"
    )
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    build(ap.parse_args(argv).jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
