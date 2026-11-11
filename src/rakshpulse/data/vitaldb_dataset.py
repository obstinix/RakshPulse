"""VitalDB adapter: surgical cases with finger PPG, ECG, arterial line and intermittent cuff (NIBP).

Source: *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients* v1.0.0, PhysioNet,
doi:10.13026/czw8-9p62, **CC BY 4.0**. Files are ``vital_files/<caseid>.vital``, downloaded from PhysioNet and
SHA-256-verified (``python -m rakshpulse.data.physionet vitaldb --records ...``), and read with the MIT-licensed
``vitaldb`` package (optional extra ``pip install -e .[vitaldb]``). Originals are kept unmodified in
``data/raw/vitaldb/``.

Why VitalDB matters for RakshPulse: it has an **invasive arterial line** (a far stronger BP reference than CNAP),
**real intermittent cuff readings** (NIBP, the reference a wearable would actually be calibrated with) and large
within-patient BP changes (induction, vasopressors), so it tests whether PPG adds information *beyond the last
cuff reading* under realistic conditions. It is a different population (anaesthetised surgical patients,
finger PPG), so it is an independent-dataset check, not a stand-in for behind-ear PPG.

Outputs (``data/vitaldb/``): ``cases.csv``, ``windows.csv`` (labels, quality, PAT, waveform position),
``features.csv`` (finger PPG + wavelet + ECG-PAT), ``nibp_readings.csv`` (cuff readings with timestamps),
``summary.csv``. Case selection is fixed and recorded in ``data/metadata/vitaldb_selection.csv``.
"""

from __future__ import annotations

import gzip
import io
import time
import urllib.request

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..features.advanced import PAT_COLUMNS, haar_relative_energies, window_pat
from ..features.feature_pipeline import PER_CHANNEL, extract_features
from .download import DATA_ROOT, USER_AGENT

RAW = DATA_ROOT / "raw" / "vitaldb"
OUT = DATA_ROOT / "vitaldb"
SELECTION = DATA_ROOT / "metadata" / "vitaldb_selection.csv"
# RakshPulse 4.0: an extension set of further cases, disjoint from SELECTION, kept in its own folder so the 3.0
# subset and its results stay immutable. It is used for *independent* external validation (models fitted on the
# 3.0 cases are evaluated on these patients without refitting).
SELECTION_EXT = DATA_ROOT / "metadata" / "vitaldb_selection_ext.csv"
OUT_EXT = DATA_ROOT / "vitaldb_ext"
GZIP_MAGIC = bytes([0x1F, 0x8B])
SETS = {"v3": (SELECTION, OUT), "ext": (SELECTION_EXT, OUT_EXT)}
TRACKS = {
    "ppg": "SNUADC/PLETH",
    "ecg": "SNUADC/ECG_II",
    "art": "SNUADC/ART",
    "art_sbp": "Solar8000/ART_SBP",
    "art_dbp": "Solar8000/ART_DBP",
    "art_mbp": "Solar8000/ART_MBP",
    "nibp_sbp": "Solar8000/NIBP_SBP",
    "nibp_dbp": "Solar8000/NIBP_DBP",
}
FS = 500.0
WIN_S, STEP_S = 10.0, 20.0
SBP_RANGE, DBP_RANGE, MIN_PULSE_PRESSURE = (40.0, 250.0), (20.0, 150.0), 10.0
MAX_ART_SBP_RANGE = 30.0
# SNUADC/PLETH lags SNUADC/ECG_II by a roughly constant ~0.4-0.5 s (monitor analog-output processing): R→foot
# delays of ~650 ms at RR ~730 ms were measured on case 20. Absolute PAT values are therefore NOT physiological;
# only within-case PAT *changes* are interpretable. The search range is widened accordingly.
PAT_RANGE_MS = (300.0, 1000.0)  # arterial SBP spread inside one window above this = flush/zeroing/artefact


def _eligible_cases() -> np.ndarray:
    """Case ids that have every track in TRACKS (public VitalDB track list; metadata only)."""
    req = urllib.request.Request("https://api.vitaldb.net/trks", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    trks = pd.read_csv(io.BytesIO(gzip.decompress(raw) if raw[:2] == GZIP_MAGIC else raw))
    have = trks[trks.tname.isin(TRACKS.values())].groupby("caseid").tname.nunique()
    return np.sort(have[have == len(TRACKS)].index.to_numpy())


def select_extension(n: int = 60, seed: int = 1) -> pd.DataFrame:
    """Draw ``n`` further eligible cases, excluding every case already in ``SELECTION``."""
    base = pd.read_csv(SELECTION)
    eligible = _eligible_cases()
    pool = np.setdiff1d(eligible, base.caseid.to_numpy())
    pick = np.sort(np.random.default_rng(seed).choice(pool, size=min(n, pool.size), replace=False))
    sel = pd.DataFrame({"caseid": pick, "file": [f"{c:04d}.vital" for c in pick]})
    sel.attrs["n_eligible"] = int(eligible.size)
    sel.assign(n_eligible_cases=int(eligible.size), seed=seed, excluded_from="vitaldb_selection.csv").to_csv(
        SELECTION_EXT, index=False
    )
    return sel


def select_cases(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Pick ``n`` cases that have every required track (metadata-only API call; no signal data)."""
    eligible = _eligible_cases()
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(eligible, size=min(n, eligible.size), replace=False))
    sel = pd.DataFrame({"caseid": pick, "file": [f"{c:04d}.vital" for c in pick]})
    sel.attrs["n_eligible"] = int(eligible.size)
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    sel.assign(n_eligible_cases=int(eligible.size), seed=seed).to_csv(SELECTION, index=False)
    return sel


def _labels(win: np.ndarray, kind: str) -> float:
    v = win[np.isfinite(win)]
    return float(np.median(v)) if v.size else np.nan


def process_case(caseid: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import vitaldb

    path = RAW / "vital_files" / f"{caseid:04d}.vital"
    vf = vitaldb.VitalFile(str(path), list(TRACKS.values()))
    w = vf.to_numpy([TRACKS[k] for k in ("ppg", "ecg", "art")], 1 / FS)
    num = vf.to_numpy([TRACKS[k] for k in ("art_sbp", "art_dbp", "art_mbp", "nibp_sbp", "nibp_dbp")], 1.0)
    ppg, ecg, art = w[:, 0], w[:, 1], w[:, 2]
    n_s = num.shape[0]
    # cuff readings: numeric samples that are present (sparse track)
    nib = pd.DataFrame({"t_s": np.arange(n_s, dtype=float), "nibp_sbp": num[:, 3], "nibp_dbp": num[:, 4]})
    nib = nib.dropna()
    nib = nib[
        (nib.nibp_sbp.between(*SBP_RANGE))
        & (nib.nibp_dbp.between(*DBP_RANGE))
        & (nib.nibp_sbp - nib.nibp_dbp >= MIN_PULSE_PRESSURE)
    ]
    nib = nib.loc[
        nib[["nibp_sbp", "nibp_dbp"]].diff().abs().sum(axis=1).ne(0) | (nib.index == nib.index[0])
        if len(nib)
        else nib.index
    ]  # numeric tracks repeat the last value; keep changes only
    nib.insert(0, "caseid", caseid)
    win_n, step_n = int(WIN_S * FS), int(STEP_S * FS)
    rows, feats = [], []
    for i, a in enumerate(range(0, ppg.size - win_n + 1, step_n)):
        b = a + win_n
        t0 = a / FS
        sp = ppg[a:b]
        s0, s1 = int(t0), int(t0 + WIN_S)
        sbp, dbp, mbp = (_labels(num[s0:s1, j], "") for j in (0, 1, 2))
        art_w = art[a:b]
        art_ok = np.isfinite(art_w).mean() > 0.9 and np.nanstd(art_w) > 1.0
        sbp_range = (
            float(np.nanmax(num[s0:s1, 0]) - np.nanmin(num[s0:s1, 0])) if np.isfinite(num[s0:s1, 0]).any() else np.nan
        )
        label_valid = bool(
            art_ok
            and np.isfinite(sbp)
            and np.isfinite(dbp)
            and SBP_RANGE[0] <= sbp <= SBP_RANGE[1]
            and DBP_RANGE[0] <= dbp <= DBP_RANGE[1]
            and sbp - dbp >= MIN_PULSE_PRESSURE
            and (not np.isfinite(sbp_range) or sbp_range <= MAX_ART_SBP_RANGE)
        )
        row = {
            "window_id": f"VDB{caseid:04d}_w{i:05d}",
            "caseid": caseid,
            "subject_id": f"vitaldb_{caseid}",
            "t_start_s": t0,
            "t_end_s": t0 + WIN_S,
            "sample_start": a,
            "sample_end": b,
            "source_fs_hz": FS,
            "label_valid": label_valid,
            "art_sbp_mmHg": sbp if label_valid else np.nan,
            "art_dbp_mmHg": dbp if label_valid else np.nan,
            "art_mbp_mmHg": mbp if label_valid else np.nan,
            "art_sbp_range_in_window_mmHg": sbp_range,
        }
        frow = {"window_id": row["window_id"]}
        finite = np.isfinite(sp).mean()
        flat = finite < 0.95 or np.nanstd(sp) < 1e-3
        if flat:
            row.update(finger_sqi_ok=False, finger_sqi_reasons="no_signal_or_missing", finger_hr_bpm=np.nan)
            frow.update({f"finger_{k}": np.nan for k in PER_CHANNEL})
            frow.update(dict.fromkeys(("wav_e1", "wav_e2", "wav_e3", "wav_e4", "wav_e5", "wav_e6", "wav_a6"), np.nan))
            frow.update(dict.fromkeys(PAT_COLUMNS, np.nan))
        else:
            sp = np.where(np.isfinite(sp), sp, np.nanmedian(sp))
            fts, proc = extract_features({"ir": sp}, FS, has_dc=False)
            pc = proc["ir"]
            row.update(
                finger_sqi_ok=bool(pc.sqi.acceptable),
                finger_sqi_reasons="|".join(pc.sqi.reasons),
                finger_hr_bpm=fts["ir_hr"],
            )
            frow.update({f"finger_{k}": fts[f"ir_{k}"] for k in PER_CHANNEL})
            frow.update(haar_relative_energies(pc.filtered))
            e = ecg[a:b]
            frow.update(window_pat(e if np.isfinite(e).mean() > 0.95 else None, sp, FS, PAT_RANGE_MS))
        rows.append(row)
        feats.append(frow)
    case = {
        "caseid": caseid,
        "subject_id": f"vitaldb_{caseid}",
        "duration_s": round(ppg.size / FS, 1),
        "n_windows": len(rows),
        "n_label_valid": int(sum(r["label_valid"] for r in rows)),
        "n_nibp_readings": len(nib),
        "source_file": f"vital_files/{caseid:04d}.vital",
    }
    return case, pd.DataFrame(rows), pd.DataFrame(feats), nib


def build(jobs: int = 3, which: str = "v3") -> pd.DataFrame:
    t0 = time.time()
    selection, out = SETS[which]
    sel = pd.read_csv(selection)
    cases = [int(c) for c in sel.caseid if (RAW / "vital_files" / f"{int(c):04d}.vital").exists()]
    res = Parallel(n_jobs=jobs, batch_size=1)(delayed(process_case)(c) for c in cases)
    out.mkdir(parents=True, exist_ok=True)
    cl = pd.read_csv(RAW / "clinical_data.csv")
    keep = ["caseid", "age", "sex", "height", "weight", "bmi", "asa", "optype", "department", "approach", "ane_type"]
    case_df = pd.DataFrame([r[0] for r in res]).merge(cl[[k for k in keep if k in cl]], on="caseid", how="left")
    win = pd.concat([r[1] for r in res], ignore_index=True)
    feat = win[["window_id", "subject_id"]].merge(pd.concat([r[2] for r in res], ignore_index=True), on="window_id")
    nib = pd.concat([r[3] for r in res], ignore_index=True)
    pat_ok = feat.set_index("window_id")["pat_foot_ms"].notna()
    win["pat_available"] = win.window_id.map(pat_ok).fillna(False).astype(bool)
    case_df.to_csv(out / "cases.csv", index=False)
    win.to_csv(out / "windows.csv", index=False, float_format="%.4f")
    feat.to_csv(out / "features.csv", index=False, float_format="%.6g")
    nib.to_csv(out / "nibp_readings.csv", index=False)
    summ = summarize(sel=sel, which=which)
    print(f"({time.time() - t0:.0f}s)")
    return summ


def summarize(sel: pd.DataFrame | None = None, which: str = "v3") -> pd.DataFrame:
    """Write ``summary.csv`` from the tables already in ``data/vitaldb`` (no re-processing)."""
    selection, out = SETS[which]
    sel = sel if sel is not None else pd.read_csv(selection)
    case_df = pd.read_csv(out / "cases.csv")
    win = pd.read_csv(out / "windows.csv")
    feat = pd.read_csv(out / "features.csv", nrows=1)
    nib = pd.read_csv(out / "nibp_readings.csv")
    lv = win[win.label_valid]
    age = pd.to_numeric(case_df.age, errors="coerce")  # clinical_data.csv stores age as text for some cases
    summ = pd.DataFrame(
        [
            ("cases", len(case_df), ""),
            ("cases_eligible_in_vitaldb", int(sel.n_eligible_cases.iloc[0]), ""),
            ("total_hours", round(case_df.duration_s.sum() / 3600, 1), "h"),
            ("windows_total", len(win), ""),
            ("windows_label_valid", len(lv), ""),
            ("windows_finger_sqi_ok", int(win.finger_sqi_ok.sum()), ""),
            ("windows_label_valid_and_sqi_ok", int((win.label_valid & win.finger_sqi_ok).sum()), ""),
            ("windows_with_pat", int(win.pat_available.sum()), ""),
            ("nibp_readings", len(nib), ""),
            ("art_sbp_mean", round(lv.art_sbp_mmHg.mean(), 1), "mmHg"),
            ("art_sbp_sd", round(lv.art_sbp_mmHg.std(), 1), "mmHg"),
            ("art_sbp_within_case_sd_median", round(lv.groupby("caseid").art_sbp_mmHg.std().median(), 1), "mmHg"),
            ("age_median", age.median(), "years"),
            ("features_per_window", feat.shape[1] - 2, ""),
        ],
        columns=["statistic", "value", "unit"],
    )
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))
    return summ


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m rakshpulse.data.vitaldb_dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--extension", action="store_true", help="4.0: draw further cases disjoint from the 3.0 set")
    for name in ("build", "summarize"):
        b = sub.add_parser(name)
        b.add_argument("--set", dest="which", choices=sorted(SETS), default="v3")
        if name == "build":
            b.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        if a.extension:
            sel, dest = select_extension(a.n or 60, 1 if a.seed is None else a.seed), SELECTION_EXT
        else:
            sel, dest = select_cases(a.n or 40, 0 if a.seed is None else a.seed), SELECTION
        print(f"selected {len(sel)} of {sel.attrs['n_eligible']} eligible cases -> {dest}")
        print(" ".join(sel.file))
    elif a.cmd == "summarize":
        summarize(which=a.which)
    else:
        build(a.jobs, a.which)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
