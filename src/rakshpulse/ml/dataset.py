"""Dataset audit and preparation for the complete ML workflow.

Primary dataset: the standardised Auricle dataset committed in ``data/auricle/`` (healthy cohort, simultaneous
earlobe PPG and continuous CNAP blood pressure, two recording sessions on different days). Every window's
features were computed with the production feature extractor (``extract_features``), and the earlobe is the
closest available site to the behind-the-ear MAX30102 target.

``build_ear_dataset`` turns those tables into one model-ready CSV. Each exclusion step is logged with window and
subject counts:

1. a reliable CNAP label (``label_valid``);
2. plausible blood pressure (SBP 60–250, DBP 30–150, pulse pressure at least 10 mmHg);
3. label alignment: SBP varied by at most 20 mmHg inside the 10 s window, so one label describes the window;
4. the production signal-quality gate passed for the ear channel;
5. no ADC saturation (rail-clipped samples at most 1 %);
6. the core beat features (heart rate, beat duration) present;
7. at least 20 windows left for the subject.

Raw datasets are never modified. Feature scaling, winsorising and imputation happen inside the model pipeline,
fitted on training subjects only (``training.py``), so no test statistics leak into training.

``make_smoke_dataset`` creates a clearly labelled SYNTHETIC dataset for pipeline smoke tests only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.download import REPO_ROOT
from ..evaluation.auricle_study import add_normalised_shape
from ..features.feature_pipeline import PER_CHANNEL, extract_features

SITE = "ear"
# shape features normalised by beat duration (HR-invariant), as computed at inference by engine.window_feature_dict
NORMALISED = ("w10_n", "w25_n", "w50_n", "w75_n", "w90_n", "t_max_slope_n", "t_apg_b_n", "max_slope_n")
SBP_RANGE, DBP_RANGE, MIN_PULSE_PRESSURE = (60.0, 250.0), (30.0, 150.0), 10.0
MAX_SBP_CHANGE_IN_WINDOW = 20.0
MAX_RAIL_PCT = 1.0
MIN_WINDOWS_PER_SUBJECT = 20


@dataclass(frozen=True)
class DatasetPaths:
    root: Path = REPO_ROOT

    @property
    def auricle(self) -> Path:
        return self.root / "data" / "auricle"

    @property
    def ml_dir(self) -> Path:
        return self.root / "data" / "ml"

    @property
    def ml_dataset(self) -> Path:
        return self.ml_dir / "auricle_ear_bp_windows.csv"

    @property
    def cleaning_log(self) -> Path:
        return self.ml_dir / "auricle_ear_bp_cleaning_log.csv"


DEFAULT_PATHS = DatasetPaths()  # immutable; safe as a default argument

# ------------------------------------------------------------------------------------------------ audit
_AUDIT = {
    # dataset: (folder, required files, label columns to check)
    "auricle": (
        "data/auricle",
        ["windows.csv", "features.csv", "subjects.csv", "recordings.csv"],
        ["sbp_mmHg", "dbp_mmHg"],
    ),
    "vitaldb": (
        "data/vitaldb",
        ["windows.csv", "features.csv", "cases.csv", "nibp_readings.csv"],
        ["art_sbp_mmHg", "art_dbp_mmHg"],
    ),
    "vitaldb_ext": (
        "data/vitaldb_ext",
        ["windows.csv", "features.csv", "cases.csv", "nibp_readings.csv"],
        ["art_sbp_mmHg", "art_dbp_mmHg"],
    ),
    "metadata": ("data/metadata", ["provenance.json", "dataset_summary.json", "validation_report.json"], []),
    "processed (local, regenerable)": ("data/processed", [], []),
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_datasets(paths: DatasetPaths = DEFAULT_PATHS) -> pd.DataFrame:
    """One row per dataset file: presence, size, rows, missing values, label validity and checksum status."""
    rows = []
    for name, (folder, required, labels) in _AUDIT.items():
        d = paths.root / folder
        files = sorted({*(d / f for f in required), *(d.glob("*.csv") if d.exists() else [])})
        if name.startswith("processed"):
            files = sorted(d.glob("*.csv")) if d.exists() else []
            if not files:
                rows.append(
                    {
                        "dataset": name,
                        "file": folder,
                        "status": "absent",
                        "note": "built locally by `python -m rakshpulse.data.build` after downloading the public "
                        "fingertip datasets; not needed by the complete pipeline",
                    }
                )
                continue
        for f in files:
            r = {"dataset": name, "file": f.relative_to(paths.root).as_posix(), "exists": f.exists()}
            if not f.exists():
                rows.append({**r, "status": "missing", "note": "required file is missing"})
                continue
            r["size_mb"] = round(f.stat().st_size / 1e6, 2)
            if f.suffix == ".csv":
                df = pd.read_csv(f, low_memory=False)
                r.update(rows_=len(df), columns=df.shape[1], missing_cells_pct=round(100 * df.isna().mean().mean(), 2))
                lab = [c for c in labels if c in df]
                if lab:
                    y = df[lab].apply(pd.to_numeric, errors="coerce")
                    r["labelled_rows"] = int(y.notna().all(axis=1).sum())
                if "subject_id" in df:
                    r["subjects"] = int(df["subject_id"].nunique())
            r["status"] = "ok"
            rows.append(r)
    # published waveform arrays (optional; only needed to recompute features or plot raw PPG)
    man = paths.auricle / "release_manifest.csv"
    if man.exists():
        m = pd.read_csv(man)
        for _, a in m[m.collection == "waveforms"].iterrows():
            p = paths.root / a.original_file
            status, note = "not downloaded", "optional: `python -m rakshpulse.data.release fetch waveforms`"
            if p.exists():
                ok = p.stat().st_size == int(a.size_bytes)
                status, note = ("ok", "size matches the release manifest") if ok else ("corrupt", "size mismatch")
            rows.append(
                {
                    "dataset": "auricle waveforms (GitHub release)",
                    "file": a.original_file,
                    "exists": p.exists(),
                    "status": status,
                    "note": note,
                }
            )
    out = pd.DataFrame(rows).rename(columns={"rows_": "rows"})
    return out


def validate_auricle(paths: DatasetPaths = DEFAULT_PATHS) -> pd.DataFrame:
    """Integrity checks of the committed Auricle tables. Returns one row per check (PASS/FAIL)."""
    w = pd.read_csv(paths.auricle / "windows.csv", low_memory=False)
    f = pd.read_csv(paths.auricle / "features.csv", usecols=["window_id", "subject_id", "session"])
    s = pd.read_csv(paths.auricle / "subjects.csv")
    r = pd.read_csv(paths.auricle / "recordings.csv")
    lab = w[w.label_valid.astype(bool)]
    checks = [
        ("window ids unique", w.window_id.is_unique, f"{len(w)} windows"),
        (
            "features match windows one-to-one",
            set(w.window_id) == set(f.window_id) and f.window_id.is_unique,
            f"{len(f)} feature rows",
        ),
        (
            "subject and session agree between tables",
            bool(
                (
                    w.set_index("window_id")[["subject_id", "session"]]
                    .join(f.set_index("window_id"), rsuffix="_f")
                    .eval("subject_id == subject_id_f and session == session_f")
                ).all()
            ),
            "",
        ),
        ("every subject documented", set(w.subject_id) <= set(s.subject_id), f"{s.subject_id.nunique()} subjects"),
        ("sessions are 1 or 2", set(w.session.unique()) <= {1, 2}, str(sorted(int(x) for x in w.session.unique()))),
        (
            "one source sampling rate",
            w.source_fs_hz.nunique() == 1,
            ", ".join(f"{x:g}" for x in sorted(w.source_fs_hz.unique())) + " Hz",
        ),
        ("all windows 10 s long", bool(np.allclose(w.t_end_s - w.t_start_s, 10.0)), ""),
        ("labels numeric where valid", bool(lab[["sbp_mmHg", "dbp_mmHg"]].notna().all().all()), f"{len(lab)} labelled"),
        ("SBP > DBP where valid", bool((lab.sbp_mmHg > lab.dbp_mmHg).all()), ""),
        (
            "raw recordings verified against PhysioNet SHA-256",
            bool(r.sha256_verified_against_manifest.astype(bool).all()),
            f"{len(r)} recordings",
        ),
    ]
    return pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL", "detail": det} for c, ok, det in checks])


def feature_reproducibility(paths: DatasetPaths = DEFAULT_PATHS, n: int = 200, seed: int = 0) -> pd.DataFrame | None:
    """Recompute ear features from the published raw waveforms with the production extractor and compare them with
    the stored features (Spearman ρ per feature). Returns None if the waveform arrays are not downloaded."""
    npz = paths.auricle / "waveforms" / "auricle_waveforms_session1.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=False)
    ids = z["window_id"]
    stored = pd.read_csv(paths.auricle / "features.csv", low_memory=False).set_index("window_id")
    w = pd.read_csv(paths.auricle / "windows.csv", usecols=["window_id", "ear_sqi_ok"]).set_index("window_id")
    ok = np.flatnonzero(w.reindex(ids).ear_sqi_ok.fillna(False).to_numpy(bool))
    pick = np.sort(np.random.default_rng(seed).choice(ok, size=min(n, ok.size), replace=False))
    raw = z["raw_ear"]
    recomputed = []
    for i in pick:
        feats, _ = extract_features({"ir": raw[i].astype(float)}, 100.0, has_dc=False)
        recomputed.append(
            {"window_id": ids[i], **{f"{SITE}_{k[3:]}": v for k, v in feats.items() if k.startswith("ir_")}}
        )
    rc = pd.DataFrame(recomputed).set_index("window_id")
    rows = []
    for c in [f"{SITE}_{x}" for x in PER_CHANNEL]:
        a, b = stored.loc[rc.index, c].astype(float), rc[c].astype(float)
        m = a.notna() & b.notna()
        rho = pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman") if m.sum() > 5 else np.nan
        rel = float(np.median(np.abs(a[m] - b[m]) / (np.abs(a[m]) + 1e-9))) if m.any() else np.nan
        rows.append(
            {
                "feature": c,
                "n_windows": int(m.sum()),
                "spearman": round(float(rho), 4),
                "median_relative_difference": round(rel, 4),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ preparation
def candidate_features() -> list[str]:
    """Ear features the hardware path can compute: every per-channel feature of the production extractor plus the
    beat-duration-normalised shape features. Amplitude-dependent raw-signal statistics are not produced by the
    extractor and are excluded (they depend on sensor gain and LED current)."""
    return [f"{SITE}_{n}" for n in PER_CHANNEL] + [f"{SITE}_{n}" for n in NORMALISED]


def _step(log: list, name: str, d: pd.DataFrame, rule: str) -> None:
    log.append(
        {
            "step": len(log),
            "filter": name,
            "rule": rule,
            "windows": len(d),
            "subjects": d["subject_id"].nunique(),
            "records": d["record_id"].nunique(),
        }
    )


def build_ear_dataset(paths: DatasetPaths = DEFAULT_PATHS, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-ready earlobe window table and its cleaning log (see module docstring)."""
    w = pd.read_csv(paths.auricle / "windows.csv", low_memory=False)
    f = pd.read_csv(paths.auricle / "features.csv", low_memory=False).drop(
        columns=["subject_id", "record_id", "session"]
    )
    d = w.merge(f, on="window_id", how="inner", validate="one_to_one")
    log: list[dict] = []
    _step(log, "all windows", d, "standardised Auricle windows (10 s, 5 s hop)")
    d = d[d.label_valid.astype(bool)]
    _step(log, "reliable CNAP label", d, "label_valid (CNAP mask valid, label_status ok)")
    sbp, dbp = d.sbp_mmHg.astype(float), d.dbp_mmHg.astype(float)
    d = d[sbp.between(*SBP_RANGE) & dbp.between(*DBP_RANGE) & (sbp - dbp >= MIN_PULSE_PRESSURE)]
    _step(log, "plausible BP", d, f"SBP {SBP_RANGE}, DBP {DBP_RANGE}, pulse pressure >= {MIN_PULSE_PRESSURE}")
    d = d[d.sbp_range_in_window_mmHg.astype(float) <= MAX_SBP_CHANGE_IN_WINDOW]
    _step(log, "label aligned", d, f"SBP range inside the window <= {MAX_SBP_CHANGE_IN_WINDOW} mmHg")
    d = d[d.ear_sqi_ok.astype(bool)]
    _step(log, "ear signal quality", d, "production SQI gate passed (beats, HR, regularity, template, spectrum)")
    d = d[d.ear_rail_pct.fillna(0).astype(float) <= MAX_RAIL_PCT]
    _step(log, "no saturation", d, f"rail-clipped samples <= {MAX_RAIL_PCT} %")
    d = d[d.ear_hr.notna() & d.ear_beat_T.notna()]
    _step(log, "core features present", d, "heart rate and beat duration extracted")
    counts = d.subject_id.value_counts()
    d = d[d.subject_id.isin(counts[counts >= MIN_WINDOWS_PER_SUBJECT].index)].copy()
    _step(log, "enough windows per subject", d, f">= {MIN_WINDOWS_PER_SUBJECT} windows per subject")

    add_normalised_shape(d, SITE)
    feats = candidate_features()
    d[feats] = d[feats].replace([np.inf, -np.inf], np.nan)
    # non-overlapping windows (every second 5 s-hop window of a record) are used for fitting; all windows are scored
    d = d.sort_values(["record_id", "t_start_s"])
    d["fit_window"] = (d.t_start_s / 5.0).round().astype(int) % 2 == 0  # windows starting at 0, 10, 20 … s
    out = d[
        ["window_id", "subject_id", "record_id", "session", "activity_name", "t_start_s", "fit_window"] + feats
    ].copy()
    out.insert(6, "sbp", d.sbp_mmHg.astype(float).round(2))
    out.insert(7, "dbp", d.dbp_mmHg.astype(float).round(2))
    out = out.rename(columns={"activity_name": "activity"})
    out["synthetic"] = False
    out = out.reset_index(drop=True)
    log_df = pd.DataFrame(log)
    if write:
        paths.ml_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(paths.ml_dataset, index=False, float_format="%.6g")
        log_df.to_csv(paths.cleaning_log, index=False)
    return out, log_df


def load_ml_dataset(paths: DatasetPaths = DEFAULT_PATHS) -> pd.DataFrame:
    d = pd.read_csv(paths.ml_dataset)
    d["fit_window"] = d["fit_window"].astype(bool)
    return d


def input_fingerprint(paths: DatasetPaths = DEFAULT_PATHS) -> str:
    """SHA-256 over the source tables, used to decide whether the prepared dataset is stale."""
    h = hashlib.sha256()
    for name in ("windows.csv", "features.csv"):
        h.update(_sha256(paths.auricle / name).encode())
    h.update(
        json.dumps(
            [
                SBP_RANGE,
                DBP_RANGE,
                MIN_PULSE_PRESSURE,
                MAX_SBP_CHANGE_IN_WINDOW,
                MAX_RAIL_PCT,
                MIN_WINDOWS_PER_SUBJECT,
                candidate_features(),
            ]
        ).encode()
    )
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ smoke test data
def make_smoke_dataset(n_subjects: int = 10, windows_per_session: int = 12, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC windows for pipeline smoke tests only.

    Signals come from the RakshPulse simulator and go through the production feature extractor. The BP labels are
    invented from the simulator's own settings plus noise, so a model that fits them proves only that the code
    runs. Every row carries ``synthetic=True``."""
    from ..acquisition.simulator import SimConfig, simulate

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        base_sbp, base_dbp = rng.normal(118, 12), rng.normal(74, 8)
        for session in (1, 2):
            for k in range(windows_per_session):
                hr = float(np.clip(rng.normal(72, 10), 50, 110))
                refl = float(np.clip(rng.normal(0.45, 0.1), 0.15, 0.8))
                cfg = SimConfig(fs=100.0, hr_bpm=hr, reflection=refl, seed=int(rng.integers(1 << 31)))
                rec = simulate(cfg, 10.0)
                feats, _ = extract_features({"ir": rec.ir}, 100.0, has_dc=True)
                r = {f"{SITE}_{key[3:]}": v for key, v in feats.items() if key.startswith("ir_")}
                r.update(
                    window_id=f"SYN{s:02d}_{session}_w{k:03d}",
                    subject_id=f"synthetic_{s:02d}",
                    record_id=f"SYN{s:02d}_{session}",
                    session=session,
                    activity="synthetic",
                    t_start_s=10.0 * k,
                    fit_window=True,
                    sbp=round(base_sbp + 0.4 * (hr - 72) + 25 * (refl - 0.45) + rng.normal(0, 4), 2),
                    dbp=round(base_dbp + 0.2 * (hr - 72) + 10 * (refl - 0.45) + rng.normal(0, 3), 2),
                    synthetic=True,
                )
                rows.append(r)
    d = pd.DataFrame(rows)
    add_normalised_shape(d, SITE)
    cols = ["window_id", "subject_id", "record_id", "session", "activity", "t_start_s", "sbp", "dbp", "fit_window"]
    return d[cols + candidate_features() + ["synthetic"]]
