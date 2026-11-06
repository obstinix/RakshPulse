"""Loaders that turn each extracted dataset into a common ``Record`` structure.

Raw files are only read, never written. Each loader documents exactly which part of the
dataset carries valid BP labels.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .download import DATA_ROOT

EXTRACTED = DATA_ROOT / "extracted"


@dataclass
class Record:
    dataset: str
    subject_id: str
    record_id: str
    fs: float
    channels: dict[str, np.ndarray]  # keys among {"ir", "red", "ecg"}
    sbp: float | None
    dbp: float | None
    has_dc: bool  # True if the optical channels still contain the DC level (raw ADC counts)
    demographics: dict[str, float] = field(default_factory=dict)
    state: str = "rest"


# --------------------------------------------------------------------------------------
# Guilin dual-wavelength (Mendeley pz2zzr8vhm v1)
# --------------------------------------------------------------------------------------
GUILIN_DIR = EXTRACTED / "guilin_dualwave" / "A dataset of simultaneous collected ECG and PPG signals"
GUILIN_FS = 250.0
GUILIN_STATES = {"1": "rest", "2": "post_exercise_a", "3": "post_exercise_b"}


def _sex_code(s: str) -> float:
    s = s.strip().lower()
    return 1.0 if s.startswith("m") else 0.0 if s.startswith("f") else np.nan


def load_guilin_info(root: Path = GUILIN_DIR) -> dict[str, dict]:
    info = {}
    with (root / "information.csv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sid = row["ID"].strip().zfill(3)
            info[sid] = {
                "age": float(row["Age"]),
                "sex": _sex_code(row["Gender"]),
                "height_cm": float(row["Height/cm"]),
                "weight_kg": float(row["Weight/kg"]),
                "sbp": float(row["SBP/mmHg"]),
                "dbp": float(row["DBP/mmHg"]),
                "hr_ref": float(row["HR/bpm"]),
            }
    return info


def read_guilin_raw(path: Path) -> dict[str, np.ndarray]:
    """Read a Raw_data CSV (ECG_I, PPG_RED, PPG_IR, trailing sampling-rate column)."""
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0, 1, 2), dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    return {"ecg": arr[:, 0], "red": arr[:, 1], "ir": arr[:, 2]}


def iter_guilin(states: tuple[str, ...] = ("1",), root: Path = GUILIN_DIR) -> Iterator[Record]:
    """Yield Guilin records.

    Only state "1" (rest) has a cuff BP measured for that recording. States 2 and 3 are
    post-exercise recordings without BP. They get ``sbp=None`` so they can never be used
    as supervised BP targets. The cuff reading is a single value per subject taken at
    rest, so every window from one subject shares the same label.
    """
    info = load_guilin_info(root)
    for sid in sorted(info):
        for st in states:
            p = root / "Raw_data" / f"{sid}_{st}.csv"
            if not p.exists():
                continue
            meta = info[sid]
            labelled = st == "1"
            yield Record(
                dataset="guilin_dualwave",
                subject_id=f"guilin_{sid}",
                record_id=f"{sid}_{st}",
                fs=GUILIN_FS,
                channels=read_guilin_raw(p),
                sbp=meta["sbp"] if labelled else None,
                dbp=meta["dbp"] if labelled else None,
                has_dc=True,
                demographics={k: meta[k] for k in ("age", "sex", "height_cm", "weight_kg")},
                state=GUILIN_STATES[st],
            )


# --------------------------------------------------------------------------------------
# Foggia multiwavelength (Mendeley gs7jvwtcgt v1), MAX86150, fingertip, 100 Hz
# --------------------------------------------------------------------------------------
FOGGIA_DIR = EXTRACTED / "foggia_multiwave" / "PPG_DATASET"


def iter_foggia(root: Path = FOGGIA_DIR) -> Iterator[Record]:
    """Signals are distributed already band-passed (0.5–10 Hz), Hampel-despiked and z-scored,
    so the DC level is lost (``has_dc=False``) and AC/DC ratio features are unavailable."""
    from scipy.io import loadmat

    x = loadmat(root / "input_data.mat")["input_data"]
    y = loadmat(root / "output_data.mat")["output_data"].astype(float)
    ph = loadmat(root / "physiological_data.mat")["physiological_data"].astype(float)
    for i in range(x.shape[0]):
        sig = np.asarray(x[i, 0], dtype=float)
        yield Record(
            dataset="foggia_multiwave",
            subject_id=f"foggia_{i + 1:03d}",
            record_id=f"{i + 1:03d}",
            fs=100.0,
            channels={"ir": sig[:, 0], "red": sig[:, 1]},
            sbp=y[i, 0],
            dbp=y[i, 1],
            has_dc=False,
            demographics={"age": ph[i, 0], "sex": ph[i, 1], "height_cm": np.nan, "weight_kg": np.nan},
        )


# --------------------------------------------------------------------------------------
# PPG-BP (figshare 5459299 v5), fingertip IR only, 1 kHz, 3 x 2.1 s per subject
# --------------------------------------------------------------------------------------
PPGBP_DIR = EXTRACTED / "ppg_bp" / "Data File"


def iter_ppg_bp(root: Path = PPGBP_DIR) -> Iterator[Record]:
    """IR-only. The red channel was not released, so it stays absent and is never synthesised."""
    import pandas as pd

    meta = pd.read_excel(root / "PPG-BP dataset.xlsx", header=1)
    for _, r in meta.iterrows():
        sid = int(r["subject_ID"])
        for k in (1, 2, 3):
            p = root / "0_subject" / f"{sid}_{k}.txt"
            if not p.exists():
                continue
            sig = np.loadtxt(p)
            sig = sig[np.isfinite(sig)]
            yield Record(
                dataset="ppg_bp",
                subject_id=f"ppgbp_{sid}",
                record_id=f"{sid}_{k}",
                fs=1000.0,
                channels={"ir": sig},
                sbp=float(r["Systolic Blood Pressure(mmHg)"]),
                dbp=float(r["Diastolic Blood Pressure(mmHg)"]),
                has_dc=False,  # hardware 0.5–12 Hz band-pass already applied
                demographics={
                    "age": float(r["Age(year)"]),
                    "sex": 1.0 if str(r["Sex(M/F)"]).lower().startswith("m") else 0.0,
                    "height_cm": float(r["Height(cm)"]),
                    "weight_kg": float(r["Weight(kg)"]),
                },
            )


LOADERS = {"guilin_dualwave": iter_guilin, "foggia_multiwave": iter_foggia, "ppg_bp": iter_ppg_bp}
