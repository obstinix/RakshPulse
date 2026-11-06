"""Validate downloaded and extracted datasets.

Checks:
* raw archives exist and match the checksum recorded by the hosting repository;
* expected files and record counts are present after extraction;
* the Guilin sampling-rate header says 250 Hz and every labelled subject has a rest recording;
* labels fall in physiologically plausible ranges.

Usage: ``python -m rakshpulse.data.dataset_validator``
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .dataset_loader import FOGGIA_DIR, GUILIN_DIR, PPGBP_DIR, load_guilin_info
from .download import DATA_ROOT, DATASETS, file_checksum

SBP_RANGE = (60, 260)
DBP_RANGE = (30, 160)


def check_raw(key: str) -> list[str]:
    errs = []
    for rf in DATASETS[key].files:
        p = DATA_ROOT / "raw" / key / rf.filename
        if not p.exists():
            errs.append(f"{key}: missing raw file {rf.filename}")
        elif file_checksum(p, rf.algo) != rf.checksum:
            errs.append(f"{key}: checksum mismatch for {rf.filename}")
    return errs


def check_guilin() -> list[str]:
    errs = []
    if not GUILIN_DIR.exists():
        return ["guilin_dualwave: not extracted"]
    info = load_guilin_info()
    if len(info) != 148:
        errs.append(f"guilin: expected 148 subjects, found {len(info)}")
    for sid, m in info.items():
        p = GUILIN_DIR / "Raw_data" / f"{sid}_1.csv"
        if not p.exists():
            errs.append(f"guilin: missing rest recording {p.name}")
        if not (SBP_RANGE[0] <= m["sbp"] <= SBP_RANGE[1] and DBP_RANGE[0] <= m["dbp"] <= DBP_RANGE[1]):
            errs.append(f"guilin: implausible BP for {sid}: {m['sbp']}/{m['dbp']}")
        if m["dbp"] >= m["sbp"]:
            errs.append(f"guilin: DBP >= SBP for {sid}")
    header = (GUILIN_DIR / "Raw_data" / "001_1.csv").read_text().splitlines()[0]
    if "'Hz': 250" not in header:
        errs.append(f"guilin: unexpected sampling-rate header {header!r}")
    n_files = len(list((GUILIN_DIR / "Raw_data").glob("*.csv")))
    if n_files != 444:
        errs.append(f"guilin: expected 444 raw recordings, found {n_files}")
    return errs


def check_foggia() -> list[str]:
    from scipy.io import loadmat

    if not FOGGIA_DIR.exists():
        return ["foggia_multiwave: not extracted"]
    errs = []
    x = loadmat(FOGGIA_DIR / "input_data.mat")["input_data"]
    y = loadmat(FOGGIA_DIR / "output_data.mat")["output_data"]
    if x.shape[0] != 127 or y.shape != (127, 2):
        errs.append(f"foggia: unexpected shapes {x.shape} {y.shape}")
    if any(np.asarray(c[0]).shape != (1000, 2) for c in x):
        errs.append("foggia: a recording is not 1000x2")
    return errs


def check_ppg_bp() -> list[str]:
    if not PPGBP_DIR.exists():
        return ["ppg_bp: not extracted"]
    n = len(list((PPGBP_DIR / "0_subject").glob("*.txt")))
    return [] if n == 657 else [f"ppg_bp: expected 657 segments, found {n}"]


def main() -> int:
    errs: list[str] = []
    for key in DATASETS:
        errs += check_raw(key)
    errs += check_guilin() + check_foggia() + check_ppg_bp()
    report = {"ok": not errs, "errors": errs}
    (DATA_ROOT / "metadata" / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
