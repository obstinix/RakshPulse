"""Summarise processed datasets (subjects, BP distribution, SQI pass rates) into
``data/metadata/dataset_summary.json``. Only aggregate statistics are written, which are safe
to commit, and no individual-level records.

Usage: ``python -m rakshpulse.data.metadata``
"""

from __future__ import annotations

import json

import pandas as pd

from .build import PROCESSED
from .download import DATA_ROOT


def summarise(name: str) -> dict:
    df = pd.read_csv(PROCESSED / f"{name}_features.csv")
    subj = df.groupby("subject_id").first()
    out = {
        "windows": len(df),
        "subjects": len(subj),
        "ir_sqi_pass": round(float(df["ir_sqi_ok"].mean()), 4),
        "red_sqi_pass": round(float(df["red_sqi_ok"].mean()), 4),
    }
    if subj["sbp"].notna().any():
        for k in ("sbp", "dbp", "demo_age"):
            s = subj[k].dropna()
            out[k] = {
                q: round(float(v), 1)
                for q, v in {
                    "min": s.min(),
                    "p25": s.quantile(0.25),
                    "median": s.median(),
                    "p75": s.quantile(0.75),
                    "max": s.max(),
                    "mean": s.mean(),
                    "sd": s.std(),
                }.items()
            }
        out["hypertensive_sbp_ge_140"] = int((subj["sbp"] >= 140).sum())
        out["hypotensive_sbp_lt_90"] = int((subj["sbp"] < 90).sum())
        out["male_fraction"] = round(float(subj["demo_sex"].mean()), 3)
    return out


def main() -> int:
    res = {}
    for f in sorted(PROCESSED.glob("*_features.csv")):
        name = f.name.replace("_features.csv", "")
        res[name] = summarise(name)
    path = DATA_ROOT / "metadata" / "dataset_summary.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
