"""Paired subject-level comparisons with bootstrap confidence intervals.

For two configurations evaluated on the *same* subjects and folds, resample subjects with
replacement (10,000×) and report the 95 % CI of the MAE difference. A CI that excludes 0
means the improvement is unlikely to be fold noise. Writes results/paired_comparisons.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation.validation import subject_level  # noqa: E402

P = ROOT / "results" / "predictions"

COMPARISONS = [
    # (label, candidate, reference)
    ("PPG-only best vs mean", "A_ppg_ir_red_xgboost", "A_ppg_ir_red_mean"),
    ("PPG-only ridge vs mean", "A_ppg_ir_red_ridge", "A_ppg_ir_red_mean"),
    ("Demographics ridge vs mean", "A_demographics_ridge", "A_demographics_mean"),
    ("PPG+demo ridge vs demographics ridge", "A_ppg_ir_red+demo_ridge", "A_demographics_ridge"),
    ("PPG+demo SVR vs demographics ridge", "A_ppg_ir_red+demo_svr", "A_demographics_ridge"),
    ("CNN red+IR vs mean", "A_cnn_ir_red", "A_ppg_ir_red_mean"),
]


def load(stem: str, target: str) -> pd.DataFrame:
    return subject_level(pd.read_csv(P / f"{stem}_{target}.csv"), target).set_index("subject_id")


def main() -> int:
    rng = np.random.default_rng(0)
    out = []
    for label, cand, ref in COMPARISONS:
        for target in ("sbp", "dbp"):
            if not (P / f"{cand}_{target}.csv").exists():
                continue
            a, b = load(cand, target), load(ref, target)
            common = a.index.intersection(b.index)
            ea = (a.loc[common, "pred"] - a.loc[common, "y"]).abs().to_numpy()
            eb = (b.loc[common, "pred"] - b.loc[common, "y"]).abs().to_numpy()
            d = ea - eb
            boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(10_000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row = {
                "comparison": label,
                "target": target,
                "n_subjects": int(d.size),
                "mae_candidate": float(ea.mean()),
                "mae_reference": float(eb.mean()),
                "mae_diff": float(d.mean()),
                "ci95": [float(lo), float(hi)],
                "significant": bool(hi < 0 or lo > 0),
            }
            out.append(row)
            print(
                f"{label:40s} {target}: {ea.mean():.2f} vs {eb.mean():.2f}  Δ {d.mean():+.2f} "
                f"[{lo:+.2f}, {hi:+.2f}] {'*' if row['significant'] else ''}"
            )
    (ROOT / "results" / "paired_comparisons.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
