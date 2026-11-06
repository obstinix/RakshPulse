"""Regenerate all cross-day / ear-vs-finger figures from saved results (run after run_cross_day.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation import figures_cross_day as F  # noqa: E402

RES = ROOT / "results" / "cross_day"
PRED = ROOT / "results" / "predictions"
FIG = ROOT / "docs" / "figures" / "cross_day"


def main() -> int:
    m = pd.read_csv(RES / "metrics.csv")
    per = pd.read_csv(RES / "per_subject.csv")
    inv = json.loads((RES / "investigations.json").read_text())
    made = [
        F.actual_vs_predicted(PRED, "sbp", FIG / "fig3_sbp_actual_vs_predicted.png"),
        F.actual_vs_predicted(PRED, "dbp", FIG / "fig4_dbp_actual_vs_predicted.png"),
        F.calibration_modes(m, FIG / "fig5a_calibration_strategies.png"),
        F.drift(PRED, m, FIG / "fig5b_cross_day_drift.png"),
        F.bland_altman(PRED, FIG / "fig6_bland_altman_day2.png"),
        F.per_subject(per, FIG / "fig7_per_subject_mae.png"),
        F.model_comparison(m, FIG / "fig8_model_comparison.png"),
        F.hr_vs_ppg(m, inv, FIG / "fig9_hr_vs_ppg_tracking.png"),
        F.calibration_transfer(m, FIG / "fig10_calibration_transfer.png"),
    ]
    for p in made:
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
