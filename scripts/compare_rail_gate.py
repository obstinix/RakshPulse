"""Quantify the effect of the amplifier rail-clipping gate on the cross-day results.

Needs both:  results/cross_day/ (gate on, default run) and results/cross_day_no_rail_gate/ (ablation run,
``run_cross_day.py --models ridge --no-rail-gate --tag _no_rail_gate``). Compares ridge configurations only.
Writes results/cross_day/rail_gate_comparison.{csv,json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.data.build_auricle import MAX_RAIL_FRACTION  # noqa: E402

ON, OFF = ROOT / "results" / "cross_day", ROOT / "results" / "cross_day_no_rail_gate"
KEYS = ["site", "target", "feature_set", "model", "scope", "mode"]


def main() -> int:
    a = pd.read_csv(ON / "metrics.csv")
    b = pd.read_csv(OFF / "metrics.csv")
    a = a[a.model.isin(["ridge", "personal_ridge", "ridge_trained_on_finger"])]
    m = a.merge(b, on=KEYS, suffixes=("_gated", "_ungated"))
    cols = ["mae", "within_subject_pearson_median", "n_windows", "n_subjects"]
    out = m[KEYS + [f"{c}_{s}" for c in cols for s in ("gated", "ungated")]].copy()
    out["mae_change"] = out.mae_gated - out.mae_ungated
    out["windows_removed"] = out.n_windows_ungated - out.n_windows_gated
    out.to_csv(ON / "rail_gate_comparison.csv", index=False)

    f = pd.read_csv(PROCESSED / "auricle_features.csv")
    f = f[f.cohort == "HC"]
    per_subj = (
        f.assign(clipped=f.finger_sat_fraction > MAX_RAIL_FRACTION)
        .groupby("subject_id")
        .clipped.mean()
        .sort_values(ascending=False)
    )
    ps_on = pd.read_csv(ON / "per_subject.csv")
    ps_off = pd.read_csv(OFF / "per_subject.csv")
    sel = dict(site="finger", target="sbp", feature_set="hr_shape", model="ridge", scope="cross_day", mode="offset")
    q_on, q_off = ps_on, ps_off
    for k, v in sel.items():
        q_on, q_off = q_on[q_on[k] == v], q_off[q_off[k] == v]
    j = (
        q_on.set_index("subject_id")
        .mae.to_frame("mae_gated")
        .join(q_off.set_index("subject_id").mae.rename("mae_ungated"))
    )
    j["clipped_fraction"] = per_subj
    j = j.dropna()
    key = out[(out.feature_set == "hr_shape") & (out.model == "ridge") & (out["mode"] == "offset")]
    summary = {
        "finger_windows_clipped_fraction_overall": float((f.finger_sat_fraction > MAX_RAIL_FRACTION).mean()),
        "ear_windows_clipped_fraction_overall": float((f.ear_sat_fraction > MAX_RAIL_FRACTION).mean()),
        "subjects_with_any_clipped_finger_window": int((per_subj > 0).sum()),
        "subjects_with_over_25pct_clipped_finger_windows": int((per_subj > 0.25).sum()),
        "top_clipped_subjects": per_subj.head(5).round(3).to_dict(),
        "hr_shape_ridge_offset": key[
            [
                "site",
                "target",
                "scope",
                "mae_gated",
                "mae_ungated",
                "mae_change",
                "within_subject_pearson_median_gated",
                "within_subject_pearson_median_ungated",
                "windows_removed",
            ]
        ]
        .round(3)
        .to_dict("records"),
        "finger_sbp_cross_day_per_subject_mae_change_vs_clipped_fraction_r": float(
            (j.mae_gated - j.mae_ungated).corr(j.clipped_fraction)
        )
        if len(j) > 3
        else None,
    }
    (ON / "rail_gate_comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
