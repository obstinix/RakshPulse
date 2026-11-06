"""Generate exploratory PPG figures into docs/figures/.

* example raw vs processed red/IR windows (good and rejected);
* ensemble-average beats by dataset and by SBP tertile (Guilin);
* SQI pass rates at rest vs post-exercise states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.data.dataset_loader import iter_foggia, iter_guilin, iter_ppg_bp  # noqa: E402
from rakshpulse.evaluation import plots  # noqa: E402
from rakshpulse.features.feature_pipeline import extract_features  # noqa: E402

FIG = ROOT / "docs" / "figures"


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(PROCESSED / "guilin_dualwave_features.csv")

    # 1. example windows: one accepted, one rejected
    ok_row = feats[feats.ir_sqi_ok & feats.red_sqi_ok].iloc[0]
    bad_row = feats[~feats.ir_sqi_ok].iloc[0]
    wanted = {ok_row.record_id: ("accepted", ok_row.t_start_s), bad_row.record_id: ("rejected", bad_row.t_start_s)}
    for rec in iter_guilin():
        if rec.record_id in wanted:
            tag, t0 = wanted.pop(rec.record_id)
            a = int(t0 * rec.fs)
            b = a + int(10 * rec.fs)
            raw = {c: rec.channels[c][a:b] for c in ("ir", "red")}
            _, proc = extract_features(raw, rec.fs)
            plots.plot_window(
                raw, rec.fs, proc, FIG / f"guilin_window_{tag}.png", f"Guilin {rec.record_id} t={t0:.0f}s ({tag})"
            )
        if not wanted:
            break

    # 2. mean beats by dataset (IR channel)
    groups: dict[str, list] = {
        "Guilin (finger?, 18–36 y)": [],
        "Foggia (fingertip, 18–86 y)": [],
        "PPG-BP (fingertip, 20–89 y)": [],
    }
    sbp_groups: dict[str, list] = {}
    rng = np.random.default_rng(0)
    g_recs = list(iter_guilin())
    sbps = np.array([r.sbp for r in g_recs])
    t1, t2 = np.quantile(sbps, [1 / 3, 2 / 3])
    for r in g_recs:
        a = int(20 * r.fs)
        win = {c: r.channels[c][a : a + int(10 * r.fs)] for c in ("ir", "red")}
        _, proc = extract_features(win, r.fs)
        groups["Guilin (finger?, 18–36 y)"].append(proc["ir"].beats)
        key = f"SBP < {t1:.0f}" if r.sbp < t1 else (f"SBP ≥ {t2:.0f}" if r.sbp >= t2 else f"SBP {t1:.0f}–{t2:.0f}")
        sbp_groups.setdefault(key, []).append(proc["ir"].beats)
    for r in iter_foggia():
        _, proc = extract_features({"ir": r.channels["ir"]}, r.fs, has_dc=False)
        groups["Foggia (fingertip, 18–86 y)"].append(proc["ir"].beats)
    for r in iter_ppg_bp():
        if rng.random() < 0.5:
            _, proc = extract_features({"ir": r.channels["ir"]}, r.fs, has_dc=False)
            groups["PPG-BP (fingertip, 20–89 y)"].append(proc["ir"].beats)
    plots.plot_mean_beats(groups, FIG / "mean_beat_by_dataset.png", "IR ensemble beat by dataset (mean ± SD)")
    order = sorted(sbp_groups, key=lambda k: ("<" not in k, "≥" in k))
    plots.plot_mean_beats(
        {k: sbp_groups[k] for k in order},
        FIG / "mean_beat_by_sbp_tertile.png",
        "Guilin IR ensemble beat by SBP tertile",
    )

    # 3. SQI by state
    un = pd.read_csv(PROCESSED / "guilin_unlabelled_features.csv")
    allw = pd.concat([feats, un])
    tab = allw.groupby("state")[["ir_sqi_ok", "red_sqi_ok"]].mean().round(3)
    tab.to_csv(FIG / "sqi_by_state.csv")
    print(tab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
