"""Personal-calibration evaluation and BP-change tracking metrics.

Calibration protocol (one-point offset, as used by commercial cuffless devices):
for each session, the first ``k_cal`` valid **rest** windows serve as the cuff calibration.
Their reference BP mean is the calibration value. The model's mean prediction on the same
windows defines an offset that is applied to every later prediction. Evaluation windows
start ≥ ``gap_s`` after the last calibration window, so overlapping windows cannot leak.

The fair baseline for any calibrated device is **carry-forward**: keep reporting the
calibration value. A calibrated model is only useful if it beats carry-forward, especially
during BP-changing manoeuvres (ESH 2023 cuffless validation recommendations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRESSORS = (1, 2, 3, 4)  # breath-hold, Valsalva, exercise, cold pressor (Auricle codes)


def within_session_r(d: pd.DataFrame, target: str, min_windows: int = 10, min_sd: float = 2.0) -> float:
    rs = []
    for _, g in d.groupby("record_id"):
        if len(g) >= min_windows and g[target].std() > min_sd and g["pred"].std() > 0:
            rs.append(np.corrcoef(g[target], g["pred"])[0, 1])
    return float(np.median(rs)) if rs else float("nan")


def calibrated_errors(d: pd.DataFrame, target: str, k_cal: int = 3, gap_s: float = 10.0) -> dict:
    """``d`` needs columns record_id, t_start_s, activity, <target>, pred."""
    em, ec, sm, sc, dt, dp = [], [], [], [], [], []
    sess_diff = []  # per-session MAE(model) - MAE(carry-forward), for a cluster bootstrap
    n_sessions = 0
    for _, g in d.groupby("record_id"):
        g = g.sort_values("t_start_s")
        rest = g[g["activity"] == 0]
        if len(rest) < k_cal:
            continue
        cal = rest.iloc[:k_cal]
        ev = g[g["t_start_s"] > cal["t_start_s"].max() + gap_s]
        if len(ev) < 5:
            continue
        n_sessions += 1
        ref = cal[target].mean()
        offset = ref - cal["pred"].mean()
        e_model = (ev["pred"] + offset - ev[target]).to_numpy()
        e_carry = (ref - ev[target]).to_numpy()
        st = ev["activity"].isin(STRESSORS).to_numpy()
        sess_diff.append(np.abs(e_model).mean() - np.abs(e_carry).mean())
        em.append(e_model)
        ec.append(e_carry)
        sm.append(e_model[st])
        sc.append(e_carry[st])
        dt.append((ev[target] - ref).to_numpy())
        dp.append((ev["pred"] - cal["pred"].mean()).to_numpy())
    if not n_sessions:
        return {"n_sessions": 0}
    cat = np.concatenate
    em, ec, sm, sc, dt, dp = (cat(x) for x in (em, ec, sm, sc, dt, dp))
    rng = np.random.default_rng(0)
    sd_ = np.asarray(sess_diff)
    boot = [sd_[rng.integers(0, sd_.size, sd_.size)].mean() for _ in range(5000)]
    return {
        "session_mae_diff_mean": float(sd_.mean()),
        "session_mae_diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_sessions": n_sessions,
        "n_windows": int(em.size),
        "calibrated_model_mae": float(np.abs(em).mean()),
        "carry_forward_mae": float(np.abs(ec).mean()),
        "calibrated_model_me": float(em.mean()),
        "calibrated_model_sd": float(em.std(ddof=1)),
        "carry_forward_sd": float(ec.std(ddof=1)),
        "stressor_windows": int(sm.size),
        "stressor_calibrated_model_mae": float(np.abs(sm).mean()) if sm.size else float("nan"),
        "stressor_carry_forward_mae": float(np.abs(sc).mean()) if sc.size else float("nan"),
        "delta_tracking_r": float(np.corrcoef(dt, dp)[0, 1]) if dp.std() > 0 and dt.std() > 0 else float("nan"),
    }


def cross_session_errors(d: pd.DataFrame, target: str, k_cal: int = 3) -> dict:
    """Calibrate on the first ``k_cal`` rest windows of session 1, evaluate on all of session 2
    (recorded on a different day). This measures calibration drift across days.

    ``d`` needs subject_id, session, record_id, t_start_s, activity, <target>, pred.
    """
    em, ec = [], []
    n = 0
    for _, g in d.groupby("subject_id"):
        s1 = g[(g["session"] == 1) & (g["activity"] == 0)].sort_values("t_start_s")
        s2 = g[g["session"] == 2]
        if len(s1) < k_cal or len(s2) < 5:
            continue
        cal = s1.iloc[:k_cal]
        ref = cal[target].mean()
        em.append((s2["pred"] + ref - cal["pred"].mean() - s2[target]).to_numpy())
        ec.append((ref - s2[target]).to_numpy())
        n += 1
    if not n:
        return {"n_subjects": 0}
    em, ec = np.concatenate(em), np.concatenate(ec)
    return {
        "n_subjects": n,
        "n_windows": int(em.size),
        "calibrated_model_mae": float(np.abs(em).mean()),
        "carry_forward_mae": float(np.abs(ec).mean()),
        "calibrated_model_me": float(em.mean()),
        "carry_forward_me": float(ec.mean()),
    }
