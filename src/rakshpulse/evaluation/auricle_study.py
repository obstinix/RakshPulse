"""Auricle within-day and cross-day BP-tracking study (reusable core; scripts call into this).

Design (all splits by **subject**; no window of a test subject is ever used for population training):

* **Population model:** fitted on the *session-1* windows of the training subjects of an outer
  subject-grouped fold, with inner GroupKFold tuning on those subjects only. It predicts all windows
  (session 1 and session 2) of the held-out subjects.
* **Calibration modes** for each held-out subject (all calibration data come from session 1):
    - ``carry_forward``: report the mean reference BP of the first K rest windows of session 1 (baseline);
    - ``offset``: population prediction + one-point offset from those K rest windows;
    - ``linear``: per-subject linear map y = a + b·pred fitted on *all* session-1 windows;
    - ``uncalibrated``: raw population prediction.
  Evaluation windows: *within-day* = session-1 windows ≥ ``gap_s`` after the last calibration window;
  *cross-day* = all session-2 windows (a different day). ``linear`` uses all of session 1, so it is only
  evaluated cross-day.
* **Personal models:** a ridge model per subject trained only on that subject's session 1, then
  evaluated on session 2 (no population data at all).
* **Feature sets** separate heart rate from pulse shape (see :data:`SHAPE_FEATURES`).

All statistics use subject-level bootstrap resampling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..models.model_factory import make_model

# Dimensionless / beat-duration-normalised descriptors. These do not change merely because the
# heart beats faster. Raw durations (s), spectral band powers and f0 do, so they are excluded.
SHAPE_BASE = [
    "sys_ratio",
    "w10_sd_ratio",
    "w25_sd_ratio",
    "w50_sd_ratio",
    "w75_sd_ratio",
    "w90_sd_ratio",
    "area_ratio",
    "beat_skew",
    "beat_kurt",
    "apg_b_a",
    "apg_c_a",
    "apg_d_a",
    "apg_e_a",
    "aging_index",
    "win_skew",
    "win_kurt",
    "h2_h1",
    "h3_h1",
    "spec_entropy",
]
SHAPE_NORMALISED = ["w10_n", "w25_n", "w50_n", "w75_n", "w90_n", "t_max_slope_n", "t_apg_b_n", "max_slope_n"]
SHAPE_FEATURES = SHAPE_BASE + SHAPE_NORMALISED


def add_normalised_shape(df: pd.DataFrame, site: str) -> pd.DataFrame:
    """Add beat-duration-normalised timing features ``<site>_w50_n`` etc. (in place, returns df)."""
    T = df[f"{site}_beat_T"]
    for w in ("w10", "w25", "w50", "w75", "w90", "t_max_slope", "t_apg_b"):
        df[f"{site}_{w}_n"] = df[f"{site}_{w}"] / T
    df[f"{site}_max_slope_n"] = df[f"{site}_max_slope"] * T  # slope of a unit-amplitude beat per beat duration
    return df


def feature_set(name: str, site: str, all_site_cols: list[str]) -> list[str]:
    if name == "hr_only":
        return [f"{site}_hr"]
    if name == "shape_only":
        return [f"{site}_{f}" for f in SHAPE_FEATURES]
    if name == "hr_shape":
        return [f"{site}_hr"] + [f"{site}_{f}" for f in SHAPE_FEATURES]
    if name == "ppg_all":
        return all_site_cols
    raise KeyError(name)


# --------------------------------------------------------------------------------------- folds
def subject_folds(subjects: pd.Series, key: pd.Series, n: int = 5) -> dict[str, int]:
    """Deterministic subject→fold map, interleaved by a per-subject key (e.g. mean SBP) for balance."""
    order = pd.DataFrame({"s": subjects, "k": key}).groupby("s")["k"].mean().sort_values()
    return {s: i % n for i, s in enumerate(order.index)}


def fit_population(model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0):
    est, grid = make_model(model, seed)
    if grid and len(np.unique(groups)) >= 3:
        est = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="neg_mean_absolute_error", n_jobs=-1)
        est = est.fit(X, y, groups=groups).best_estimator_
    else:
        est.fit(X, y)
    return est


def population_predictions(
    s1: pd.DataFrame,
    all_windows: pd.DataFrame,
    train_cols: list[str],
    target: str,
    model: str,
    fold_of: dict[str, int],
    test_cols: list[str] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions for ``all_windows`` (training uses only ``s1`` of other-fold subjects)."""
    test_cols = test_cols or train_cols
    pred = np.full(len(all_windows), np.nan)
    f_all = all_windows["subject_id"].map(fold_of).to_numpy()
    f_s1 = s1["subject_id"].map(fold_of).to_numpy()
    for k in sorted(set(fold_of.values())):
        tr = f_s1 != k
        te = f_all == k
        if not te.any():
            continue
        assert not set(s1.loc[tr, "subject_id"]) & set(all_windows.loc[te, "subject_id"]), "subject leakage"
        est = fit_population(
            model,
            s1.loc[tr, train_cols].to_numpy(float),
            s1.loc[tr, target].to_numpy(float),
            s1.loc[tr, "subject_id"].to_numpy(),
            seed,
        )
        pred[te] = est.predict(all_windows.loc[te, test_cols].to_numpy(float))
    return pred


# --------------------------------------------------------------------------------- calibration
@dataclass
class CalibratedOutputs:
    """Per-window long table: subject, scope (within_day / cross_day), mode, y, yhat, activity."""

    table: pd.DataFrame


def calibrate(d: pd.DataFrame, target: str, k_cal: int = 3, gap_s: float = 10.0) -> CalibratedOutputs:
    """``d``: subject_id, session, t_start_s, activity, <target>, pred (population prediction)."""
    out = []
    for sid, g in d.groupby("subject_id"):
        s1 = g[g.session == 1].sort_values("t_start_s")
        s2 = g[g.session == 2]
        rest = s1[s1.activity == 0]
        if len(rest) < k_cal:
            continue
        cal = rest.iloc[:k_cal]
        ref, pref = cal[target].mean(), cal.pred.mean()
        scopes = {"within_day": s1[s1.t_start_s > cal.t_start_s.max() + gap_s], "cross_day": s2}
        lin = None
        if len(s1) >= 10 and s1.pred.std() > 0:
            b, a = np.polyfit(s1.pred, s1[target], 1)
            lin = (a, b)
        for scope, ev in scopes.items():
            if len(ev) < 5:
                continue
            base = {"subject_id": sid, "scope": scope, "y": ev[target].to_numpy(), "activity": ev.activity.to_numpy()}
            modes = {
                "carry_forward": np.full(len(ev), ref),
                "offset": ev.pred.to_numpy() + ref - pref,
                "uncalibrated": ev.pred.to_numpy(),
            }
            if lin is not None and scope == "cross_day":
                modes["linear"] = lin[0] + lin[1] * ev.pred.to_numpy()
            for mode, yhat in modes.items():
                out.append(pd.DataFrame({**base, "mode": mode, "yhat": yhat}))
    return CalibratedOutputs(pd.concat(out, ignore_index=True) if out else pd.DataFrame())


def personal_model_predictions(
    d: pd.DataFrame, cols: list[str], target: str, alpha: float = 10.0, min_windows: int = 20
) -> pd.DataFrame:
    """Per-subject ridge fitted on session 1 only, evaluated on session 2 (cross-day)."""
    rows = []
    for sid, g in d.groupby("subject_id"):
        s1, s2 = g[g.session == 1], g[g.session == 2]
        if len(s1) < min_windows or len(s2) < 5:
            continue
        X1, X2 = s1[cols].to_numpy(float), s2[cols].to_numpy(float)
        med = np.nanmedian(X1, 0)
        med = np.where(np.isfinite(med), med, 0.0)
        X1 = np.where(np.isfinite(X1), X1, med)
        X2 = np.where(np.isfinite(X2), X2, med)
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X1, s1[target].to_numpy(float))
        rows.append(
            pd.DataFrame(
                {
                    "subject_id": sid,
                    "scope": "cross_day",
                    "mode": "personal_s1",
                    "y": s2[target].to_numpy(),
                    "yhat": m.predict(X2),
                    "activity": s2.activity.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------------------------- metrics
def _within_r(g: pd.DataFrame, fn: Callable) -> list[float]:
    rs = []
    for _, s in g.groupby("subject_id"):
        if len(s) >= 10 and s.y.std() > 2 and s.yhat.std() > 1e-9:
            rs.append(fn(s.y, s.yhat))
    return rs


def point_metrics(t: pd.DataFrame) -> dict[str, float]:
    e = t.yhat.to_numpy() - t.y.to_numpy()
    subj_bias = t.assign(e=e).groupby("subject_id").e.mean()
    pr = _within_r(t, lambda a, b: stats.pearsonr(a, b)[0])
    sr = _within_r(t, lambda a, b: stats.spearmanr(a, b)[0])
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(e.mean()),
        "sd_error": float(e.std(ddof=1)),
        "pooled_pearson": float(stats.pearsonr(t.y, t.yhat)[0]) if t.yhat.std() > 1e-9 else float("nan"),
        "pooled_spearman": float(stats.spearmanr(t.y, t.yhat)[0]) if t.yhat.std() > 1e-9 else float("nan"),
        "within_subject_pearson_median": float(np.median(pr)) if pr else float("nan"),
        "within_subject_spearman_median": float(np.median(sr)) if sr else float("nan"),
        "subject_bias_sd": float(subj_bias.std(ddof=1)) if len(subj_bias) > 1 else float("nan"),
        "n_subjects": int(t.subject_id.nunique()),
        "n_windows": int(len(t)),
    }


def subject_bootstrap(
    t: pd.DataFrame, stat: Callable[[pd.DataFrame], float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """(estimate, 2.5 %, 97.5 %) with subjects (not windows) resampled with replacement."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in t.groupby("subject_id")}
    keys = list(groups)
    est = stat(t)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        boots.append(stat(pd.concat([groups[keys[i]] for i in pick], ignore_index=True)))
    boots = np.asarray(boots, float)
    boots = boots[np.isfinite(boots)]
    if not boots.size:
        return float(est), float("nan"), float("nan")
    return float(est), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_subject_diff(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean over subjects of [MAE_a(subject) − MAE_b(subject)], bootstrap CI over subjects."""
    ma = (a.yhat - a.y).abs().groupby(a.subject_id).mean()
    mb = (b.yhat - b.y).abs().groupby(b.subject_id).mean()
    d = (ma - mb).dropna().to_numpy()
    if not d.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def calibration_drift(t: pd.DataFrame) -> dict[str, float]:
    """Cross-day drift = per-subject mean error on day 2 after day-1 calibration (bias distribution)."""
    b = (t.yhat - t.y).groupby(t.subject_id).mean()
    return {
        "drift_mean": float(b.mean()),
        "drift_abs_mean": float(b.abs().mean()),
        "drift_sd": float(b.std(ddof=1)) if len(b) > 1 else float("nan"),
        "drift_p90_abs": float(np.percentile(b.abs(), 90)),
    }
