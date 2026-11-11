"""Cross-day distribution shift diagnostics and leakage-safe feature normalisation (RakshPulse 4.0).

Diagnostics (per feature, using resting windows, activity 0):
* ``cross_shift_z``: |median day 2 − median day 1| / within-day robust SD, per subject;
* ``within_shift_z``: the same between the first and second half of day-1 rest (the "no new session" null);
* ``shift_excess``: cross − within, tested over subjects with a sign-flip test;
* ``ks_pooled``: two-sample KS statistic, day-1 vs day-2 rest pooled over subjects (descriptive only: subjects
  are mixed);
* ``slope_agreement``: whether the across-subject relation between the *day-to-day change* in a feature and the
  day-to-day change in resting SBP has the same sign as the pooled within-day relation. If within-day feature–BP
  coupling held across days, the two would agree.

Normalisations (``normalise``) map one subject's features to ``z`` before the change model ``Δz → ΔBP``:

====  ==========================  =========================================================================
A     ``raw``                     no normalisation (3.0 behaviour)
B     ``session``                 subtract the session's own warm-up median (first ``warmup_s`` of PPG)
C     ``subject``                 z-score with the subject's day-1 mean/SD (all accepted day-1 windows)
D     ``robust``                  median/MAD of day 1, winsorised at ±4
E     ``adaptive``                subtract a causal EWMA baseline (half-life ``half_life_s``) running across days
F     ``domain``                  moment-match day 2 to day 1: (x − m₂)/s₂·s₁ + m₁, with m₂, s₂ from the day-2 warm-up
====  ==========================  =========================================================================

Leakage rules: only that subject's **unlabelled** PPG that is available at prediction time is used (day 1, plus
the first ``warmup_s`` of day 2 for B/E/F). Evaluation windows start after the warm-up for **every** method.
No BP label is used except the day-1 reference reading itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NORMALISATIONS = ("raw", "session", "subject", "robust", "adaptive", "domain")
LETTER = dict(zip(NORMALISATIONS, "ABCDEF", strict=True))


def _rsd(x: np.ndarray) -> np.ndarray:
    """Robust SD (1.4826 × MAD) per column, never 0."""
    med = np.nanmedian(x, 0)
    s = 1.4826 * np.nanmedian(np.abs(x - med), 0)
    fallback = np.nanstd(x, 0)
    s = np.where(s > 1e-12, s, fallback)
    return np.where(s > 1e-12, s, 1.0)


def shift_diagnostics(d: pd.DataFrame, cols: list[str], target: str = "sbp_mmHg") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature and per-subject cross-day shift tables. ``d``: SQI-accepted windows with session, activity."""
    rest = d[d.activity == 0]
    feat_rows, subj = [], {}
    per_feat: dict[str, list] = {c: [] for c in cols}
    for sid, g in rest.groupby("subject_id"):
        g1 = g[g.session == 1].sort_values("t_start_s")
        g2 = g[g.session == 2]
        if len(g1) < 12 or len(g2) < 6:
            continue
        x1, x2 = g1[cols].to_numpy(float), g2[cols].to_numpy(float)
        s = _rsd(x1)
        h = len(g1) // 2
        cross = np.abs(np.nanmedian(x2, 0) - np.nanmedian(x1, 0)) / s
        within = np.abs(np.nanmedian(x1[h:], 0) - np.nanmedian(x1[:h], 0)) / s
        dfeat = np.nanmedian(x2, 0) - np.nanmedian(x1, 0)
        dy = float(g2[target].median() - g1[target].median()) if target in g else np.nan
        for i, c in enumerate(cols):
            per_feat[c].append((cross[i], within[i], dfeat[i] / s[i], dy))
        subj[sid] = {
            "subject_id": sid,
            "mean_cross_shift_z": float(np.nanmean(cross)),
            "mean_within_shift_z": float(np.nanmean(within)),
            "rest_sbp_change": dy,
            "n_rest_day1": len(g1),
            "n_rest_day2": len(g2),
        }
    from scipy import stats

    for c in cols:
        a = np.array(per_feat[c], float)
        if a.size == 0:
            continue
        excess = a[:, 0] - a[:, 1]
        excess = excess[np.isfinite(excess)]
        rng = np.random.default_rng(0)
        null = (rng.choice((-1.0, 1.0), (5000, excess.size)) * excess).mean(1) if excess.size else np.array([np.nan])
        p = float((np.sum(np.abs(null) >= abs(excess.mean())) + 1) / (null.size + 1)) if excess.size else np.nan
        v1 = rest.loc[rest.session == 1, c].dropna()
        v2 = rest.loc[rest.session == 2, c].dropna()
        ok = np.isfinite(a[:, 2]) & np.isfinite(a[:, 3])
        r_between = float(stats.spearmanr(a[ok, 2], a[ok, 3])[0]) if ok.sum() >= 8 else np.nan
        feat_rows.append(
            {
                "feature": c,
                "cross_shift_z_median": float(np.nanmedian(a[:, 0])),
                "within_shift_z_median": float(np.nanmedian(a[:, 1])),
                "shift_excess_mean": float(excess.mean()) if excess.size else np.nan,
                "shift_excess_p": p,
                "ks_pooled": float(stats.ks_2samp(v1, v2).statistic) if len(v1) and len(v2) else np.nan,
                "between_day_spearman_dfeature_dsbp": r_between,
                "n_subjects": int(a.shape[0]),
            }
        )
    return pd.DataFrame(feat_rows), pd.DataFrame(list(subj.values()))


def _ewma_baseline(t: np.ndarray, x: np.ndarray, half_life_s: float) -> np.ndarray:
    """Causal EWMA in *time* (irregular sampling): baseline before each sample, starting from the first one."""
    b = np.empty_like(x)
    cur = np.where(np.isfinite(x[0]), x[0], 0.0)
    last_t = t[0]
    for i in range(len(t)):
        b[i] = cur
        a = 1.0 - 0.5 ** ((t[i] - last_t) / half_life_s) if i else 0.0
        a = max(a, 0.05)
        xi = np.where(np.isfinite(x[i]), x[i], cur)
        cur = (1 - a) * cur + a * xi
        last_t = t[i]
    return b


def normalise(
    method: str,
    day1: pd.DataFrame,
    day2: pd.DataFrame,
    ref: pd.DataFrame,
    cols: list[str],
    warmup_s: float = 180.0,
    half_life_s: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_ref (n_features,), z_day2 (n_day2, n_features)) for one subject.

    ``day1``/``day2``: that subject's SQI-accepted windows (labels not used), time-sorted; ``ref``: the day-1
    reference windows. ``day2`` must contain the warm-up windows (the first ``warmup_s``); the caller evaluates
    only windows after the warm-up.
    """
    x1 = day1[cols].to_numpy(float)
    x2 = day2[cols].to_numpy(float)
    xr = ref[cols].to_numpy(float).mean(0)
    t2 = day2.t_start_s.to_numpy(float)
    warm2 = x2[t2 < t2.min() + warmup_s]
    t1 = day1.t_start_s.to_numpy(float)
    warm1 = x1[t1 < t1.min() + warmup_s]
    if method == "raw":
        return xr, x2
    if method == "session":
        return xr - np.nanmedian(warm1, 0), x2 - np.nanmedian(warm2, 0)
    if method == "subject":
        mu, sd = np.nanmean(x1, 0), np.nanstd(x1, 0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (xr - mu) / sd, (x2 - mu) / sd
    if method == "robust":
        med, s = np.nanmedian(x1, 0), _rsd(x1)
        return np.clip((xr - med) / s, -4, 4), np.clip((x2 - med) / s, -4, 4)
    if method == "adaptive":
        t_all = np.r_[t1, t2 + 1e7]  # day 2 follows day 1; the long gap makes the EWMA forget day 1
        x_all = np.vstack([x1, x2])
        b = _ewma_baseline(t_all, x_all, half_life_s)
        b1, b2 = b[: len(t1)], b[len(t1) :]
        tr = ref.t_start_s.to_numpy(float)
        br = b1[np.searchsorted(t1, tr, side="left").clip(0, len(t1) - 1)].mean(0)
        # the day-2 baseline needs warm-up data: windows before the warm-up end are never evaluated
        return xr - br, x2 - b2
    if method == "domain":
        m1, s1 = np.nanmedian(x1, 0), _rsd(x1)
        m2, s2 = np.nanmedian(warm2, 0), _rsd(warm2)
        return xr, (x2 - m2) / s2 * s1 + m1
    raise KeyError(method)
