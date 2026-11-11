"""Hierarchical Bayesian calibration with uncertainty (RakshPulse 4.0).

For one person, the residual of a base prediction ``f`` (a PPG population model, or the training-mean BP for
the matched **no-PPG** control) at time *t* in session *j* is modelled as::

    r(t) = y(t) − f(t) = b + c_j + d_j(t)

* ``b ~ N(0, σ_b²)``: the person's stable offset (across days);
* ``c_j ~ N(0, σ_c²)``: the session (day) offset: sensor placement, physiology of the day;
* ``d_j(t)``: within-session deviation. A reading taken at time *s* has error variance ``σ_r²``. A prediction
  Δt seconds after the most recent reading of the same session has deviation variance ``γ(Δt) = min(σ_r² + κ·Δt,
  σ_w²)`` (a capped linear variogram). Without a reading in the session, the stationary ``σ_w²`` applies.

Hyper-parameters are estimated by method of moments on **training subjects only** (:meth:`fit`). Given any
reference readings (earlier sessions and/or the current one), the posterior of ``μ = b + c_current`` is exact
Gaussian conditioning. The predictive distribution is ``f(t) + E[μ]`` with variance ``Var(μ | readings) + γ``.
Readings from earlier days inform ``b`` only, so cross-day transfer is shrunk automatically by
``σ_b² / (σ_b² + σ_c²)``.

``conformal_scale`` (optional) multiplies the predictive SD by a factor estimated on training subjects so that
the nominal coverage is reached (normalised split conformal). Interval widths therefore stay adaptive to Δt and
to the number of readings, while coverage is calibrated empirically rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class HierParams:
    sigma_b2: float
    sigma_c2: float
    sigma_r2: float
    sigma_w2: float
    kappa: float  # variance growth per second since the last reading (mmHg² / s)

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def fit_params(res: pd.DataFrame, max_lag_s: float = 1800.0) -> HierParams:
    """Method-of-moments estimates from training residuals.

    ``res``: columns subject_id, session, t_start_s, r (= y − f). Uses within-session variograms for σ_r², κ
    and σ_w², session-mean differences within subjects for σ_c², and subject means for σ_b².
    """
    lags, half_sq = [], []
    within_var, sess_means = [], {}
    for (sid, sess), g in res.groupby(["subject_id", "session"]):
        g = g.sort_values("t_start_s")
        t, r = g.t_start_s.to_numpy(float), g.r.to_numpy(float)
        if len(r) < 6:
            continue
        within_var.append(r.var(ddof=1))
        sess_means[(sid, sess)] = (r.mean(), len(r))
        # sparse pairs for the variogram (every window vs a few later ones)
        for step in (1, 3, 6, 12, 24, 48, 96):
            if step >= len(r):
                break
            dt = t[step:] - t[:-step]
            ok = dt <= max_lag_s
            lags.append(dt[ok])
            half_sq.append(0.5 * (r[step:] - r[:-step])[ok] ** 2)
    sigma_w2 = float(np.mean(within_var)) if within_var else 1.0
    lag, hs = np.concatenate(lags), np.concatenate(half_sq)
    bins = np.array([0, 15, 30, 60, 120, 240, 480, 900, max_lag_s])
    centers, vals = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (lag >= lo) & (lag < hi)
        if m.sum() >= 20:
            centers.append(lag[m].mean())
            vals.append(hs[m].mean())
    centers, vals = np.array(centers), np.array(vals)
    if centers.size >= 2:
        kappa, intercept = np.polyfit(centers, vals, 1)
        kappa, sigma_r2 = max(float(kappa), 0.0), float(np.clip(intercept, 1.0, sigma_w2))
    else:
        kappa, sigma_r2 = 0.0, sigma_w2
    # session offsets: subjects with two sessions
    diffs, subj_means = [], []
    by_subj: dict = {}
    for (sid, _sess), (m, n) in sess_means.items():
        by_subj.setdefault(sid, []).append((m, n))
    for v in by_subj.values():
        subj_means.append(np.mean([m for m, _ in v]))
        if len(v) >= 2:
            (m1, n1), (m2, n2) = v[:2]
            diffs.append(0.5 * (m1 - m2) ** 2 - 0.5 * sigma_w2 * (1 / n1 + 1 / n2))
    sigma_c2 = float(max(np.mean(diffs), 0.0)) if diffs else 0.0
    sigma_b2 = float(max(np.var(subj_means, ddof=1) - sigma_c2 / 2, 0.0)) if len(subj_means) > 1 else 0.0
    return HierParams(sigma_b2, sigma_c2, sigma_r2, sigma_w2, kappa)


class HierarchicalCalibrator:
    def __init__(self, params: HierParams, conformal_scale: float = 1.0):
        self.p = params
        self.scale = conformal_scale

    def posterior(self, obs_r: np.ndarray, obs_current: np.ndarray) -> tuple[float, float]:
        """Posterior mean and variance of μ = b + c_current given reading residuals.

        ``obs_r``: residuals of the readings; ``obs_current``: bool, the reading is from the current session.
        Readings from earlier sessions are treated as coming from *one* other session per distinct day only
        if the caller passes them as such; here all non-current readings share one earlier-session offset.
        """
        p = self.p
        prior = p.sigma_b2 + p.sigma_c2
        r = np.asarray(obs_r, float)
        if r.size == 0:
            return 0.0, prior
        cur = np.asarray(obs_current, bool)
        same = cur[:, None] == cur[None, :]
        K = p.sigma_b2 + p.sigma_c2 * same + p.sigma_r2 * np.eye(r.size)
        k = p.sigma_b2 + p.sigma_c2 * cur
        w = np.linalg.solve(K, k)
        return float(w @ r), float(max(prior - k @ w, 1e-9))

    def predict(
        self,
        f: np.ndarray,
        dt_since_reading: np.ndarray,
        obs_r: np.ndarray,
        obs_current: np.ndarray,
        level: float = 0.90,
    ) -> pd.DataFrame:
        """Point prediction and ``level`` interval for windows with base predictions ``f``.

        ``dt_since_reading``: seconds since the most recent *current-session* reading (NaN if none).
        """
        mu, v = self.posterior(obs_r, obs_current)
        dt = np.asarray(dt_since_reading, float)
        gamma = np.where(
            np.isfinite(dt), np.minimum(self.p.sigma_r2 + self.p.kappa * dt, self.p.sigma_w2), self.p.sigma_w2
        )
        sd = np.sqrt(v + gamma) * self.scale
        z = stats.norm.ppf(0.5 + level / 2)
        yhat = np.asarray(f, float) + mu
        return pd.DataFrame({"yhat": yhat, "sd": sd, "lo": yhat - z * sd, "hi": yhat + z * sd})


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float, subject: np.ndarray) -> dict:
    """Coverage, width, Winkler interval score and per-subject coverage spread (subject-weighted)."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    alpha = 1 - level
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width + (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    df = pd.DataFrame({"s": subject, "inside": inside, "width": width, "score": score})
    per = df.groupby("s").agg(cov=("inside", "mean"), width=("width", "mean"), score=("score", "mean"))
    return {
        "coverage": float(per["cov"].mean()),
        "mean_width": float(per.width.mean()),
        "winkler": float(per.score.mean()),
        "subject_coverage_p10": float(per["cov"].quantile(0.10)),
        "frac_subjects_cov_ge_80pct_of_nominal": float((per["cov"] >= 0.8 * level).mean()),
        "n_subjects": int(per.shape[0]),
    }
