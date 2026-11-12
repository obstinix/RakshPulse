"""Modular calibration strategies, policy and leakage-safe evaluation (RakshPulse 5.0).

Strategies share one interface: ``fit(train)`` on training subjects only, then ``predict(readings, windows)``
for one subject, where ``readings`` are that subject's calibration readings (cuff values at known times) and
``windows`` are the windows to estimate. Frames use the columns ``subject_id, session_id, t, y`` (truth; never
used by ``predict``), ``pred`` (PPG model output) and ``pmean`` (training-mean BP).

====================  ===================================================================================
``population``        training-mean BP. No PPG, no cuff (the floor every method must beat)
``ppg_only``          the PPG population model, uncalibrated
``cuff_last``         last cuff reading carried forward (no PPG)
``cuff_mean``         mean of all cuff readings so far (no PPG)
``cuff_hierarchical`` hierarchical Bayesian shrinkage of the cuff readings towards the population (no PPG;
                      the matched control for ``combined``)
``combined``          the same hierarchical model on the PPG prediction (cuff + PPG)
``subject_linear``    subject-specific ``y = a + b·pred`` with the slope shrunk towards 1 (≥ 2 readings)
====================  ===================================================================================

:class:`CalibrationPolicy` decides when a calibration is valid. It needs at least ``min_readings`` readings
``min_spacing_s`` apart, and the calibration **expires** when the last reading is older than ``expiry_s``, when
the predictive 90 % half-width exceeds ``max_halfwidth_mmhg``, or (on reassessment) when a new reading's
residual exceeds ``max_residual_mmhg``. Readings from an earlier session never count as a current-session
calibration unless ``allow_cross_session`` is set.

:func:`split_readings` enforces the leakage rule: **windows that overlap a calibration reading (± ``exclude_s``)
are never evaluated**, and evaluation windows come strictly after the last reading of their session.
:func:`evaluate` reports calibration fit (error *at* the readings, in-sample) and independent prediction error
(held-out windows) in separate rows, plus calibration-retention curves by time since the last reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .hierarchical import HierarchicalCalibrator, fit_params

Z90 = 1.6448536269514722


@dataclass
class CalibrationPolicy:
    min_readings: int = 1
    min_spacing_s: float = 60.0
    exclude_s: float = 10.0
    expiry_s: float = 3600.0
    max_halfwidth_mmhg: float | None = None
    max_residual_mmhg: float = 15.0
    allow_cross_session: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Strategy:
    name = "base"
    uses_ppg = False
    uses_cuff = False

    def fit(self, train: pd.DataFrame) -> Strategy:
        self.train_sd = float((train.y - train.pmean).std())
        return self

    def predict(self, readings: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Population(Strategy):
    name = "population"

    def predict(self, readings, windows):
        return pd.DataFrame({"yhat": windows.pmean.to_numpy(), "sd": self.train_sd}, index=windows.index)


class PPGOnly(Strategy):
    name, uses_ppg = "ppg_only", True

    def fit(self, train):
        self.sd = float((train.y - train.pred).std())
        return self

    def predict(self, readings, windows):
        return pd.DataFrame({"yhat": windows.pred.to_numpy(), "sd": self.sd}, index=windows.index)


class CuffLast(Strategy):
    name, uses_cuff = "cuff_last", True

    def predict(self, readings, windows):
        if readings.empty:  # nothing to carry forward: fall back to the population baseline
            return pd.DataFrame({"yhat": windows.pmean.to_numpy(), "sd": np.nan}, index=windows.index)
        last = float(readings.sort_values("t").y.iloc[-1])
        return pd.DataFrame({"yhat": last, "sd": np.nan}, index=windows.index)


class CuffMean(Strategy):
    name, uses_cuff = "cuff_mean", True

    def predict(self, readings, windows):
        if readings.empty:
            return pd.DataFrame({"yhat": windows.pmean.to_numpy(), "sd": np.nan}, index=windows.index)
        return pd.DataFrame({"yhat": float(readings.y.mean()), "sd": np.nan}, index=windows.index)


class _Hier(Strategy):
    base = "pmean"

    def fit(self, train):
        res = train.assign(r=train.y - train[self.base], t_start_s=train.t, session=train.session_id)
        self.cal = HierarchicalCalibrator(fit_params(res))
        return self

    def predict(self, readings, windows):
        cur_session = windows.session_id.iloc[0] if len(windows) else None
        cur = (readings.session_id == cur_session).to_numpy() if len(readings) else np.array([], bool)
        last_cur = readings.t[cur].max() if cur.any() else np.nan
        dt = windows.t.to_numpy() - last_cur if np.isfinite(last_cur) else np.full(len(windows), np.nan)
        p = self.cal.predict(windows[self.base].to_numpy(), dt, (readings.y - readings[self.base]).to_numpy(), cur)
        return pd.DataFrame({"yhat": p.yhat.to_numpy(), "sd": p.sd.to_numpy()}, index=windows.index)


class CuffHierarchical(_Hier):
    name, uses_cuff, base = "cuff_hierarchical", True, "pmean"


class Combined(_Hier):
    name, uses_cuff, uses_ppg, base = "combined", True, True, "pred"


class SubjectLinear(Strategy):
    """y = mean(readings) + b·(pred − mean(pred at readings)), with b shrunk towards 1 (ridge-to-prior)."""

    name, uses_cuff, uses_ppg = "subject_linear", True, True
    lam_per_reading = 25.0

    def predict(self, readings, windows):
        if len(readings) < 2:
            off = float((readings.y - readings.pred).mean()) if len(readings) else 0.0
            return pd.DataFrame({"yhat": windows.pred.to_numpy() + off, "sd": np.nan}, index=windows.index)
        dp = readings.pred - readings.pred.mean()
        dy = readings.y - readings.y.mean()
        lam = self.lam_per_reading * len(readings)
        b = float((dp @ dy + lam) / (dp @ dp + lam))
        yhat = readings.y.mean() + b * (windows.pred.to_numpy() - readings.pred.mean())
        return pd.DataFrame({"yhat": yhat, "sd": np.nan}, index=windows.index)


STRATEGIES = {c.name: c for c in (Population, PPGOnly, CuffLast, CuffMean, CuffHierarchical, Combined, SubjectLinear)}


# ------------------------------------------------------------------------------------------------ policy
def pick_readings(g: pd.DataFrame, k: int, policy: CalibrationPolicy) -> pd.DataFrame:
    """First ``k`` windows of ``g`` (time-sorted) at least ``min_spacing_s`` apart."""
    out, last = [], -np.inf
    for i, t in zip(g.index, g.t, strict=True):
        if len(out) == k:
            break
        if t - last >= policy.min_spacing_s:
            out.append(i)
            last = t
    return g.loc[out]


def split_readings(g: pd.DataFrame, readings: pd.DataFrame, policy: CalibrationPolicy) -> pd.DataFrame:
    """Evaluation windows for one subject: never overlapping a reading (± exclude_s), and after the last reading
    of their own session. Raises if a reading window would be evaluated (defensive leakage check)."""
    ev = g.drop(index=readings.index, errors="ignore")
    for _, r in readings.iterrows():
        same = ev.session_id == r.session_id
        ev = ev[~(same & ((ev.t - r.t).abs() <= policy.exclude_s))]
    last = readings.groupby("session_id").t.max()
    after = ev.session_id.map(last)
    ev = ev[after.isna() | (ev.t > after)]
    if set(ev.index) & set(readings.index):
        raise AssertionError("leakage: a calibration reading is in the evaluation set")
    return ev


def calibration_status(
    readings: pd.DataFrame, t: float, session_id, policy: CalibrationPolicy, halfwidth: float | None = None
) -> tuple[bool, str]:
    """(valid, reason) of the calibration at time ``t`` in ``session_id``."""
    r = readings if policy.allow_cross_session else readings[readings.session_id == session_id]
    if len(r) < policy.min_readings:
        return False, "too_few_readings_in_session" if len(readings) else "no_readings"
    age = t - r.t.max()
    if age > policy.expiry_s:
        return False, "expired_age"
    if policy.max_halfwidth_mmhg is not None and halfwidth is not None and halfwidth > policy.max_halfwidth_mmhg:
        return False, "expired_uncertainty"
    return True, "valid"


def evaluate(
    frame: pd.DataFrame,
    folds: dict,
    strategies: list[str],
    policy: CalibrationPolicy,
    scenarios: dict[str, dict],
    retention_bins_s: tuple[float, ...] = (0, 300, 600, 1200, 2400, np.inf),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-window predictions, calibration-fit rows). ``scenarios``: name → {"k": readings,
    "reading_session": 1, "eval_sessions": (1,)}; readings come from the first windows of ``reading_session``."""
    preds, fit_rows = [], []
    for k_fold in sorted(set(folds.values())):
        test = {s for s, f in folds.items() if f == k_fold}
        train = frame[~frame.subject_id.isin(test)]
        fitted = {n: STRATEGIES[n]().fit(train) for n in strategies}
        for sid, g in frame[frame.subject_id.isin(test)].groupby("subject_id"):
            g = g.sort_values(["session_id", "t"])
            for scen, spec in scenarios.items():
                src = g[g.session_id == spec["reading_session"]]
                readings = pick_readings(src, spec["k"], policy)
                if len(readings) < spec["k"]:
                    continue
                ev = split_readings(g[g.session_id.isin(spec["eval_sessions"])], readings, policy)
                if ev.empty:
                    continue
                for name, strat in fitted.items():
                    for sess, e in ev.groupby("session_id"):
                        p = strat.predict(readings, e)
                        last = (
                            readings[readings.session_id == sess].t.max()
                            if (readings.session_id == sess).any()
                            else np.nan
                        )
                        valid = [
                            calibration_status(readings, t, sess, policy, Z90 * s if np.isfinite(s) else None)
                            for t, s in zip(e.t, p.sd, strict=True)
                        ]
                        preds.append(
                            pd.DataFrame(
                                {
                                    "scenario": scen,
                                    "strategy": name,
                                    "subject_id": sid,
                                    "session_id": sess,
                                    "t": e.t.to_numpy(),
                                    "y": e.y.to_numpy(),
                                    "yhat": p.yhat.to_numpy(),
                                    "sd": p.sd.to_numpy(),
                                    "since_reading_s": e.t.to_numpy() - last,
                                    "calibration_valid": [v for v, _ in valid],
                                    "calibration_reason": [r for _, r in valid],
                                }
                            )
                        )
                    pr = strat.predict(readings, readings.assign(session_id=readings.session_id))
                    fit_rows.append(
                        {
                            "scenario": scen,
                            "strategy": name,
                            "subject_id": sid,
                            "fit_mae_at_readings": float(np.mean(np.abs(pr.yhat - readings.y))),
                        }
                    )
    if not preds:
        raise ValueError("no subject had enough calibration readings and later windows for any scenario")
    P = pd.concat(preds, ignore_index=True)
    P["retention_bin_s"] = pd.cut(P.since_reading_s, list(retention_bins_s), right=False).astype(str)
    return P, pd.DataFrame(fit_rows)
