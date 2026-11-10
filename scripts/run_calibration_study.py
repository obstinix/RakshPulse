"""RakshPulse 3.0, Phase 5: how many cuff calibrations are needed, and what do they buy?

Uses the cached RakshPulse 2.0 out-of-fold population predictions (``results/predictions/
personalization_pop_{site}_ridge_{target}.csv.gz``, subject-held-out, trained on session 1 only), so no model is
retrained here and 2.0 results stay untouched.

Calibration points: the first ``k`` accepted, label-valid windows of session 1 that are at least ``CAL_SPACING_S``
apart (as if a cuff were read once a minute at the start of use). k ∈ {0, 1, 2, 3, 5, 10}.

Scenarios:
* ``within_day``: evaluate session-1 windows starting ≥ 10 s after the last calibration point;
* ``cross_day``: evaluate all session-2 windows with the session-1 calibration (no new reading).

Methods (k = calibration count):
* ``population``: uncalibrated 2.0 prediction (k = 0 only);
* ``reference_mean``: mean of the k cuff readings, carried forward (no PPG);
* ``offset``: population + mean residual over the k references (2.0 engine default);
* ``bayes_offset``: offset shrunk towards 0 by k / (k + τ), with τ = σ²_within / σ²_between estimated on the
  other subjects (empirical Bayes, leave-one-subject-out);
* ``linear_free``: per-subject OLS y = a + b·pred on the k references (k ≥ 2);
* ``linear_shrunk``: same, but the slope is shrunk towards 1 (ridge-to-prior, λ = LAMBDA_SLOPE · k);
* ``bayes_reference``: the no-PPG control for ``bayes_offset``. The cuff mean is shrunk towards the out-of-fold
  training-mean BP with the same empirical-Bayes rule (τ from truth instead of residuals). ``bayes_offset`` vs
  ``bayes_reference`` isolates what the waveform adds; the other comparisons also credit plain shrinkage.

Uncertainty: for every method and k, a 90 % interval ± q, with q the 90th percentile of absolute errors of the
**other** subjects (cross-subject split conformal). Coverage and width are reported.
Drift: offset-method error by time since the last calibration (within day) is reported in ``drift.csv``.

Outputs: results/calibration_study/{metrics,paired,drift}.csv and a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation.auricle_study import paired_subject_diff, point_metrics  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402

AUR = ROOT / "data" / "auricle"
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "calibration_study"
KS = (0, 1, 2, 3, 5, 10)
CAL_SPACING_S = 60.0
LAMBDA_SLOPE = 25.0  # mmHg² of prediction spread per calibration point, pre-specified
COVERAGE = 0.90
SEED, N_BOOT = 0, 2000
log = logging.getLogger("calib")


def pick_calibration(g: pd.DataFrame, k: int) -> pd.DataFrame:
    """First k windows of the (time-sorted) session that are ≥ CAL_SPACING_S apart."""
    out, last = [], -np.inf
    for i, t in zip(g.index, g.t_start_s, strict=True):
        if len(out) == k:
            break
        if t - last >= CAL_SPACING_S:
            out.append(i)
            last = t
    return g.loc[out]


def tau_loso(resid: pd.DataFrame, subject: str) -> float:
    """σ²_within / σ²_between of population residuals in session 1, leaving ``subject`` out."""
    r = resid[resid.subject_id != subject]
    between = r.groupby("subject_id").e.mean().var(ddof=1)
    within = r.groupby("subject_id").e.var(ddof=1).mean()
    return float(within / max(between, 1e-6))


def calibrate(
    cal: pd.DataFrame, ev: pd.DataFrame, k: int, tau: float, tau0: float | None = None
) -> dict[str, np.ndarray]:
    p, y = cal.pred.to_numpy(), cal.y.to_numpy()
    pe = ev.pred.to_numpy()
    has_mean = "pmean" in ev and tau0 is not None
    if k == 0:
        return {"population": pe, **({"training_mean": ev.pmean.to_numpy()} if has_mean else {})}
    off = float(np.mean(y - p))
    out = {
        "reference_mean": np.full(len(ev), y.mean()),
        "offset": pe + off,
        "bayes_offset": pe + off * k / (k + tau),
    }
    if has_mean:  # the no-PPG control for bayes_offset: shrink the cuff mean towards the training mean
        off0 = float(np.mean(y - cal.pmean.to_numpy()))
        out["bayes_reference"] = ev.pmean.to_numpy() + off0 * k / (k + tau0)
    if k >= 2:
        dp, dy = p - p.mean(), y - y.mean()
        b_free = float(dp @ dy / (dp @ dp)) if dp @ dp > 1e-9 else 1.0
        lam = LAMBDA_SLOPE * k
        b_shr = float((dp @ dy + lam) / (dp @ dp + lam))
        out["linear_free"] = y.mean() + b_free * (pe - p.mean())
        out["linear_shrunk"] = y.mean() + b_shr * (pe - p.mean())
    return out


def subject_mae_ci(t: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    m = (t.yhat - t.y).abs().groupby(t.subject_id).mean().to_numpy()
    b = [m[rng.integers(0, m.size, m.size)].mean() for _ in range(N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_site(site: str) -> tuple[list, list, list]:
    w = pd.read_csv(AUR / "windows.csv")
    w = w[w.label_valid & w[f"{site}_sqi_ok"]]
    rows, paired, drift = [], [], []
    for target, col in (("sbp", "sbp_mmHg"), ("dbp", "dbp_mmHg")):
        p = pd.read_csv(PRED / f"personalization_pop_{site}_ridge_{target}.csv.gz")
        pm = pd.read_csv(PRED / f"personalization_pop_{site}_mean_{target}.csv.gz").rename(columns={"pred": "pmean"})
        d = w.merge(p, on="window_id").merge(pm, on="window_id")[
            ["window_id", "subject_id", "session", "t_start_s", "activity_code", col, "pred", "pmean"]
        ]
        d = d.rename(columns={col: "y", "activity_code": "activity"}).sort_values(
            ["subject_id", "session", "t_start_s"]
        )
        resid = d[d.session == 1].assign(e=lambda x: x.y - x.pred)
        resid0 = d[d.session == 1].assign(e=lambda x: x.y - x.pmean)
        recs = []  # (scenario, k, method, subject_id, t_since_cal, activity, y, yhat)
        for sid, g in d.groupby("subject_id"):
            s1, s2 = g[g.session == 1], g[g.session == 2]
            if s1.empty:
                continue
            tau, tau0 = tau_loso(resid, sid), tau_loso(resid0, sid)
            for k in KS:
                cal = pick_calibration(s1, k)
                if len(cal) < k:
                    continue
                t_cal = cal.t_start_s.max() if k else -np.inf
                scen = {"within_day": s1[s1.t_start_s >= (t_cal + 10 if k else 0)], "cross_day": s2}
                for sc, ev in scen.items():
                    if ev.empty:
                        continue
                    since = ev.t_start_s.to_numpy() - (t_cal if k else 0) if sc == "within_day" else np.nan
                    for meth, yhat in calibrate(cal, ev, k, tau, tau0).items():
                        recs.append(
                            pd.DataFrame(
                                {
                                    "scenario": sc,
                                    "k": k,
                                    "method": meth,
                                    "subject_id": sid,
                                    "since_s": since,
                                    "activity": ev.activity.to_numpy(),
                                    "y": ev.y.to_numpy(),
                                    "yhat": yhat,
                                }
                            )
                        )
        R = pd.concat(recs, ignore_index=True)
        rng = np.random.default_rng(SEED)
        # Pairs are made only within the same evaluation windows (same scenario and k), so they are comparable.
        for (sc, k), gk in R.groupby(["scenario", "k"]):
            base_subjects = None
            for meth, t in gk.groupby("method"):
                t = t.reset_index(drop=True)
                # cross-subject conformal half-width
                ae = (t.yhat - t.y).abs()
                q = np.empty(len(t))
                for sid in t.subject_id.unique():
                    m = (t.subject_id == sid).to_numpy()
                    q[m] = np.quantile(ae[~m], COVERAGE)
                lo, hi = subject_mae_ci(t, rng)
                rows.append(
                    {
                        "site": site,
                        "target": target,
                        "scenario": sc,
                        "k": k,
                        "method": meth,
                        **point_metrics(t),
                        "mae_ci_low": lo,
                        "mae_ci_high": hi,
                        "subject_mean_mae": float(ae.groupby(t.subject_id).mean().mean()),
                        "pi90_coverage": float((ae <= q).mean()),
                        "pi90_halfwidth_mean": float(q.mean()),
                    }
                )
                base_subjects = t.subject_id.nunique() if base_subjects is None else base_subjects
            ref = "training_mean" if k == 0 else "reference_mean"
            pairs = [(m_, ref) for m_ in gk.method.unique() if m_ != ref]
            if k > 0:
                pairs.append(("bayes_offset", "bayes_reference"))  # the PPG's own contribution
            for meth, ref in pairs:
                t, b = gk[gk.method == meth], gk[gk.method == ref]
                dd, lo, hi = paired_subject_diff(t, b, N_BOOT, SEED)
                paired.append(
                    {
                        "site": site,
                        "target": target,
                        "scenario": sc,
                        "k": k,
                        "comparison": f"{meth} vs {ref}",
                        "mae_diff": dd,
                        "ci_low": lo,
                        "ci_high": hi,
                        "significant": bool(hi < 0 or lo > 0),
                    }
                )
        # each k against the population (k=0) for the offset method: what does calibration itself buy?
        for sc in ("within_day", "cross_day"):
            pop = R[(R.scenario == sc) & (R.k == 0) & (R.method == "population")]
            for k in KS[1:]:
                t = R[(R.scenario == sc) & (R.k == k) & (R.method == "offset")]
                if t.empty:
                    continue
                dd, lo, hi = paired_subject_diff(t, pop[pop.subject_id.isin(t.subject_id)], N_BOOT, SEED)
                paired.append(
                    {
                        "site": site,
                        "target": target,
                        "scenario": sc,
                        "k": k,
                        "comparison": "offset vs population (k=0)",
                        "mae_diff": dd,
                        "ci_low": lo,
                        "ci_high": hi,
                        "significant": bool(hi < 0 or lo > 0),
                    }
                )
        wd = R[(R.scenario == "within_day") & (R.method.isin(["offset", "reference_mean"])) & (R.k == 3)].copy()
        wd["since_bin_min"] = pd.cut(
            wd.since_s / 60, [0, 5, 10, 20, 40, 1e9], labels=["0-5", "5-10", "10-20", "20-40", "40+"]
        )
        for (meth, b), t in wd.groupby(["method", "since_bin_min"], observed=True):
            drift.append(
                {
                    "site": site,
                    "target": target,
                    "k": 3,
                    "method": meth,
                    "minutes_since_calibration": b,
                    "mae": float((t.yhat - t.y).abs().mean()),
                    "bias": float((t.yhat - t.y).mean()),
                    "n_subjects": t.subject_id.nunique(),
                    "n_windows": len(t),
                }
            )
        log.info("%s %s done", site, target)
    return rows, paired, drift


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, paired, drift = [], [], []
    for site in ("ear", "finger"):
        a, b, c = run_site(site)
        rows, paired, drift = rows + a, paired + b, drift + c
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(paired).to_csv(OUT / "paired.csv", index=False, float_format="%.4f")
    pd.DataFrame(drift).to_csv(OUT / "drift.csv", index=False, float_format="%.4f")
    run = ExperimentRun(
        "calibration_count_study",
        [AUR / "windows.csv", *sorted(PRED.glob("personalization_pop_*_ridge_*.csv.gz"))],
        {
            "k": ";".join(map(str, KS)),
            "cal_spacing_s": CAL_SPACING_S,
            "lambda_slope": LAMBDA_SLOPE,
            "coverage": COVERAGE,
            "n_boot": N_BOOT,
            "population_model": "RakshPulse 2.0 ridge (cached OOF)",
        },
        seed=SEED,
        t0=t0,
    )
    run.add_metrics(met)
    run.finish()
    v = met[(met.site == "ear") & (met.target == "sbp")].pivot_table(
        index=["scenario", "method"], columns="k", values="mae"
    )
    print(v.round(2).to_string())
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
