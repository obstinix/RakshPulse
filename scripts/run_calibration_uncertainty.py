"""RakshPulse 4.0: hierarchical Bayesian calibration with matched no-PPG controls, plus uncertainty.

Base predictions are the cached RakshPulse 2.0 out-of-fold predictions (subject-held-out): ``pred`` (ridge,
HR + shape) and ``pmean`` (training-mean BP; no PPG). For each of 5 subject-grouped folds, hyper-parameters
(variance components, drift κ, conformal scales, quantile models) are estimated on the training subjects only.

Scenarios (readings are the first label-valid windows ≥ 60 s apart; never scored):
* ``within_day_k1`` / ``within_day_k3``: 1 or 3 readings at the start of day 1; score later day-1 windows;
* ``cross_day_transfer``: the 3 day-1 readings only; score all day-2 windows;
* ``cross_day_plus_one``: the 3 day-1 readings + the first day-2 reading; score later day-2 windows.

Point methods (identical windows within a scenario): ``population`` (2.0 model), ``training_mean``,
``last_cuff``, ``historical_mean``, ``offset_2.0`` (2.0 engine: base + mean reading residual, preferring
current-session readings), ``regularized_ppg_3.0`` (3.0 shrunk offset), ``bayes_reference_hier`` (hierarchical
on ``pmean``: the **matched no-PPG control**) and ``bayes_ppg_hier`` (hierarchical on ``pred``).

Uncertainty (90 % unless stated), for the hierarchical methods: Gaussian posterior predictive (``model``);
normalised split conformal (``conformal_scaled``: the posterior SD scaled on training subjects); plain
cross-subject split conformal (``conformal_const``); and quantile gradient boosting of the residual on
[Δt since reading, readings in session, base prediction, quality score] (``quantile_gb``). Reported: coverage,
width, Winkler score, per-subject coverage spread, and calibration error across 50/80/90/95 %. Under shift:
cross-day vs within-day coverage, and coverage by feature Mahalanobis distance (top decile = out-of-distribution
proxy).

Adaptive recalibration interval: from the fitted variance components, the time after a reading until the
90 % half-width exceeds 10 / 15 mmHg, for PPG and reference-only.

Outputs: results/calibration_v4/{point,paired,intervals,levels,ood,params,recal_interval}.csv and a run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.statistics import holm, paired_test, subject_errors  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.personalization.hierarchical import HierarchicalCalibrator, fit_params, interval_metrics  # noqa: E402

AUR = ROOT / "data" / "auricle"
PRED = ROOT / "results" / "predictions"
QUAL = ROOT / "data" / "interim" / "auricle_quality_v4.csv.gz"
OUT = ROOT / "results" / "calibration_v4"
SEED, N_BOOT, N_FOLDS, SPACING = 0, 2000, 5, 60.0
LEVELS = (0.5, 0.8, 0.9, 0.95)
SCENARIOS = ("within_day_k1", "within_day_k3", "cross_day_transfer", "cross_day_plus_one")
log = logging.getLogger("calib4")


def load(site: str, target: str) -> pd.DataFrame:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w[w.label_valid & w[f"{site}_sqi_ok"] & (w.cohort == "HC")].merge(f, on="window_id")
    st.add_normalised_shape(d, site)
    p = pd.read_csv(PRED / f"personalization_pop_{site}_ridge_{target}.csv.gz")
    pm = pd.read_csv(PRED / f"personalization_pop_{site}_mean_{target}.csv.gz").rename(columns={"pred": "pmean"})
    d = d.merge(p, on="window_id").merge(pm, on="window_id")
    if QUAL.exists():
        q = pd.read_csv(QUAL)
        d = d.merge(q[q.site == site][["window_id", "score"]], on="window_id", how="left")
    else:
        d["score"] = np.nan
    d["y"] = d[f"{target}_mmHg"]
    return d.sort_values(["subject_id", "session", "t_start_s"]).reset_index(drop=True)


def readings(g: pd.DataFrame, k: int) -> pd.DataFrame:
    out, last = [], -np.inf
    for i, t in zip(g.index, g.t_start_s, strict=True):
        if len(out) == k:
            break
        if t - last >= SPACING:
            out.append(i)
            last = t
    return g.loc[out]


def scenario_sets(d: pd.DataFrame) -> dict[str, list[tuple]]:
    """Per scenario: list of (subject, readings frame with 'current' flag, evaluation frame)."""
    out = {s: [] for s in SCENARIOS}
    for sid, g in d.groupby("subject_id"):
        g1, g2 = g[g.session == 1], g[g.session == 2]
        for k, name in ((1, "within_day_k1"), (3, "within_day_k3")):
            r = readings(g1, k)
            if len(r) == k:
                ev = g1[g1.t_start_s > r.t_start_s.max() + 10]
                if len(ev) >= 5:
                    out[name].append((sid, r.assign(current=True), ev))
        r1 = readings(g1, 3)
        if len(r1) == 3 and len(g2) >= 6:
            out["cross_day_transfer"].append((sid, r1.assign(current=False), g2))
            r2 = readings(g2, 1)
            ev = g2[g2.t_start_s > r2.t_start_s.max() + 10]
            if len(ev) >= 5:
                rr = pd.concat([r1.assign(current=False), r2.assign(current=True)])
                out["cross_day_plus_one"].append((sid, rr, ev))
    return out


def point_methods(r: pd.DataFrame, ev: pd.DataFrame, cal: dict, tau: float) -> dict[str, pd.DataFrame]:
    """Point predictions (and, for hierarchical methods, SD) for one subject × scenario."""
    cur = r[r.current]
    use = cur if len(cur) else r
    last = r.sort_values(["session", "t_start_s"]).iloc[-1]
    dt = ev.t_start_s.to_numpy() - cur.t_start_s.max() if len(cur) else np.full(len(ev), np.nan)
    if len(cur):  # Δt only within the current session
        dt = np.where(ev.session.to_numpy() == cur.session.iloc[0], dt, np.nan)
    off = float((use.y - use.pred).mean())
    k = len(use)
    out = {
        "population": ev.pred.to_numpy(),
        "training_mean": ev.pmean.to_numpy(),
        "last_cuff": np.full(len(ev), last.y),
        "historical_mean": np.full(len(ev), r.y.mean()),
        "offset_2.0": ev.pred.to_numpy() + off,
        "regularized_ppg_3.0": ev.pred.to_numpy() + off * k / (k + tau),
    }
    sds = {}
    for name, base in (("bayes_ppg_hier", "pred"), ("bayes_reference_hier", "pmean")):
        hc = cal[base]
        p = hc.predict(ev[base].to_numpy(), dt, (r.y - r[base]).to_numpy(), r.current.to_numpy())
        out[name] = p.yhat.to_numpy()
        sds[name] = p.sd.to_numpy() / hc.scale  # unscaled model SD
    return out, sds, dt


def run_fold(d: pd.DataFrame, sets: dict, train: set, test: set) -> tuple[list, list]:
    tr = d[d.subject_id.isin(train)]
    params = {b: fit_params(tr.assign(r=tr.y - tr[b])) for b in ("pred", "pmean")}
    cal = {b: HierarchicalCalibrator(params[b]) for b in params}
    res_s1 = tr[tr.session == 1].assign(e=lambda x: x.y - x.pred)
    tau = float(res_s1.groupby("subject_id").e.var().mean() / max(res_s1.groupby("subject_id").e.mean().var(), 1e-6))
    recs = {"train": [], "test": []}
    for scen, items in sets.items():
        for sid, r, ev in items:
            part = "train" if sid in train else "test" if sid in test else None
            if part is None:
                continue
            pts, sds, dt = point_methods(r, ev, cal, tau)
            base = pd.DataFrame(
                {
                    "scenario": scen,
                    "subject_id": sid,
                    "session": ev.session.to_numpy(),
                    "y": ev.y.to_numpy(),
                    "dt": dt,
                    "k_current": int(r.current.sum()),
                    "score": ev.score.to_numpy(),
                    "f_ppg": ev.pred.to_numpy(),
                }
            )
            for m, v in pts.items():
                base[f"p_{m}"] = v
            for m, v in sds.items():
                base[f"sd_{m}"] = v
            recs[part].append(base)
    trn, tst = pd.concat(recs["train"], ignore_index=True), pd.concat(recs["test"], ignore_index=True)
    # conformal and quantile models, per scenario, fitted on the TRAINING subjects' scenario predictions
    for m in ("bayes_ppg_hier", "bayes_reference_hier"):
        for scen in SCENARIOS:
            a, b = trn[trn.scenario == scen], tst.scenario == scen
            if a.empty or not b.any():
                continue
            zabs = np.abs(a.y - a[f"p_{m}"]) / a[f"sd_{m}"]
            eabs = np.abs(a.y - a[f"p_{m}"])
            for lv in LEVELS:
                zq = np.quantile(zabs, lv)
                tst.loc[b, f"hw_scaled_{m}_{lv}"] = zq * tst.loc[b, f"sd_{m}"]
                tst.loc[b, f"hw_const_{m}_{lv}"] = np.quantile(eabs, lv)
                tst.loc[b, f"hw_model_{m}_{lv}"] = stats.norm.ppf(0.5 + lv / 2) * tst.loc[b, f"sd_{m}"]

            def X(q, m=m):
                return np.c_[np.nan_to_num(q.dt, nan=1e5), q.k_current, q[f"p_{m}"], np.nan_to_num(q.score, nan=50)]

            res = (a.y - a[f"p_{m}"]).to_numpy()
            for lv in (0.9,):
                lo_m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=(1 - lv) / 2, max_iter=150, random_state=SEED
                )
                hi_m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=1 - (1 - lv) / 2, max_iter=150, random_state=SEED
                )
                xb = X(tst[b])
                tst.loc[b, f"qlo_{m}"] = lo_m.fit(X(a), res).predict(xb)
                tst.loc[b, f"qhi_{m}"] = hi_m.fit(X(a), res).predict(xb)
    return tst, [{"base": b, **params[b].as_dict(), "tau_3.0": tau} for b in params]


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run = ExperimentRun(
        "calibration_uncertainty_v4",
        [
            AUR / "windows.csv",
            *sorted(PRED.glob("personalization_pop_*_ridge_*.csv.gz")),
            *sorted(PRED.glob("personalization_pop_*_mean_*.csv.gz")),
        ],
        {
            "scenarios": ";".join(SCENARIOS),
            "reading_spacing_s": SPACING,
            "levels": ";".join(map(str, LEVELS)),
            "n_boot": N_BOOT,
            "folds": N_FOLDS,
        },
        seed=SEED,
        dataset_versions={"auricle": "PhysioNet 1.0.0, standardised 2.0", "base_predictions": "RakshPulse 2.0 OOF"},
        model_config={
            "hierarchical": "b + c_session + capped linear variogram; method of moments on training subjects",
            "quantile": "HistGradientBoosting quantile, 150 iterations",
        },
    )
    point, paired, intervals, levels, ood, params_rows = [], [], [], [], [], []
    for site in ("ear", "finger"):
        for target in ("sbp", "dbp"):
            with run.phase("train"):
                d = load(site, target)
                sets = scenario_sets(d)
                subj = d.groupby("subject_id").y.mean()
                folds = st.subject_folds(pd.Series(subj.index), pd.Series(subj.values), n=N_FOLDS)
                run.splits.update(folds)
                parts = []
                for k in range(N_FOLDS):
                    test = {s for s, f in folds.items() if f == k}
                    tst, pr = run_fold(d, sets, set(folds) - test, test)
                    parts.append(tst)
                    params_rows += [{"site": site, "target": target, "fold": k, **x} for x in pr]
                P = pd.concat(parts, ignore_index=True)
            with run.phase("evaluate"):
                methods = [c[2:] for c in P.columns if c.startswith("p_")]
                for scen, g in P.groupby("scenario"):
                    errs = {}
                    for m in methods:
                        t = g.assign(yhat=g[f"p_{m}"])
                        point.append(
                            {"site": site, "target": target, "scenario": scen, "method": m, **st.point_metrics(t)}
                        )
                        errs[m] = subject_errors(t)
                    fam = []
                    for a, b in (
                        ("bayes_ppg_hier", "bayes_reference_hier"),
                        ("bayes_ppg_hier", "last_cuff"),
                        ("bayes_reference_hier", "last_cuff"),
                        ("bayes_reference_hier", "historical_mean"),
                        ("bayes_ppg_hier", "regularized_ppg_3.0"),
                        ("bayes_ppg_hier", "offset_2.0"),
                        ("bayes_ppg_hier", "population"),
                    ):
                        fam.append(
                            {
                                "site": site,
                                "target": target,
                                "scenario": scen,
                                "comparison": f"{a} vs {b}",
                                **paired_test(errs[a], errs[b], N_BOOT, seed=SEED),
                            }
                        )
                    for r, p in zip(fam, holm([r["p_perm"] for r in fam]), strict=True):
                        r["p_holm"] = p
                    paired += fam
                    for m in ("bayes_ppg_hier", "bayes_reference_hier"):
                        yv, ph = g.y.to_numpy(), g[f"p_{m}"].to_numpy()
                        for kind in ("model", "scaled", "const"):
                            hw = g[f"hw_{kind}_{m}_0.9"].to_numpy()
                            intervals.append(
                                {
                                    "site": site,
                                    "target": target,
                                    "scenario": scen,
                                    "method": m,
                                    "interval": {
                                        "model": "gaussian_model",
                                        "scaled": "conformal_scaled",
                                        "const": "conformal_const",
                                    }[kind],
                                    **interval_metrics(yv, ph - hw, ph + hw, 0.9, g.subject_id.to_numpy()),
                                }
                            )
                            cov = [np.mean(np.abs(yv - ph) <= g[f"hw_{kind}_{m}_{lv}"].to_numpy()) for lv in LEVELS]
                            levels.append(
                                {
                                    "site": site,
                                    "target": target,
                                    "scenario": scen,
                                    "method": m,
                                    "interval": kind,
                                    **{f"cov_{lv}": c for lv, c in zip(LEVELS, cov, strict=True)},
                                    "calibration_error": float(np.mean(np.abs(np.array(cov) - np.array(LEVELS)))),
                                }
                            )
                        intervals.append(
                            {
                                "site": site,
                                "target": target,
                                "scenario": scen,
                                "method": m,
                                "interval": "quantile_gb",
                                **interval_metrics(
                                    yv, ph + g[f"qlo_{m}"], ph + g[f"qhi_{m}"], 0.9, g.subject_id.to_numpy()
                                ),
                            }
                        )
                    # out-of-distribution proxy: distance of the base prediction's inputs is not cached, so use
                    # |population prediction − training mean| scaled by its training spread as a shift score
                    sc = np.abs(g.f_ppg - g.p_training_mean)
                    top = sc >= np.quantile(sc, 0.9)
                    for m in ("bayes_ppg_hier",):
                        hw = g[f"hw_scaled_{m}_0.9"]
                        inside = np.abs(g.y - g[f"p_{m}"]) <= hw
                        ood.append(
                            {
                                "site": site,
                                "target": target,
                                "scenario": scen,
                                "method": m,
                                "coverage_top_decile_shift": float(inside[top].mean()),
                                "coverage_rest": float(inside[~top].mean()),
                                "mae_top_decile_shift": float(np.abs(g.y - g[f"p_{m}"])[top].mean()),
                                "mae_rest": float(np.abs(g.y - g[f"p_{m}"])[~top].mean()),
                            }
                        )
            log.info("%s %s done", site, target)
    pr = pd.DataFrame(params_rows)
    rec = []
    for (site, target, base), g in pr.groupby(["site", "target", "base"]):
        p = g.mean(numeric_only=True)
        for thr in (10.0, 15.0):
            # after one reading in the session: Var = post(1 reading) + γ(Δt); solve for Δt
            from rakshpulse.personalization.hierarchical import HierParams

            hp = HierParams(p.sigma_b2, p.sigma_c2, p.sigma_r2, p.sigma_w2, p.kappa)
            v_mu = HierarchicalCalibrator(hp).posterior(np.array([0.0]), np.array([True]))[1]
            need = (thr / 1.645) ** 2 - v_mu - hp.sigma_r2
            # NaN = the half-width already exceeds the threshold right after a reading (no safe interval exists);
            # inf = it never exceeds it (capped variogram stays below)
            if need <= 0:
                t_s = np.nan
            else:
                t_s = need / hp.kappa if hp.kappa > 0 else np.inf
            cap = need > 0 and v_mu + hp.sigma_w2 <= (thr / 1.645) ** 2
            rec.append(
                {
                    "site": site,
                    "target": target,
                    "base": base,
                    "halfwidth_threshold_mmHg": thr,
                    "halfwidth_at_reading": float(1.645 * np.sqrt(v_mu + hp.sigma_r2)),
                    "seconds_until_threshold": float(np.inf if cap else t_s),
                }
            )
    for name, rows in (
        ("point", point),
        ("paired", paired),
        ("intervals", intervals),
        ("levels", levels),
        ("ood", ood),
        ("params", params_rows),
        ("recal_interval", rec),
    ):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    run.add_metrics(pd.DataFrame(point))
    run.finish()
    pp = pd.DataFrame(paired)
    print(
        pp[pp.comparison.str.contains("bayes_ppg_hier vs bayes_reference_hier|bayes_reference_hier vs last_cuff")][
            ["site", "target", "scenario", "comparison", "mean_diff", "ci_low", "ci_high", "p_holm"]
        ]
        .round(3)
        .to_string(index=False)
    )
    iv = pd.DataFrame(intervals)
    print(
        iv[(iv.target == "sbp") & (iv.method == "bayes_ppg_hier")][
            ["site", "scenario", "interval", "coverage", "mean_width", "winkler", "subject_coverage_p10"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(pd.DataFrame(rec).round(1).to_string(index=False))
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
