"""Population + personalisation experiments on the standardised Auricle dataset (data/auricle/*.csv).

Experiments (healthy cohort; site = ear primary, finger secondary; HR + pulse-shape features):

P  Population models on UNSEEN subjects: subject-grouped 5-fold CV, trained on training subjects'
   session-1 windows only, evaluated uncalibrated on held-out subjects' session 1 and session 2.
S  Cross-day strategy comparison on one common evaluation set (session-2 windows after the first 3 session-2
   rest calibration windows). All strategies use the ridge population model (pre-specified; no model was
   better than ridge in the earlier cross-day study):
     population             uncalibrated population prediction
     pop+S1cal              engine offset calibrated on 3 session-1 rest references
     pop+S1cal+S2update     same engine, then updated with the 3 session-2 rest references (adaptive)
     pop+S2cal              fresh engine calibrated on the 3 session-2 rest references only
     pop+S1linear           engine 'linear' calibrated on session-1 references taken every 2 min
     personal_ridge / personal_elasticnet   per-subject feature models fitted on session 1 only
     baseline_S1            BaselineTracker median of the 3 session-1 references (no PPG)
     baseline_S1+S2         BaselineTracker rolling-5 mean after adding the 3 session-2 references (no PPG)
A  Adaptive within-day experiment (each session separately, strictly causal): a reference reading every
   2 minutes. Between readings each method predicts the following windows using only past references.
D  Drift monitoring: does the engine's recommendation on day 2 (after the first new reference) identify
   subjects whose day-1 calibration is wrong?
B  Baseline tracking: day-2 resting BP relative to the day-1 personal baseline.

Outputs: results/personalization/*.csv (committed) and models/auricle_population_ear_ridge.joblib.
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

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.features.feature_pipeline import PER_CHANNEL  # noqa: E402
from rakshpulse.inference.model_loader import ModelBundle  # noqa: E402
from rakshpulse.models.model_factory import PERSONALIZATION_MODELS  # noqa: E402
from rakshpulse.personalization import BaselineTracker, EngineConfig, PersonalizationEngine, RefPoint  # noqa: E402

DATA = ROOT / "data" / "auricle"
OUT = ROOT / "results" / "personalization"
PRED = ROOT / "results" / "predictions"  # window-level OOF population predictions (git-ignored cache)
SITES = ("ear", "finger")
TARGETS = ("sbp", "dbp")
K_CAL, GAP_S, REF_EVERY_S, DAY_S = 3, 10.0, 120.0, 86400.0
SEED, N_BOOT = 0, 2000
log = logging.getLogger("personalization")


def load() -> pd.DataFrame:
    w = pd.read_csv(DATA / "windows.csv", low_memory=False)
    f = pd.read_csv(DATA / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w.merge(f, on="window_id")
    d = d[(d.cohort == "HC") & d.label_valid].reset_index(drop=True)
    d = d.rename(columns={"sbp_mmHg": "sbp", "dbp_mmHg": "dbp", "activity_code": "activity"})
    for s in SITES:
        st.add_normalised_shape(d, s)
    return d


def metrics_row(t: pd.DataFrame, **keys) -> dict:
    return {**keys, **st.point_metrics(t), **st.fast_subject_cis(t, N_BOOT, SEED)}


def long_table(df: pd.DataFrame, target: str, pred_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": df.subject_id.to_numpy(),
            "y": df[target].to_numpy(),
            "yhat": df[pred_col].to_numpy(),
            "activity": df.activity.to_numpy(),
        }
    )


def personal_model(s1: pd.DataFrame, ev: pd.DataFrame, cols, target, kind: str) -> np.ndarray:
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X1, X2 = s1[cols].to_numpy(float), ev[cols].to_numpy(float)
    med = np.nanmedian(X1, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    X1, X2 = np.where(np.isfinite(X1), X1, med), np.where(np.isfinite(X2), X2, med)
    est = Ridge(alpha=10.0) if kind == "ridge" else ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000)
    m = make_pipeline(StandardScaler(), est).fit(X1, s1[target].to_numpy(float))
    return m.predict(X2)


def nearest_on_grid(g: pd.DataFrame, every_s: float) -> pd.DataFrame:
    """Labelled windows nearest to a regular grid (simulated intermittent reference readings)."""
    t = g.t_start_s.to_numpy()
    grid = np.arange(t.min(), t.max() + 1e-9, every_s)
    return g.iloc[sorted({int(np.abs(t - x).argmin()) for x in grid})]


def ref_points(rows: pd.DataFrame, t_offset: float = 0.0) -> list[RefPoint]:
    return [RefPoint(r.t_start_s + t_offset, r.pred_sbp, r.pred_dbp, r.sbp, r.dbp) for r in rows.itertuples()]


def run() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    fold_of = st.subject_folds(d.subject_id, d.sbp, n=5)
    pop_rows, strat_rows, paired_rows, adapt_rows, drift_rows, base_rows, cfg_rows = [], [], [], [], [], [], []
    strategy_preds = []
    for site in SITES:
        ds = d[d[f"{site}_sqi_ok"]].reset_index(drop=True)
        s1 = ds[ds.session == 1].reset_index(drop=True)
        cols = st.feature_set("hr_shape", site, [f"{site}_{n}" for n in PER_CHANNEL])
        preds = {}
        # ------------------------------------------------------------- P: population models
        for model in ("mean",) + PERSONALIZATION_MODELS:  # "mean" = training-subject mean BP (baseline)
            for target in TARGETS:
                cache = PRED / f"personalization_pop_{site}_{model}_{target}.csv.gz"
                if cache.exists() and len(pd.read_csv(cache)) == len(ds):
                    p = pd.read_csv(cache).pred.to_numpy()
                else:
                    p = st.population_predictions(s1, ds, cols, target, model, fold_of, seed=SEED)
                    PRED.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"window_id": ds.window_id, "pred": p}).to_csv(cache, index=False)
                preds[(model, target)] = p
                for sess in (1, 2):
                    m = ds.session.to_numpy() == sess
                    t = pd.DataFrame({"subject_id": ds.subject_id[m], "y": ds[target][m], "yhat": p[m]})
                    pop_rows.append(
                        metrics_row(
                            t,
                            site=site,
                            model=model,
                            target=target,
                            evaluated_on=f"unseen subjects, session {sess}, uncalibrated",
                        )
                    )
            log.info("%s population %-13s done (%.0fs)", site, model, time.time() - t0)
        cfg_rows.append(
            {
                "site": site,
                "features": ";".join(cols),
                "n_features": len(cols),
                "folds": 5,
                "training_windows": "session-1 windows of training subjects only",
                "seed": SEED,
            }
        )
        for model in PERSONALIZATION_MODELS:  # does any population model beat the training-mean predictor?
            for target in TARGETS:
                for sess in (1, 2):
                    m = ds.session.to_numpy() == sess
                    base = {"subject_id": ds.subject_id[m].to_numpy(), "y": ds[target][m].to_numpy()}
                    ta = pd.DataFrame({**base, "yhat": preds[(model, target)][m]})
                    tb = pd.DataFrame({**base, "yhat": preds[("mean", target)][m]})
                    dm, lo, hi = st.paired_subject_diff(ta, tb, N_BOOT, SEED)
                    paired_rows.append(
                        {
                            "site": site,
                            "target": target,
                            "comparison": f"population {model} vs mean (unseen subjects, session {sess})",
                            "mae_diff": dm,
                            "ci_low": lo,
                            "ci_high": hi,
                            "significant": bool(hi < 0 or lo > 0),
                        }
                    )
        ds["pred_sbp"], ds["pred_dbp"] = preds[("ridge", "sbp")], preds[("ridge", "dbp")]

        # ------------------------------------------------------------- S: cross-day strategies
        recs = []
        for sid, g in ds.groupby("subject_id"):
            g1 = g[g.session == 1].sort_values("t_start_s")
            g2 = g[g.session == 2].sort_values("t_start_s")
            c1 = g1[g1[f"{site}_calibration_eligible"]].iloc[:K_CAL]
            c2 = g2[g2[f"{site}_calibration_eligible"]].iloc[:K_CAL]
            if len(c1) < K_CAL or len(c2) < K_CAL:
                continue
            ev = g2[g2.t_start_s > c2.t_start_s.max() + GAP_S]
            if len(ev) < 5:
                continue
            out = ev[["window_id", "subject_id", "t_start_s", "activity", "sbp", "dbp", "pred_sbp", "pred_dbp"]].copy()
            e1 = PersonalizationEngine(sid, EngineConfig(method="offset"), "ridge")
            e1.calibrate(ref_points(c1))
            app = [e1.apply(r.pred_sbp, r.pred_dbp) for r in ev.itertuples()]
            out["pop+S1cal_sbp"], out["pop+S1cal_dbp"] = [a["sbp"] for a in app], [a["dbp"] for a in app]
            # drift monitor: first day-2 reference evaluated against the day-1 calibration
            first = c2.iloc[0]
            rec = e1.update(RefPoint(first.t_start_s + DAY_S, first.pred_sbp, first.pred_dbp, first.sbp, first.dbp))
            status = e1.status(first.t_start_s + DAY_S)
            for r in ref_points(c2.iloc[1:], DAY_S):
                e1.update(r)
            app = [e1.apply(r.pred_sbp, r.pred_dbp, r.t_start_s + DAY_S) for r in ev.itertuples()]
            out["pop+S1cal+S2update_sbp"], out["pop+S1cal+S2update_dbp"] = (
                [a["sbp"] for a in app],
                [a["dbp"] for a in app],
            )
            e2 = PersonalizationEngine(sid, EngineConfig(method="offset"), "ridge")
            e2.calibrate(ref_points(c2, DAY_S))
            app = [e2.apply(r.pred_sbp, r.pred_dbp, r.t_start_s + DAY_S) for r in ev.itertuples()]
            out["pop+S2cal_sbp"], out["pop+S2cal_dbp"] = [a["sbp"] for a in app], [a["dbp"] for a in app]
            # linear engine with intermittent day-1 references (every 2 min)
            sched = nearest_on_grid(g1, REF_EVERY_S)
            el = PersonalizationEngine(sid, EngineConfig(method="linear", half_life_s=1e9), "ridge")
            el.calibrate(ref_points(sched))
            app = [el.apply(r.pred_sbp, r.pred_dbp) for r in ev.itertuples()]
            out["pop+S1linear_sbp"], out["pop+S1linear_dbp"] = [a["sbp"] for a in app], [a["dbp"] for a in app]
            for kind in ("ridge", "elasticnet"):
                for target in TARGETS:
                    out[f"personal_{kind}_{target}"] = personal_model(g1, ev, cols, target, kind)
            bt = BaselineTracker(sid)
            for r in c1.itertuples():
                bt.add(r.t_start_s, r.sbp, r.dbp)
            s_ = bt.stats()
            out["baseline_S1_sbp"], out["baseline_S1_dbp"] = s_["sbp_median"], s_["dbp_median"]
            for r in c2.itertuples():
                bt.add(r.t_start_s + DAY_S, r.sbp, r.dbp)
            s2_ = bt.stats()
            out["baseline_S1+S2_sbp"], out["baseline_S1+S2_dbp"] = s2_["sbp_rolling5_mean"], s2_["dbp_rolling5_mean"]
            out["population_sbp"], out["population_dbp"] = out.pred_sbp, out.pred_dbp
            day2_err = float(np.abs(out["pop+S1cal_sbp"] - out.sbp).mean())
            drift_rows.append(
                {
                    "site": site,
                    "subject_id": sid,
                    "first_day2_residual_sbp": rec["residual_sbp"],
                    "recommended": status["recalibration_recommended"],
                    "reasons": "; ".join(status["reasons"]),
                    "day2_mae_with_day1_cal": day2_err,
                }
            )
            # B: baseline tracking (day-2 rest vs day-1 baseline)
            dev = BaselineTracker(sid)
            for r in g1[g1[f"{site}_calibration_eligible"]].itertuples():
                dev.add(r.t_start_s, r.sbp, r.dbp)
            day2_rest = g2[g2[f"{site}_calibration_eligible"]]
            dv = dev.deviation(day2_rest.sbp.median(), day2_rest.dbp.median())
            base_rows.append(
                {
                    "site": site,
                    "subject_id": sid,
                    "n_day1_refs": dev.stats()["n"],
                    "day1_sbp_median": dev.stats()["sbp_median"],
                    "day1_sbp_sd": dev.stats()["sbp_sd"],
                    "day2_rest_sbp_median": day2_rest.sbp.median(),
                    **dv,
                }
            )
            out["site"] = site
            recs.append(out)
        S = pd.concat(recs, ignore_index=True)
        strategy_preds.append(S)
        methods = [
            "population",
            "pop+S1cal",
            "pop+S1cal+S2update",
            "pop+S2cal",
            "pop+S1linear",
            "personal_ridge",
            "personal_elasticnet",
            "baseline_S1",
            "baseline_S1+S2",
        ]
        tabs = {}
        for target in TARGETS:
            for meth in methods:
                t = long_table(S, target, f"{meth}_{target}")
                tabs[(meth, target)] = t
                strat_rows.append(
                    metrics_row(
                        t,
                        site=site,
                        target=target,
                        method=meth,
                        information=(
                            "none"
                            if meth == "population"
                            else "S1 features+labels"
                            if meth.startswith("personal")
                            else "S1+S2 refs"
                            if "S2" in meth
                            else "S1 refs"
                        ),
                    )
                )
            for a, b in (
                ("pop+S1cal", "baseline_S1"),
                ("pop+S1cal+S2update", "baseline_S1+S2"),
                ("pop+S2cal", "baseline_S1+S2"),
                ("pop+S1cal+S2update", "pop+S1cal"),
                ("pop+S1linear", "pop+S1cal"),
                ("personal_ridge", "baseline_S1"),
                ("personal_elasticnet", "baseline_S1"),
                ("population", "baseline_S1"),
                ("personal_elasticnet", "personal_ridge"),
            ):
                dm, lo, hi = st.paired_subject_diff(tabs[(a, target)], tabs[(b, target)], N_BOOT, SEED)
                paired_rows.append(
                    {
                        "site": site,
                        "target": target,
                        "comparison": f"{a} vs {b}",
                        "mae_diff": dm,
                        "ci_low": lo,
                        "ci_high": hi,
                        "significant": bool(hi < 0 or lo > 0),
                    }
                )
        # ------------------------------------------------------------- A: adaptive within-day
        for sess in (1, 2):
            rows = []
            for sid, g in ds[ds.session == sess].groupby("subject_id"):
                g = g.sort_values("t_start_s")
                refs = g[g[f"{site}_calibration_eligible"] | (g.activity != 0)]  # any trusted label can be a ref
                R = nearest_on_grid(refs, REF_EVERY_S)
                if len(R) < 3:
                    continue
                eo = PersonalizationEngine(sid, EngineConfig(method="offset", half_life_s=600.0))
                el = PersonalizationEngine(sid, EngineConfig(method="linear", half_life_s=1e9))
                hist: list[float] = []
                for i, r in enumerate(R.itertuples()):
                    nxt = R.t_start_s.iloc[i + 1] if i + 1 < len(R) else np.inf
                    pt = RefPoint(r.t_start_s, r.pred_sbp, r.pred_dbp, r.sbp, r.dbp)
                    (eo.calibrate([pt]) if i == 0 else eo.update(pt))
                    (el.calibrate([pt]) if i == 0 else el.update(pt))
                    hist.append(r.sbp)
                    ev = g[(g.t_start_s >= r.t_start_s + GAP_S) & (g.t_start_s < nxt - GAP_S)]
                    for e in ev.itertuples():
                        rows.append(
                            {
                                "subject_id": sid,
                                "y": e.sbp,
                                "activity": e.activity,
                                "last_reference": r.sbp,
                                "rolling3_reference": float(np.mean(hist[-3:])),
                                "engine_offset": eo.apply(e.pred_sbp, e.pred_dbp, e.t_start_s)["sbp"],
                                "engine_linear": el.apply(e.pred_sbp, e.pred_dbp, e.t_start_s)["sbp"],
                                "population": e.pred_sbp,
                            }
                        )
            A = pd.DataFrame(rows)
            for meth in ("last_reference", "rolling3_reference", "engine_offset", "engine_linear", "population"):
                t = A[["subject_id", "y", "activity"]].assign(yhat=A[meth])
                adapt_rows.append(
                    metrics_row(t, site=site, session=sess, target="sbp", method=meth, reference_interval_s=REF_EVERY_S)
                )
                if meth != "last_reference":
                    dm, lo, hi = st.paired_subject_diff(
                        t, A[["subject_id", "y"]].assign(yhat=A.last_reference), N_BOOT, SEED
                    )
                    paired_rows.append(
                        {
                            "site": site,
                            "target": "sbp",
                            "comparison": f"adaptive S{sess}: {meth} vs last_reference",
                            "mae_diff": dm,
                            "ci_low": lo,
                            "ci_high": hi,
                            "significant": bool(hi < 0 or lo > 0),
                        }
                    )
        log.info("%s strategies + adaptive done (%.0fs)", site, time.time() - t0)

    pd.DataFrame(pop_rows).to_csv(OUT / "population_models.csv", index=False, float_format="%.4f")
    pd.DataFrame(strat_rows).to_csv(OUT / "cross_day_strategies.csv", index=False, float_format="%.4f")
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_comparisons.csv", index=False, float_format="%.4f")
    pd.DataFrame(adapt_rows).to_csv(OUT / "adaptive_within_day.csv", index=False, float_format="%.4f")
    pd.DataFrame(drift_rows).to_csv(OUT / "drift_monitor.csv", index=False, float_format="%.4f")
    pd.DataFrame(base_rows).to_csv(OUT / "baseline_tracking.csv", index=False, float_format="%.4f")
    pd.DataFrame(cfg_rows).to_csv(OUT / "experiment_config.csv", index=False)
    pd.concat(strategy_preds).to_csv(OUT / "cross_day_strategy_predictions.csv", index=False, float_format="%.3f")

    # ------------------------------------------------------------- deployable population bundle (ear, ridge)
    from rakshpulse.evaluation.auricle_study import fit_population

    ear = d[d.ear_sqi_ok & (d.session == 1)]
    cols = st.feature_set("hr_shape", "ear", [])
    P = pd.DataFrame(pop_rows)
    val = {}
    models = {}
    for target in TARGETS:
        models[target] = fit_population(
            "ridge", ear[cols].to_numpy(float), ear[target].to_numpy(float), ear.subject_id.to_numpy(), SEED
        )
        r = P[
            (P.site == "ear") & (P.model == "ridge") & (P.target == target) & P.evaluated_on.str.contains("session 1")
        ].iloc[0]
        val[target] = {"mae": float(r.mae), "sd_error": float(r.sd_error), "n_subjects": int(r.n_subjects)}
    ModelBundle(
        models["sbp"],
        models["dbp"],
        cols,
        ("ear",),
        False,
        val,
        "auricle-ear-ridge-1",
        trained_on="Auricle v1.0.0 healthy cohort, session 1, ear PPG (Nonin earlobe), CNAP labels",
    ).save(ROOT / "models" / "auricle_population_ear_ridge.joblib")
    log.info("finished in %.0fs", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
