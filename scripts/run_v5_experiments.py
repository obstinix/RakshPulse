"""RakshPulse 5.0 experiments (real Auricle data unless marked SIMULATED).

X1 Cross-day domain adaptation of the absolute population model (strict cross-day split: training subjects'
   day 1 → held-out subjects' day 2). Ridge on ear HR + shape with (a) no adaptation, (b) per-session feature
   standardisation, (c) CORAL to the training distribution, plus (d) the training mean. Each is scored
   absolute, and after transferring a day-1 cuff calibration (3 readings, hierarchical; the matched no-PPG
   control is the same calibration on the training mean). Adaptation uses unlabelled day-2 features only.
X2 Evaluation-protocol comparison: the same ridge under subject-independent (all sessions), strict cross-day,
   and session-independent-with-seen-subjects splits. This shows how much each protocol flatters a model.
X3 Calibration retention and expiry with the 5.0 policy engine on the cached RakshPulse 2.0 out-of-fold
   predictions: within-session error by time since the last reading, and error in windows the policy marks
   valid vs expired for expiry settings of 5, 10, 20 and 40 min.
X4 SIMULATED acquisition-perturbation sensitivity (software/tool validation, not physiology): how much each
   ear-feature-equivalent IR feature moves (in baseline between-window SDs) under contact/perfusion amplitude
   changes, ADC gain/DC changes, baseline wander, sensor noise and waveform reflection changes, and whether the
   quality score flags them.
B  Desktop-CPU benchmarks of the 5.0 platform (simulated input): recording throughput, pipeline ms/window,
   inference-engine latency p50/p95, peak Python memory, validation speed. Not embedded measurements.

Outputs: results/v5/*.csv, docs/figures/v5_*.png and provenance runs.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.acquisition.ingest import Recorder  # noqa: E402
from rakshpulse.acquisition.simulator import SimConfig, simulate  # noqa: E402
from rakshpulse.acquisition.sources import SimulatedSensor  # noqa: E402
from rakshpulse.datastore.store import DatasetStore, SessionMetadata  # noqa: E402
from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.splits import SplitSpec, make_splits, subject_folds  # noqa: E402
from rakshpulse.evaluation.statistics import holm, paired_test, subject_errors  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.inference.engine import EngineConfig, InferenceEngine  # noqa: E402
from rakshpulse.models.domain_adaptation import CORAL, SessionStandardizer  # noqa: E402
from rakshpulse.models.model_factory import make_model  # noqa: E402
from rakshpulse.models.registry import TrainConfig, train_model  # noqa: E402
from rakshpulse.personalization.strategies import CalibrationPolicy, evaluate  # noqa: E402
from rakshpulse.preprocessing.pipeline import PipelineConfig, process_recording  # noqa: E402

AUR = ROOT / "data" / "auricle"
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "v5"
FIG = ROOT / "docs" / "figures"
SEED, N_BOOT = 0, 2000
log = logging.getLogger("v5")


def auricle(site: str = "ear") -> tuple[pd.DataFrame, list[str]]:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w.merge(f, on="window_id")
    d = d[(d.cohort == "HC") & d.label_valid & d[f"{site}_sqi_ok"]].copy()
    st.add_normalised_shape(d, site)
    d["session_id"] = "D" + d.session.astype(str)
    d["subject_id"] = d.subject_id.astype(str)
    cols = [f"{site}_hr"] + [f"{site}_{c}" for c in st.SHAPE_FEATURES]
    return d.sort_values(["subject_id", "session_id", "t_start_s"]).reset_index(drop=True), cols


# ------------------------------------------------------------------------------------------------ X1
def x1_domain_adaptation(run: ExperimentRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    d, cols = auricle()
    both = d.groupby("subject_id").session_id.nunique()
    d = d[d.subject_id.isin(both[both == 2].index)].reset_index(drop=True)
    spec = SplitSpec("cross_day", 5, SEED, ("D1",))
    folds = make_splits(d, spec)
    run.splits.update(subject_folds(d.subject_id, 5, SEED))
    rows, paired = [], []
    for target in ("sbp", "dbp"):
        y = d[f"{target}_mmHg"].to_numpy(float)
        preds = {m: np.full(len(d), np.nan) for m in ("training_mean", "ridge", "ridge_session_std", "ridge_coral")}
        for tr, te in folds:
            Xtr, Xte = d.loc[tr, cols].to_numpy(float), d.loc[te, cols].to_numpy(float)
            med = np.nanmedian(Xtr, 0)
            Xtr, Xte = np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(Xte), Xte, med)
            preds["training_mean"][te] = y[tr].mean()
            m, _ = make_model("ridge", SEED)
            m.set_params(model__alpha=100.0)
            preds["ridge"][te] = m.fit(Xtr, y[tr]).predict(Xte)
            gtr = (d.subject_id + "|" + d.session_id).to_numpy()
            ss = SessionStandardizer()
            m2, _ = make_model("ridge", SEED)
            m2.set_params(model__alpha=100.0)
            m2.fit(ss.transform(Xtr, gtr[tr]), y[tr])
            preds["ridge_session_std"][te] = m2.predict(ss.transform(Xte, gtr[te]))
            coral = CORAL(eps=1.0).fit(Xtr)
            al = np.empty_like(Xte)
            for g in np.unique(gtr[te]):  # one target domain per held-out subject-session (unlabelled)
                k = gtr[te] == g
                al[k] = coral.transform(Xte[k])
            preds["ridge_coral"][te] = m.predict(al)
        ev = d.assign(y=y)
        test_rows = np.zeros(len(d), bool)
        for _, te in folds:
            test_rows[te] = True
        errs = {}
        for meth, p in preds.items():
            t = ev[test_rows].assign(yhat=p[test_rows])
            rows.append(
                {"experiment": "X1", "target": target, "mode": "absolute", "method": meth, **st.point_metrics(t)}
            )
            errs[("absolute", meth)] = subject_errors(t)
        # transfer a day-1 cuff calibration (3 readings ≥ 60 s apart, hierarchical) to day 2
        frame = ev.assign(t=ev.t_start_s)
        pol = CalibrationPolicy(min_spacing_s=60.0, allow_cross_session=True)
        scen = {"day1_to_day2": {"k": 3, "reading_session": "D1", "eval_sessions": ("D2",)}}
        sfold = subject_folds(d.subject_id, 5, SEED)
        for meth in ("ridge", "ridge_session_std", "ridge_coral"):
            # OOF predictions exist only for day 2; day-1 predictions for the readings come from the same fold models
            base = frame.assign(pred=_day1_preds(d, cols, y, folds, meth, preds), pmean=np.nan)
            for k in sorted(set(sfold.values())):
                trs = [s for s, f in sfold.items() if f != k]
                base.loc[base.subject_id.map(sfold) == k, "pmean"] = ev[
                    ev.subject_id.isin(trs) & (ev.session_id == "D1")
                ].y.mean()
            P, _ = evaluate(base.dropna(subset=["pred"]), sfold, ["combined", "cuff_hierarchical"], pol, scen)
            for strat, g in P.groupby("strategy"):
                key = "cuff_hierarchical" if strat == "cuff_hierarchical" else f"combined_{meth}"
                if ("calibrated", key) in errs:
                    continue
                t = g.rename(columns={"subject_id": "subject_id"})
                rows.append(
                    {
                        "experiment": "X1",
                        "target": target,
                        "mode": "day1_cuff_transferred",
                        "method": key,
                        **st.point_metrics(t.assign(activity=-1)),
                    }
                )
                errs[("calibrated", key)] = subject_errors(t)
        fam = []
        for a, b, mode in (
            ("ridge", "training_mean", "absolute"),
            ("ridge_session_std", "ridge", "absolute"),
            ("ridge_coral", "ridge", "absolute"),
            ("combined_ridge", "cuff_hierarchical", "calibrated"),
            ("combined_ridge_session_std", "cuff_hierarchical", "calibrated"),
            ("combined_ridge_coral", "cuff_hierarchical", "calibrated"),
        ):
            fam.append(
                {
                    "experiment": "X1",
                    "target": target,
                    "comparison": f"{a} vs {b} ({mode})",
                    **paired_test(errs[(mode, a)], errs[(mode, b)], N_BOOT, seed=SEED),
                }
            )
        for r, p in zip(fam, holm([r["p_perm"] for r in fam]), strict=True):
            r["p_holm"] = p
        paired += fam
        log.info("X1 %s done", target)
    return pd.DataFrame(rows), pd.DataFrame(paired)


def _day1_preds(d, cols, y, folds, meth, oof):
    """Day-2 OOF predictions plus day-1 predictions of the held-out subjects from the same fold's model/transform."""
    out = oof[meth].copy()
    gkey = (d.subject_id + "|" + d.session_id).to_numpy()
    for tr, te in folds:
        test_subj = set(d.loc[te, "subject_id"])
        d1 = np.flatnonzero(d.subject_id.isin(test_subj).to_numpy() & (d.session_id == "D1").to_numpy())
        Xtr, X1 = d.loc[tr, cols].to_numpy(float), d.loc[d1, cols].to_numpy(float)
        med = np.nanmedian(Xtr, 0)
        Xtr, X1 = np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(X1), X1, med)
        m, _ = make_model("ridge", SEED)
        m.set_params(model__alpha=100.0)
        if meth == "ridge_session_std":
            ss = SessionStandardizer()
            m.fit(ss.transform(Xtr, gkey[tr]), y[tr])
            out[d1] = m.predict(ss.transform(X1, gkey[d1]))
        else:
            m.fit(Xtr, y[tr])
            if meth == "ridge_coral":
                coral = CORAL(eps=1.0).fit(Xtr)
                al = np.empty_like(X1)
                for g in np.unique(gkey[d1]):
                    k = gkey[d1] == g
                    al[k] = coral.transform(X1[k])
                X1 = al
            out[d1] = m.predict(X1)
    return out


# ------------------------------------------------------------------------------------------------ X2
def x2_protocols(run: ExperimentRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    d, cols = auricle()
    both = d.groupby("subject_id").session_id.nunique()
    d = d[d.subject_id.isin(both[both == 2].index)].reset_index(drop=True)
    rows, paired = [], []
    for target in ("sbp", "dbp"):
        tab = d.assign(y_sbp=d.sbp_mmHg, y_dbp=d.dbp_mmHg, status="usable")
        errs = {}
        for name, spec in (
            ("subject_independent_all_sessions", SplitSpec("subject", 5, SEED)),
            ("cross_day_unseen_subjects", SplitSpec("cross_day", 5, SEED, ("D1",))),
            ("cross_session_seen_subjects", SplitSpec("session", 5, SEED, ("D1",), same_subjects=True)),
        ):
            for model in ("mean", "ridge"):
                cfg = TrainConfig(
                    name="x2",
                    model=model,
                    features=cols,
                    split=spec,
                    hyperparams={"alpha": 100.0} if model == "ridge" else {},
                )
                _, cv, oof = train_model(tab, cfg)
                t = oof.dropna(subset=[f"p_{target}"])
                tt = pd.DataFrame(
                    {"subject_id": t.subject_id, "y": t[f"y_{target}"], "yhat": t[f"p_{target}"], "activity": -1}
                )
                rows.append(
                    {"experiment": "X2", "target": target, "protocol": name, "model": model, **st.point_metrics(tt)}
                )
                errs[(name, model)] = subject_errors(tt)
        fam = [
            {
                "experiment": "X2",
                "target": target,
                "comparison": f"ridge vs mean ({name})",
                **paired_test(errs[(name, "ridge")], errs[(name, "mean")], N_BOOT, seed=SEED),
            }
            for name in dict.fromkeys(k[0] for k in errs)
        ]
        for r, p in zip(fam, holm([r["p_perm"] for r in fam]), strict=True):
            r["p_holm"] = p
        paired += fam
        log.info("X2 %s done", target)
    return pd.DataFrame(rows), pd.DataFrame(paired)


# ------------------------------------------------------------------------------------------------ X3
def x3_retention(run: ExperimentRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    w = w[w.label_valid & w.ear_sqi_ok & (w.cohort == "HC")]
    ret, exp = [], []
    for target in ("sbp", "dbp"):
        p = pd.read_csv(PRED / f"personalization_pop_ear_ridge_{target}.csv.gz")
        pm = pd.read_csv(PRED / f"personalization_pop_ear_mean_{target}.csv.gz").rename(columns={"pred": "pmean"})
        f = w.merge(p, on="window_id").merge(pm, on="window_id")
        f = f.assign(
            subject_id=f.subject_id.astype(str),
            session_id="D" + f.session.astype(str),
            t=f.t_start_s,
            y=f[f"{target}_mmHg"],
        )[["subject_id", "session_id", "t", "y", "pred", "pmean"]]
        folds = subject_folds(f.subject_id, 5, SEED)
        run.splits.update(folds)
        pol = CalibrationPolicy(min_spacing_s=60.0, exclude_s=10.0, expiry_s=np.inf)
        scen = {"within_session_k3": {"k": 3, "reading_session": "D1", "eval_sessions": ("D1",)}}
        P, fit = evaluate(
            f,
            folds,
            ["cuff_last", "cuff_hierarchical", "combined", "subject_linear", "ppg_only"],
            pol,
            scen,
            retention_bins_s=(0, 300, 600, 1200, 2400, np.inf),
        )
        P["ae"] = (P.yhat - P.y).abs()
        for (strat, b), g in P.groupby(["strategy", "retention_bin_s"]):
            per = g.groupby("subject_id").ae.mean()
            ret.append(
                {
                    "target": target,
                    "strategy": strat,
                    "minutes_since_reading": b,
                    "subject_mae": float(per.mean()),
                    "n_subjects": int(per.size),
                    "n_windows": len(g),
                }
            )
        fm = fit.groupby("strategy").fit_mae_at_readings.mean()
        for strat, v in fm.items():
            ret.append(
                {
                    "target": target,
                    "strategy": strat,
                    "minutes_since_reading": "calibration_fit_at_readings",
                    "subject_mae": float(v),
                    "n_subjects": int(fit[fit.strategy == strat].subject_id.nunique()),
                    "n_windows": np.nan,
                }
            )
        for expiry_min in (5, 10, 20, 40):
            for strat, g in P.groupby("strategy"):
                valid = g.since_reading_s <= expiry_min * 60
                exp.append(
                    {
                        "target": target,
                        "strategy": strat,
                        "expiry_min": expiry_min,
                        "share_windows_valid": float(valid.mean()),
                        "mae_valid": float(g.ae[valid].mean()) if valid.any() else np.nan,
                        "mae_expired": float(g.ae[~valid].mean()) if (~valid).any() else np.nan,
                    }
                )
        log.info("X3 %s done", target)
    return pd.DataFrame(ret), pd.DataFrame(exp)


# ------------------------------------------------------------------------------------------------ X4 (simulated)
PERTURBATIONS = {
    "baseline": {},
    "contact_perfusion_x0.5": {"perfusion_ir": 0.006, "perfusion_red": 0.0035},
    "contact_perfusion_x2": {"perfusion_ir": 0.024, "perfusion_red": 0.014},
    "adc_gain_dc_x1.5": {"dc_ir": 180_000.0, "dc_red": 135_000.0},
    "baseline_wander_x5": {"resp_baseline_frac": 0.01},
    "sensor_noise_x4": {"noise_frac": 0.0016},
    "reflection_0.30": {"reflection": 0.30},
    "reflection_0.60": {"reflection": 0.60},
}


def x4_sensitivity() -> pd.DataFrame:
    """SIMULATED: change of each feature under one perturbation, in baseline between-window SDs."""
    base_cfg = dict(seed=21, hr_drift_bpm=0.0, hr_bpm=70.0)
    res = {}
    for name, kw in PERTURBATIONS.items():
        rec = simulate(SimConfig(**{**base_cfg, **kw}), 300)
        w, _ = process_recording(
            pd.DataFrame({"timestamp": rec.t, "sample_index": rec.sample_index, "red": rec.red, "infrared": rec.ir}),
            PipelineConfig(),
        )
        res[name] = w
    cols = ["ir_hr"] + [f"ir_{c}" for c in st.SHAPE_FEATURES]
    b = res["baseline"]
    sd = b[cols].std().replace(0, np.nan)
    rows = []
    for name, w in res.items():
        z = ((w[cols].median() - b[cols].median()) / sd).abs()
        rows.append(
            {
                "perturbation": name,
                "simulated": True,
                "usable_share": float((w.status == "usable").mean()),
                "median_quality_score": float(w.quality_score.median()),
                "median_abs_shift_sd": float(z.median()),
                "max_abs_shift_sd": float(z.max()),
                "most_affected": ";".join(z.sort_values(ascending=False).index[:3]),
                "n_features_shift_gt_2sd": int((z > 2).sum()),
                "n_features": len(cols),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ B
def benchmarks() -> pd.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        store = DatasetStore(tmp)
        src = SimulatedSensor(SimConfig(seed=31), 600)
        t0 = time.perf_counter()
        with store.create_session(SessionMetadata("B", "D1", 100.0, sensor_type="simulator")) as w:
            Recorder(src, [w.write_raw]).run()
        rec_s = time.perf_counter() - t0
        rows.append({"stage": "record_to_csv", "metric": "samples_per_s", "value": 60_000 / rec_s})
        t0 = time.perf_counter()
        store.validate()
        rows.append({"stage": "validate", "metric": "s_per_10min_session", "value": time.perf_counter() - t0})
        t0 = time.perf_counter()
        win, _ = process_recording(store.raw("B", "D1", chunksize=20_000), PipelineConfig(), "B", "D1")
        rows.append(
            {
                "stage": "pipeline_chunked",
                "metric": "ms_per_window",
                "value": (time.perf_counter() - t0) * 1000 / len(win),
            }
        )
        tracemalloc.start()
        process_recording(store.raw("B", "D1", chunksize=20_000), PipelineConfig(), "B", "D1")
        rows.append(
            {
                "stage": "pipeline_chunked",
                "metric": "peak_python_mem_mb",
                "value": tracemalloc.get_traced_memory()[1] / 1e6,
            }
        )
        tracemalloc.stop()
    tab = win.assign(
        y_sbp=120.0 + np.random.default_rng(0).normal(0, 5, len(win)),
        y_dbp=75.0,
        subject_id=[f"B{i % 6}" for i in range(len(win))],
    )
    bundle, _, _ = train_model(tab, TrainConfig(hyperparams={"alpha": 10.0}))
    rec = simulate(SimConfig(seed=32), 300)
    raw = pd.DataFrame({"timestamp": rec.t, "sample_index": rec.sample_index, "red": rec.red, "infrared": rec.ir})
    for hop in (1.0, 5.0):
        eng = InferenceEngine(bundle, EngineConfig(fs=100.0, sensor="unspecified", hop_s=hop))
        for i in range(0, len(raw), 32):
            eng.push(raw.iloc[i : i + 32])
        lat = eng.predictions().query("status != 'warming_up'").latency_ms
        rows += [
            {"stage": f"inference_engine_hop_{hop:g}s", "metric": "latency_ms_p50", "value": float(lat.median())},
            {"stage": f"inference_engine_hop_{hop:g}s", "metric": "latency_ms_p95", "value": float(lat.quantile(0.95))},
        ]
    eng = InferenceEngine(bundle, EngineConfig(fs=100.0, sensor="unspecified", hop_s=5.0))
    tracemalloc.start()
    for i in range(0, len(raw), 32):
        eng.push(raw.iloc[i : i + 32])
    rows.append(
        {
            "stage": "inference_engine_hop_5s",
            "metric": "peak_python_mem_mb",
            "value": tracemalloc.get_traced_memory()[1] / 1e6,
        }
    )
    tracemalloc.stop()
    out = pd.DataFrame(rows)
    out["note"] = "desktop CPU, simulated input; not an embedded measurement"
    return out


def figures(ret: pd.DataFrame, x1: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["[0.0, 300.0)", "[300.0, 600.0)", "[600.0, 1200.0)", "[1200.0, 2400.0)", "[2400.0, inf)"]
    labels = ["0–5", "5–10", "10–20", "20–40", "40+"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, target in zip(ax, ("sbp", "dbp"), strict=True):
        r = ret[(ret.target == target) & ret.minutes_since_reading.isin(order)]
        for strat, g in r.groupby("strategy"):
            g = g.set_index("minutes_since_reading").reindex(order)
            a.plot(labels, g.subject_mae, marker="o", label=strat)
        a.set(
            xlabel="minutes since the last cuff reading",
            ylabel="subject MAE (mmHg)",
            title=f"{target.upper()}: calibration retention within a session (Auricle ear)",
        )
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v5_calibration_retention.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    m = x1[x1.target == "sbp"]
    ax.barh(
        m["mode"] + ": " + m.method, m.mae, color=["tab:gray" if x == "absolute" else "tab:blue" for x in m["mode"]]
    )
    ax.set(xlabel="window MAE (mmHg)", title="SBP, strict cross-day (day 2 of unseen subjects)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "v5_cross_day_adaptation.png", dpi=130)
    plt.close(fig)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = [AUR / "windows.csv", AUR / "features.csv", *sorted(PRED.glob("personalization_pop_ear_*_*.csv.gz"))]
    ds = [p for p in ds if "ridge" in p.name or "mean" in p.name or p.suffix == ".csv"]
    run = ExperimentRun(
        "v5_cross_day_calibration",
        ds,
        {
            "x1": "strict cross-day ridge alpha=100; session-std / CORAL on unlabelled day-2 features",
            "x2": "subject / cross_day / session-seen protocols",
            "x3": "policy engine on 2.0 OOF",
            "n_boot": N_BOOT,
        },
        seed=SEED,
        dataset_versions={"auricle": "PhysioNet 1.0.0, standardised tables 2.0"},
        feature_config={"features": "ear HR + 27 shape (3.0 hr_shape)"},
        model_config={"model": "ridge alpha=100", "calibration": "personalization.strategies"},
    )
    with run.phase("x1"):
        x1, x1p = x1_domain_adaptation(run)
    with run.phase("x2"):
        x2, x2p = x2_protocols(run)
    with run.phase("x3"):
        ret, exp = x3_retention(run)
    for name, df in (
        ("x1_cross_day_adaptation", x1),
        ("x1_paired", x1p),
        ("x2_protocols", x2),
        ("x2_paired", x2p),
        ("x3_retention", ret),
        ("x3_expiry", exp),
    ):
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    run.add_metrics(x1)
    run.finish()
    sim = ExperimentRun(
        "v5_simulation_and_benchmarks",
        [],
        {"x4": "SIMULATED perturbations, 300 s each, seed 21", "b": "desktop CPU, simulated input"},
        seed=21,
    )
    with sim.phase("x4"):
        x4 = x4_sensitivity()
    with sim.phase("benchmarks"):
        b = benchmarks()
    x4.to_csv(OUT / "x4_simulated_perturbations.csv", index=False, float_format="%.4f")
    b.to_csv(OUT / "benchmarks.csv", index=False, float_format="%.4f")
    sim.add_metrics(x4)
    sim.finish()
    figures(ret, x1)
    print(x1[["target", "mode", "method", "mae", "n_subjects"]].round(2).to_string(index=False))
    print(x1p[["target", "comparison", "mean_diff", "ci_low", "ci_high", "p_holm"]].round(3).to_string(index=False))
    print(
        x2[["target", "protocol", "model", "mae", "within_subject_pearson_median", "n_subjects"]]
        .round(2)
        .to_string(index=False)
    )
    print(ret.round(2).to_string(index=False))
    print(exp.round(2).to_string(index=False))
    print(x4.round(2).to_string(index=False))
    print(b.round(2).to_string(index=False))
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
