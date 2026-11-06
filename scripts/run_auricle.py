"""Finger vs earlobe PPG experiments on Auricle (paired sites, continuous CNAP BP labels).

E1  Calibration-free, subject-independent (GroupKFold by subject): ear vs finger features,
    mean / demographics / ridge / XGBoost.
E2  Cross-site transfer: train on finger PPG, test on earlobe PPG of held-out subjects,
    (a) naively, (b) with unsupervised per-site feature standardisation (statistics from the
    *training* subjects' unlabelled ear windows), against ear→ear as the reference.
E3  Cross-dataset + cross-site: model trained on Guilin (fingertip IR) applied to Auricle.
E4  Personal calibration and BP-change tracking: for each held-out session, the first K valid
    rest windows are used as cuff-style calibration (their CNAP BP). Evaluated on all later
    windows against the "carry the calibration value forward" baseline. Also reports
    within-session correlation of predicted vs true BP *changes*.

Primary analysis: healthy cohort (HC, closest to the target population). The heart-failure
cohort (HFC) is reported separately. Writes results/auricle_experiments.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiments import IR  # noqa: E402

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.evaluation.calibration import calibrated_errors, cross_session_errors, within_session_r  # noqa: E402
from rakshpulse.evaluation.metrics import regression_metrics  # noqa: E402
from rakshpulse.features.feature_pipeline import PER_CHANNEL  # noqa: E402
from rakshpulse.models.baseline import DEMOGRAPHIC_FEATURES  # noqa: E402
from rakshpulse.models.model_factory import make_model  # noqa: E402

RES = ROOT / "results"
TARGETS = ("sbp", "dbp")
K_CAL = 3  # calibration windows (~20 s of rest)


def site_cols(site: str) -> list[str]:
    return [f"{site}_{n}" for n in PER_CHANNEL]


def fit(model: str, X, y, groups):
    est, grid = make_model(model, 0)
    if model == "ridge":  # cheap inner tuning; tree models use fixed defaults here
        from sklearn.model_selection import GridSearchCV

        est = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="neg_mean_absolute_error", n_jobs=-1)
        return est.fit(X, y, groups=groups).best_estimator_
    if model == "xgboost":
        est.set_params(model__max_depth=3, model__learning_rate=0.05)
    return est.fit(X, y)


def folds(df: pd.DataFrame, n=5):
    subj = df.groupby("subject_id")["sbp"].mean().sort_values()
    order = {s: i % n for i, s in enumerate(subj.index)}  # interleaved by mean SBP -> balanced folds
    f = df.subject_id.map(order).to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n)]


def summarise(d: pd.DataFrame, target: str) -> dict:
    """Window-level metrics, subject-level (mean over windows) metrics, within-session tracking r."""
    w = regression_metrics(d[target], d.pred)
    s = d.groupby("subject_id")[[target, "pred"]].mean()
    sm = regression_metrics(s[target], s.pred)
    return {
        "window_mae": w["mae"],
        "window_me": w["mean_error"],
        "window_sd": w["sd_error"],
        "window_r": w["pearson_r"],
        "subject_mae": sm["mae"],
        "subject_r": sm["pearson_r"],
        "within_session_r_median": within_session_r(d, target),
        "n_windows": w["n"],
        "n_subjects": sm["n"],
    }


def gate(df: pd.DataFrame, *sites: str) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for s in sites:
        m &= df[f"{s}_sqi_ok"].astype(bool).to_numpy()
    return df[m].reset_index(drop=True)


def cv_predict(df, train_cols, test_cols, target, model, adapt=False):
    pred = np.full(len(df), np.nan)
    for tr, te in folds(df):
        Xtr = df.loc[tr, train_cols].to_numpy(float)
        Xte = df.loc[te, test_cols].to_numpy(float)
        if adapt:  # per-site standardisation with *training-subject* statistics only
            mu_s, sd_s = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
            tgt = df.loc[tr, test_cols].to_numpy(float)  # unlabelled target-site windows, train subjects
            mu_t, sd_t = np.nanmean(tgt, 0), np.nanstd(tgt, 0) + 1e-9
            Xte = (Xte - mu_t) / sd_t * sd_s + mu_s
        est = fit(model, Xtr, df.loc[tr, target].to_numpy(float), df.loc[tr, "subject_id"].to_numpy())
        pred[te] = est.predict(Xte)
    out = df[["subject_id", "record_id", "t_start_s", "activity", target]].copy()
    out["pred"] = pred
    return out


def tracking_figure() -> None:
    """Calibrated finger vs ear SBP tracking for the session with the largest BP excursion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = pd.read_csv(RES / "predictions" / "auricle_HC_finger_ridge_sbp.csv")
    e = pd.read_csv(RES / "predictions" / "auricle_HC_ear_ridge_sbp.csv")
    rid = f.groupby("record_id").sbp.agg(np.ptp).idxmax()
    fig, ax = plt.subplots(figsize=(11, 4))
    for d, name, col in ((f, "finger PPG", "C0"), (e, "earlobe PPG", "C3")):
        g = d[d.record_id == rid].sort_values("t_start_s")
        cal = g[g.activity == 0].iloc[:K_CAL]
        ax.plot(
            g.t_start_s,
            g.pred - cal.pred.mean() + cal.sbp.mean(),
            ".",
            color=col,
            ms=4,
            label=f"{name} (ridge, 1-point calibrated)",
        )
    g = f[f.record_id == rid].sort_values("t_start_s")
    ax.plot(g.t_start_s, g.sbp, "k-", lw=1.2, label="CNAP reference SBP")
    for code, name in ((1, "breath-hold"), (2, "Valsalva"), (3, "exercise"), (4, "cold pressor")):
        t = g.t_start_s[g.activity == code]
        if len(t):
            ax.axvspan(t.min(), t.max() + 10, alpha=0.08, color="grey")
            ax.text(t.min(), ax.get_ylim()[1], name, fontsize=7, va="top")
    ax.set_xlabel("time in session (s)")
    ax.set_ylabel("SBP (mmHg)")
    ax.set_title(f"Auricle {rid}: calibrated SBP tracking, finger vs earlobe (held-out subject)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "auricle_tracking_example.png", dpi=110)
    plt.close(fig)


def main() -> int:
    if "--figure-only" in sys.argv:
        tracking_figure()
        return 0
    t0 = time.time()
    a = pd.read_csv(PROCESSED / "auricle_features.csv")
    a = a[a.session == 1].reset_index(drop=True)  # session-1 analysis; cross-day lives in run_cross_day.py
    out: dict = {"meta": {}, "E1": [], "E2": [], "E3": [], "E4": []}
    out["meta"] = {
        "records": int(a.record_id.nunique()),
        "subjects": int(a.subject_id.nunique()),
        "windows": len(a),
        "by_cohort_subjects": a.groupby("cohort").subject_id.nunique().to_dict(),
        "ear_sqi_pass": float(a.ear_sqi_ok.mean()),
        "finger_sqi_pass": float(a.finger_sqi_ok.mean()),
        "sqi_pass_by_activity": a.groupby("activity")[["ear_sqi_ok", "finger_sqi_ok"]].mean().round(3).to_dict(),
        "sbp_within_subject_sd_median": float(a.groupby("subject_id").sbp.std().median()),
        "sbp_between_subject_sd": float(a.groupby("subject_id").sbp.mean().std()),
        "k_cal": K_CAL,
    }
    print(json.dumps(out["meta"], default=str), flush=True)
    for cohort in ("HC", "ALL"):
        base = a if cohort == "ALL" else a[a.cohort == cohort]
        both = gate(base, "ear", "finger")  # identical windows for a fair site comparison
        for target in TARGETS:
            # ---------- E1 within-site
            for site in ("finger", "ear"):
                for model, cols in (
                    ("mean", site_cols(site)),
                    ("ridge_demo_only", DEMOGRAPHIC_FEATURES),
                    ("ridge_hr_only", [f"{site}_hr"]),
                    ("ridge", site_cols(site)),
                    ("xgboost", site_cols(site)),
                ):
                    m = "ridge" if model.startswith("ridge_") else model
                    pdf = cv_predict(both, cols, cols, target, m)
                    if model == "ridge":
                        (RES / "predictions").mkdir(exist_ok=True)
                        pdf.assign(session=both["session"].to_numpy()).to_csv(
                            RES / "predictions" / f"auricle_{cohort}_{site}_ridge_{target}.csv", index=False
                        )
                    r = {"cohort": cohort, "target": target, "site": site, "model": model, **summarise(pdf, target)}
                    out["E1"].append(r)
                    print(
                        f"[E1 {cohort}] {target} {site:6s} {model:16s} win MAE {r['window_mae']:.2f} "
                        f"subj MAE {r['subject_mae']:.2f} track r {r['within_session_r_median']:.2f}",
                        flush=True,
                    )
                    if model in ("ridge", "xgboost", "ridge_hr_only"):
                        c = calibrated_errors(pdf.assign(activity=both["activity"].to_numpy()), target, K_CAL)
                        out["E4"].append({"cohort": cohort, "target": target, "site": site, "model": model, **c})
                        x = cross_session_errors(
                            pdf.assign(activity=both["activity"].to_numpy(), session=both["session"].to_numpy()),
                            target,
                            K_CAL,
                        )
                        if x.get("n_subjects"):
                            out.setdefault("E5_cross_day", []).append(
                                {"cohort": cohort, "target": target, "site": site, "model": model, **x}
                            )
                            print(
                                f"[E5 {cohort}] {target} {site:6s} {model:8s} cross-day calibrated MAE "
                                f"{x['calibrated_model_mae']:.2f} vs carry-forward {x['carry_forward_mae']:.2f} "
                                f"(n={x['n_subjects']})",
                                flush=True,
                            )
                        print(
                            f"[E4 {cohort}] {target} {site:6s} {model:8s} "
                            f"calibrated MAE {c['calibrated_model_mae']:.2f} "
                            f"vs carry-forward {c['carry_forward_mae']:.2f}; stressors "
                            f"{c['stressor_calibrated_model_mae']:.2f} vs {c['stressor_carry_forward_mae']:.2f}; "
                            f"Δ-tracking r {c['delta_tracking_r']:.2f}; session ΔMAE {c['session_mae_diff_mean']:+.2f} "
                            f"CI [{c['session_mae_diff_ci95'][0]:+.2f}, {c['session_mae_diff_ci95'][1]:+.2f}]",
                            flush=True,
                        )
            # ---------- E2 finger -> ear transfer
            for model in ("ridge", "xgboost"):
                for adapt in (False, True):
                    pdf = cv_predict(both, site_cols("finger"), site_cols("ear"), target, model, adapt=adapt)
                    r = {
                        "cohort": cohort,
                        "target": target,
                        "model": model,
                        "train": "finger",
                        "test": "ear",
                        "adaptation": "per-site standardisation" if adapt else "none",
                        **summarise(pdf, target),
                    }
                    out["E2"].append(r)
                    print(
                        f"[E2 {cohort}] {target} finger->ear {model:8s} adapt={adapt!s:5s} win MAE "
                        f"{r['window_mae']:.2f} bias {r['window_me']:+.2f} track r {r['within_session_r_median']:.2f}",
                        flush=True,
                    )
    # ---------- E3 Guilin (fingertip IR) -> Auricle finger / ear (healthy cohort)
    g = pd.read_csv(PROCESSED / "guilin_dualwave_features.csv")
    g = g[g.ir_sqi_ok.astype(bool)]
    hc = gate(a[a.cohort == "HC"], "ear", "finger")
    for target in TARGETS:
        for model in ("mean", "ridge", "xgboost"):
            est = fit(model, g[IR].to_numpy(float), g[target].to_numpy(float), g.subject_id.to_numpy())
            for site in ("finger", "ear"):
                pdf = hc[["subject_id", "record_id", "t_start_s", "activity", target]].copy()
                pdf["pred"] = est.predict(hc[site_cols(site)].to_numpy(float))
                r = {
                    "target": target,
                    "model": model,
                    "train": "guilin_ir",
                    "test": f"auricle_HC_{site}",
                    **summarise(pdf, target),
                }
                out["E3"].append(r)
                print(
                    f"[E3] {target} guilin->{site:6s} {model:8s} win MAE {r['window_mae']:.2f} bias "
                    f"{r['window_me']:+.2f} track r {r['within_session_r_median']:.2f}",
                    flush=True,
                )
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    tracking_figure()
    (RES / "auricle_experiments.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
