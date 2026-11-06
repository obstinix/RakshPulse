"""Run the feature-based BP experiments and write results/.

Experiments
-----------
A. Model comparison (Guilin, nested subject-wise 5-fold CV): baselines, linear, ridge, SVR,
   random forest and XGBoost on (i) PPG-only red+IR features and (ii) PPG + demographics.
B. Channel ablation: IR vs red vs red+IR (ridge, XGBoost), 3 CV seeds.
C. External generalisation: train on all of Guilin, test on Foggia (red+IR) and PPG-BP (IR).

Every model sees identical outer folds (one subject→fold map per seed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.evaluation import plots  # noqa: E402
from rakshpulse.evaluation.metrics import bhs_grade, by_range, iso81060_like, regression_metrics  # noqa: E402
from rakshpulse.evaluation.validation import nested_cv, outer_splits, subject_level  # noqa: E402
from rakshpulse.features.feature_pipeline import CROSS_FEATURES, PER_CHANNEL  # noqa: E402
from rakshpulse.models.baseline import DEMOGRAPHIC_FEATURES  # noqa: E402
from rakshpulse.models.model_factory import FEATURE_MODELS, make_model  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
TARGETS = ("sbp", "dbp")

IR = [f"ir_{n}" for n in PER_CHANNEL]
RED = [f"red_{n}" for n in PER_CHANNEL]
FEATURE_SETS = {
    "demographics": DEMOGRAPHIC_FEATURES,
    "ppg_ir": IR,
    "ppg_red": RED,
    "ppg_ir_red": IR + RED + list(CROSS_FEATURES),
    "ppg_ir_red+demo": IR + RED + list(CROSS_FEATURES) + DEMOGRAPHIC_FEATURES,
}
# which SQI flags must pass for a window to be used with a feature set
SQI_REQ = {
    "demographics": ["ir_sqi_ok"],
    "ppg_ir": ["ir_sqi_ok"],
    "ppg_red": ["red_sqi_ok"],
    "ppg_ir_red": ["ir_sqi_ok", "red_sqi_ok"],
    "ppg_ir_red+demo": ["ir_sqi_ok", "red_sqi_ok"],
}


def gate(df: pd.DataFrame, fs_name: str) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for c in SQI_REQ[fs_name]:
        m &= df[c].astype(bool).to_numpy()
    return df[m].reset_index(drop=True)


def fold_map(df: pd.DataFrame, seed: int) -> dict[str, int]:
    subj = df.groupby("subject_id").first().reset_index()
    sp = outer_splits(subj, 5, seed)
    return {s: k for k, (_, te) in enumerate(sp) for s in subj.loc[te, "subject_id"]}


def splits_from_map(df: pd.DataFrame, fmap: dict[str, int]):
    f = df["subject_id"].map(fmap).to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]


def evaluate(pred_df: pd.DataFrame, target: str) -> dict:
    s = subject_level(pred_df, target)
    ms = regression_metrics(s.y, s.pred)
    mw = regression_metrics(pred_df[target], pred_df.pred)
    return {
        "subject": ms,
        "window": mw,
        "iso81060_like_subject": iso81060_like(ms),
        "bhs_subject": bhs_grade(ms),
        "by_range_subject": by_range(s.y, s.pred, target),
    }


def main() -> int:
    RES.mkdir(exist_ok=True)
    (RES / "predictions").mkdir(exist_ok=True)
    g = pd.read_csv(PROCESSED / "guilin_dualwave_features.csv")
    log: dict = {"A_model_comparison": [], "B_channel_ablation": [], "C_external": [], "meta": {}}
    t0 = time.time()
    fmaps = {s: fold_map(g, s) for s in (0, 1, 2)}

    # ---------------- A: model comparison
    for fs_name in ("demographics", "ppg_ir_red", "ppg_ir_red+demo"):
        d = gate(g, fs_name)
        models = ("mean", "ridge", "xgboost") if fs_name == "demographics" else FEATURE_MODELS
        for model in models:
            for target in TARGETS:
                pdf, params = nested_cv(
                    d, FEATURE_SETS[fs_name], target, model, seed=0, splits=splits_from_map(d, fmaps[0])
                )
                ev = evaluate(pdf, target)
                pdf.to_csv(RES / "predictions" / f"A_{fs_name}_{model}_{target}.csv", index=False)
                row = {
                    "feature_set": fs_name,
                    "model": model,
                    "target": target,
                    "n_subjects": ev["subject"]["n"],
                    "n_windows": ev["window"]["n"],
                    **{f"subj_{k}": v for k, v in ev["subject"].items() if k != "n"},
                    "win_mae": ev["window"]["mae"],
                    "bhs_subject": ev["bhs_subject"],
                    "iso_like_subject": ev["iso81060_like_subject"],
                    "best_params": params,
                    "by_range_subject": ev["by_range_subject"],
                }
                log["A_model_comparison"].append(row)
                print(
                    f"[A] {fs_name:16s} {model:14s} {target}: subj MAE {row['subj_mae']:.2f} "
                    f"ME {row['subj_mean_error']:+.2f} SD {row['subj_sd_error']:.2f} r {row['subj_pearson_r']:.2f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    # ---------------- B: channel ablation (3 seeds)
    for fs_name in ("ppg_ir", "ppg_red", "ppg_ir_red"):
        d = gate(g, fs_name)
        for model in ("ridge", "xgboost"):
            for target in TARGETS:
                maes, rs = [], []
                for seed in (0, 1, 2):
                    pdf, _ = nested_cv(
                        d, FEATURE_SETS[fs_name], target, model, seed=seed, splits=splits_from_map(d, fmaps[seed])
                    )
                    ev = evaluate(pdf, target)
                    maes.append(ev["subject"]["mae"])
                    rs.append(ev["subject"]["pearson_r"])
                row = {
                    "feature_set": fs_name,
                    "model": model,
                    "target": target,
                    "subj_mae_mean": float(np.mean(maes)),
                    "subj_mae_sd": float(np.std(maes)),
                    "subj_r_mean": float(np.mean(rs)),
                    "seeds": [0, 1, 2],
                }
                log["B_channel_ablation"].append(row)
                print(
                    f"[B] {fs_name:10s} {model:8s} {target}: MAE {row['subj_mae_mean']:.2f}±{row['subj_mae_sd']:.2f} "
                    f"r {row['subj_r_mean']:.2f}",
                    flush=True,
                )

    # ---------------- C: external generalisation
    fog = pd.read_csv(PROCESSED / "foggia_multiwave_features.csv")
    ppgbp = pd.read_csv(PROCESSED / "ppg_bp_features.csv")
    # PPG-BP segments are 2.1 s: require >=1 complete beat and a cardiac-dominant spectrum
    # instead of the 4-beat rule (documented relaxation, used only for this external test).
    ppgbp = ppgbp[(ppgbp.ir_n_beats >= 1) & (ppgbp.ir_sqi_cardiac_power >= 0.35)].reset_index(drop=True)
    ext_sets = {
        # ratio_of_ratios needs the DC level, which Foggia/PPG-BP do not provide -> excluded
        "foggia_multiwave": (
            fog[fog.ir_sqi_ok & fog.red_sqi_ok],
            IR + RED + ["red_ir_corr"],
            ["ir_sqi_ok", "red_sqi_ok"],
        ),
        "ppg_bp": (ppgbp, IR, ["ir_sqi_ok"]),
    }
    for ext_name, (ext_df, feats, req) in ext_sets.items():
        train = g.copy()
        for c in req:
            train = train[train[c].astype(bool)]
        for model in ("mean", "ridge", "random_forest", "xgboost"):
            for target in TARGETS:
                est, grid = make_model(model, 0)
                if grid:  # tune on Guilin only (subject-grouped), never on the external set
                    from sklearn.model_selection import GridSearchCV, GroupKFold

                    est = (
                        GridSearchCV(est, grid, cv=GroupKFold(3), scoring="neg_mean_absolute_error", n_jobs=-1)
                        .fit(
                            train[feats].to_numpy(float),
                            train[target].to_numpy(float),
                            groups=train["subject_id"].to_numpy(),
                        )
                        .best_estimator_
                    )
                else:
                    est.fit(train[feats].to_numpy(float), train[target].to_numpy(float))
                pdf = ext_df[["subject_id", "record_id", "t_start_s", target]].copy()
                pdf["pred"] = est.predict(ext_df[feats].to_numpy(float))
                pdf["fold"] = -1
                ev = evaluate(pdf, target)
                pdf.to_csv(RES / "predictions" / f"C_{ext_name}_{model}_{target}.csv", index=False)
                row = {
                    "train": "guilin_dualwave",
                    "test": ext_name,
                    "model": model,
                    "target": target,
                    "n_subjects": ev["subject"]["n"],
                    **{f"subj_{k}": v for k, v in ev["subject"].items() if k != "n"},
                }
                log["C_external"].append(row)
                print(
                    f"[C] {ext_name:16s} {model:14s} {target}: MAE {row['subj_mae']:.2f} "
                    f"ME {row['subj_mean_error']:+.2f} r {row['subj_pearson_r']:.2f}",
                    flush=True,
                )

    log["meta"] = {"runtime_s": round(time.time() - t0, 1), "outer_folds": 5, "inner_folds": 3, "seeds": [0, 1, 2]}
    (RES / "feature_experiments.json").write_text(json.dumps(log, indent=2, default=float))

    # ---------------- figures for the best PPG-only model per target
    A = pd.DataFrame(log["A_model_comparison"])
    for target in TARGETS:
        sub = A[(A.feature_set == "ppg_ir_red") & (A.target == target)].sort_values("subj_mae")
        best = sub.iloc[0]
        pdf = pd.read_csv(RES / "predictions" / f"A_ppg_ir_red_{best.model}_{target}.csv")
        s = subject_level(pdf, target)
        plots.bland_altman(s.y, s.pred, FIG / f"bland_altman_{target}_{best.model}.png", target.upper())
        plots.pred_vs_true(s.y, s.pred, FIG / f"pred_vs_true_{target}_{best.model}.png", target.upper())
        mean_pdf = pd.read_csv(RES / "predictions" / f"A_ppg_ir_red_mean_{target}.csv")
        sm = subject_level(mean_pdf, target)
        plots.error_hist(
            {f"{best.model} (PPG)": s.pred - s.y, "mean baseline": sm.pred - sm.y},
            FIG / f"error_hist_{target}.png",
            target.upper(),
        )
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
