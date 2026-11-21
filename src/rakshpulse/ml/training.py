"""Model training, selection and evaluation for the complete ML workflow.

Protocol (all splits are by subject, so no person appears in both training and test data):

* **Independent test set:** about 20 % of subjects, chosen deterministically and stratified by their mean SBP.
  They are never touched by feature selection, tuning or model selection.
* **Feature selection** on training subjects only: drop features missing in more than 20 % of windows or constant,
  then drop the later feature of every pair with |Spearman ρ| > 0.95.
* **Hyper-parameter optimisation:** grid search with 5-fold GroupKFold over training subjects, separately for SBP and
  DBP. Each subject contributes at most ``max_fit_windows_per_subject`` non-overlapping windows, so subjects with long
  recordings do not dominate.
* **Selection:** for each target, the candidate with the lowest cross-validated MAE. The test set is not used for
  selection, so its numbers are an unbiased estimate for new people from the same population.
* **Baseline:** predicting the training-set mean. A model is only useful if it beats it on the test set; the
  difference is reported with a subject-level bootstrap 95 % confidence interval.
* **Cross-session:** the selected configuration refitted on training subjects' session 1, scored on test subjects'
  session 2 (new people *and* a different day).
* **Uncertainty:** a 90 % interval half-width from the selected model's out-of-fold residuals (split conformal), with
  its coverage checked on the test set. It is only valid for data like the training data.

Pipelines (winsorise → impute → scale → model) are fitted on training folds only.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

from ..inference.contract import CONTRACT_VERSION, DEFAULT_PROCESSING
from ..inference.model_loader import ModelBundle
from ..models.model_factory import make_model
from .dataset import NORMALISED, SITE

TARGETS = ("sbp", "dbp")
CANDIDATES = ("ridge", "svr", "random_forest", "hist_gb", "xgboost", "lightgbm")
MODEL_LABELS = {
    "mean": "Baseline (training mean)",
    "ridge": "Ridge regression",
    "svr": "Support vector regression (RBF)",
    "random_forest": "Random forest",
    "hist_gb": "Gradient boosting (histogram)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
# Grids searched by GroupKFold CV. Keys use the ``model__`` prefix of the production model factory.
PARAM_GRIDS = {
    "mean": {},
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]},
    "svr": {"model__C": [3.0, 10.0, 30.0], "model__epsilon": [2.0, 5.0], "model__gamma": ["scale", 0.01]},
    "random_forest": {"model__min_samples_leaf": [5, 20, 50], "model__max_features": [0.3, 0.6]},
    "hist_gb": {
        "model__max_depth": [3, None],
        "model__min_samples_leaf": [20, 80],
        "model__learning_rate": [0.03, 0.1],
    },
    "xgboost": {"model__max_depth": [2, 4, 6], "model__learning_rate": [0.03, 0.1]},
    "lightgbm": {
        "model__num_leaves": [7, 31],
        "model__learning_rate": [0.03, 0.1],
        "model__min_child_samples": [30, 100],
    },
}
# Neural networks are not trained here: the 3.0 benchmark (results/deep_benchmark) found that CNN, GRU, CNN-LSTM,
# TCN and Transformer models on raw waveforms did not beat these feature models on the same data.


@dataclass
class PipelineConfig:
    seed: int = 0
    test_fraction: float = 0.2
    cv_folds: int = 5
    models: tuple[str, ...] = CANDIDATES
    max_fit_windows_per_subject: int = 150
    max_missing: float = 0.2
    corr_threshold: float = 0.95
    n_bootstrap: int = 2000
    n_jobs: int = -1


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each feature to quantiles learned on the training data (robust to feature outliers)."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            self.lo_ = np.nanquantile(X, self.lower, axis=0)
            self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        lo = np.where(np.isfinite(self.lo_), self.lo_, -np.inf)
        hi = np.where(np.isfinite(self.hi_), self.hi_, np.inf)
        return np.clip(X, lo, hi)  # NaN stays NaN and is imputed next


def make_pipeline(name: str, seed: int = 0):
    """The production model-factory pipeline (impute → scale → model) with a winsorising step in front."""
    pipe, _ = make_model(name, seed)
    pipe.steps.insert(0, ("clip", QuantileClipper()))
    return pipe


# ------------------------------------------------------------------------------------------------ metrics
def metrics(y, p, subjects=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    sd = float(np.std(e, ddof=1)) if e.size > 1 else np.nan
    out = {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if e.size > 1 else np.nan,
        "bias": float(e.mean()),
        "sd_error": sd,
        "loa_low": float(e.mean() - 1.96 * sd),
        "loa_high": float(e.mean() + 1.96 * sd),
        "n_windows": int(e.size),
    }
    if subjects is not None:
        s = pd.Series(np.abs(e)).groupby(np.asarray(subjects))
        out["n_subjects"] = int(s.ngroups)
        out["subject_mean_mae"] = float(s.mean().mean())
    return out


def bootstrap_mae_difference(y, p_model, p_base, subjects, n: int = 2000, seed: int = 0) -> dict:
    """Subject-level (cluster) bootstrap of MAE(model) − MAE(baseline). Negative = model better."""
    d = pd.DataFrame({"s": subjects, "a": np.abs(np.asarray(p_model) - y), "b": np.abs(np.asarray(p_base) - y)})
    per = d.groupby("s")[["a", "b"]].agg(["sum", "count"])
    sums_a, sums_b, cnt = per[("a", "sum")].to_numpy(), per[("b", "sum")].to_numpy(), per[("a", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cnt), size=(n, len(cnt)))
    diff = (sums_a[idx].sum(1) - sums_b[idx].sum(1)) / cnt[idx].sum(1)
    point = float(d.a.mean() - d.b.mean())
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return {
        "mae_difference": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "model_better": bool(hi < 0),
        "baseline_better": bool(lo > 0),
    }


# ------------------------------------------------------------------------------------------------ splits and selection
def split_subjects(d: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic subject split stratified by mean SBP: subjects ordered by mean SBP, every k-th to test."""
    order = d.groupby("subject_id")["sbp"].mean().sort_values().index.tolist()
    k = max(2, round(1 / test_fraction))
    offset = seed % k
    test = [s for i, s in enumerate(order) if i % k == offset]
    train = [s for s in order if s not in test]
    return sorted(train), sorted(test)


def select_features(train: pd.DataFrame, candidates: list[str], max_missing: float, corr: float) -> pd.DataFrame:
    """Feature selection on training rows only. Priority: heart rate, HR-invariant shape, then the rest."""
    prio = [f"{SITE}_hr"] + [f"{SITE}_{n}" for n in NORMALISED]
    ordered = prio + [c for c in candidates if c not in prio]
    rows, kept = [], []
    miss = train[ordered].isna().mean()
    for c in ordered:
        r = {"feature": c, "missing_rate": round(float(miss[c]), 4)}
        x = train[c]
        if miss[c] > max_missing:
            r.update(selected=False, reason=f"missing in > {max_missing:.0%} of windows")
        elif x.nunique(dropna=True) <= 1 or not np.isfinite(x.std()) or x.std() == 0:
            r.update(selected=False, reason="constant")
        else:
            partner = None
            for k in kept:
                rho = spearmanr(x, train[k], nan_policy="omit").statistic
                if np.isfinite(rho) and abs(rho) > corr:
                    partner = (k, rho)
                    break
            if partner:
                r.update(selected=False, reason=f"|rho|={abs(partner[1]):.3f} with {partner[0]}")
            else:
                r.update(selected=True, reason="")
                kept.append(c)
        rows.append(r)
    return pd.DataFrame(rows)


def _cap(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """At most n non-overlapping windows per subject, spread evenly over the recording."""
    fit = d[d.fit_window]
    parts = []
    for _, g in fit.groupby("subject_id", sort=True):
        if len(g) > n:
            g = g.iloc[np.linspace(0, len(g) - 1, n).round().astype(int)]
        parts.append(g)
    return pd.concat(parts)


def tune(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv, cfg: PipelineConfig):
    """Grid search of one candidate with group-aware CV. Returns (best unfitted-clone-ready estimator, params)."""
    pipe = make_pipeline(name, cfg.seed)
    grid = PARAM_GRIDS.get(name, {})
    if not grid:
        return clone(pipe), {}
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=cfg.n_jobs, error_score="raise")
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, {k.replace("model__", ""): v for k, v in gs.best_params_.items()}


# ------------------------------------------------------------------------------------------------ main entry
def run_training(
    d: pd.DataFrame, features: list[str] | None = None, cfg: PipelineConfig | None = None, log=print
) -> dict:
    """Full protocol on a prepared window table. Returns tables, predictions and the fitted selected models."""
    from .dataset import candidate_features

    cfg = cfg or PipelineConfig()
    d = d.copy()
    train_s, test_s = split_subjects(d, cfg.test_fraction, cfg.seed)
    d["split"] = np.where(d.subject_id.isin(test_s), "test", "train")
    train, test = d[d.split == "train"], d[d.split == "test"]
    sel = select_features(train, features or candidate_features(), cfg.max_missing, cfg.corr_threshold)
    feats = sel.loc[sel.selected, "feature"].tolist()
    hpo = _cap(train, cfg.max_fit_windows_per_subject)
    fit_all = train[train.fit_window]
    log(
        f"subjects: {len(train_s)} train / {len(test_s)} test; windows: {len(train)} train "
        f"({len(hpo)} used for tuning), {len(test)} test; {len(feats)} features selected"
    )
    cv = GroupKFold(n_splits=min(cfg.cv_folds, len(train_s)))
    Xh, gh = hpo[feats].to_numpy(float), hpo.subject_id.to_numpy()
    rows, fitted, oof_store, test_pred = [], {}, {}, []
    for target in TARGETS:
        yh = hpo[target].to_numpy(float)
        for name in ("mean",) + tuple(cfg.models):
            t0 = time.perf_counter()
            best, params = tune(name, Xh, yh, gh, cv, cfg)
            oof = cross_val_predict(clone(best), Xh, yh, cv=cv, groups=gh, n_jobs=cfg.n_jobs)
            model = clone(best).fit(fit_all[feats].to_numpy(float), fit_all[target].to_numpy(float))
            p_train = model.predict(fit_all[feats].to_numpy(float))
            p_test = model.predict(test[feats].to_numpy(float))
            secs = time.perf_counter() - t0
            m_cv = metrics(yh, oof, gh)
            m_tr = metrics(fit_all[target], p_train, fit_all.subject_id)
            m_te = metrics(test[target], p_test, test.subject_id)
            row = {
                "target": target,
                "model": name,
                "label": MODEL_LABELS[name],
                "params": params,
                "fit_seconds": round(secs, 1),
            }
            row.update({f"cv_{k}": v for k, v in m_cv.items()})
            row.update({f"train_{k}": v for k, v in m_tr.items()})
            row.update({f"test_{k}": v for k, v in m_te.items()})
            rows.append(row)
            fitted[(target, name)] = model
            oof_store[(target, name)] = oof
            test_pred.append(
                pd.DataFrame(
                    {
                        "window_id": test.window_id.values,
                        "subject_id": test.subject_id.values,
                        "session": test.session.values,
                        "activity": test.activity.values,
                        "target": target,
                        "model": name,
                        "y": test[target].values,
                        "pred": p_test,
                    }
                )
            )
            log(
                f"  {target.upper()} {name:<14} CV MAE {m_cv['mae']:6.2f}  test MAE {m_te['mae']:6.2f}  "
                f"R² {m_te['r2']:6.3f}  ({secs:.0f} s) {params}"
            )
    comp = pd.DataFrame(rows)
    preds = pd.concat(test_pred, ignore_index=True)

    selection, cross_session, comparisons, models, uncertainty = [], [], [], {}, {}
    for target in TARGETS:
        cand = comp[(comp.target == target) & (comp.model != "mean")]
        best = cand.sort_values("cv_mae").iloc[0]
        name = best.model
        models[target] = fitted[(target, name)]
        base = preds[(preds.target == target) & (preds.model == "mean")]
        mine = preds[(preds.target == target) & (preds.model == name)]
        cmp_ = bootstrap_mae_difference(
            mine.y.values, mine.pred.values, base.pred.values, mine.subject_id.values, cfg.n_bootstrap, cfg.seed
        )
        comparisons.append({"target": target, "model": name, **cmp_})
        # split-conformal 90 % half-width from out-of-fold residuals, coverage on the test set
        hw = float(np.quantile(np.abs(oof_store[(target, name)] - hpo[target].to_numpy(float)), 0.9))
        cover = float(np.mean(np.abs(mine.pred - mine.y) <= hw))
        uncertainty[target] = {"halfwidth_90": hw, "test_coverage": cover}
        selection.append(
            {
                "target": target,
                "model": name,
                "label": best.label,
                "params": best.params,
                "cv_mae": best.cv_mae,
                "test_mae": best.test_mae,
                "test_rmse": best.test_rmse,
                "test_r2": best.test_r2,
                "halfwidth_90": hw,
                "test_coverage_90": cover,
                "beats_baseline_on_test": cmp_["model_better"],
            }
        )
        # cross-session: train subjects' session 1 → test subjects' session 2
        tr1 = train[(train.session == 1) & train.fit_window]
        te2 = test[test.session == 2]
        if len(tr1) and len(te2):
            for nm in ("mean", name):
                m = clone(fitted[(target, nm)]).fit(tr1[feats].to_numpy(float), tr1[target].to_numpy(float))
                p = m.predict(te2[feats].to_numpy(float))
                cross_session.append(
                    {
                        "target": target,
                        "model": nm,
                        "train": "train subjects, session 1",
                        "test": "test subjects, session 2",
                        **metrics(te2[target], p, te2.subject_id),
                    }
                )
    return {
        "config": asdict(cfg),
        "features": feats,
        "feature_selection": sel,
        "train_subjects": train_s,
        "test_subjects": test_s,
        "data": d,
        "comparison": comp,
        "selection": pd.DataFrame(selection),
        "baseline_comparison": pd.DataFrame(comparisons),
        "cross_session": pd.DataFrame(cross_session),
        "test_predictions": preds,
        "models": models,
        "uncertainty": uncertainty,
        "training_rows": fit_all,
    }


# ------------------------------------------------------------------------------------------------ export
def export_bundle(result: dict, version: str, synthetic: bool = False, calibration: dict | None = None) -> ModelBundle:
    """Wrap the selected SBP and DBP pipelines in a contract-carrying bundle for the production inference engine.

    Mode: ``experimental`` for Auricle earlobe data (the population it was trained and tested on);
    ``demonstration`` for MAX30102 and the simulator, because it has never seen MAX30102 recordings."""
    feats, fit_all = result["features"], result["training_rows"]
    sel = result["selection"].set_index("target")
    fills = [float(v) for v in result["models"]["sbp"].named_steps["impute"].statistics_]
    nan_rate = fit_all[feats].isna().mean()
    spec = [
        {
            "name": n,
            "source": "ir_" + n[len(SITE) + 1 :],
            "kind": "signal",
            "required": bool(nan_rate[n] == 0),
            "fill": fill,
            "training_missing_rate": float(nan_rate[n]),
        }
        for n, fill in zip(feats, fills, strict=True)
    ]
    per_window_missing = fit_all[feats].isna().mean(axis=1)
    test_perf = {f"{t}_{k}": float(sel.loc[t, f"test_{k}"]) for t in TARGETS for k in ("mae", "rmse", "r2")}
    base = result["comparison"].query("model == 'mean'").set_index("target")
    test_perf.update({f"{t}_baseline_mae": float(base.loc[t, "test_mae"]) for t in TARGETS})
    limitations = [
        "trained and tested on Auricle earlobe transmissive PPG (Nonin) with CNAP labels, healthy adults",
        "never tested on MAX30102 recordings: on MAX30102 input it runs in demonstration mode, not a BP measurement",
        "uncertainty interval only valid on Auricle earlobe data",
    ]
    if synthetic:
        limitations.insert(0, "SMOKE TEST: trained on SYNTHETIC data; demonstrates the software only")
    contract = {
        "contract_version": CONTRACT_VERSION,
        "model_id": "rakshpulse-ear-bp" + ("-smoke-test" if synthetic else ""),
        "features": spec,
        "channels": ["ir"],
        "demographics": [],
        "fs_range_hz": [50.0, 1000.0],
        "processing": dict(DEFAULT_PROCESSING),
        "max_imputed_fraction": float(max(0.1, np.quantile(per_window_missing, 0.99))),
        "validated_for": [],
        "experimental_for": [] if synthetic else ["auricle_earlobe"],
        "demonstration_for": ["auricle_earlobe", "max30102", "simulator"] if synthetic else ["max30102", "simulator"],
        "uncertainty": {
            "method": "subject_wise_cv_conformal_90",
            "halfwidth_90": {t: result["uncertainty"][t]["halfwidth_90"] for t in TARGETS},
            "valid_for": [] if synthetic else ["auricle_earlobe"],
            "test_coverage": {t: result["uncertainty"][t]["test_coverage"] for t in TARGETS},
        },
        "calibration": _calibration_contract(calibration, synthetic),
        "performance": test_perf,
        "selected_models": {t: {"model": sel.loc[t, "model"], "params": sel.loc[t, "params"]} for t in TARGETS},
        "limitations": limitations,
    }
    validation = {
        t: {
            "mae": float(sel.loc[t, "test_mae"]),
            "rmse": float(sel.loc[t, "test_rmse"]),
            "r2": float(sel.loc[t, "test_r2"]),
        }
        for t in TARGETS
    }
    validation.update(
        n_subjects_train=len(result["train_subjects"]),
        n_subjects_test=len(result["test_subjects"]),
        protocol="subject-wise held-out test set; GroupKFold CV for tuning and selection",
    )
    extras = {"calibration_models": dict(calibration["models"])} if calibration else {}
    return ModelBundle(
        extras=extras,
        sbp=result["models"]["sbp"],
        dbp=result["models"]["dbp"],
        features=list(feats),
        channels=("ir",),
        uses_demographics=False,
        validation=validation,
        version=version,
        trained_on="SYNTHETIC smoke-test windows" if synthetic else "Auricle earlobe PPG windows (data/ml)",
        contract=contract,
    )


def _calibration_contract(cal: dict | None, synthetic: bool) -> dict:
    """Describe the optional per-person calibration (``BPEstimator.calibrate``) and its measured test performance."""
    if not cal:
        return {"supported": False}
    summ = cal["summary"]
    perf = {}
    for _, r in summ[summ.method.isin(["delta_ridge", "carry_forward"])].iterrows():
        perf.setdefault(r.scenario, {})[f"{r.target}_{r.method}_mae"] = float(r.mae)
    return {
        "supported": True,
        "method": "delta_ridge",
        "description": "cuff reading + ridge model of the BP change predicted from the PPG feature change since the "
        "reading (trained on training subjects; one reading per recording, scored >= "
        f"{cal['gap_s']:.0f} s later)",
        "valid_for": [] if synthetic else ["auricle_earlobe"],
        "test_performance": perf,
        "params": cal["delta_params"],
    }
