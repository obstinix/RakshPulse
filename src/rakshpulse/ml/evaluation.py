"""Robustness and calibration analyses on top of the held-out-test protocol in ``training.py``.

* :func:`nested_cv`: nested subject-wise cross-validation over **all** subjects. Every subject is predicted once by a
  model whose features, hyper-parameters and model family were chosen without that subject (inner GroupKFold on the
  outer-training subjects). This gives an estimate over 40 people instead of 8 and per-subject errors.
* :func:`calibration_study`: per-person cuff calibration, the realistic use case. In each recording the first
  labelled rest window plays the role of one cuff reading; windows at least ``gap_s`` seconds later are scored.
  Methods (all fitted on training subjects only, scored on the held-out test subjects):

  - ``population``: the uncalibrated population model;
  - ``carry_forward``: report the calibration reading (cuff only, no PPG) — the baseline every PPG method must beat;
  - ``offset``: population model + (cuff − model) at the calibration window;
  - ``delta_ridge``: a ridge model of the BP *change* since calibration from the feature *change* since calibration.

  Within-session (calibration and scoring in the same recording) and cross-session (calibration in session 1,
  scoring in session 2 of the same person) are reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_score

from .training import (
    TARGETS,
    PipelineConfig,
    _cap,
    bootstrap_mae_difference,
    metrics,
    select_features,
    tune,
)


# ------------------------------------------------------------------------------------------------ nested CV
def _folds(d: pd.DataFrame, k: int) -> dict[str, int]:
    """Subjects ordered by mean SBP and dealt round-robin into k outer folds (balanced BP levels)."""
    order = d.groupby("subject_id")["sbp"].mean().sort_values().index
    return {s: i % k for i, s in enumerate(order)}


def nested_cv(
    d: pd.DataFrame, candidates: list[str], cfg: PipelineConfig | None = None, outer_folds: int = 5, log=print
) -> dict:
    """Outer subject folds → inner feature selection, tuning and model choice → out-of-fold predictions."""
    cfg = cfg or PipelineConfig()
    fold = _folds(d, outer_folds)
    d = d.assign(outer_fold=d.subject_id.map(fold))
    rows, choices = [], []
    for k in range(outer_folds):
        tr, te = d[d.outer_fold != k], d[d.outer_fold == k]
        sel = select_features(tr, candidates, cfg.max_missing, cfg.corr_threshold)
        feats = sel.loc[sel.selected, "feature"].tolist()
        hpo, fit_all = _cap(tr, cfg.max_fit_windows_per_subject), tr[tr.fit_window]
        cv = GroupKFold(n_splits=min(cfg.cv_folds, hpo.subject_id.nunique()))
        X, g = hpo[feats].to_numpy(float), hpo.subject_id.to_numpy()
        out = te[["window_id", "subject_id", "session", "activity", "sbp", "dbp"]].copy()
        out["outer_fold"] = k
        for t in TARGETS:
            y = hpo[t].to_numpy(float)
            best_name, best_score, best_est, best_params = None, np.inf, None, {}
            for name in cfg.models:
                est, params = tune(name, X, y, g, cv, cfg)
                score = -cross_val_score(
                    clone(est), X, y, groups=g, cv=cv, scoring="neg_mean_absolute_error", n_jobs=cfg.n_jobs
                ).mean()
                if score < best_score:
                    best_name, best_score, best_est, best_params = name, score, est, params
            model = clone(best_est).fit(fit_all[feats].to_numpy(float), fit_all[t].to_numpy(float))
            out[f"pred_{t}"] = model.predict(te[feats].to_numpy(float))
            out[f"base_{t}"] = float(fit_all[t].mean())
            choices.append(
                {
                    "outer_fold": k,
                    "target": t,
                    "model": best_name,
                    "params": best_params,
                    "inner_cv_mae": best_score,
                    "n_features": len(feats),
                    "test_subjects": te.subject_id.nunique(),
                }
            )
            log(
                f"  outer fold {k}: {t.upper()} → {best_name} (inner CV MAE {best_score:.2f}) "
                f"outer MAE {np.mean(np.abs(out[f'pred_{t}'] - te[t])):.2f}"
            )
        rows.append(out)
    pred = pd.concat(rows, ignore_index=True)
    summary, per_subject = [], []
    for t in TARGETS:
        m, b = (
            metrics(pred[t], pred[f"pred_{t}"], pred.subject_id),
            metrics(pred[t], pred[f"base_{t}"], pred.subject_id),
        )
        c = bootstrap_mae_difference(
            pred[t].to_numpy(),
            pred[f"pred_{t}"].to_numpy(),
            pred[f"base_{t}"].to_numpy(),
            pred.subject_id.to_numpy(),
            cfg.n_bootstrap,
            cfg.seed,
        )
        summary += [{"target": t, "model": "nested-selected", **m, **c}, {"target": t, "model": "mean", **b}]
        g = pred.assign(
            e_m=(pred[f"pred_{t}"] - pred[t]).abs(),
            e_b=(pred[f"base_{t}"] - pred[t]).abs(),
            bias=pred[f"pred_{t}"] - pred[t],
        ).groupby("subject_id")
        ps = g.agg(
            windows=("e_m", "size"),
            mae_model=("e_m", "mean"),
            mae_baseline=("e_b", "mean"),
            bias_model=("bias", "mean"),
            mean_reference=(t, "mean"),
        ).reset_index()
        ps["target"] = t
        per_subject.append(ps)
    return {
        "predictions": pred,
        "summary": pd.DataFrame(summary),
        "choices": pd.DataFrame(choices),
        "per_subject": pd.concat(per_subject, ignore_index=True),
    }


# ------------------------------------------------------------------------------------------------ calibration
def _calibration_points(d: pd.DataFrame) -> pd.DataFrame:
    """One calibration window per recording: the first labelled rest window (else the first window)."""
    d = d.sort_values(["record_id", "t_start_s"])
    rest = d[d.activity == "rest"].groupby("record_id").head(1)
    first = d.groupby("record_id").head(1)
    cal = pd.concat([rest, first[~first.record_id.isin(rest.record_id)]])
    return cal.set_index("record_id")


def calibration_study(
    d: pd.DataFrame,
    train_s: list[str],
    test_s: list[str],
    population: dict,
    feats: list[str],
    cfg: PipelineConfig | None = None,
    gap_s: float = 60.0,
) -> dict:
    """See the module docstring. ``population``: fitted population pipelines per target (from ``run_training``)."""
    cfg = cfg or PipelineConfig()
    cal = _calibration_points(d)
    d = d.join(cal[["t_start_s"] + feats + list(TARGETS)].add_prefix("cal_"), on="record_id")
    # delta model on training subjects: change since calibration, same recording, after the gap
    tr = d[d.subject_id.isin(train_s) & d.fit_window & (d.t_start_s >= d.cal_t_start_s + gap_s)]
    Xd = tr[feats].to_numpy(float) - tr[[f"cal_{f}" for f in feats]].to_numpy(float)
    g = tr.subject_id.to_numpy()
    cv = GroupKFold(n_splits=min(cfg.cv_folds, len(set(g))))
    delta = {}
    for t in TARGETS:
        est, params = tune("ridge", Xd, (tr[t] - tr[f"cal_{t}"]).to_numpy(float), g, cv, cfg)
        delta[t] = (clone(est).fit(Xd, (tr[t] - tr[f"cal_{t}"]).to_numpy(float)), params)

    te = d[d.subject_id.isin(test_s)].copy()
    rows, scenarios = [], {}
    within = te[te.t_start_s >= te.cal_t_start_s + gap_s].copy()
    scenarios["within-session"] = within
    # cross-session: session-1 calibration applied to the same subject's session-2 windows
    s1 = cal[cal.session == 1].reset_index().set_index("subject_id")
    cross = te[te.session == 2].drop(columns=[c for c in te if c.startswith("cal_")])
    cross = cross[cross.subject_id.isin(s1.index)]
    cross = cross.join(s1[["t_start_s"] + feats + list(TARGETS)].add_prefix("cal_"), on="subject_id")
    scenarios["cross-session (S1 → S2)"] = cross
    preds = []
    for scen, s in scenarios.items():
        if s.empty:
            continue
        Xs, Xc = s[feats].to_numpy(float), s[[f"cal_{f}" for f in feats]].to_numpy(float)
        for t in TARGETS:
            pop = population[t].predict(Xs)
            pop_cal = population[t].predict(Xc)
            methods = {
                "population": pop,
                "carry_forward": s[f"cal_{t}"].to_numpy(float),
                "offset": pop + (s[f"cal_{t}"].to_numpy(float) - pop_cal),
                "delta_ridge": s[f"cal_{t}"].to_numpy(float) + delta[t][0].predict(Xs - Xc),
            }
            y = s[t].to_numpy(float)
            for name, p in methods.items():
                r = {"scenario": scen, "target": t, "method": name, **metrics(y, p, s.subject_id)}
                if name != "carry_forward":
                    r.update(
                        bootstrap_mae_difference(
                            y, p, methods["carry_forward"], s.subject_id.to_numpy(), cfg.n_bootstrap, cfg.seed
                        )
                    )
                rows.append(r)
                preds.append(
                    pd.DataFrame(
                        {
                            "scenario": scen,
                            "target": t,
                            "method": name,
                            "window_id": s.window_id.values,
                            "subject_id": s.subject_id.values,
                            "y": y,
                            "pred": p,
                        }
                    )
                )
    return {
        "summary": pd.DataFrame(rows),
        "predictions": pd.concat(preds, ignore_index=True),
        "delta_params": {t: delta[t][1] for t in TARGETS},
        "models": {t: delta[t][0] for t in TARGETS},
        "gap_s": gap_s,
        "n_calibration_points": int(len(cal)),
    }


__all__ = ["calibration_study", "nested_cv"]
