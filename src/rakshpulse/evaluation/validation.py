"""Subject-independent nested cross-validation.

* **Outer loop:** ``StratifiedGroupKFold`` on subject IDs (stratified by subject SBP quintile),
  so every window of a test subject is unseen during training, scaling, imputation,
  hyper-parameter search and model selection.
* **Inner loop:** ``GroupKFold`` on the training subjects only, used for hyper-parameter tuning.
* **Scoring unit:** the reference BP is a single cuff value per subject, so windows are
  pseudo-replicates. Window predictions are aggregated per subject (median), and
  **subject-level metrics are the primary result** (n = number of subjects). Window-level
  metrics are reported as secondary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

from ..models.model_factory import make_model


def outer_splits(df: pd.DataFrame, n_splits: int, seed: int, strat_col: str = "sbp"):
    subj = df.groupby("subject_id")[strat_col].first()
    bins = pd.qcut(subj.rank(method="first"), q=min(5, len(subj) // n_splits), labels=False)
    y_bins = df["subject_id"].map(bins).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(df, y_bins, groups=df["subject_id"].to_numpy()))


def nested_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
    seed: int = 0,
    n_outer: int = 5,
    n_inner: int = 3,
    splits=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return (window predictions with fold index, best-params per fold)."""
    df = df.reset_index(drop=True)
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    g = df["subject_id"].to_numpy()
    pred = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    params = []
    for k, (tr, te) in enumerate(splits or outer_splits(df, n_outer, seed)):
        assert not set(g[tr]) & set(g[te]), "subject leakage between train and test"
        est, grid = make_model(model_name, seed)
        if grid:
            search = GridSearchCV(
                est, grid, cv=GroupKFold(n_splits=n_inner), scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
            )
            search.fit(X[tr], y[tr], groups=g[tr])
            est = search.best_estimator_
            params.append(search.best_params_)
        else:
            est.fit(X[tr], y[tr])
            params.append({})
        pred[te] = est.predict(X[te])
        fold[te] = k
    out = df[["subject_id", "record_id", "t_start_s", target]].copy()
    out["pred"] = pred
    out["fold"] = fold
    return out, params


def subject_level(pred_df: pd.DataFrame, target: str) -> pd.DataFrame:
    return (
        pred_df.groupby("subject_id")
        .agg(y=(target, "first"), pred=("pred", "median"), n_windows=("pred", "size"), fold=("fold", "first"))
        .reset_index()
    )
