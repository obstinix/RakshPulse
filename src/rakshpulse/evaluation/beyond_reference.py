"""Does PPG carry BP information *beyond the last reference reading*?

For every evaluation window *t* the most recent reference reading *r* (at least ``min_gap_s`` earlier) is known.
Every method predicts::

    BP_hat(t) = reported_ref(r) + Δ_hat(t, r)

* ``last_reference``: Δ_hat = 0 (carry the reading forward);
* ``historical_mean``: BP_hat = mean of all reference readings up to *t* (personal history);
* ``population_mean``: BP_hat = mean truth of the training subjects (zero-shot);
* ``mean_change``: Δ_hat = average BP change since a reference in the training subjects (no signal). This is the
  control for the delta models, whose intercept learns exactly this;
* ``delta_<featureset>``: Δ_hat = f(x(t) − x(r)). This is a ridge model trained **on other subjects only**
  (subject-grouped folds) to map *feature changes* to *true BP changes*, truth(t) − truth(r).

Learning from feature *changes* removes each person's absolute offset, which no population model could estimate
(RakshPulse 2.0). If ``delta_*`` cannot beat ``mean_change``, the signal carries no usable information about BP
changes since the reference, regardless of how well a model fits absolute BP.

``reported_ref`` may differ from ``truth(r)``. In VitalDB the reference is the intermittent cuff (NIBP) and the
truth is the arterial line, so the cuff–line disagreement is a real error source, as it would be in use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .auricle_study import paired_subject_diff, point_metrics, subject_folds


def make_pairs(
    windows: pd.DataFrame,
    refs: pd.DataFrame,
    feature_cols: list[str],
    truth_col: str,
    min_gap_s: float = 10.0,
    max_gap_s: float | None = None,
) -> pd.DataFrame:
    """Join each evaluation window with its most recent reference (same subject and session).

    ``windows`` needs: subject_id, session, t_start_s, truth_col, feature_cols.
    ``refs`` needs: subject_id, session, t_ref_s, ref_reported, ref_truth, and ``ref_<feature>`` for each feature.
    """
    out = []
    w = windows.sort_values("t_start_s")
    for (sid, sess), g in w.groupby(["subject_id", "session"]):
        r = refs[(refs.subject_id == sid) & (refs.session == sess)].sort_values("t_ref_s")
        if r.empty:
            continue
        idx = np.searchsorted(r.t_ref_s.to_numpy(), g.t_start_s.to_numpy() - min_gap_s, side="right") - 1
        ok = idx >= 0
        if not ok.any():
            continue
        gg, rr = g[ok].reset_index(drop=True), r.iloc[idx[ok]].reset_index(drop=True)
        gap = gg.t_start_s.to_numpy() - rr.t_ref_s.to_numpy()
        keep = np.ones(len(gg), bool) if max_gap_s is None else gap <= max_gap_s
        # personal historical mean of all readings up to and including r
        cum = r.ref_reported.expanding().mean().to_numpy()[idx[ok]]
        pair = pd.DataFrame(
            {
                "subject_id": sid,
                "session": sess,
                "t_s": gg.t_start_s,
                "gap_s": gap,
                "y": gg[truth_col],
                "ref_reported": rr.ref_reported,
                "ref_truth": rr.ref_truth,
                "historical_mean": cum,
                "activity": gg["activity"] if "activity" in gg else -1,
            }
        )
        for c in feature_cols:
            pair[f"d_{c}"] = gg[c].to_numpy() - rr[f"ref_{c}"].to_numpy()
        out.append(pair[keep])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(
    pairs: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    population_truth: pd.Series | None = None,
    n_folds: int = 5,
    n_boot: int = 2000,
    seed: int = 0,
    extra_methods: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (metrics, paired comparisons vs last_reference, per-window predictions)."""
    fold_of = subject_folds(pairs.subject_id, pairs.y, n=n_folds)
    f = pairs.subject_id.map(fold_of).to_numpy()
    preds = {
        "last_reference": pairs.ref_reported.to_numpy(float),
        "historical_mean": pairs.historical_mean.to_numpy(float),
    }
    if population_truth is not None:  # mean truth of the *training* subjects in each fold
        pm = np.full(len(pairs), np.nan)
        for k in range(n_folds):
            tr_subj = [s for s, kk in fold_of.items() if kk != k]
            pm[f == k] = population_truth[population_truth.index.isin(tr_subj)].mean()
        preds["population_mean"] = pm
    dy = (pairs.y - pairs.ref_truth).to_numpy(float)
    # Control: mean BP change since a reference in the *training* subjects (no signal at all). A delta model's
    # intercept learns exactly this, so the signal's own contribution is delta_<set> vs mean_change.
    mc = np.full(len(pairs), np.nan)
    for k in range(n_folds):
        if (f == k).any():
            mc[f == k] = dy[f != k].mean()
    preds["mean_change"] = pairs.ref_reported.to_numpy(float) + mc
    for name, cols in feature_sets.items():
        X = pairs[[f"d_{c}" for c in cols]].to_numpy(float)
        p = np.full(len(pairs), np.nan)
        for k in range(n_folds):
            tr, te = f != k, f == k
            if not te.any():
                continue
            m = make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0),
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-1, 4, 12)),
            )
            m.fit(X[tr], dy[tr])
            p[te] = m.predict(X[te])
        preds[f"delta_{name}"] = pairs.ref_reported.to_numpy(float) + p
    for k, v in (extra_methods or {}).items():
        preds[k] = v
    rows, paired = [], []
    base = pd.DataFrame(
        {"subject_id": pairs.subject_id, "y": pairs.y, "activity": pairs.activity, "yhat": preds["last_reference"]}
    )
    for meth, yhat in preds.items():
        t = pd.DataFrame({"subject_id": pairs.subject_id, "y": pairs.y, "activity": pairs.activity, "yhat": yhat})
        t = t[np.isfinite(t.yhat)]
        m = point_metrics(t)
        # tracking of BP *changes* since the reference (meaningful only for delta methods)
        dpred = t.yhat - pairs.ref_reported[t.index]
        dtrue = pairs.y[t.index] - pairs.ref_truth[t.index]
        m["delta_r"] = float(np.corrcoef(dpred, dtrue)[0, 1]) if dpred.std() > 1e-9 else float("nan")
        rows.append({"method": meth, **m})
        for ref in ["last_reference"] + (["mean_change"] if meth.startswith("delta_") else []):
            if meth == ref:
                continue
            b = base.loc[t.index].assign(yhat=np.asarray(preds[ref])[t.index])
            d, lo, hi = paired_subject_diff(t, b, n_boot, seed)
            paired.append(
                {
                    "comparison": f"{meth} vs {ref}",
                    "mae_diff": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(hi < 0 or lo > 0),
                }
            )
    pred_df = pairs[["subject_id", "session", "t_s", "gap_s", "y", "ref_reported", "ref_truth"]].copy()
    for k, v in preds.items():
        pred_df[k] = v
    return pd.DataFrame(rows), pd.DataFrame(paired), pred_df
