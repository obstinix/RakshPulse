"""RakshPulse 3.0, Phase 3: feature redundancy, within-subject relevance and cross-day stability (Auricle).

For each site (ear, finger), on label-valid SQI-accepted windows, over HR + pulse-shape + wavelet + PAT features
(plus ear→finger PTT on the ear table):

* ``redundancy``: variance inflation factor (from the inverse correlation matrix of session-1 features) and the
  most-correlated partner of each feature;
* ``relevance``: per-subject Spearman correlation of feature vs SBP within session 1 and within session 2,
  the median over subjects with a subject-bootstrap CI, the fraction of subjects with the majority sign, and
  per-subject sign agreement between days. A feature that tracks BP *within* people, consistently and on both
  days, is the only kind a reference-anchored model can use;
* ``stability``: day-to-day reliability of each subject's resting feature value (ICC(3,1) between session-1
  and session-2 rest means), i.e. whether a calibration taken on one day still describes the person the next;
* ``selection``: stability selection. For 200 subject-bootstrap resamples, Lasso (fixed α chosen once by
  GroupKFold CV) is fitted on within-subject-centred features → within-subject-centred SBP, and the selection
  frequency of each feature is recorded.

Outputs: results/feature_analysis/{features,selection}.csv and a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation.auricle_study import SHAPE_FEATURES, add_normalised_shape  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.features.advanced import PAT_COLUMNS, WAVELET_COLUMNS  # noqa: E402

AUR = ROOT / "data" / "auricle"
OUT = ROOT / "results" / "feature_analysis"
SEED, N_BOOT, N_SEL = 0, 2000, 200
log = logging.getLogger("feat")


def load(site: str) -> tuple[pd.DataFrame, list[str]]:
    w = pd.read_csv(AUR / "windows.csv")
    f = pd.read_csv(AUR / "features.csv")
    a = pd.read_csv(AUR / "features_advanced.csv")
    d = w.merge(f[["window_id"] + [c for c in f.columns if c.startswith(f"{site}_")]], on="window_id")
    d = d.merge(a.drop(columns=[c for c in ("subject_id", "record_id", "session") if c in a]), on="window_id")
    d = d[d.label_valid & d[f"{site}_sqi_ok"]].copy()
    add_normalised_shape(d, site)
    cols = [f"{site}_hr"] + [f"{site}_{c}" for c in SHAPE_FEATURES]
    cols += [f"{site}_{c}" for c in WAVELET_COLUMNS] + [f"{site}_{c}" for c in PAT_COLUMNS[:2]]
    if site == "ear":
        cols += ["ear_finger_ptt_ms"]
    return d, [c for c in cols if c in d]


def icc31(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(3,1), two-way mixed, consistency, for two measurements per subject."""
    x = np.c_[a, b]
    n = len(x)
    if n < 5:
        return float("nan")
    gm = x.mean()
    ss_r = 2 * ((x.mean(1) - gm) ** 2).sum()
    ss_c = n * ((x.mean(0) - gm) ** 2).sum()
    ss_e = ((x - gm) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / (n - 1)
    return float((ms_r - ms_e) / (ms_r + ms_e))


def within_r(d: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for sid, g in d.groupby("subject_id"):
        g = g[[col, "sbp_mmHg"]].dropna()
        if len(g) >= 20 and g[col].std() > 0 and g.sbp_mmHg.std() > 2:
            out[sid] = stats.spearmanr(g[col], g.sbp_mmHg)[0]
    return pd.Series(out, dtype=float)


def boot_median(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if v.size < 3:
        return float("nan"), float("nan")
    b = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyse(site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d, cols = load(site)
    rng = np.random.default_rng(SEED)
    s1, s2 = d[d.session == 1], d[d.session == 2]
    # ---- redundancy
    c = s1[cols].corr(method="spearman")
    try:
        vif = np.diag(np.linalg.pinv(c.to_numpy()))
    except np.linalg.LinAlgError:
        vif = np.full(len(cols), np.nan)
    rows = []
    for i, col in enumerate(cols):
        partner = c[col].drop(col).abs().idxmax()
        r1, r2 = within_r(s1, col), within_r(s2, col)
        both = r1.index.intersection(r2.index)
        lo1, hi1 = boot_median(r1.to_numpy(), rng)
        maj = np.sign(np.median(r1)) if r1.size else 0
        rest1 = s1[s1.activity_code == 0].groupby("subject_id")[col].mean()
        rest2 = s2[s2.activity_code == 0].groupby("subject_id")[col].mean()
        rs = rest1.index.intersection(rest2.index)
        ok = np.isfinite(rest1[rs]) & np.isfinite(rest2[rs])
        rows.append(
            {
                "site": site,
                "feature": col,
                "available_frac": float(d[col].notna().mean()),
                "vif": float(vif[i]),
                "most_correlated_with": partner,
                "abs_spearman_with_partner": float(abs(c.loc[col, partner])),
                "within_r_s1_median": float(np.median(r1)) if r1.size else np.nan,
                "within_r_s1_ci_low": lo1,
                "within_r_s1_ci_high": hi1,
                "within_r_s2_median": float(np.median(r2)) if r2.size else np.nan,
                "majority_sign_frac_s1": float((np.sign(r1) == maj).mean()) if r1.size else np.nan,
                "sign_agreement_s1_s2": float((np.sign(r1[both]) == np.sign(r2[both])).mean()) if both.size else np.nan,
                "n_subjects_s1": int(r1.size),
                "n_subjects_both_days": int(both.size),
                "rest_icc31_day1_day2": icc31(rest1[rs][ok].to_numpy(), rest2[rs][ok].to_numpy()),
                "n_subjects_rest_both_days": int(ok.sum()),
            }
        )
    feats = pd.DataFrame(rows)
    # ---- stability selection on within-subject-centred session-1 data
    z = s1[["subject_id", "sbp_mmHg", *cols]].copy()
    z[cols] = z[cols].fillna(z[cols].median())
    gm = z.groupby("subject_id")
    Xc = (z[cols] - gm[cols].transform("mean")).to_numpy(float)
    Xc /= Xc.std(0) + 1e-12
    yc = (z.sbp_mmHg - gm.sbp_mmHg.transform("mean")).to_numpy(float)
    g = z.subject_id.to_numpy()
    alpha = LassoCV(cv=list(GroupKFold(5).split(Xc, yc, g)), alphas=np.logspace(-3, 1, 30)).fit(Xc, yc).alpha_
    subj = np.unique(g)
    freq = np.zeros(len(cols))
    for _ in range(N_SEL):
        pick = rng.choice(subj, subj.size, replace=True)
        idx = np.concatenate([np.flatnonzero(g == s) for s in pick])
        freq += Lasso(alpha=alpha, max_iter=5000).fit(Xc[idx], yc[idx]).coef_ != 0
    sel = pd.DataFrame({"site": site, "feature": cols, "selection_freq": freq / N_SEL, "lasso_alpha": alpha})
    full = Lasso(alpha=alpha, max_iter=5000).fit(Xc, yc)
    sel["coef_full_std"] = full.coef_
    log.info("%s: %d features, alpha=%.3f", site, len(cols), alpha)
    return feats, sel.sort_values("selection_freq", ascending=False)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fs, ss = zip(*(analyse(s) for s in ("ear", "finger")), strict=True)
    feats, sel = pd.concat(fs, ignore_index=True), pd.concat(ss, ignore_index=True)
    feats.to_csv(OUT / "features.csv", index=False, float_format="%.4f")
    sel.to_csv(OUT / "selection.csv", index=False, float_format="%.4f")
    run = ExperimentRun(
        "feature_analysis",
        [AUR / "windows.csv", AUR / "features.csv", AUR / "features_advanced.csv"],
        {"n_boot": N_BOOT, "n_stability_resamples": N_SEL, "within_r": "Spearman, subjects with >=20 windows"},
        seed=SEED,
        t0=t0,
    )
    run.add_metrics(feats)
    run.finish()
    show = [
        "feature",
        "vif",
        "within_r_s1_median",
        "within_r_s1_ci_low",
        "within_r_s1_ci_high",
        "within_r_s2_median",
        "sign_agreement_s1_s2",
        "rest_icc31_day1_day2",
    ]
    print(feats[feats.site == "ear"][show].round(2).to_string(index=False))
    print(sel[sel.site == "ear"].head(12).round(3).to_string(index=False))
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
