"""RakshPulse 4.0, physiological pulse-contour features: are they related to SBP/DBP, stable, and useful?

Per feature (HR + 3.0 shape + 4.0 contour landmarks) and per target (SBP, DBP, analysed separately):
within-subject Spearman r on day 1 and day 2 (median over subjects, subject-bootstrap CI for day 1), sign
agreement between days, and resting ICC(3,1) between days.

Composites: principal components of the within-subject-centred, standardised day-1 features (label-free, all
subjects). Are combinations more stable than single features?

Predictive test (within day, a reference every 300 s, as 3.0 V1): change models on ``hr_shape`` (3.0),
``physio``, ``hr_shape_physio`` vs the ``mean_change`` control, and ``hr_shape_physio`` vs ``hr_shape`` (does the
contour set add to 3.0?). Subject-level sign-flip tests with Holm correction per site × target.

Outputs: results/physio_features/{features,composites,predictive}.csv and a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_beyond_reference import grid_refs  # noqa: E402

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.beyond_reference import evaluate, make_pairs  # noqa: E402
from rakshpulse.evaluation.statistics import holm, icc31, paired_test, subject_errors  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.features.physiological import PHYSIO_FEATURES  # noqa: E402

AUR = ROOT / "data" / "auricle"
OUT = ROOT / "results" / "physio_features"
SEED, N_BOOT, N_PC = 0, 2000, 6
log = logging.getLogger("physio")


def load(site: str) -> pd.DataFrame:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    p = pd.read_csv(AUR / "features_physio.csv")
    d = w.merge(f, on="window_id").merge(p, on="window_id")
    d = d[(d.cohort == "HC") & d.label_valid & d[f"{site}_sqi_ok"]].rename(columns={"activity_code": "activity"})
    st.add_normalised_shape(d, site)
    return d.reset_index(drop=True)


def within_r(d: pd.DataFrame, col: str, y: str) -> pd.Series:
    out = {}
    for sid, g in d.groupby("subject_id"):
        g = g[[col, y]].dropna()
        if len(g) >= 20 and g[col].std() > 0 and g[y].std() > 2:
            out[sid] = stats.spearmanr(g[col], g[y])[0]
    return pd.Series(out, dtype=float)


def describe(d: pd.DataFrame, cols: list[str], site: str, kind: str, rng) -> list[dict]:
    rows = []
    s1, s2 = d[d.session == 1], d[d.session == 2]
    r1m = s1[s1.activity == 0].groupby("subject_id")[cols].mean()
    r2m = s2[s2.activity == 0].groupby("subject_id")[cols].mean()
    both = r1m.index.intersection(r2m.index)
    for target in ("sbp", "dbp"):
        y = f"{target}_mmHg"
        for c in cols:
            a, b = within_r(s1, c, y), within_r(s2, c, y)
            j = a.index.intersection(b.index)
            boot = (
                [np.median(a.to_numpy()[rng.integers(0, a.size, a.size)]) for _ in range(N_BOOT)]
                if a.size
                else [np.nan]
            )
            rows.append(
                {
                    "site": site,
                    "kind": kind,
                    "target": target,
                    "feature": c,
                    "within_r_day1": float(a.median()) if a.size else np.nan,
                    "within_r_day1_ci_low": float(np.nanpercentile(boot, 2.5)),
                    "within_r_day1_ci_high": float(np.nanpercentile(boot, 97.5)),
                    "within_r_day2": float(b.median()) if b.size else np.nan,
                    "sign_agreement": float((np.sign(a[j]) == np.sign(b[j])).mean()) if j.size else np.nan,
                    "rest_icc_day1_day2": icc31(r1m.loc[both, c], r2m.loc[both, c]),
                    "available_frac": float(d[c].notna().mean()),
                    "n_subjects": int(a.size),
                }
            )
    return rows


def composites(d: pd.DataFrame, cols: list[str], site: str) -> pd.DataFrame:
    """Add PC1..PCk of within-subject-centred standardised day-1 features (loadings from day 1 only)."""
    s1 = d[d.session == 1]
    x = s1[cols].fillna(s1[cols].median())
    xc = x - x.groupby(s1.subject_id).transform("mean")
    mu, sd = xc.mean(), xc.std().replace(0, 1)
    u, s, vt = np.linalg.svd(((xc - mu) / sd).to_numpy(), full_matrices=False)
    load = vt[:N_PC]
    # Project raw (uncentred) features: within-session correlations are unaffected by a per-session constant,
    # and the between-day ICC then measures the composite's own day-to-day reproducibility.
    X = d[cols].fillna(s1[cols].median())
    pcs = ((X - X.mean()) / sd).to_numpy() @ load.T
    names = [f"{site}_pc{i + 1}" for i in range(N_PC)]
    for i, n in enumerate(names):
        d[n] = pcs[:, i]
    expl = (s**2 / (s**2).sum())[:N_PC]
    return pd.DataFrame(
        {
            "site": site,
            "component": names,
            "explained_var": expl,
            "top_loadings": [";".join(np.array(cols)[np.argsort(-np.abs(v))[:4]]) for v in load],
        }
    )


def predictive(d: pd.DataFrame, site: str) -> list[dict]:
    hr_shape = [f"{site}_hr"] + [f"{site}_{c}" for c in st.SHAPE_FEATURES]
    physio = [f"{site}_{c}" for c in PHYSIO_FEATURES]
    fsets = {"hr_shape": hr_shape, "physio": physio, "hr_shape_physio": hr_shape + physio}
    cols = hr_shape + physio
    rows = []
    for target in ("sbp", "dbp"):
        y = f"{target}_mmHg"
        refs = grid_refs(d, 300.0, cols, y)
        pairs = make_pairs(d, refs, cols, y)
        _, _, pred = evaluate(pairs, fsets, d.groupby("subject_id")[y].mean(), n_boot=200, seed=SEED)
        errs = {m: subject_errors(pred.assign(yhat=pred[m])) for m in ["mean_change"] + [f"delta_{k}" for k in fsets]}
        fam = []
        for a, b in (
            ("delta_hr_shape", "mean_change"),
            ("delta_physio", "mean_change"),
            ("delta_hr_shape_physio", "mean_change"),
            ("delta_hr_shape_physio", "delta_hr_shape"),
        ):
            fam.append(
                {
                    "site": site,
                    "target": target,
                    "scenario": "within_day_ref_every_300s",
                    "comparison": f"{a} vs {b}",
                    "mae_a": float(errs[a].mean()),
                    "mae_b": float(errs[b].mean()),
                    **paired_test(errs[a], errs[b], N_BOOT, seed=SEED),
                }
            )
        p = holm([r["p_perm"] for r in fam])
        for r, ph in zip(fam, p, strict=True):
            r["p_holm"] = ph
        rows += fam
    return rows


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    feats, comps, preds = [], [], []
    for site in ("ear", "finger"):
        d = load(site)
        hr_shape = [f"{site}_hr"] + [f"{site}_{c}" for c in st.SHAPE_FEATURES]
        physio = [f"{site}_{c}" for c in PHYSIO_FEATURES]
        feats += describe(d, hr_shape, site, "3.0_hr_shape", rng)
        feats += describe(d, physio, site, "4.0_physio", rng)
        cmp = composites(d, hr_shape + physio, site)
        comps.append(cmp)
        feats += describe(d, list(cmp.component), site, "pca_composite", rng)
        preds += predictive(d, site)
        log.info("%s done", site)
    f = pd.DataFrame(feats)
    f.to_csv(OUT / "features.csv", index=False, float_format="%.4f")
    pd.concat(comps).to_csv(OUT / "composites.csv", index=False, float_format="%.4f")
    pr = pd.DataFrame(preds)
    pr.to_csv(OUT / "predictive.csv", index=False, float_format="%.4f")
    run = ExperimentRun(
        "physio_features_v4",
        [AUR / "windows.csv", AUR / "features.csv", AUR / "features_physio.csv"],
        {"n_boot": N_BOOT, "n_pc": N_PC, "predictive_scenario": "within day, reference every 300 s"},
        seed=SEED,
        t0=t0,
        dataset_versions={"auricle": "PhysioNet 1.0.0, standardised tables 2.0, features_physio 4.0"},
        feature_config={
            "physio": ";".join(PHYSIO_FEATURES),
            "composites": "PCA of within-subject-centred day-1 features",
        },
        model_config={"delta_model": "RidgeCV on feature changes (3.0 evaluate)"},
    )
    run.add_metrics(pr)
    run.finish()
    summ = (
        f.assign(abs_r=f.within_r_day1.abs())
        .groupby(["site", "kind", "target"])[["abs_r", "sign_agreement", "rest_icc_day1_day2"]]
        .median()
    )
    print(summ.round(2).to_string())
    print(
        pr[["site", "target", "comparison", "mean_diff", "ci_low", "ci_high", "p_holm"]].round(3).to_string(index=False)
    )
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
