"""RakshPulse 4.0, priority 1: why does within-session PPG information fail across days, and does
normalisation fix it?

Part 1, diagnostics (resting windows): per-feature cross-day shift vs the within-day null, pooled KS,
between-day feature–BP coupling, and per-subject drivers: pulse-amplitude ratio (contact / perfusion / gain
proxy; Auricle PPG is AC-coupled, so a true perfusion index is not available), template similarity between days
vs within day 1 (sensor placement and morphology), SQI change, HR change and resting SBP change. Temperature is
not recorded in Auricle, so it cannot be analysed.

Part 2, normalisation comparison (cross-day, change model ŷ = day-1 reference + f(Δz)):
* A raw · B session · C subject · D robust · E adaptive · F domain (see ``evaluation/domain_shift.py``);
* feature sets: ``hr_shape`` (3.0 primary), ``physio`` (4.0 contour landmarks), ``hr_shape_physio`` and
  ``stable`` (the 10 features with the highest day-to-day resting ICC, chosen on the training subjects of each
  fold only, from the hr_shape + physio pool; label-free);
* controls: ``last_reference`` (carry the day-1 reading) and ``mean_change`` (training-subject mean change,
  no signal);
* every method is evaluated on the same day-2 windows (after a 180 s PPG warm-up), with 5 subject-grouped folds.
  Subject-level paired tests use sign-flip permutation and bootstrap CIs, with Holm and BH correction within each
  site × target family.

Outputs: results/cross_day_v4/{shift_features,shift_subjects,drivers,metrics,paired}.csv,
docs/figures/v4_cross_day_shift.png and a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.domain_shift import LETTER, NORMALISATIONS, normalise, shift_diagnostics  # noqa: E402
from rakshpulse.evaluation.statistics import benjamini_hochberg, holm, icc31, paired_test, subject_errors  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.features.physiological import PHYSIO_FEATURES  # noqa: E402

AUR = ROOT / "data" / "auricle"
OUT = ROOT / "results" / "cross_day_v4"
FIG = ROOT / "docs" / "figures" / "v4_cross_day_shift.png"
TEMPLATES = ROOT / "data" / "interim" / "auricle_templates.npz"
SEED, N_BOOT, N_FOLDS = 0, 2000, 5
WARMUP_S, N_STABLE = 180.0, 10
log = logging.getLogger("crossday")


def load(site: str) -> pd.DataFrame:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    p = pd.read_csv(AUR / "features_physio.csv")
    d = w.merge(f, on="window_id").merge(p, on="window_id")
    d = d[(d.cohort == "HC") & d[f"{site}_sqi_ok"]].rename(columns={"activity_code": "activity"})
    st.add_normalised_shape(d, site)
    return d.sort_values(["subject_id", "session", "t_start_s"]).reset_index(drop=True)


def feature_sets(site: str) -> dict[str, list[str]]:
    hr_shape = [f"{site}_hr"] + [f"{site}_{c}" for c in st.SHAPE_FEATURES]
    physio = [f"{site}_{c}" for c in PHYSIO_FEATURES]
    return {"hr_shape": hr_shape, "physio": physio, "hr_shape_physio": hr_shape + physio}


def template_similarity(site: str, d: pd.DataFrame) -> pd.DataFrame:
    """Per subject: correlation of median rest templates, day 1 vs day 2, and day-1 first vs second half."""
    if not TEMPLATES.exists():
        return pd.DataFrame(columns=["subject_id"])
    z = np.load(TEMPLATES)
    pos = {}
    for s in (1, 2):
        for i, wid in enumerate(z[f"window_id_s{s}"]):
            pos[wid] = (s, i)
    tp = {s: z[f"{site}_s{s}"].astype(float) for s in (1, 2)}
    rows = []
    for sid, g in d[d.activity == 0].groupby("subject_id"):
        mats = {}
        for s in (1, 2):
            ids = [pos[w][1] for w in g[g.session == s].window_id if w in pos]
            m = tp[s][ids] if ids else np.empty((0, 100))
            mats[s] = m[np.isfinite(m).all(1)]
        if len(mats[1]) < 12 or len(mats[2]) < 6:
            continue
        h = len(mats[1]) // 2
        med = lambda m: np.median(m, 0)  # noqa: E731
        rows.append(
            {
                "subject_id": sid,
                "template_corr_cross_day": float(np.corrcoef(med(mats[1]), med(mats[2]))[0, 1]),
                "template_corr_within_day": float(np.corrcoef(med(mats[1][:h]), med(mats[1][h:]))[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def subject_drivers(site: str, d: pd.DataFrame) -> pd.DataFrame:
    rest = d[d.activity == 0]
    rows = []
    for sid, g in rest.groupby("subject_id"):
        g1, g2 = g[g.session == 1], g[g.session == 2]
        if len(g1) < 12 or len(g2) < 6:
            continue
        a1, a2 = g1[f"{site}_pulse_amp"].median(), g2[f"{site}_pulse_amp"].median()
        rows.append(
            {
                "subject_id": sid,
                "log_amp_ratio": float(np.log(a2 / a1)) if a1 > 0 and a2 > 0 else np.nan,
                "sqi_template_corr_change": float(
                    g2[f"{site}_template_corr"].median() - g1[f"{site}_template_corr"].median()
                ),
                "hr_change": float(g2[f"{site}_hr"].median() - g1[f"{site}_hr"].median()),
            }
        )
    return pd.DataFrame(rows).merge(template_similarity(site, d), on="subject_id", how="left")


def build_subjects(d: pd.DataFrame, cols: list[str], site: str, y: str) -> dict[str, dict]:
    """Per eligible subject: day-1 reference, day-2 evaluation windows, and normalised Δz per normalisation."""
    out = {}
    for sid, g in d.groupby("subject_id"):
        d1, d2 = g[g.session == 1], g[g.session == 2]
        ref = d1[d1[f"{site}_calibration_eligible"] & d1.label_valid].iloc[:3]
        if len(ref) < 3 or d2.empty:
            continue
        t_warm = d2.t_start_s.min() + WARMUP_S
        warm = d2[d2.t_start_s < t_warm]
        ev_mask = (d2.t_start_s >= t_warm).to_numpy() & d2.label_valid.to_numpy()
        if len(warm) < 6 or ev_mask.sum() < 5:
            continue
        rec = {
            "y": d2[y].to_numpy()[ev_mask],
            "ref": float(ref[y].mean()),
            "t": d2.t_start_s.to_numpy()[ev_mask],
            "activity": d2.activity.to_numpy()[ev_mask],
            "dz": {},
            "rest1": d1.loc[d1.activity == 0, cols].median().to_numpy(float),
            "rest2": d2.loc[d2.activity == 0, cols].median().to_numpy(float),
        }
        for nm in NORMALISATIONS:
            zr, z2 = normalise(nm, d1, d2, ref, cols, warmup_s=WARMUP_S)
            rec["dz"][nm] = z2[ev_mask] - zr
        out[sid] = rec
    return out


def stable_subset(subjects: dict, train: list[str], cols: list[str], pool: list[str]) -> list[str]:
    """Top-N_STABLE features of ``pool`` by day-to-day resting ICC on the TRAINING subjects (label-free)."""
    r1 = np.array([subjects[s]["rest1"] for s in train])
    r2 = np.array([subjects[s]["rest2"] for s in train])
    scores = {c: icc31(r1[:, cols.index(c)], r2[:, cols.index(c)]) for c in pool}
    return sorted(pool, key=lambda c: -np.nan_to_num(scores[c], nan=-9))[:N_STABLE]


def run_site_target(site: str, target: str, d: pd.DataFrame) -> tuple[list, list, dict]:
    y = f"{target}_mmHg"
    fsets = feature_sets(site)
    cols = fsets["hr_shape_physio"]
    subjects = build_subjects(d, cols, site, y)
    sids = sorted(subjects)
    folds = st.subject_folds(pd.Series(sids), pd.Series([subjects[s]["ref"] for s in sids]), n=N_FOLDS)
    rows = []
    for s in sids:
        r = subjects[s]
        rows.append(pd.DataFrame({"subject_id": s, "y": r["y"], "ref": r["ref"], "activity": r["activity"]}))
    base = pd.concat(rows, ignore_index=True)
    preds = {"last_reference": base.ref.to_numpy(), "mean_change": np.full(len(base), np.nan)}
    owner = base.subject_id.to_numpy()
    dy_all = (base.y - base.ref).to_numpy()
    fold_of = np.array([folds[s] for s in owner])
    chosen = {}
    for k in range(N_FOLDS):
        tr, te = fold_of != k, fold_of == k
        preds["mean_change"][te] = base.ref.to_numpy()[te] + dy_all[tr].mean()
        train_s = [s for s in sids if folds[s] != k]
        stable = stable_subset(subjects, train_s, cols, cols)
        chosen[k] = stable
        sets = {**fsets, "stable": stable}
        for fs_name, fcols in sets.items():
            idx = [cols.index(c) for c in fcols]
            for nm in NORMALISATIONS:
                X = np.vstack([subjects[s]["dz"][nm][:, idx] for s in sids])
                key = f"delta_{fs_name}_{LETTER[nm]}_{nm}"
                if key not in preds:
                    preds[key] = np.full(len(base), np.nan)
                m = make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0.0),
                    StandardScaler(),
                    RidgeCV(np.logspace(-1, 4, 12)),
                )
                m.fit(X[tr], dy_all[tr])
                preds[key][te] = base.ref.to_numpy()[te] + m.predict(X[te])
    met, par = [], []
    errs = {}
    for meth, yhat in preds.items():
        t = base.assign(yhat=yhat)
        met.append({"site": site, "target": target, "method": meth, **st.point_metrics(t)})
        errs[meth] = subject_errors(t)
    fam = []
    for meth in preds:
        for ref in ("mean_change", "last_reference"):
            if meth == ref or (meth == "last_reference" and ref == "mean_change"):
                continue  # each delta model vs both controls, and mean_change vs last_reference
            r = paired_test(errs[meth], errs[ref], N_BOOT, seed=SEED)
            fam.append({"site": site, "target": target, "comparison": f"{meth} vs {ref}", **r})
    fam = pd.DataFrame(fam)
    vs_mc = fam.comparison.str.endswith("vs mean_change") & fam.comparison.str.startswith("delta_")
    fam.loc[vs_mc, "p_holm"] = holm(fam.loc[vs_mc, "p_perm"].to_numpy())
    fam.loc[vs_mc, "p_bh"] = benjamini_hochberg(fam.loc[vs_mc, "p_perm"].to_numpy())
    par = fam.to_dict("records")
    sel = pd.Series([c for v in chosen.values() for c in v]).value_counts()
    info = {"n_subjects": len(sids), "folds": folds, "stable_selection_counts": sel.to_dict(), "errors": errs}
    return met, par, info


def figure(shift: pd.DataFrame, subj: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for site, mk in (("ear", "o"), ("finger", "s")):
        s = shift[shift.site == site]
        ax[0].scatter(s.within_shift_z_median, s.cross_shift_z_median, marker=mk, alpha=0.7, label=site)
        q = subj[subj.site == site].dropna(subset=["template_corr_cross_day"])
        ax[1].scatter(q.template_corr_within_day, q.template_corr_cross_day, marker=mk, alpha=0.7, label=site)
    lim = max(shift.cross_shift_z_median.max(), shift.within_shift_z_median.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0].set(
        xlabel="within-day shift (|Δ median| / robust SD), day-1 halves",
        ylabel="cross-day shift, day 1 → day 2",
        title="Feature shift per feature (median over subjects)",
    )
    ax[1].plot([0.8, 1], [0.8, 1], "k--", lw=0.8)
    ax[1].set(
        xlabel="rest-template correlation, within day 1",
        ylabel="rest-template correlation, day 1 vs day 2",
        title="Pulse-shape similarity per subject",
    )
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run = ExperimentRun(
        "cross_day_generalization_v4",
        [AUR / "windows.csv", AUR / "features.csv", AUR / "features_physio.csv"],
        {
            "warmup_s": WARMUP_S,
            "n_boot": N_BOOT,
            "folds": N_FOLDS,
            "reference": "first 3 eligible day-1 rest windows",
            "evaluation": "day-2 label-valid windows after the warm-up; identical for all methods",
        },
        seed=SEED,
        dataset_versions={"auricle": "PhysioNet 1.0.0 (doi:10.13026/swme-vp69), standardised tables 2.0"},
        feature_config={
            "normalisations": ";".join(f"{LETTER[n]}={n}" for n in NORMALISATIONS),
            "stable": f"top {N_STABLE} by training-subject rest ICC(3,1)",
        },
        model_config={"delta_model": "SimpleImputer(0) -> StandardScaler -> RidgeCV(logspace(-1,4,12))"},
    )
    shifts, subjs, met, par, drivers = [], [], [], [], []
    with run.phase("diagnostics"):
        for site in ("ear", "finger"):
            d = load(site)
            cols = feature_sets(site)["hr_shape_physio"]
            sf, ss = shift_diagnostics(d, cols)
            sf["site"], ss["site"] = site, site
            dr = subject_drivers(site, d)
            dr["site"] = site
            shifts.append(sf)
            subjs.append(ss.merge(dr, on=["subject_id", "site"], how="left"))
    shift, subj = pd.concat(shifts, ignore_index=True), pd.concat(subjs, ignore_index=True)
    with run.phase("train_evaluate"):
        for site in ("ear", "finger"):
            d = load(site)
            for target in ("sbp", "dbp"):
                m, p, info = run_site_target(site, target, d)
                met += m
                par += p
                run.splits.update(info["folds"])
                # does a subject's cross-day failure relate to the shift drivers? (A raw hr_shape vs control)
                e = (info["errors"]["delta_hr_shape_A_raw"] - info["errors"]["mean_change"]).rename("excess_error")
                q = subj[subj.site == site].set_index("subject_id").join(e, how="inner")
                for drv in (
                    "mean_cross_shift_z",
                    "log_amp_ratio",
                    "template_corr_cross_day",
                    "sqi_template_corr_change",
                    "hr_change",
                    "rest_sbp_change",
                ):
                    ok = q[[drv, "excess_error"]].dropna()
                    r, p_ = stats.spearmanr(ok[drv], ok.excess_error) if len(ok) >= 8 else (np.nan, np.nan)
                    drivers.append(
                        {
                            "site": site,
                            "target": target,
                            "driver": drv,
                            "spearman_r": float(r),
                            "p": float(p_),
                            "n_subjects": len(ok),
                        }
                    )
                for c, n in info["stable_selection_counts"].items():
                    drivers.append(
                        {
                            "site": site,
                            "target": target,
                            "driver": f"stable_selected:{c}",
                            "spearman_r": np.nan,
                            "p": np.nan,
                            "n_subjects": int(n),
                        }
                    )
                log.info("%s %s: %d subjects", site, target, info["n_subjects"])
    drv = pd.DataFrame(drivers)
    drv.loc[drv.p.notna(), "p_holm"] = holm(drv.loc[drv.p.notna(), "p"].to_numpy())
    for name, df in (
        ("shift_features", shift),
        ("shift_subjects", subj),
        ("drivers", drv),
        ("metrics", pd.DataFrame(met)),
        ("paired", pd.DataFrame(par)),
    ):
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    figure(shift, subj)
    run.add_metrics(pd.DataFrame(met))
    run.finish()
    pm = pd.DataFrame(par)
    show = pm[pm.comparison.str.endswith("vs mean_change") & (pm.target == "sbp")]
    print(
        show[["site", "comparison", "mean_diff", "ci_low", "ci_high", "p_perm", "p_holm"]]
        .round(3)
        .to_string(index=False)
    )
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
