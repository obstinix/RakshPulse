"""RakshPulse 4.0: does the graded PPG quality score track prediction error, and is rejection worth it?

1. Grade every Auricle window at both sites (``preprocessing/quality_score.py``). Relative amplitude is causal:
   pulse amplitude / running median of the previous ``good``-or-``degraded`` windows of the same recording.
2. Predict SBP for every label-valid window that has features, *including* windows the 3.0 gate rejected. The
   model is a population ridge on HR + shape, trained per subject-grouped fold on SQI-accepted session-1 windows
   of the training subjects. Two error types:
   * absolute |ŷ − y|;
   * offset-calibrated: per subject × session, minus the mean residual of the first 3 ``good`` windows (those 3
     are not scored).
3. Report subject-level MAE per class, paired within-subject comparisons (degraded / unreliable vs good;
   sign-flip tests, Holm), the within-subject Spearman correlation of score with calibrated error, and a
   retention curve (score threshold → share of windows kept, subject-weighted MAE, subjects with ≥ 10 windows).

Outputs: results/quality_v4/{class_summary,paired,retention,score_error}.csv; per-window grades in
data/interim/auricle_quality_v4.csv.gz (git-ignored); a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.statistics import holm, paired_test, subject_errors  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.preprocessing.quality_score import CLASSES, components, grade  # noqa: E402

AUR = ROOT / "data" / "auricle"
OUT = ROOT / "results" / "quality_v4"
CACHE = ROOT / "data" / "interim" / "auricle_quality_v4.csv.gz"
SEED, N_BOOT = 0, 2000
THRESHOLDS = (0, 25, 50, 75, 90, 99.9)
log = logging.getLogger("quality")


def _components_block(proc: np.ndarray, raw: np.ndarray) -> list[dict]:
    out = []
    for p, r in zip(proc, raw, strict=True):
        if not (np.isfinite(r).all() and np.isfinite(p).all()) or r.std() < 0.003:
            out.append({"missing": True, "raw_sd": float(np.nanstd(r))})
            continue
        out.append(components(p, r, 100.0))
    return out


def grade_all(jobs: int = 3) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE)
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    frames = []
    for sess in (1, 2):
        z = np.load(AUR / "waveforms" / f"auricle_waveforms_session{sess}.npz")
        wid = z["window_id"]
        for site in ("ear", "finger"):
            P, R = z[f"proc_{site}"], z[f"raw_{site}"]
            chunks = np.array_split(np.arange(len(P)), jobs * 8)
            res = Parallel(n_jobs=jobs)(delayed(_components_block)(P[c], R[c]) for c in chunks)
            rows = [r for rr in res for r in rr]
            df = pd.DataFrame(rows)
            df.insert(0, "window_id", wid)
            df.insert(1, "site", site)
            frames.append(df)
            log.info("components: session %d %s", sess, site)
    c = pd.concat(frames, ignore_index=True)
    c = c.merge(w[["window_id", "record_id", "t_start_s", "ear_rail_pct", "finger_rail_pct"]], on="window_id")
    out = []
    for (_, site), g in c.sort_values("t_start_s").groupby(["record_id", "site"], sort=False):
        ref_amps: list[float] = []
        for _, r in g.iterrows():
            rail = r[f"{site}_rail_pct"]
            if r.get("missing") in (True, 1.0) and not pd.isna(r.get("missing")):
                out.append(
                    {
                        "window_id": r.window_id,
                        "site": site,
                        "quality_class": "missing",
                        "score": 0.0,
                        "reasons": "no_signal_flatline",
                    }
                )
                continue
            comp = {
                k: r[k]
                for k in (
                    "template_corr",
                    "cardiac_power_ratio",
                    "rr_cv",
                    "n_beats",
                    "spike_frac",
                    "baseline_ratio",
                    "hf_noise_ratio",
                    "pulse_amplitude",
                    "plateau_frac",
                )
            }
            comp["n_beats"] = 0 if not np.isfinite(comp["n_beats"]) else int(comp["n_beats"])
            reasons = r["gate_reasons"]
            comp["gate_reasons"] = (
                list(reasons)
                if isinstance(reasons, list)
                else (
                    [x for x in str(reasons).strip("[]").replace("'", "").split(", ") if x]
                    if isinstance(reasons, str)
                    else []
                )
            )
            ref = float(np.median(ref_amps[-60:])) if len(ref_amps) >= 3 else None
            comp["rel_amplitude"] = comp["pulse_amplitude"] / ref if ref else np.nan
            q = grade(comp, rail_pct=rail)
            if q.quality_class in ("good", "degraded"):
                ref_amps.append(comp["pulse_amplitude"])
            out.append(
                {
                    "window_id": r.window_id,
                    "site": site,
                    "quality_class": q.quality_class,
                    "score": q.score,
                    "reasons": "|".join(q.reasons),
                    "rel_amplitude": comp["rel_amplitude"],
                }
            )
    q = pd.DataFrame(out)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(CACHE, index=False, float_format="%.4f")
    return q


def predict_all(site: str, q: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w.merge(f, on="window_id")
    d = d[(d.cohort == "HC") & d.label_valid].copy()
    st.add_normalised_shape(d, site)
    d = d.merge(q[q.site == site][["window_id", "quality_class", "score"]], on="window_id")
    cols = [f"{site}_hr"] + [f"{site}_{c}" for c in st.SHAPE_FEATURES]
    d = d[d[cols].notna().sum(1) >= len(cols) // 2].reset_index(drop=True)  # need features to predict at all
    s1ok = (d.session == 1) & d[f"{site}_sqi_ok"]
    folds = st.subject_folds(d.subject_id[s1ok], d.sbp_mmHg[s1ok])
    d = d[d.subject_id.isin(folds)].reset_index(drop=True)
    k = d.subject_id.map(folds).to_numpy()
    d["yhat"] = np.nan
    for f_ in range(5):
        tr = (k != f_) & (d.session == 1).to_numpy() & d[f"{site}_sqi_ok"].to_numpy()
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(np.logspace(-2, 4, 13)))
        m.fit(d.loc[tr, cols], d.loc[tr, "sbp_mmHg"])
        d.loc[k == f_, "yhat"] = m.predict(d.loc[k == f_, cols])
    d["y"] = d.sbp_mmHg
    d["ae"] = (d.yhat - d.y).abs()
    d["yhat_cal"] = np.nan
    for _, g in d.sort_values("t_start_s").groupby(["subject_id", "session"]):
        good = g[g.quality_class == "good"]
        if len(good) < 4:
            continue
        cal = good.iloc[:3]
        rest = g.index.difference(cal.index)
        d.loc[rest, "yhat_cal"] = d.loc[rest, "yhat"] - (cal.yhat - cal.y).mean()
    d["ae_cal"] = (d.yhat_cal - d.y).abs()
    d["site"] = site
    d.attrs["folds"] = folds
    return d


def analyse(d: pd.DataFrame) -> tuple[list, list, list, list]:
    site = d.site.iloc[0]
    summ, par, ret, se = [], [], [], []
    for err in ("ae", "ae_cal"):
        e = d[np.isfinite(d[err])]
        for cls in CLASSES[:3]:
            g = e[e.quality_class == cls]
            if g.empty:
                continue
            per = g.groupby("subject_id")[err].mean()
            summ.append(
                {
                    "site": site,
                    "error": err,
                    "quality_class": cls,
                    "window_share": len(g) / len(e),
                    "subject_mae": float(per.mean()),
                    "window_mae": float(g[err].mean()),
                    "n_subjects": int(per.size),
                    "n_windows": len(g),
                }
            )
        fam = []
        e = e.assign(yhat=e.y + e[err])  # subject_errors(|yhat − y|) then equals this error type
        ref = e[e.quality_class == "good"]
        for cls in ("degraded", "unreliable"):
            g = e[e.quality_class == cls]
            fam.append(
                {
                    "site": site,
                    "error": err,
                    "comparison": f"{cls} vs good",
                    **paired_test(subject_errors(g), subject_errors(ref), N_BOOT, seed=SEED),
                }
            )
        for r, p in zip(fam, holm([r["p_perm"] for r in fam]), strict=True):
            r["p_holm"] = p
        par += fam
        rs = []
        for _, g in e.groupby("subject_id"):
            if len(g) >= 20 and g.score.std() > 0:
                rs.append(stats.spearmanr(g.score, g[err])[0])
        se.append(
            {
                "site": site,
                "error": err,
                "within_subject_spearman_median": float(np.median(rs)),
                "frac_subjects_negative": float(np.mean(np.array(rs) < 0)),
                "n_subjects": len(rs),
            }
        )
        for thr in THRESHOLDS:
            k = e[e.score >= thr]
            per = k.groupby("subject_id")[err].mean()
            ret.append(
                {
                    "site": site,
                    "error": err,
                    "min_score": thr,
                    "window_share_kept": len(k) / len(e),
                    "subject_mae": float(per.mean()) if per.size else np.nan,
                    "n_subjects_ge10_windows": int((k.groupby("subject_id").size() >= 10).sum()),
                }
            )
    return summ, par, ret, se


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run = ExperimentRun(
        "quality_score_v4",
        [AUR / "windows.csv", AUR / "features.csv", *sorted((AUR / "waveforms").glob("*.npz"))],
        {
            "n_boot": N_BOOT,
            "thresholds": ";".join(map(str, THRESHOLDS)),
            "calibration": "offset from first 3 good windows per subject-session",
        },
        seed=SEED,
        dataset_versions={"auricle": "PhysioNet 1.0.0, standardised tables 2.0"},
        feature_config={"quality": "preprocessing/quality_score.py (pre-specified margins)"},
        model_config={"population": "ridge HR + shape, SQI-accepted session-1 training windows"},
    )
    with run.phase("grade"):
        q = grade_all()
    summ, par, ret, se, dist = [], [], [], [], []
    with run.phase("evaluate"):
        for site in ("ear", "finger"):
            qs = q[q.site == site]
            dist.append({"site": site, **qs.quality_class.value_counts(normalize=True).to_dict(), "n_windows": len(qs)})
            d = predict_all(site, q)
            run.splits.update({f"{site}:{k}": v for k, v in d.attrs["folds"].items()})
            a, b, c, e = analyse(d)
            summ, par, ret, se = summ + a, par + b, ret + c, se + e
    for name, rows in (
        ("class_summary", summ),
        ("paired", par),
        ("retention", ret),
        ("score_error", se),
        ("class_distribution", dist),
    ):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    run.add_metrics(pd.DataFrame(summ))
    run.finish()
    print(pd.DataFrame(dist).round(3).to_string(index=False))
    print(pd.DataFrame(summ).round(2).to_string(index=False))
    print(
        pd.DataFrame(par)[["site", "error", "comparison", "mean_diff", "ci_low", "ci_high", "p_holm"]]
        .round(3)
        .to_string(index=False)
    )
    print(pd.DataFrame(se).round(3).to_string(index=False))
    print(pd.DataFrame(ret).round(3).to_string(index=False))
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
