"""Auricle cross-day BP-tracking study (healthy cohort). Calibrate on session 1, evaluate on session 2.

Outputs (lightweight, committed):  results/cross_day/{summary.json, metrics.csv, paired_tests.csv,
                                   per_subject.csv, investigations.json}
Window-level predictions (regenerable, git-ignored): results/predictions/cross_day_*.csv.gz

Usage:  python scripts/run_cross_day.py [--quick]   (--quick: ridge only, 500 bootstrap draws)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.features.feature_pipeline import PER_CHANNEL  # noqa: E402
from rakshpulse.models.model_factory import EXTENDED_MODELS  # noqa: E402

OUT = ROOT / "results" / "cross_day"
PRED = ROOT / "results" / "predictions"
SITES = ("ear", "finger")
TARGETS = ("sbp", "dbp")
FEATURE_SETS = ("hr_only", "shape_only", "hr_shape", "ppg_all")
K_CAL = 3
SEED = 0
log = logging.getLogger("cross_day")


def load() -> pd.DataFrame:
    a = pd.read_csv(PROCESSED / "auricle_features.csv")
    a = a[a.cohort == "HC"].reset_index(drop=True)
    for s in SITES:
        st.add_normalised_shape(a, s)
    return a


def evaluate_table(t: pd.DataFrame, n_boot: int) -> dict:
    m = st.point_metrics(t)
    m.update(st.fast_subject_cis(t, n_boot, SEED))
    return m


def run(quick: bool = False) -> None:
    t0 = time.time()
    n_boot = 500 if quick else 2000
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    a = load()
    both_sessions = a.groupby("subject_id").session.nunique()
    summary = {
        "subjects": int(a.subject_id.nunique()),
        "subjects_with_both_sessions": int((both_sessions == 2).sum()),
        "windows_by_session": a.groupby("session").size().to_dict(),
        "k_cal_rest_windows": K_CAL,
        "n_boot": n_boot,
        "seed": SEED,
    }
    log.info("data: %s", summary)
    fold_of = st.subject_folds(a.subject_id, a.sbp, n=5)
    metric_rows, paired_rows, subj_rows, invest = [], [], [], {}
    store: dict[tuple, pd.DataFrame] = {}

    def record(site, target, fset, model, t: pd.DataFrame, extra: dict | None = None):
        for (scope, mode), g in t.groupby(["scope", "mode"]):
            r = {
                "site": site,
                "target": target,
                "feature_set": fset,
                "model": model,
                "scope": scope,
                "mode": mode,
                **(extra or {}),
                **evaluate_table(g, n_boot),
            }
            if scope == "cross_day" and mode in ("offset", "linear", "personal_s1", "carry_forward"):
                r.update(st.calibration_drift(g))
            metric_rows.append(r)
            store[(site, target, fset, model, scope, mode)] = g
            ps = (g.yhat - g.y).abs().groupby(g.subject_id).mean()
            for sid, v in ps.items():
                subj_rows.append(
                    {
                        "site": site,
                        "target": target,
                        "feature_set": fset,
                        "model": model,
                        "scope": scope,
                        "mode": mode,
                        "subject_id": sid,
                        "mae": v,
                    }
                )

    models = ("ridge",) if quick else EXTENDED_MODELS
    for site in SITES:
        d = a[a[f"{site}_sqi_ok"].astype(bool)].reset_index(drop=True)
        s1 = d[d.session == 1].reset_index(drop=True)
        all_cols = [f"{site}_{n}" for n in PER_CHANNEL]
        for target in TARGETS:
            for fset in FEATURE_SETS:
                cols = st.feature_set(fset, site, all_cols)
                for model in models if fset == "hr_shape" else ("ridge",):
                    pred = st.population_predictions(s1, d, cols, target, model, fold_of, seed=SEED)
                    dd = d.assign(pred=pred)
                    t = st.calibrate(dd, target, K_CAL).table
                    record(site, target, fset, model, t)
                    dd[["subject_id", "session", "t_start_s", "activity", target, "pred"]].to_csv(
                        PRED / f"cross_day_{site}_{target}_{fset}_{model}.csv.gz", index=False
                    )
                    log.info("%s %s %-10s %-13s done (%.0fs)", site, target, fset, model, time.time() - t0)
                if fset in ("hr_only", "hr_shape"):
                    pm = st.personal_model_predictions(d, cols, target)
                    record(site, target, fset, "personal_ridge", pm)
        # ---- calibration transfer: finger-trained population model applied to ear windows
        if site == "ear":
            d_e = d
            s1_f = a[(a.session == 1) & a.finger_sqi_ok.astype(bool)].reset_index(drop=True)
            for target in TARGETS:
                f_cols = st.feature_set("hr_shape", "finger", [])
                e_cols = st.feature_set("hr_shape", "ear", [])
                pred = st.population_predictions(s1_f, d_e, f_cols, target, "ridge", fold_of, test_cols=e_cols)
                t = st.calibrate(d_e.assign(pred=pred), target, K_CAL).table
                record("ear", target, "hr_shape", "ridge_trained_on_finger", t)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(OUT / "per_subject.csv", index=False)

    # ---- paired subject-level comparisons
    def pair(label, ka, kb):
        if ka in store and kb in store:
            dmean, lo, hi = st.paired_subject_diff(store[ka], store[kb], n_boot, SEED)
            paired_rows.append(
                {
                    "comparison": label,
                    "a": "|".join(map(str, ka)),
                    "b": "|".join(map(str, kb)),
                    "mae_diff_a_minus_b": dmean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(np.isfinite(hi) and (hi < 0 or lo > 0)),
                }
            )

    for site in SITES:
        for target in TARGETS:
            for scope in ("within_day", "cross_day"):
                k = lambda fs, mode, model="ridge", _s=site, _t=target, _sc=scope: (_s, _t, fs, model, _sc, mode)  # noqa: E731
                pair(
                    f"{site} {target} {scope}: offset hr_shape vs carry-forward",
                    k("hr_shape", "offset"),
                    k("hr_shape", "carry_forward"),
                )
                pair(
                    f"{site} {target} {scope}: offset hr_only vs carry-forward",
                    k("hr_only", "offset"),
                    k("hr_only", "carry_forward"),
                )
                pair(
                    f"{site} {target} {scope}: offset hr_shape vs offset hr_only",
                    k("hr_shape", "offset"),
                    k("hr_only", "offset"),
                )
                pair(
                    f"{site} {target} {scope}: offset shape_only vs offset hr_only",
                    k("shape_only", "offset"),
                    k("hr_only", "offset"),
                )
            ck = lambda fs, mode, model="ridge", _s=site, _t=target: (_s, _t, fs, model, "cross_day", mode)  # noqa: E731
            pair(
                f"{site} {target} cross_day: linear hr_shape vs offset hr_shape",
                ck("hr_shape", "linear"),
                ck("hr_shape", "offset"),
            )
            pair(
                f"{site} {target} cross_day: personal hr_shape vs personal hr_only",
                ck("hr_shape", "personal_s1", "personal_ridge"),
                ck("hr_only", "personal_s1", "personal_ridge"),
            )
            pair(
                f"{site} {target} cross_day: personal hr_shape vs carry-forward",
                ck("hr_shape", "personal_s1", "personal_ridge"),
                ck("hr_shape", "carry_forward"),
            )
    for target in TARGETS:
        for mode in ("offset", "linear"):
            pair(
                f"ear {target} cross_day: finger-trained vs ear-trained ({mode})",
                ("ear", target, "hr_shape", "ridge_trained_on_finger", "cross_day", mode),
                ("ear", target, "hr_shape", "ridge", "cross_day", mode),
            )
        pair(
            f"{target} cross_day: ear vs finger (offset hr_shape, each site's own windows)",
            ("ear", target, "hr_shape", "ridge", "cross_day", "offset"),
            ("finger", target, "hr_shape", "ridge", "cross_day", "offset"),
        )
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_tests.csv", index=False)

    # ---- investigations
    for site in SITES:
        d = a[a[f"{site}_sqi_ok"].astype(bool)]
        inv = {}
        # (1) how much of within-subject BP variation co-varies with heart rate?
        r_hr = [
            stats.pearsonr(g[f"{site}_hr"], g.sbp)[0]
            for _, g in d.groupby("record_id")
            if len(g) >= 10 and g.sbp.std() > 2 and g[f"{site}_hr"].std() > 0
        ]
        inv["within_session_r_sbp_vs_hr_median"] = float(np.median(r_hr))
        # (2) partial correlation: does the shape prediction explain BP beyond HR, within sessions?
        p = pd.read_csv(PRED / f"cross_day_{site}_sbp_shape_only_ridge.csv.gz")
        p = p.merge(d[["subject_id", "session", "t_start_s", f"{site}_hr"]], on=["subject_id", "session", "t_start_s"])
        part = []
        for _, g in p.groupby(["subject_id", "session"]):
            if len(g) < 15 or g.sbp.std() <= 2:
                continue
            X = np.column_stack([np.ones(len(g)), g[f"{site}_hr"]])
            ry = g.sbp - X @ np.linalg.lstsq(X, g.sbp, rcond=None)[0]
            rp = g.pred - X @ np.linalg.lstsq(X, g.pred, rcond=None)[0]
            if ry.std() > 0 and rp.std() > 0:
                part.append(stats.pearsonr(ry, rp)[0])
        inv["within_session_partial_r_sbp_shape_given_hr_median"] = float(np.median(part)) if part else None
        inv["within_session_partial_r_n_sessions"] = len(part)
        if part:
            boot = np.median(
                np.asarray(part)[np.random.default_rng(SEED).integers(0, len(part), (n_boot, len(part)))], 1
            )
            inv["partial_r_median_ci"] = np.percentile(boot, [2.5, 97.5]).tolist()
        # (3) subject dominance in calibration-free predictions
        q = pd.read_csv(PRED / f"cross_day_{site}_sbp_hr_shape_ridge.csv.gz")
        between = q.groupby("subject_id").pred.mean().var()
        inv["uncalibrated_prediction_between_subject_variance_fraction"] = float(between / q.pred.var())
        sm = q.groupby("subject_id")[["sbp", "pred"]].mean()
        inv["subject_mean_r_truth_vs_pred"] = float(stats.pearsonr(sm.sbp, sm.pred)[0])
        inv["truth_between_subject_variance_fraction"] = float(q.groupby("subject_id").sbp.mean().var() / q.sbp.var())
        # (4) session-to-session change of each subject's resting BP (what calibration drift must absorb)
        rest = d[d.activity == 0].groupby(["subject_id", "session"]).sbp.median().unstack()
        ch = (rest[2] - rest[1]).dropna() if 2 in rest else pd.Series(dtype=float)
        inv["rest_sbp_day2_minus_day1"] = {
            "n": int(ch.size),
            "mean": float(ch.mean()) if ch.size else None,
            "sd": float(ch.std()) if ch.size > 1 else None,
            "abs_mean": float(ch.abs().mean()) if ch.size else None,
        }
        # (5) error by activity for the main configuration
        key = (site, "sbp", "hr_shape", "ridge", "cross_day", "offset")
        if key in store:
            g = store[key]
            inv["cross_day_offset_mae_by_activity"] = (
                (g.yhat - g.y).abs().groupby(g.activity).mean().round(2).rename(str).to_dict()
            )
        invest[site] = inv
    summary["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (OUT / "investigations.json").write_text(json.dumps(invest, indent=2, default=float))
    log.info("finished in %.0fs", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(quick="--quick" in sys.argv)
