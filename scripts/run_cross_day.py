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


def load(rail_gate: bool = True) -> pd.DataFrame:
    a = pd.read_csv(PROCESSED / "auricle_features.csv")
    a = a[a.cohort == "HC"].reset_index(drop=True)
    if not rail_gate:  # ablation: undo only the rail-clipping rejection
        for s in SITES:
            only_rail = a[f"{s}_sqi_reasons"].fillna("") == "clipped_at_rail"
            a[f"{s}_sqi_ok"] = a[f"{s}_sqi_ok"].astype(bool) | only_rail
    for s in SITES:
        st.add_normalised_shape(a, s)
    return a


def evaluate_table(t: pd.DataFrame, n_boot: int) -> dict:
    m = st.point_metrics(t)
    m.update(st.fast_subject_cis(t, n_boot, SEED))
    return m


def _corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    rs = [np.corrcoef(x[i], y[i])[0, 1] for i in idx if np.std(x[i]) > 0 and np.std(y[i]) > 0]
    return np.percentile(rs, [2.5, 97.5]).tolist()


def day_effects(p: pd.DataFrame, target: str, n_boot: int) -> dict:
    """Per-subject analysis of what calibration carries across days.

    * rest_shift: day-2 minus day-1 median reference BP at rest (the true day-to-day change);
    * pred_shift: the same for the population prediction (does the model see the change?);
    * offset_d1 / offset_d2: per-subject calibration offset (reference - prediction at rest) on each day.
    Day-2 error of the offset-calibrated model at rest equals pred_shift - rest_shift, so the stability of
    the offset is exactly what determines cross-day absolute accuracy.
    """
    rows = []
    for sid, g in p.groupby("subject_id"):
        r1 = g[(g.session == 1) & (g.activity == 0)]
        r2 = g[(g.session == 2) & (g.activity == 0)]
        if len(r1) < K_CAL or len(r2) < K_CAL:
            continue
        rows.append(
            {
                "subject_id": sid,
                "rest_shift": r2[target].median() - r1[target].median(),
                "pred_shift": r2.pred.median() - r1.pred.median(),
                "offset_d1": (r1[target] - r1.pred).median(),
                "offset_d2": (r2[target] - r2.pred).median(),
            }
        )
    t = pd.DataFrame(rows)
    if len(t) < 5:
        return {"n_subjects": len(t)}
    x, y = t.rest_shift.to_numpy(), t.pred_shift.to_numpy()
    o1, o2 = t.offset_d1.to_numpy(), t.offset_d2.to_numpy()
    resid = y - x
    return {
        "n_subjects": len(t),
        "rest_shift_mean": float(x.mean()),
        "rest_shift_sd": float(x.std(ddof=1)),
        "rest_shift_abs_mean": float(np.abs(x).mean()),
        "r_pred_shift_vs_rest_shift": float(np.corrcoef(x, y)[0, 1]),
        "r_pred_shift_vs_rest_shift_ci": _corr_ci(x, y, n_boot),
        "slope_pred_shift_on_rest_shift": float(np.polyfit(x, y, 1)[0]),
        "offset_day1_day2_r": float(np.corrcoef(o1, o2)[0, 1]),
        "offset_day1_day2_r_ci": _corr_ci(o1, o2, n_boot),
        "offset_change_abs_mean": float(np.abs(o2 - o1).mean()),
        "offset_change_sd": float(np.std(o2 - o1, ddof=1)),
        "rest_error_after_offset_abs_mean": float(np.abs(resid).mean()),
        "rest_error_carry_forward_abs_mean": float(np.abs(x).mean()),
    }


def run(quick: bool = False, models_arg: str | None = None, rail_gate: bool = True, tag: str = "") -> None:
    global OUT
    OUT = ROOT / "results" / f"cross_day{tag}"
    t0 = time.time()
    n_boot = 500 if quick else 2000
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    a = load(rail_gate)
    both_sessions = a.groupby("subject_id").session.nunique()
    summary = {
        "subjects": int(a.subject_id.nunique()),
        "subjects_with_both_sessions": int((both_sessions == 2).sum()),
        "windows_by_session": a.groupby("session").size().to_dict(),
        "k_cal_rest_windows": K_CAL,
        "n_boot": n_boot,
        "seed": SEED,
        "rail_gate": rail_gate,
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

    models = tuple(models_arg.split(",")) if models_arg else (("ridge",) if quick else EXTENDED_MODELS)
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
                        PRED / f"cross_day{tag}_{site}_{target}_{fset}_{model}.csv.gz", index=False
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
    for site in SITES:  # every population model vs ridge, same features, cross-day offset calibration
        for target in TARGETS:
            for mod in models:
                if mod != "ridge":
                    pair(
                        f"{site} {target} cross_day: {mod} vs ridge (offset hr_shape)",
                        (site, target, "hr_shape", mod, "cross_day", "offset"),
                        (site, target, "hr_shape", "ridge", "cross_day", "offset"),
                    )
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_tests.csv", index=False)

    # ---- generalisability ranking: cross-day error, its CI, and the within->cross-day gap, not correlation
    pop = metrics[(metrics.feature_set == "hr_shape") & (metrics["mode"] == "offset") & metrics.model.isin(models)]
    if len(pop):
        cd = pop[pop.scope == "cross_day"].set_index(["model", "site", "target"]).mae
        wd = pop[pop.scope == "within_day"].set_index(["model", "site", "target"]).mae
        rank = cd.groupby(["site", "target"]).rank()
        table = pd.DataFrame(
            {
                "cross_day_mae_mean": cd.groupby("model").mean(),
                "mean_rank": rank.groupby("model").mean(),
                "worst_rank": rank.groupby("model").max(),
                "within_to_cross_day_gap": (cd - wd).groupby("model").mean(),
            }
        ).sort_values("mean_rank")
        table.to_csv(OUT / "model_ranking.csv")
        summary["model_ranking"] = table.round(3).reset_index().to_dict("records")

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
        p = pd.read_csv(PRED / f"cross_day{tag}_{site}_sbp_shape_only_ridge.csv.gz")
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
        q = pd.read_csv(PRED / f"cross_day{tag}_{site}_sbp_hr_shape_ridge.csv.gz")
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
        for target in TARGETS:
            inv[f"{target}_day_effects"] = day_effects(
                pd.read_csv(PRED / f"cross_day{tag}_{site}_{target}_hr_shape_ridge.csv.gz"), target, n_boot
            )
            key_u = (site, target, "hr_shape", "ridge", "cross_day", "uncalibrated")
            key_c = (site, target, "hr_shape", "ridge", "cross_day", "carry_forward")
            key_o = (site, target, "hr_shape", "ridge", "cross_day", "offset")
            if key_u in store and key_c in store:
                # absolute estimation without calibration vs the trivial baselines
                inv[f"{target}_uncalibrated_vs_carry_forward"] = st.paired_subject_diff(
                    store[key_u], store[key_c], n_boot
                )
                inv[f"{target}_offset_vs_uncalibrated"] = st.paired_subject_diff(store[key_o], store[key_u], n_boot)
        invest[site] = inv
    summary["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (OUT / "investigations.json").write_text(json.dumps(invest, indent=2, default=float))
    log.info("finished in %.0fs", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="ridge only, 500 bootstrap draws")
    ap.add_argument("--models", help="comma-separated population models (default: all registered)")
    ap.add_argument("--no-rail-gate", action="store_true", help="ablation: keep rail-clipped windows")
    ap.add_argument("--tag", default="", help="suffix for output directory / prediction files")
    a = ap.parse_args()
    run(a.quick, a.models, not a.no_rail_gate, a.tag)
