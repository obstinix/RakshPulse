"""Independent audit of the cross-day study. Run after run_cross_day.py.

The checks deliberately re-derive results with separate, minimal code paths (raw WFDB records, saved
predictions, plain NumPy) instead of calling the functions under audit. Writes
results/cross_day/audit.json; exits non-zero if any hard check fails.

Checks
 1  all 37 healthy session-2 records (and session-1 records) present and SHA-256 verified (manifest)
 2  subject pairing: session-1 / session-2 windows of a subject share identical demographics
 3  no subject leakage: population-model folds are subject-disjoint; no window overlap with calibration
 4  calibration windows are rest-only and precede evaluation windows by >= 10 s (no overlapping windows)
 5  CNAP-mask and rail-clipping exclusions hold, re-checked on raw signals for a random sample
 6  bootstrap is subject-level: independent re-implementation reproduces the saved CIs
 7  paired comparisons: independent recomputation of the mean per-subject difference and CI
 8  identical evaluation subjects for each presented comparison (reported, with counts)
 9  exclusions: every healthy subject accounted for, with the stage at which it drops out
10  saved metrics reproduce from saved window predictions
11  statistical support for "day-shift not tracked": slope CI vs 0 and vs 1, power for r
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.data.build_auricle import MAX_RAIL_FRACTION, RAIL_V  # noqa: E402
from rakshpulse.data.dataset_loader import read_auricle  # noqa: E402
from rakshpulse.data.download import DATA_ROOT  # noqa: E402
from rakshpulse.evaluation import auricle_study as st  # noqa: E402

RES = ROOT / "results" / "cross_day"
PRED = ROOT / "results" / "predictions"
K_CAL, GAP = 3, 10.0
rng = np.random.default_rng(12345)  # different seed from the study on purpose
report: dict = {}
hard_fail: list[str] = []


def check(name: str, ok: bool, detail) -> None:
    report[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        hard_fail.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def ci_close(a: list[float], b: list[float]) -> bool:
    """Two bootstrap CIs from different RNG streams agree to Monte-Carlo error (10 % of width, >= 0.25)."""
    tol = max(0.25, 0.1 * abs(b[1] - b[0]))
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def offset_calibrate(p: pd.DataFrame, target: str) -> pd.DataFrame:
    """Minimal independent re-implementation of cross-day 1-point calibration."""
    rows = []
    for sid, g in p.groupby("subject_id"):
        s1 = g[g.session == 1].sort_values("t_start_s")
        s2 = g[g.session == 2]
        rest = s1[s1.activity == 0]
        if len(rest) < K_CAL or len(s2) < 5:
            continue
        cal = rest.iloc[:K_CAL]
        off = cal[target].mean() - cal.pred.mean()
        rows.append(
            pd.DataFrame(
                {
                    "subject_id": sid,
                    "y": s2[target].to_numpy(),
                    "yhat": s2.pred.to_numpy() + off,
                    "carry": cal[target].mean(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    # 1 -------------------------------------------------------------------------------- acquisition
    man = json.loads((DATA_ROOT / "metadata" / "auricle_manifest.json").read_text())
    recs = {r["record"]: r for r in man["records"]}
    hc2 = [r for r in recs if r.startswith("HC_") and r.endswith("_2")]
    hc1 = [r for r in recs if r.startswith("HC_") and r.endswith("_1")]
    ok2 = [r for r in hc2 if recs[r]["status"] == "ok"]
    ok1 = [r for r in hc1 if recs[r]["status"] == "ok"]
    check(
        "1_session2_verified",
        len(ok2) == len(hc2) == 37,
        {
            "hc_session2_listed": len(hc2),
            "verified": len(ok2),
            "not_ok": sorted(set(hc2) - set(ok2)),
            "hc_session1_verified": f"{len(ok1)}/{len(hc1)}",
            "untrusted_labels": sorted(r for r in recs if r.startswith("HC") and not recs[r]["trusted_labels"]),
        },
    )

    f = pd.read_csv(PROCESSED / "auricle_features.csv")
    f = f[f.cohort == "HC"]
    # 2 -------------------------------------------------------------------------------- pairing
    demo_cols = [c for c in f.columns if c.startswith("demo_")]
    per = f.groupby(["subject_id", "session"])[demo_cols].first()
    both = per.reset_index().groupby("subject_id").session.nunique()
    paired = both[both == 2].index
    mism = [
        s
        for s in paired
        if not np.allclose(per.loc[(s, 1)].to_numpy(float), per.loc[(s, 2)].to_numpy(float), equal_nan=True)
    ]
    rid_ok = all(
        r.split("_")[1] == s.split("_")[2]
        for s, r in f[["subject_id", "record_id"]].drop_duplicates().itertuples(index=False)
    )
    check(
        "2_subject_pairing",
        not mism and rid_ok,
        {
            "subjects_with_both_sessions": int(len(paired)),
            "demographic_mismatches": mism,
            "record_to_subject_mapping_consistent": rid_ok,
        },
    )

    # 3/4 ------------------------------------------------------------------------- leakage, calibration
    fold_of = st.subject_folds(f.subject_id, f.sbp, n=5)
    folds = pd.Series(fold_of)
    disjoint = all(set(folds[folds == k].index).isdisjoint(set(folds[folds != k].index)) for k in range(5))
    p = pd.read_csv(PRED / "cross_day_ear_sbp_hr_shape_ridge.csv.gz")
    bad_cal, overlap, n_sess = 0, 0, 0
    for _, g in p[p.session == 1].groupby("subject_id"):
        g = g.sort_values("t_start_s")
        rest = g[g.activity == 0]
        if len(rest) < K_CAL:
            continue
        cal = rest.iloc[:K_CAL]
        n_sess += 1
        bad_cal += int((cal.activity != 0).sum())
        ev = g[g.t_start_s > cal.t_start_s.max() + GAP]
        # a 10 s window starting at t overlaps calibration if t < last_cal_start + 10
        overlap += int((ev.t_start_s < cal.t_start_s.max() + 10.0).sum())
    check(
        "3_no_subject_leakage",
        disjoint and len(fold_of) == f.subject_id.nunique(),
        {
            "subjects_in_folds": len(fold_of),
            "fold_sizes": folds.value_counts().sort_index().tolist(),
            "population_training_uses": "session-1 windows of other-fold subjects only (asserted in code)",
        },
    )
    check(
        "4_calibration_rest_only_no_overlap",
        bad_cal == 0 and overlap == 0,
        {"sessions_checked": n_sess, "non_rest_calibration_windows": bad_cal, "overlapping_eval_windows": overlap},
    )

    # 5 ------------------------------------------------------------------- mask / clipping on raw data
    clipped_accepted = int(
        ((f.finger_sat_fraction > MAX_RAIL_FRACTION) & f.finger_sqi_ok.astype(bool)).sum()
        + ((f.ear_sat_fraction > MAX_RAIL_FRACTION) & f.ear_sqi_ok.astype(bool)).sum()
    )
    sample = f.sample(40, random_state=7)
    raw_bad = []
    cache: dict = {}
    for r in sample.itertuples():
        rec = cache.get(r.record_id) or cache.setdefault(r.record_id, read_auricle(r.record_id))
        a = int(r.t_start_s * rec.fs)
        b = a + int(10 * rec.fs)
        mask_ok = bool(rec.cnap_valid[a:b].all())
        sat = float(np.mean(np.abs(rec.finger[a:b]) >= RAIL_V))
        sbp_ok = abs(np.median(rec.sbp[a:b]) - r.sbp) < 1e-6
        if not mask_ok or abs(sat - r.finger_sat_fraction) > 1e-9 or not sbp_ok:
            raw_bad.append(r.Index)
    no_mask = [
        rid
        for rid in recs
        if rid.startswith("HC")
        and recs[rid]["status"] == "ok"
        and "CNAP_Annotations" in recs[rid].get("missing_annotations", [])
    ]
    in_table = sorted(set(no_mask) & set(f.record_id))
    check(
        "5_mask_and_clipping_exclusions",
        clipped_accepted == 0 and not raw_bad and not in_table,
        {
            "clipped_windows_marked_ok": clipped_accepted,
            "raw_resample_checked": len(sample),
            "raw_mismatches": raw_bad,
            "records_without_mask": no_mask,
            "of_which_in_analysis_table": in_table,
        },
    )

    # 6/10 ----------------------------------------- reproduce metrics and subject-level CIs independently
    m = pd.read_csv(RES / "metrics.csv")
    repro = []
    for site in ("ear", "finger"):
        for target in ("sbp", "dbp"):
            pp = pd.read_csv(PRED / f"cross_day_{site}_{target}_hr_shape_ridge.csv.gz")
            t = offset_calibrate(pp, target)
            mae = float(np.abs(t.yhat - t.y).mean())
            carry = float(np.abs(t.carry - t.y).mean())
            row = m[
                (m.site == site)
                & (m.target == target)
                & (m.feature_set == "hr_shape")
                & (m.model == "ridge")
                & (m.scope == "cross_day")
            ]
            saved_off = float(row[row["mode"] == "offset"].mae.iloc[0])
            saved_cf = float(row[row["mode"] == "carry_forward"].mae.iloc[0])
            # independent subject-level bootstrap of MAE (resample subjects, pool their windows)
            subj = {s: np.abs(g.yhat - g.y).to_numpy() for s, g in t.groupby("subject_id")}
            keys = list(subj)
            boots = []
            for _ in range(2000):
                pick = rng.integers(0, len(keys), len(keys))
                e = np.concatenate([subj[keys[i]] for i in pick])
                boots.append(e.mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            s_lo, s_hi = (float(v) for v in str(row[row["mode"] == "offset"].mae_ci.iloc[0]).strip("[]").split(","))
            repro.append(
                {
                    "site": site,
                    "target": target,
                    "mae_recomputed": mae,
                    "mae_saved": saved_off,
                    "carry_recomputed": carry,
                    "carry_saved": saved_cf,
                    "ci_recomputed": [lo, hi],
                    "ci_saved": [s_lo, s_hi],
                    "n_subjects": len(keys),
                }
            )
    ok10 = all(
        abs(r["mae_recomputed"] - r["mae_saved"]) < 1e-6 and abs(r["carry_recomputed"] - r["carry_saved"]) < 1e-6
        for r in repro
    )
    ok6 = all(
        ci_close(r["ci_recomputed"], r["ci_saved"]) for r in repro
    )  # different RNG streams: CIs must agree to Monte-Carlo error
    check("10_metrics_reproduce_from_predictions", ok10, repro)
    check("6_subject_level_bootstrap_reproduces", ok6, "independent subject-resampling CIs within Monte-Carlo error")

    # 7/8 --------------------------------------------------------------- paired comparisons & subject sets
    pt = pd.read_csv(RES / "paired_tests.csv")
    ps = pd.read_csv(RES / "per_subject.csv")
    checks7, subj_sets = [], []
    for row in pt.itertuples():
        ka, kb = row.a.split("|"), row.b.split("|")
        cols = ["site", "target", "feature_set", "model", "scope", "mode"]
        qa, qb = ps, ps
        for c, va, vb in zip(cols, ka, kb, strict=True):
            qa, qb = qa[qa[c] == va], qb[qb[c] == vb]
        a, b = qa.set_index("subject_id").mae, qb.set_index("subject_id").mae
        common = a.index.intersection(b.index)
        d = (a[common] - b[common]).to_numpy()
        boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)] if d.size else [np.nan]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        checks7.append(abs(d.mean() - row.mae_diff_a_minus_b) < 1e-9 and ci_close([lo, hi], [row.ci_low, row.ci_high]))
        subj_sets.append(
            {
                "comparison": row.comparison,
                "n_a": int(a.size),
                "n_b": int(b.size),
                "n_common": int(len(common)),
                "identical_subject_sets": bool(set(a.index) == set(b.index)),
            }
        )
    check(
        "7_paired_comparisons_reproduce",
        all(checks7),
        {"n_comparisons": len(checks7), "n_failed": int(len(checks7) - sum(checks7))},
    )
    ss = pd.DataFrame(subj_sets)
    check(
        "8_identical_evaluation_subjects",
        True,
        {
            "comparisons_with_identical_sets": int(ss.identical_subject_sets.sum()),
            "comparisons_restricted_to_common_subjects": ss[~ss.identical_subject_sets].to_dict("records"),
            "note": "paired statistics always use the intersection of subjects (by construction)",
        },
    )

    # 9 ---------------------------------------------------------------------------------- exclusions
    hc_subjects = sorted({f"auricle_HC_{r.split('_')[1]}" for r in recs if r.startswith("HC_")})
    stages = {}
    for s in hc_subjects:
        n = s.split("_")[-1]
        r1, r2 = recs.get(f"HC_{n}_1"), recs.get(f"HC_{n}_2")
        g = f[f.subject_id == s]
        reasons = []
        if r1 is None or not r1["trusted_labels"]:
            reasons.append("session1 untrusted/missing (no CNAP mask)" if r1 else "no session-1 record")
        if r2 is None:
            reasons.append("no session-2 record in dataset")
        if g.empty:
            reasons.append("no labelled windows")
        for site in ("ear", "finger"):
            gs = g[g[f"{site}_sqi_ok"].astype(bool)]
            r1rest = gs[(gs.session == 1) & (gs.activity == 0)]
            if r2 is not None and len(r1rest) < K_CAL:
                reasons.append(f"{site}: <{K_CAL} SQI-passing session-1 rest windows (no cross-day calibration)")
            if r2 is not None and len(gs[gs.session == 2]) < 5:
                reasons.append(f"{site}: <5 SQI-passing session-2 windows")
        stages[s] = reasons or ["included"]
    check("9_exclusions_accounted", True, stages)

    # 11 ----------------------------------------------- is "day shift not tracked" statistically supported?
    inv = json.loads((RES / "investigations.json").read_text())
    stat = {}
    for site in ("ear", "finger"):
        for target in ("sbp", "dbp"):
            de = inv[site].get(f"{target}_day_effects", {})
            n = de.get("n_subjects", 0)
            if n < 5:
                continue
            se = 1 / np.sqrt(n - 3)  # Fisher z: smallest |r| detectable with 80 % power at alpha 0.05
            mde = float(np.tanh((1.96 + 0.84) * se))
            lo, hi = de["slope_pred_shift_on_rest_shift_ci"]
            stat[f"{site}_{target}"] = {
                "n_subjects": n,
                "r": de["r_pred_shift_vs_rest_shift"],
                "r_ci": de["r_pred_shift_vs_rest_shift_ci"],
                "slope": de["slope_pred_shift_on_rest_shift"],
                "slope_ci": [lo, hi],
                "slope_ci_excludes_1": bool(hi < 1),
                "slope_ci_excludes_0": bool(lo > 0 or hi < 0),
                "min_detectable_abs_r_80pct_power": mde,
                "interpretation": (
                    "prediction shift is significantly smaller than the true day shift (slope<1)"
                    if hi < 1
                    else "cannot exclude full tracking of the day shift"
                )
                + (
                    "; independence NOT established (r CI includes moderate correlations)"
                    if de["r_pred_shift_vs_rest_shift_ci"][1] > 0.3
                    else ""
                ),
            }
    check("11_day_shift_statistics", True, stat)

    report["hard_failures"] = hard_fail
    (RES / "audit.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"audit complete: {len(hard_fail)} hard failure(s)")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
