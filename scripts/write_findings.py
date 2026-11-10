"""Render the results tables of docs/RESEARCH_FINDINGS.md from audited outputs (no hand-typed numbers).

Only the block between ``<!-- BEGIN AUTO -->`` and ``<!-- END AUTO -->`` is rewritten. Interpretation text
outside the markers is preserved. Refuses to run if the audit has hard failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "cross_day"
DOC = ROOT / "docs" / "RESEARCH_FINDINGS.md"
B, E = "<!-- BEGIN AUTO -->", "<!-- END AUTO -->"
BP_, EP_ = "<!-- BEGIN PERSONALIZATION -->", "<!-- END PERSONALIZATION -->"
PERS = ROOT / "results" / "personalization"
BV, EV = "<!-- BEGIN V3 -->", "<!-- END V3 -->"


def ci(s) -> str:
    try:
        lo, hi = (float(x) for x in str(s).strip("[]").split(","))
        return f"[{lo:.2f}, {hi:.2f}]"
    except ValueError:
        return "—"


def f2(x) -> str:
    return "—" if pd.isna(x) else f"{x:.2f}"


def table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()])


def personalization_block() -> str:
    """Tables for the population + personalisation experiments (results/personalization/)."""
    pop = pd.read_csv(PERS / "population_models.csv")
    pop["session"] = pop.evaluated_on.str.extract(r"session (\d)")[0]
    strat = pd.read_csv(PERS / "cross_day_strategies.csv")
    pair = pd.read_csv(PERS / "paired_comparisons.csv")
    ad = pd.read_csv(PERS / "adaptive_within_day.csv")
    dr = pd.read_csv(PERS / "drift_monitor.csv")
    bl = pd.read_csv(PERS / "baseline_tracking.csv")
    out = ["### P1. Population models on unseen subjects (uncalibrated; HR + shape features)", ""]
    rows = []
    for (site, target, model), g in pop.groupby(["site", "target", "model"], sort=False):
        r1, r2 = g[g.session == "1"].iloc[0], g[g.session == "2"].iloc[0]
        rows.append(
            [
                site,
                target.upper(),
                model,
                f2(r1.mae),
                ci(r1.mae_ci),
                f2(r2.mae),
                ci(r2.mae_ci),
                f2(r1.within_subject_pearson_median),
                int(r1.n_subjects),
            ]
        )
    out += [
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Target",
                    "Model",
                    "MAE S1",
                    "95% CI",
                    "MAE S2",
                    "95% CI",
                    "Within-subj r (S1)",
                    "n subj (S1)",
                ],
            )
        ),
        "",
    ]
    vm = pair[pair.comparison.str.contains("vs mean")]
    better = vm[vm.significant & (vm.mae_diff < 0)]
    worse = vm[vm.significant & (vm.mae_diff > 0)]
    worse_models = sorted(set(worse.comparison.str.extract(r"population (\w+) vs")[0]))
    out += [
        f"Paired tests of each model against the training-mean predictor: {len(vm)} tests, "
        f"{len(better)} significantly better, {len(worse)} significantly worse ({', '.join(worse_models)}).",
        "",
    ]
    out += ["### P2. Cross-day strategies (common day-2 evaluation set)", ""]
    rows = [
        [
            r.site,
            r.target.upper(),
            r.method,
            r.information,
            f2(r.mae),
            ci(r.mae_ci),
            f2(r.rmse),
            f2(r.bias),
            f2(r.sd_error),
            f2(r.pooled_pearson),
            f2(r.pooled_spearman),
            f2(r.within_subject_pearson_median),
            int(r.n_subjects),
        ]
        for r in strat.itertuples()
    ]
    out += [
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Target",
                    "Strategy",
                    "Information used",
                    "MAE",
                    "95% CI",
                    "RMSE",
                    "Bias",
                    "SD err",
                    "Pearson",
                    "Spearman",
                    "Within-subj r",
                    "n subj",
                ],
            )
        ),
        "",
    ]
    keep = pair[~pair.comparison.str.contains("vs mean")]
    out += [
        "### P3. Paired subject-level comparisons (Δ = MAE(a) − MAE(b); negative favours a)",
        "",
        table(keep[["site", "target", "comparison", "mae_diff", "ci_low", "ci_high", "significant"]].round(2)),
        "",
    ]
    rows = [[r.site, int(r.session), r.method, f2(r.mae), ci(r.mae_ci), int(r.n_subjects)] for r in ad.itertuples()]
    out += [
        "### P4. Adaptive within-day tracking (a reference every 2 min; strictly causal; SBP)",
        "",
        table(pd.DataFrame(rows, columns=["Site", "Session", "Method", "MAE", "95% CI", "n subj"])),
        "",
    ]
    rows = []
    for site, g in dr.groupby("site"):
        rec = g[g.recommended]
        rows.append(
            [
                site,
                len(g),
                len(rec),
                f2(rec.day2_mae_with_day1_cal.mean()),
                f2(g[~g.recommended].day2_mae_with_day1_cal.mean()),
                f2(g.first_day2_residual_sbp.abs().corr(g.day2_mae_with_day1_cal)),
            ]
        )
    out += [
        "### P5. Drift monitor (first day-2 reference vs the day-1 calibration)",
        "",
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Subjects",
                    "Recalibration recommended",
                    "Day-2 MAE if recommended",
                    "Day-2 MAE otherwise",
                    "r(abs first residual, day-2 MAE)",
                ],
            )
        ),
        "",
    ]
    rows = []
    for site, g in bl.groupby("site"):
        rows.append(
            [
                site,
                len(g),
                f2(g.n_day1_refs.median()),
                f2(g.sbp_minus_baseline.mean()),
                f2(g.sbp_minus_baseline.std()),
                f2((g.sbp_z.abs() > 2).mean() * 100),
            ]
        )
    out += [
        "### P6. Baseline tracking (day-2 resting SBP vs the day-1 personal reference baseline)",
        "",
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Subjects",
                    "Median day-1 references",
                    "Mean shift (mmHg)",
                    "SD of shift",
                    "% subjects with abs(z) > 2",
                ],
            )
        ),
        "",
    ]
    return "\n".join(out)


def _d(r) -> str:
    """Paired difference with CI; * marks a CI excluding 0."""
    if r is None:
        return "—"
    star = "*" if bool(r["significant"]) else ""
    return f"{r['mae_diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]{star}"


def v3_block() -> str:
    """RakshPulse 3.0 tables (beyond-reference, calibration count, features, deep models, VitalDB, latency)."""
    out = []
    br = ROOT / "results" / "beyond_reference"
    if (br / "metrics.csv").exists():
        m, p = pd.read_csv(br / "metrics.csv"), pd.read_csv(br / "paired.csv")
        out += [
            "### V1. Does the waveform add information beyond the last reference? (Δ MAE vs the `mean_change` "
            "control, mmHg; negative = the signal helps; * = 95 % subject-bootstrap CI excludes 0)",
            "",
        ]
        rows = []
        keys = ["dataset", "site", "target", "scenario"]
        for k, g in m.groupby(keys, sort=False):
            pp = p.merge(pd.DataFrame([dict(zip(keys, k, strict=True))]), on=keys)

            def get(comp, pp=pp):
                q = pp[pp.comparison == comp]
                return q.iloc[0] if len(q) else None

            mae = g.set_index("method").mae
            rows.append(
                [
                    *k,
                    f2(mae.get("last_reference")),
                    f2(mae.get("population_mean")),
                    _d(get("delta_hr_only vs mean_change")),
                    _d(get("delta_hr_shape vs mean_change")),
                    _d(get("delta_ppg_all vs mean_change")),
                    _d(get("delta_hr_shape_pat vs mean_change")),
                    _d(get("rakshpulse2_offset vs last_reference")),
                    int(g.n_subjects.iloc[0]),
                ]
            )
        cols = [
            "Data",
            "Site",
            "Target",
            "Reference scenario",
            "MAE last ref",
            "MAE pop. mean",
            "HR only",
            "HR + shape",
            "All PPG",
            "HR + shape + PAT",
            "2.0 model offset vs last ref",
            "n subj",
        ]
        out += [table(pd.DataFrame(rows, columns=cols)), ""]
    cs = ROOT / "results" / "calibration_study"
    if (cs / "metrics.csv").exists():
        m, p = pd.read_csv(cs / "metrics.csv"), pd.read_csv(cs / "paired.csv")
        out += ["### V2. Calibration count (SBP MAE, mmHg; 2.0 ridge population model, cached OOF predictions)", ""]
        for site in ("ear", "finger"):
            v = m[(m.site == site) & (m.target == "sbp")].pivot_table(
                index=["scenario", "method"], columns="k", values="mae"
            )
            v = v.map(f2).reset_index()
            v.columns = ["Scenario", "Method", *[f"k={c}" for c in v.columns[2:]]]
            out += [f"**{site}**", "", table(v), ""]
        comps = ["bayes_offset vs bayes_reference", "bayes_offset vs reference_mean", "offset vs reference_mean"]
        q = p[(p.target == "sbp") & p.comparison.isin(comps)]
        rows = [
            [r["site"], r["scenario"], r["k"], r["comparison"], _d(r)] for _, r in q.iterrows() if r["k"] in (1, 3, 10)
        ]
        out += [
            "Paired (Δ MAE; negative favours the first method. `bayes_offset vs bayes_reference` is the PPG's own "
            "contribution; the other rows also credit shrinkage):",
            "",
            table(pd.DataFrame(rows, columns=["Site", "Scenario", "k", "Comparison", "Δ MAE [95% CI]"])),
            "",
        ]
        cov = m[(m.target == "sbp") & (m.k == 3)][
            ["site", "scenario", "method", "pi90_coverage", "pi90_halfwidth_mean"]
        ]
        cov = cov.assign(pi90_coverage=cov.pi90_coverage.map(f2), pi90_halfwidth_mean=cov.pi90_halfwidth_mean.map(f2))
        out += [
            "Cross-subject conformal 90 % intervals (k = 3):",
            "",
            table(cov.set_axis(["Site", "Scenario", "Method", "Coverage", "Mean half-width (mmHg)"], axis=1)),
            "",
        ]
        dr = pd.read_csv(cs / "drift.csv")
        dr = dr[dr.target == "sbp"][["site", "method", "minutes_since_calibration", "mae", "bias", "n_subjects"]]
        dr = dr.assign(mae=dr.mae.map(f2), bias=dr.bias.map(f2))
        out += [
            "Error vs time since calibration (within day, k = 3, SBP):",
            "",
            table(dr.set_axis(["Site", "Method", "Minutes since calibration", "MAE", "Bias", "n subj"], axis=1)),
            "",
        ]
    fa = ROOT / "results" / "feature_analysis"
    if (fa / "features.csv").exists():
        f, sel = pd.read_csv(fa / "features.csv"), pd.read_csv(fa / "selection.csv")
        f = f.merge(sel[["site", "feature", "selection_freq"]], on=["site", "feature"], how="left")
        f = f.reindex(f.within_r_s1_median.abs().sort_values(ascending=False).index)
        rows = []
        for site in ("ear", "finger"):
            for _, r in f[f.site == site].head(12).iterrows():
                rows.append(
                    [
                        site,
                        r["feature"],
                        f"{r['within_r_s1_median']:+.2f} "
                        f"[{r['within_r_s1_ci_low']:+.2f}, {r['within_r_s1_ci_high']:+.2f}]",
                        f"{r['within_r_s2_median']:+.2f}",
                        f2(r["sign_agreement_s1_s2"]),
                        f2(r["rest_icc31_day1_day2"]),
                        f"{r['vif']:.1f}",
                        f2(r["selection_freq"]),
                    ]
                )
        cols = [
            "Site",
            "Feature",
            "Within-subj Spearman day 1 [95% CI]",
            "Day 2",
            "Sign agreement day1/day2",
            "Rest ICC(3,1) day1/day2",
            "VIF",
            "Stability-selection freq.",
        ]
        out += [
            "### V3. Features: within-subject association with SBP, cross-day stability, redundancy "
            "(top 12 by |within-subject r|)",
            "",
            table(pd.DataFrame(rows, columns=cols)),
            "",
        ]
    db = ROOT / "results" / "deep_benchmark"
    if (db / "metrics.csv").exists():
        m, p = pd.read_csv(db / "metrics.csv"), pd.read_csv(db / "paired.csv")
        rows = []
        for (target, meth), g in m.groupby(["target", "method"], sort=False):
            gm = {(a, b): v for a, b, v in zip(g["scope"], g["mode"], g["mae"], strict=True)}
            pp = p[(p.target == target) & p.comparison.str.startswith(f"{meth} vs")]
            pm = {(r["scope"], r["mode"]): r for _, r in pp.iterrows()}

            def dd(scope, mode, gm=gm, pm=pm):
                if (scope, mode) not in gm:
                    return "—"
                v = f2(gm[(scope, mode)])
                return v + (f" ({_d(pm[(scope, mode)])})" if (scope, mode) in pm else "")

            rows.append(
                [
                    target.split("_")[0].upper(),
                    meth,
                    dd("same_day_heldout_subjects", "absolute"),
                    dd("other_day_heldout_subjects", "absolute"),
                    dd("same_day_heldout_subjects", "offset_calibrated"),
                    dd("other_day_heldout_subjects", "offset_calibrated"),
                ]
            )
        cols = [
            "Target",
            "Method",
            "Same day, absolute",
            "Other day, absolute",
            "Same day, offset-calibrated",
            "Other day, offset-calibrated",
        ]
        out += [
            "### V4. Raw-waveform deep models vs baselines (ear; held-out subjects; MAE mmHg; in brackets Δ vs "
            "`training_mean` (absolute) or vs `reference_only` (calibrated))",
            "",
            table(pd.DataFrame(rows, columns=cols)),
            "",
        ]
    vs = ROOT / "data" / "vitaldb" / "summary.csv"
    if vs.exists():
        out += ["### V5. VitalDB subset (surgical patients, CC BY 4.0)", "", table(pd.read_csv(vs)), ""]
    ib = ROOT / "results" / "benchmarks" / "inference_benchmark.csv"
    if ib.exists():
        b = pd.read_csv(ib).set_index("fs_hz").T.reset_index()
        b.columns = ["Metric", *[f"{c} Hz" for c in b.columns[1:]]]
        out += ["### V6. Streaming inference cost (desktop CPU; not a microcontroller measurement)", "", table(b), ""]
    return "\n".join(out)


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    if begin not in text:
        text = text.rstrip() + f"\n\n{begin}\n{end}\n"
    a, b = text.index(begin) + len(begin), text.index(end)
    return text[:a] + "\n\n" + body + "\n" + text[b:]


def main() -> int:
    audit = json.loads((RES / "audit.json").read_text())
    if audit.get("hard_failures"):
        print("audit has hard failures; refusing to publish:", audit["hard_failures"])
        return 1
    summ = json.loads((RES / "summary.json").read_text())
    inv = json.loads((RES / "investigations.json").read_text())
    m = pd.read_csv(RES / "metrics.csv")
    pt = pd.read_csv(RES / "paired_tests.csv")
    out = []
    acq = audit["1_session2_verified"]["detail"]
    out += [
        "### R1. Data and audit",
        "",
        f"- Healthy session-2 records verified: **{acq['verified']}/{acq['hc_session2_listed']}**; "
        f"session-1: {acq['hc_session1_verified']}. Records with untrusted labels (no CNAP mask): "
        f"{', '.join(acq['untrusted_labels']) or 'none'}.",
        f"- Subjects analysed: {summ['subjects']} (both sessions: {summ['subjects_with_both_sessions']}); windows by "
        f"session: {summ['windows_by_session']}. Bootstrap: {summ['n_boot']} subject-level draws, seed {summ['seed']}.",
        "- Independent audit (`results/cross_day/audit.json`): "
        + ", ".join(
            f"{k.split('_', 1)[0]} {'PASS' if v['pass'] else 'FAIL'}"
            for k, v in audit.items()
            if isinstance(v, dict) and "pass" in v
        ),
        "",
    ]

    # main cross-day table: ridge, HR+shape, by calibration mode
    rows = []
    for target in ("sbp", "dbp"):
        for site in ("ear", "finger"):
            for scope in ("within_day", "within_day_s2", "cross_day"):
                for mode in ("carry_forward", "uncalibrated", "offset", "linear"):
                    q = m[
                        (m.site == site)
                        & (m.target == target)
                        & (m.feature_set == "hr_shape")
                        & (m.model == "ridge")
                        & (m.scope == scope)
                        & (m["mode"] == mode)
                    ]
                    if not len(q):
                        continue
                    r = q.iloc[0]
                    rows.append(
                        [
                            target.upper(),
                            site,
                            scope,
                            mode,
                            f2(r.mae),
                            ci(r.mae_ci),
                            f2(r.rmse),
                            f2(r.bias),
                            f2(r.sd_error),
                            f2(r.within_subject_pearson_median),
                            f2(r.within_subject_spearman_median),
                            int(r.n_subjects),
                        ]
                    )
            q = m[
                (m.site == site) & (m.target == target) & (m.feature_set == "hr_shape") & (m.model == "personal_ridge")
            ]
            for r in q.itertuples():
                rows.append(
                    [
                        target.upper(),
                        site,
                        r.scope,
                        "personal model (day-1 fit)",
                        f2(r.mae),
                        ci(r.mae_ci),
                        f2(r.rmse),
                        f2(r.bias),
                        f2(r.sd_error),
                        f2(r.within_subject_pearson_median),
                        f2(r.within_subject_spearman_median),
                        int(r.n_subjects),
                    ]
                )
    out += [
        "### R2. Within-day vs cross-day, by calibration strategy (ridge, HR + shape features)",
        "",
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Target",
                    "Site",
                    "Scope",
                    "Calibration",
                    "MAE",
                    "MAE 95% CI",
                    "RMSE",
                    "Bias",
                    "SD err",
                    "Within-subj Pearson",
                    "Within-subj Spearman",
                    "n subj",
                ],
            )
        ),
        "",
    ]

    # feature sets (HR vs shape), cross-day offset
    rows = []
    for target in ("sbp", "dbp"):
        for site in ("ear", "finger"):
            for fs in ("hr_only", "shape_only", "hr_shape", "ppg_all"):
                for scope in ("within_day", "cross_day"):
                    q = m[
                        (m.site == site)
                        & (m.target == target)
                        & (m.feature_set == fs)
                        & (m.model == "ridge")
                        & (m.scope == scope)
                        & (m["mode"] == "offset")
                    ]
                    if len(q):
                        r = q.iloc[0]
                        rows.append(
                            [
                                target.upper(),
                                site,
                                fs,
                                scope,
                                f2(r.mae),
                                ci(r.mae_ci),
                                f2(r.within_subject_pearson_median),
                                ci(r.get("within_pearson_ci")),
                            ]
                        )
    out += [
        "### R3. Heart rate vs pulse shape (ridge, 1-point offset calibration)",
        "",
        table(
            pd.DataFrame(
                rows, columns=["Target", "Site", "Features", "Scope", "MAE", "MAE 95% CI", "Within-subj r", "r 95% CI"]
            )
        ),
        "",
    ]

    # model ranking
    rk = pd.read_csv(RES / "model_ranking.csv")
    out += [
        "### R4. Population models (HR + shape, cross-day, offset calibration)",
        "",
        "Ranking criterion (fixed in advance): mean rank of cross-day MAE across the four site × target conditions, "
        "then worst rank, then within→cross-day gap. Correlation is not used for selection.",
        "",
        table(rk.round(2)),
        "",
    ]
    mv = pt[pt.comparison.str.contains("vs ridge")]
    out += [table(mv[["comparison", "mae_diff_a_minus_b", "ci_low", "ci_high", "significant"]].round(2)), ""]

    # key paired tests
    key = pt[~pt.comparison.str.contains("vs ridge")]
    out += [
        "### R5. Key paired subject-level comparisons (Δ = MAE(a) − MAE(b); negative favours a)",
        "",
        table(key[["comparison", "mae_diff_a_minus_b", "ci_low", "ci_high", "significant"]].round(2)),
        "",
    ]

    # day effects
    rows = []
    for site in ("ear", "finger"):
        for target in ("sbp", "dbp"):
            d = inv[site].get(f"{target}_day_effects", {})
            if d.get("n_subjects", 0) < 5:
                continue
            a11 = audit["11_day_shift_statistics"]["detail"].get(f"{site}_{target}", {})
            rows.append(
                [
                    site,
                    target.upper(),
                    d["n_subjects"],
                    f2(d["rest_shift_mean"]),
                    f2(d["rest_shift_sd"]),
                    f2(d["slope_pred_shift_on_rest_shift"]),
                    ci(d["slope_pred_shift_on_rest_shift_ci"]),
                    f2(d["r_pred_shift_vs_rest_shift"]),
                    ci(d["r_pred_shift_vs_rest_shift_ci"]),
                    f2(a11.get("min_detectable_abs_r_80pct_power")),
                    f2(d["offset_day1_day2_r"]),
                    ci(d["offset_day1_day2_r_ci"]),
                    f2(d["offset_change_abs_mean"]),
                ]
            )
    out += [
        "### R6. Day-to-day resting BP shift and calibration stability",
        "",
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Target",
                    "n",
                    "True rest shift mean",
                    "SD",
                    "Slope pred-shift~true",
                    "Slope 95% CI",
                    "r",
                    "r 95% CI",
                    "Min detectable |r|",
                    "Offset day1~day2 r",
                    "r 95% CI",
                    "Mean |offset change|",
                ],
            )
        ),
        "",
    ]

    # clipping ablation
    rg = json.loads((RES / "rail_gate_comparison.json").read_text())
    rows = [
        [
            r["site"],
            r["target"].upper(),
            r["scope"],
            f2(r["mae_ungated"]),
            f2(r["mae_gated"]),
            f2(r["mae_change"]),
            f2(r["within_subject_pearson_median_ungated"]),
            f2(r["within_subject_pearson_median_gated"]),
            r["windows_removed"],
        ]
        for r in rg["hr_shape_ridge_offset"]
    ]
    out += [
        "### R7. Amplifier rail-clipping gate ablation (ridge, HR + shape, offset)",
        "",
        f"Finger windows clipped: {rg['finger_windows_clipped_fraction_overall'] * 100:.1f} %; ear: "
        f"{rg['ear_windows_clipped_fraction_overall'] * 100:.1f} %. Subjects with any clipped finger window: "
        f"{rg['subjects_with_any_clipped_finger_window']}; with > 25 % clipped: "
        f"{rg['subjects_with_over_25pct_clipped_finger_windows']}.",
        "",
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Target",
                    "Scope",
                    "MAE no gate",
                    "MAE gated",
                    "Δ",
                    "r no gate",
                    "r gated",
                    "Windows removed",
                ],
            )
        ),
        "",
    ]
    # investigations
    rows = []
    for site in ("ear", "finger"):
        i = inv[site]
        rows.append(
            [
                site,
                f2(i["within_session_r_sbp_vs_hr_median"]),
                f2(i["within_session_partial_r_sbp_shape_given_hr_median"]),
                ci(i.get("partial_r_median_ci")),
                f2(i["uncalibrated_prediction_between_subject_variance_fraction"]),
                f2(i["subject_mean_r_truth_vs_pred"]),
            ]
        )
    out += [
        "### R8. Confounds and subject structure (SBP)",
        "",
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "Site",
                    "Within-session r(SBP, HR)",
                    "Partial r(SBP, shape pred | HR)",
                    "95% CI",
                    "Between-subject share of pred. variance",
                    "r(subject-mean pred, truth)",
                ],
            )
        ),
        "",
    ]
    auto = "\n".join(out)
    text = DOC.read_text(encoding="utf-8") if DOC.exists() else f"# Research Findings\n\n{B}\n{E}\n"
    a, b = text.index(B) + len(B), text.index(E)
    text = text[:a] + "\n\n" + auto + "\n" + text[b:]
    if (PERS / "population_models.csv").exists():
        text = _replace_block(text, BP_, EP_, personalization_block())
        text = _replace_block(text, BV, EV, v3_block())
    DOC.write_text(text, encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)} ({len(auto)} chars of tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
