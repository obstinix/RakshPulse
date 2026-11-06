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
    DOC.write_text(text[:a] + "\n\n" + auto + "\n" + text[b:], encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)} ({len(auto)} chars of tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
