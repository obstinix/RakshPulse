"""Regenerate MODEL_EVALUATION.md §F1–F3 (Auricle session-1 analysis) from results/auricle_experiments.json.

Rewrites the region between ``### F1.`` and ``### F4.`` so that every table number comes from the saved
results (``scripts/run_auricle.py``). Prints the key sentences' numbers for the other documents.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "MODEL_EVALUATION.md"
LAB = {
    "mean": "Mean",
    "ridge_demo_only": "Demographics ridge",
    "ridge_hr_only": "HR-only ridge",
    "ridge": "Ridge (PPG features)",
    "xgboost": "XGBoost",
}


def main() -> int:
    d = json.loads((ROOT / "results" / "auricle_experiments.json").read_text())
    e1 = [r for r in d["E1"] if r["cohort"] == "HC"]
    e4 = [r for r in d["E4"] if r["cohort"] == "HC"]
    e2 = [r for r in d["E2"] if r["cohort"] == "HC"]

    def g(rows, **k):
        return next(r for r in rows if all(r[a] == b for a, b in k.items()))

    t1 = [
        "| Site | Model | SBP win MAE | SBP subj MAE | SBP tracking r | DBP win MAE | DBP subj MAE | DBP tracking r |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for site in ("finger", "ear"):
        for m in ("mean", "ridge_demo_only", "ridge_hr_only", "ridge", "xgboost"):
            if site == "ear" and m in ("mean", "ridge_demo_only"):
                continue
            s, b = g(e1, site=site, model=m, target="sbp"), g(e1, site=site, model=m, target="dbp")
            base = m in ("mean", "ridge_demo_only")

            def tr(x, base=base):
                return "—" if base else f"{x['within_session_r_median']:.2f}"

            t1.append(
                f"| {'—' if base else site.capitalize()} | {LAB[m]} | {s['window_mae']:.2f} | {s['subject_mae']:.2f} "
                f"| {tr(s)} | {b['window_mae']:.2f} | {b['subject_mae']:.2f} | {tr(b)} |"
            )
    t2 = [
        "| Site | Model | SBP calibrated MAE | Carry-forward | Stressor windows (model vs carry) | ΔMAE [95 % CI] | "
        "Δ-tracking r |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for site in ("finger", "ear"):
        for m in ("ridge_hr_only", "ridge", "xgboost"):
            c = g(e4, site=site, model=m, target="sbp")
            lo, hi = c["session_mae_diff_ci95"]
            b = "**" if (hi < 0 or lo > 0) else ""
            t2.append(
                f"| {site.capitalize()} | {LAB[m].replace(' (PPG features)', '')} | {c['calibrated_model_mae']:.2f} | "
                f"{c['carry_forward_mae']:.2f} | {c['stressor_calibrated_model_mae']:.2f} vs "
                f"{c['stressor_carry_forward_mae']:.2f} | {b}{c['session_mae_diff_mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]{b} "
                f"| {c['delta_tracking_r']:.2f} |"
            )
    dbp = []
    for site in ("finger", "ear"):
        for m in ("ridge", "xgboost"):
            c = g(e4, site=site, model=m, target="dbp")
            lo, hi = c["session_mae_diff_ci95"]
            dbp.append(
                f"{site} {LAB[m].split(' (')[0]} {c['calibrated_model_mae']:.2f} vs {c['carry_forward_mae']:.2f}, "
                f"ΔMAE {c['session_mae_diff_mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            )
    t3 = [
        "| Model | Adaptation | SBP win MAE | SBP bias | SBP tracking r | DBP win MAE | DBP bias |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in ("ridge", "xgboost"):
        for ad in ("none", "per-site standardisation"):
            s, b = g(e2, model=m, adaptation=ad, target="sbp"), g(e2, model=m, adaptation=ad, target="dbp")
            t3.append(
                f"| {LAB[m].split(' (')[0]} | {ad} | {s['window_mae']:.2f} | {s['window_me']:+.2f} | "
                f"{s['within_session_r_median']:.2f} | {b['window_mae']:.2f} | {b['window_me']:+.2f} |"
            )
    er, ed = g(e1, site="ear", model="ridge", target="sbp"), g(e1, site="ear", model="ridge", target="dbp")
    fr = g(e1, site="finger", model="ridge", target="sbp")
    t3.append(
        f"| *Ear → ear (reference)* | — | *{er['window_mae']:.2f}* | — | *{er['within_session_r_median']:.2f}* | "
        f"*{ed['window_mae']:.2f}* | — |"
    )
    eh, fh = (g(e1, site=s, model="ridge_hr_only", target="sbp")["within_session_r_median"] for s in ("ear", "finger"))
    meta = d["meta"]
    block = f"""### F1. Calibration-free, subject-independent (window MAE / subject-mean MAE / median within-session r)

Windows must pass SQI (including the rail-clipping and flat-signal gates) at **both** sites.
Finger SQI pass rate {meta["finger_sqi_pass"] * 100:.1f} %, ear {meta["ear_sqi_pass"] * 100:.1f} %.

{chr(10).join(t1)}

PPG morphology tracks **within-session SBP changes** a little better than heart rate alone (finger r
{fr["within_session_r_median"]:.2f} vs {fh:.2f}; ear r {er["within_session_r_median"]:.2f} vs {eh:.2f}). Absolute
calibration-free errors stay around 14–15 mmHg, and DBP shows essentially no PPG signal.

### F2. One-point personal calibration (first 3 rest windows ≈ 20 s of each session)

The comparison baseline is **carry-forward** (keep reporting the calibration value). ΔMAE = per-session MAE(model) −
MAE(carry-forward), with a 95 % bootstrap CI over sessions. **Bold** = CI excludes 0.

{chr(10).join(t2)}

DBP: {"; ".join(dbp)}.

### F3. Finger → ear transfer (train on finger PPG, test on earlobe PPG of held-out subjects)

{chr(10).join(t3)}

A finger-trained model loses most of its tracking ability at the ear. Per-site standardisation removes the bias but
does not restore tracking (`docs/figures/cross_day/fig1_ear_vs_finger_waveforms.png`).

"""
    text = DOC.read_text(encoding="utf-8")
    a, b = text.index("### F1."), text.index("### F4.")
    DOC.write_text(text[:a] + block + text[b:], encoding="utf-8")
    c_ear, c_f = g(e4, site="ear", model="ridge", target="sbp"), g(e4, site="finger", model="ridge", target="sbp")
    print(
        f"ear r {er['within_session_r_median']:.2f} vs HR {eh:.2f}; finger r {fr['within_session_r_median']:.2f} vs HR {fh:.2f}"
    )
    print(
        f"ear calibrated {c_ear['calibrated_model_mae']:.2f} vs carry {c_ear['carry_forward_mae']:.2f}, "
        f"Δ {c_ear['session_mae_diff_mean']:+.2f} {[round(x, 2) for x in c_ear['session_mae_diff_ci95']]}"
    )
    print(f"finger Δ {c_f['session_mae_diff_mean']:+.2f} {[round(x, 2) for x in c_f['session_mae_diff_ci95']]}")
    print(
        "transfer r:",
        [
            round(g(e2, model=m, adaptation=a, target="sbp")["within_session_r_median"], 2)
            for m in ("ridge", "xgboost")
            for a in ("none", "per-site standardisation")
        ],
    )
    print(
        f"sqi ear {meta['ear_sqi_pass']:.3f} finger {meta['finger_sqi_pass']:.3f}; by activity {meta['sqi_pass_by_activity']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
