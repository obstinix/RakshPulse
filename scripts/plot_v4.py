"""RakshPulse 4.0 figures, drawn only from the committed result CSVs (results/*_v4/).

docs/figures/v4_quality_retention.png: share of windows kept vs subject MAE as the quality threshold rises.
docs/figures/v4_interval_coverage.png: coverage vs mean width of 90 % intervals per method and scenario.
docs/figures/v4_external.png: MAE of controls vs change models on the independent VitalDB patients.
docs/figures/v4_cross_day_shift.png: redrawn here from results/cross_day_v4/*.csv (the experiment script draws a
first version; this one fits the axis limits to the data).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "results", ROOT / "docs" / "figures"


def quality() -> None:
    rt = pd.read_csv(R / "quality_v4" / "retention.csv")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (site, err), g in rt.groupby(["site", "error"]):
        ax.plot(
            g.window_share_kept * 100,
            g.subject_mae,
            marker="o",
            label=f"{site}, {'calibrated' if err == 'ae_cal' else 'absolute'}",
        )
        for _, r in g.iterrows():
            ax.annotate(
                f"≥{r.min_score:g}",
                (r.window_share_kept * 100, r.subject_mae),
                fontsize=7,
                textcoords="offset points",
                xytext=(3, 3),
            )
    ax.invert_xaxis()
    ax.set(
        xlabel="windows kept (%)",
        ylabel="subject-level SBP MAE (mmHg)",
        title="Rejecting low-quality windows (quality score threshold)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v4_quality_retention.png", dpi=130)
    plt.close(fig)


def intervals() -> None:
    iv = pd.read_csv(R / "calibration_v4" / "intervals.csv")
    iv = iv[(iv.target == "sbp") & (iv.method == "bayes_ppg_hier")]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    marks = {"gaussian_model": "o", "conformal_scaled": "s", "conformal_const": "^", "quantile_gb": "x"}
    for (interval, site), g in iv.groupby(["interval", "site"]):
        ax.scatter(g.mean_width, g.coverage, marker=marks.get(interval, "o"), label=f"{interval} ({site})", alpha=0.75)
    ax.axhline(0.9, color="k", lw=0.8, ls="--")
    ax.set(
        xlabel="mean 90 % interval width (mmHg)",
        ylabel="empirical coverage",
        title="SBP prediction intervals: coverage vs width (all scenarios)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "v4_interval_coverage.png", dpi=130)
    plt.close(fig)


def external() -> None:
    m = pd.read_csv(R / "external_v4" / "metrics.csv")
    m = m[(m.test_set == "vitaldb_ext") & ~m.method.str.startswith("E1")]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=False)
    for a, target in zip(ax, ("sbp", "dbp"), strict=True):
        g = m[m.target == target].sort_values("mae")
        a.barh(
            g.method,
            g.mae,
            color=["tab:gray" if not x.startswith(("E2_delta", "E3_delta", "E4")) else "tab:blue" for x in g.method],
        )
        a.set(xlabel="MAE vs arterial line (mmHg)", title=f"{target.upper()}, independent VitalDB patients")
        a.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "v4_external.png", dpi=130)
    plt.close(fig)


def cross_day() -> None:
    sf = pd.read_csv(R / "cross_day_v4" / "shift_features.csv")
    ss = pd.read_csv(R / "cross_day_v4" / "shift_subjects.csv").dropna(subset=["template_corr_cross_day"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for site, mk in (("ear", "o"), ("finger", "s")):
        s = sf[sf.site == site]
        ax[0].scatter(s.within_shift_z_median, s.cross_shift_z_median, marker=mk, alpha=0.7, label=site)
        q = ss[ss.site == site]
        ax[1].scatter(q.template_corr_within_day, q.template_corr_cross_day, marker=mk, alpha=0.7, label=site)
    lim = max(sf.cross_shift_z_median.max(), sf.within_shift_z_median.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0].set(
        xlabel="within-day shift, day-1 halves (|Δ median| / robust SD)",
        ylabel="cross-day shift, day 1 → day 2",
        title="Per feature: cross-day vs within-day shift",
    )
    lo = min(ss.template_corr_cross_day.min(), ss.template_corr_within_day.min()) - 0.005
    ax[1].plot([lo, 1], [lo, 1], "k--", lw=0.8)
    ax[1].set(
        xlim=(lo, 1.001),
        ylim=(lo, 1.001),
        xlabel="rest-template correlation within day 1",
        ylabel="rest-template correlation day 1 vs day 2",
        title="Per subject: pulse-shape similarity",
    )
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "v4_cross_day_shift.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    cross_day()
    quality()
    intervals()
    external()
    print("figures written to", FIG)
