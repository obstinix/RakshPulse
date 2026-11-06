"""Publication figures for the Auricle finger-vs-ear and cross-day study.

Every figure is generated from saved results (results/cross_day/*, results/predictions/cross_day_*)
or directly from verified raw records, so it can be regenerated with
``python scripts/make_cross_day_figures.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SITE_COLOR = {"ear": "#c0392b", "finger": "#2c7fb8"}
MODE_LABEL = {
    "carry_forward": "Carry-forward (no PPG)",
    "offset": "1-point offset",
    "linear": "Per-subject linear",
    "personal_s1": "Personal model",
    "uncalibrated": "Uncalibrated",
}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _row(m: pd.DataFrame, **kw) -> pd.Series | None:
    q = m
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0] if len(q) else None


def _ci(r: pd.Series, key: str = "mae_ci") -> tuple[float, float]:
    lo, hi = (float(x) for x in str(r[key]).strip("[]").split(","))
    return lo, hi


def model_comparison(m: pd.DataFrame, out: Path) -> Path:
    """Fig 8: cross-day 1-point-calibrated MAE (95 % subject-bootstrap CI) per model and site."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, target in zip(axes, ("sbp", "dbp"), strict=True):
        sub = m[
            (m.scope == "cross_day") & (m["mode"] == "offset") & (m.feature_set == "hr_shape") & (m.target == target)
        ]
        sub = sub[~sub.model.isin(["personal_ridge", "ridge_trained_on_finger"])]
        models = sorted(sub.model.unique())
        x = np.arange(len(models))
        for j, site in enumerate(("ear", "finger")):
            vals, lo, hi = [], [], []
            for mod in models:
                r = _row(sub, site=site, model=mod)
                vals.append(r.mae if r is not None else np.nan)
                c = _ci(r) if r is not None else (np.nan, np.nan)
                lo.append(r.mae - c[0] if r is not None else 0)
                hi.append(c[1] - r.mae if r is not None else 0)
            ax.bar(
                x + (j - 0.5) * 0.38,
                vals,
                0.38,
                yerr=[lo, hi],
                color=SITE_COLOR[site],
                alpha=0.85,
                label=site,
                capsize=2,
            )
        cf = _row(
            m, scope="cross_day", mode="carry_forward", feature_set="hr_shape", target=target, site="ear", model="ridge"
        )
        if cf is not None:
            ax.axhline(cf.mae, color="k", ls="--", lw=1, label="carry-forward (ear windows)")
        ax.set_xticks(x, models, rotation=30, ha="right")
        ax.set_ylabel(f"cross-day {target.upper()} MAE (mmHg)")
        ax.set_title(f"{target.upper()}: day-1 calibrated, day-2 evaluated")
        ax.legend(fontsize=7)
    fig.suptitle("Model comparison (HR + shape features, 1-point offset calibration; error bars: 95 % CI)")
    return _save(fig, out)


def calibration_modes(m: pd.DataFrame, out: Path) -> Path:
    """Fig 5a: cross-day MAE by calibration strategy and feature set (ridge)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sets = ["hr_only", "shape_only", "hr_shape", "ppg_all"]
    for ax, target in zip(axes, ("sbp", "dbp"), strict=True):
        for j, site in enumerate(("ear", "finger")):
            for i, mode in enumerate(("carry_forward", "offset", "linear", "personal_s1")):
                for k, fs in enumerate(sets):
                    model = "personal_ridge" if mode == "personal_s1" else "ridge"
                    r = _row(m, scope="cross_day", mode=mode, feature_set=fs, target=target, site=site, model=model)
                    if r is None or (mode == "carry_forward" and fs != "hr_shape"):
                        continue
                    lo, hi = _ci(r)
                    xpos = i * 1.2 + j * 0.5 + k * 0.1
                    ax.errorbar(
                        xpos,
                        r.mae,
                        yerr=[[r.mae - lo], [hi - r.mae]],
                        fmt="os^D"[k],
                        color=SITE_COLOR[site],
                        ms=4,
                        capsize=2,
                        label=f"{site} {fs}" if i == 1 else None,
                    )
        ax.set_xticks(
            [0.4, 1.6, 2.8, 4.0],
            [MODE_LABEL[x] for x in ("carry_forward", "offset", "linear", "personal_s1")],
            rotation=15,
        )
        ax.set_ylabel(f"cross-day {target.upper()} MAE (mmHg)")
        ax.set_title(target.upper())
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Calibration strategy on day 1 → error on day 2 (markers: feature sets; 95 % subject CI)")
    return _save(fig, out)


def drift(store_dir: Path, m: pd.DataFrame, out: Path) -> Path:
    """Fig 5b: distribution of per-subject day-2 bias after day-1 calibration (calibration drift)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, target in zip(axes, ("sbp", "dbp"), strict=True):
        data, labels = [], []
        for site in ("ear", "finger"):
            p = pd.read_csv(store_dir / f"cross_day_{site}_{target}_hr_shape_ridge.csv.gz")
            from .auricle_study import calibrate

            t = calibrate(p, target).table
            t = t[t.scope == "cross_day"]
            for mode in ("carry_forward", "offset", "linear"):
                g = t[t["mode"] == mode]
                data.append((g.yhat - g.y).groupby(g.subject_id).mean().to_numpy())
                labels.append(
                    f"{site}\n" + {"carry_forward": "carry-fwd", "offset": "offset", "linear": "linear"}[mode]
                )
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        for i, dd in enumerate(data, 1):
            ax.plot(
                np.full(dd.size, i) + np.random.default_rng(i).normal(0, 0.05, dd.size),
                dd,
                ".",
                alpha=0.5,
                color=SITE_COLOR["ear" if i <= 3 else "finger"],
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"day-2 mean error per subject, {target.upper()} (mmHg)")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{target.upper()} calibration drift across days")
    return _save(fig, out)


def actual_vs_predicted(store_dir: Path, target: str, out: Path) -> Path:
    """Figs 3/4: day-2 reference vs 1-point-calibrated estimate, ear and finger."""
    from .auricle_study import calibrate

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, site in zip(axes, ("ear", "finger"), strict=True):
        p = pd.read_csv(store_dir / f"cross_day_{site}_{target}_hr_shape_ridge.csv.gz")
        t = calibrate(p, target).table
        g = t[(t.scope == "cross_day") & (t["mode"] == "offset")]
        ax.scatter(g.y, g.yhat, s=4, alpha=0.3, color=SITE_COLOR[site])
        lo, hi = np.nanpercentile(np.r_[g.y, g.yhat], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        r = np.corrcoef(g.y, g.yhat)[0, 1]
        ax.set_title(
            f"{site}: MAE {np.abs(g.yhat - g.y).mean():.1f} mmHg, pooled r {r:.2f}, n={g.subject_id.nunique()}"
        )
        ax.set_xlabel(f"CNAP reference {target.upper()} (mmHg)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"estimated {target.upper()} (mmHg)")
    fig.suptitle(f"{target.upper()}, day 2, calibrated on day 1 (ridge, HR + shape features)")
    return _save(fig, out)


def bland_altman(store_dir: Path, out: Path) -> Path:
    """Fig 6: Bland–Altman, day 2, ear and finger, SBP and DBP."""
    from .auricle_study import calibrate

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for i, target in enumerate(("sbp", "dbp")):
        for j, site in enumerate(("ear", "finger")):
            p = pd.read_csv(store_dir / f"cross_day_{site}_{target}_hr_shape_ridge.csv.gz")
            t = calibrate(p, target).table
            g = t[(t.scope == "cross_day") & (t["mode"] == "offset")]
            mean, diff = (g.y + g.yhat) / 2, g.yhat - g.y
            md, sd = diff.mean(), diff.std(ddof=1)
            ax = axes[i, j]
            ax.scatter(mean, diff, s=3, alpha=0.3, color=SITE_COLOR[site])
            for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
                ax.axhline(v, color="k", ls=ls, lw=0.9)
            ax.set_title(
                f"{site} {target.upper()}: bias {md:+.1f}, LoA [{md - 1.96 * sd:.1f}, {md + 1.96 * sd:.1f}] mmHg"
            )
            ax.set_xlabel("mean of reference and estimate (mmHg)")
            ax.set_ylabel("estimate − reference (mmHg)")
    fig.suptitle(
        "Bland–Altman, day 2 after day-1 one-point calibration (windows; LoA ignore within-subject clustering)"
    )
    return _save(fig, out)


def per_subject(per_subj: pd.DataFrame, out: Path) -> Path:
    """Fig 7: per-subject day-2 MAE, carry-forward vs calibrated PPG model (paired)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, ("sbp", "dbp"), strict=True):
        for k, site in enumerate(("ear", "finger")):
            q = per_subj[
                (per_subj.site == site)
                & (per_subj.target == target)
                & (per_subj.feature_set == "hr_shape")
                & (per_subj.model == "ridge")
                & (per_subj.scope == "cross_day")
            ]
            a = q[q["mode"] == "carry_forward"].set_index("subject_id").mae
            b = q[q["mode"] == "offset"].set_index("subject_id").mae
            common = a.index.intersection(b.index)
            for s in common:
                ax.plot([k * 2, k * 2 + 1], [a[s], b[s]], "-", color="grey", lw=0.5, alpha=0.6)
            ax.plot(np.full(len(common), k * 2), a[common], "o", color="k", ms=3)
            ax.plot(np.full(len(common), k * 2 + 1), b[common], "o", color=SITE_COLOR[site], ms=3)
            better = (b[common] < a[common]).mean() * 100
            ax.text(
                k * 2 + 0.5,
                ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 30,
                f"{better:.0f}% improved",
                ha="center",
                fontsize=7,
            )
        ax.set_xticks([0, 1, 2, 3], ["ear\ncarry-fwd", "ear\nPPG offset", "finger\ncarry-fwd", "finger\nPPG offset"])
        ax.set_ylabel(f"day-2 MAE per subject, {target.upper()} (mmHg)")
        ax.set_title(target.upper())
    fig.suptitle("Per-subject day-2 error: carry-forward vs PPG model (lines join the same subject)")
    return _save(fig, out)


def hr_vs_ppg(m: pd.DataFrame, invest: dict, out: Path) -> Path:
    """Fig 9: within-subject tracking r by feature set (calibration-independent) and partial r beyond HR."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1.3]})
    sets = ["hr_only", "shape_only", "hr_shape", "ppg_all"]
    for j, site in enumerate(("ear", "finger")):
        for t_i, target in enumerate(("sbp", "dbp")):
            vals, lo, hi = [], [], []
            for fs in sets:
                r = _row(m, scope="within_day", mode="offset", feature_set=fs, target=target, site=site, model="ridge")
                vals.append(r.within_subject_pearson_median if r is not None else np.nan)
                c = (
                    _ci(r, "within_pearson_ci")
                    if r is not None and isinstance(r.get("within_pearson_ci"), str)
                    else (np.nan, np.nan)
                )
                lo.append(vals[-1] - c[0])
                hi.append(c[1] - vals[-1])
            x = np.arange(len(sets)) + (j * 2 + t_i - 1.5) * 0.2
            axes[0].errorbar(
                x,
                vals,
                yerr=[lo, hi],
                fmt="o" if target == "sbp" else "s",
                color=SITE_COLOR[site],
                capsize=2,
                ms=4,
                label=f"{site} {target.upper()}",
            )
    axes[0].set_xticks(range(len(sets)), sets)
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].set_ylabel("median within-session Pearson r\n(estimate vs reference)")
    axes[0].set_title("Within-day tracking by feature set (95 % subject CI)")
    axes[0].legend(fontsize=7, ncol=2)
    for j, site in enumerate(("ear", "finger")):
        inv = invest.get(site, {})
        v = inv.get("within_session_partial_r_sbp_shape_given_hr_median")
        ci = inv.get("partial_r_median_ci", [np.nan, np.nan])
        if v is not None:
            axes[1].errorbar(j, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=SITE_COLOR[site], capsize=3)
    axes[1].set_xticks([0, 1], ["ear", "finger"])
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_ylabel("partial r (SBP, shape model | HR)")
    axes[1].set_title("Shape information beyond HR")
    return _save(fig, out)


def calibration_transfer(m: pd.DataFrame, out: Path) -> Path:
    """Fig 10: on ear windows, ear-trained vs finger-trained population model under each calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, target in zip(axes, ("sbp", "dbp"), strict=True):
        for k, (model, lab) in enumerate((("ridge", "ear-trained"), ("ridge_trained_on_finger", "finger-trained"))):
            for i, mode in enumerate(("uncalibrated", "offset", "linear")):
                r = _row(
                    m, scope="cross_day", mode=mode, feature_set="hr_shape", target=target, site="ear", model=model
                )
                if r is None:
                    continue
                lo, hi = _ci(r)
                ax.bar(
                    i + (k - 0.5) * 0.38,
                    r.mae,
                    0.38,
                    yerr=[[r.mae - lo], [hi - r.mae]],
                    capsize=2,
                    color=SITE_COLOR["ear" if k == 0 else "finger"],
                    label=lab if i == 0 else None,
                )
        ax.set_xticks([0, 1, 2], [MODE_LABEL[x] for x in ("uncalibrated", "offset", "linear")])
        ax.set_ylabel(f"day-2 ear {target.upper()} MAE (mmHg)")
        ax.set_title(target.upper())
        ax.legend(fontsize=7)
    fig.suptitle("Calibration transfer: can per-subject calibration rescue a finger-trained model at the ear?")
    return _save(fig, out)
