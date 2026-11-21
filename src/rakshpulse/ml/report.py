"""Figures, CSV tables and the Markdown evaluation report of the complete ML workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .training import MODEL_LABELS, TARGETS  # noqa: E402

NAVY, TEAL, ORANGE, RED, GREY = "#1B2A4A", "#0F8B8D", "#D9731A", "#C0392B", "#9AA3AE"
SMOKE_BANNER = (
    "SMOKE TEST ON SYNTHETIC DATA. These numbers only show that the code runs end to end. They are "
    "not model validation and say nothing about blood pressure."
)


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "font.size": 9,
        }
    )


# ------------------------------------------------------------------------------------------------ figures
def plot_dataset(d: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    per = d.groupby(["subject_id", "session"]).size().unstack(fill_value=0).sort_index()
    per.plot.bar(stacked=True, ax=ax[0], color=[NAVY, TEAL], width=0.85, legend=True)
    ax[0].set(title="Windows per subject and session", xlabel="subject", ylabel="windows")
    ax[0].set_xticklabels([])
    ax[0].legend(title="session", frameon=False)
    for i, (t, c) in enumerate((("sbp", NAVY), ("dbp", TEAL))):
        ax[1 + i].hist(d[t], bins=40, color=c, alpha=0.85)
        ax[1 + i].set(title=f"{t.upper()} label distribution", xlabel="mmHg", ylabel="windows")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_model_comparison(comp: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, t in zip(ax, TARGETS, strict=True):
        c = comp[comp.target == t].reset_index(drop=True)
        x = np.arange(len(c))
        a.bar(x - 0.27, c.train_mae, 0.27, label="training (in-sample)", color=GREY)
        a.bar(x, c.cv_mae, 0.27, label="cross-validation (train subjects)", color=TEAL)
        a.bar(x + 0.27, c.test_mae, 0.27, label="independent test subjects", color=NAVY)
        base = float(c.loc[c.model == "mean", "test_mae"].iloc[0])
        a.axhline(base, color=RED, ls="--", lw=1, label="baseline test MAE")
        a.set_xticks(x, [m.replace("_", "\n") for m in c.model])
        a.set(title=f"{t.upper()}: MAE by model", ylabel="MAE (mmHg)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_actual_vs_predicted(preds: pd.DataFrame, selection: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, t in zip(ax, TARGETS, strict=True):
        m = selection.set_index("target").loc[t, "model"]
        p = preds[(preds.target == t) & (preds.model == m)]
        lo, hi = np.nanpercentile(np.r_[p.y, p.pred], [0.5, 99.5])
        a.scatter(p.y, p.pred, s=6, alpha=0.35, color=NAVY, edgecolors="none")
        a.plot([lo, hi], [lo, hi], color=RED, lw=1, label="identity")
        base = preds[(preds.target == t) & (preds.model == "mean")].pred.iloc[0]
        a.axhline(base, color=ORANGE, lw=1, ls="--", label="baseline (training mean)")
        a.set(
            xlim=(lo, hi),
            ylim=(lo, hi),
            xlabel=f"reference {t.upper()} (mmHg)",
            ylabel=f"predicted {t.upper()} (mmHg)",
            title=f"{t.upper()}: {MODEL_LABELS[m]} on test subjects",
        )
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_residuals(preds: pd.DataFrame, selection: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for j, t in enumerate(TARGETS):
        m = selection.set_index("target").loc[t, "model"]
        p = preds[(preds.target == t) & (preds.model == m)]
        e = p.pred - p.y
        ax[0, j].hist(e, bins=50, color=TEAL, alpha=0.85)
        ax[0, j].axvline(0, color=RED, lw=1)
        ax[0, j].set(title=f"{t.upper()} residuals (predicted − reference)", xlabel="mmHg", ylabel="windows")
        mean = (p.pred + p.y) / 2
        bias, sd = e.mean(), e.std(ddof=1)
        ax[1, j].scatter(mean, e, s=5, alpha=0.3, color=NAVY, edgecolors="none")
        for v, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            ax[1, j].axhline(v, color=RED, ls=ls, lw=1)
        ax[1, j].set(
            title=f"{t.upper()} Bland–Altman (bias {bias:+.1f}, LoA ±{1.96 * sd:.1f} mmHg)",
            xlabel="mean of predicted and reference (mmHg)",
            ylabel="difference (mmHg)",
        )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ tables
def metrics_summary(result: dict) -> pd.DataFrame:
    """One row per target and evaluation: the selected model and the baseline, every metric requested."""
    comp, sel = result["comparison"], result["selection"].set_index("target")
    rows = []
    for t in TARGETS:
        for name in (sel.loc[t, "model"], "mean"):
            r = comp[(comp.target == t) & (comp.model == name)].iloc[0]
            for ev in ("train", "cv", "test"):
                rows.append(
                    {
                        "target": t.upper(),
                        "model": name,
                        "role": "selected" if name != "mean" else "baseline",
                        "evaluation": {
                            "train": "training (in-sample)",
                            "cv": "cross-validation",
                            "test": "independent test",
                        }[ev],
                        **{
                            k: r[f"{ev}_{k}"]
                            for k in (
                                "mae",
                                "rmse",
                                "r2",
                                "bias",
                                "sd_error",
                                "loa_low",
                                "loa_high",
                                "n_windows",
                                "n_subjects",
                            )
                        },
                    }
                )
        cs = result["cross_session"]
        for _, r in cs[cs.target == t].iterrows():
            rows.append(
                {
                    "target": t.upper(),
                    "model": r.model,
                    "role": "selected" if r.model != "mean" else "baseline",
                    "evaluation": "cross-session (train S1 → test subjects S2)",
                    **{
                        k: r[k]
                        for k in (
                            "mae",
                            "rmse",
                            "r2",
                            "bias",
                            "sd_error",
                            "loa_low",
                            "loa_high",
                            "n_windows",
                            "n_subjects",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def write_tables(result: dict, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    comp = result["comparison"].copy()
    comp["params"] = comp["params"].map(json.dumps)
    sel = result["selection"].copy()
    sel["params"] = sel["params"].map(json.dumps)
    files = {
        "metrics_summary": metrics_summary(result),
        "model_comparison": comp,
        "selected_models": sel,
        "baseline_comparison": result["baseline_comparison"],
        "cross_session": result["cross_session"],
        "feature_selection": result["feature_selection"],
        "test_predictions": result["test_predictions"],
        "subject_split": pd.DataFrame(
            {
                "subject_id": result["train_subjects"] + result["test_subjects"],
                "split": ["train"] * len(result["train_subjects"]) + ["test"] * len(result["test_subjects"]),
            }
        ),
    }
    paths = {}
    for name, df in files.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False, float_format="%.4f")
        paths[name] = p
    return paths


# ------------------------------------------------------------------------------------------------ markdown
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def write_markdown(result: dict, ctx: dict, out: Path, figures: dict[str, Path], smoke: bool = False) -> Path:
    """``ctx``: dataset audit / cleaning log / validation / reproducibility tables and run metadata."""
    sel, comp = result["selection"].set_index("target"), result["comparison"]
    base = comp[comp.model == "mean"].set_index("target")
    bc = result["baseline_comparison"].set_index("target")
    d = result["data"]
    rel = lambda p: Path(p).relative_to(out.parent).as_posix()  # noqa: E731
    L = []
    title = "RakshPulse: Smoke-Test Report (synthetic data)" if smoke else "RakshPulse: Final Model Evaluation"
    L += [f"# {title}", ""]
    if smoke:
        L += [f"> [!CAUTION]\n> **{SMOKE_BANNER}**", ""]
    L += [
        f"Generated by `python scripts/run_complete_pipeline.py{' --smoke-test' if smoke else ''}` on "
        f"{ctx['generated_utc']} (commit `{ctx['commit']}`, rakshpulse {ctx['version']}). Every number below is "
        "computed by that run; nothing is copied from elsewhere.",
        "",
    ]
    if not smoke:
        L += [
            "> [!IMPORTANT]\n> This is **offline validation on a public research dataset** (earlobe PPG, Nonin "
            "transmissive sensor, CNAP continuous BP reference, healthy adults). It is **not** a validation on "
            "MAX30102 hardware: no MAX30102 recordings with reference BP exist yet. On MAX30102 input the exported "
            "model runs in `demonstration` mode, and its numbers are not blood-pressure measurements.",
            "",
        ]

    # ---- summary
    L += ["## 1. Summary", ""]
    rows = []
    for t in TARGETS:
        rows.append(
            {
                "Target": t.upper(),
                "Selected model": MODEL_LABELS[sel.loc[t, "model"]],
                "Test MAE": sel.loc[t, "test_mae"],
                "Test RMSE": sel.loc[t, "test_rmse"],
                "Test R²": sel.loc[t, "test_r2"],
                "Baseline test MAE": base.loc[t, "test_mae"],
                "MAE difference vs baseline [95 % CI]": f"{bc.loc[t, 'mae_difference']:+.2f} "
                f"[{bc.loc[t, 'ci_low']:+.2f}, {bc.loc[t, 'ci_high']:+.2f}]",
            }
        )
    L += [_md_table(pd.DataFrame(rows)), ""]
    verdict = []
    for t in TARGETS:
        if bc.loc[t, "model_better"]:
            verdict.append(
                f"**{t.upper()}: the selected model beats the baseline** on the independent test subjects "
                "(the 95 % CI of the MAE difference excludes 0)."
            )
        elif bc.loc[t, "baseline_better"]:
            verdict.append(
                f"**{t.upper()}: the baseline is significantly better** than the selected model on the "
                "independent test subjects."
            )
        else:
            verdict.append(
                f"**{t.upper()}: no significant difference from the baseline** (predicting the training "
                "mean) on the independent test subjects; the 95 % CI of the MAE difference includes 0."
            )
    L += ["- " + "\n- ".join(verdict), ""]
    if not smoke:
        L += [
            "In plain words: from a single 10 s earlobe PPG window, without a personal cuff calibration, none of the "
            "models predicts an unseen person's blood pressure meaningfully better than the population average. "
            "This agrees with the project's earlier research (docs/RESEARCH_FINDINGS.md). R² values near or below "
            "0 mean the models explain essentially none of the between-person BP variation.",
            "",
        ]

    # ---- data
    L += ["## 2. Dataset", ""]
    L += [f"**{ctx['dataset_name']}**", ""]
    L += [
        f"- Usable windows: **{len(d)}** from **{d.subject_id.nunique()} subjects**, "
        f"{d.record_id.nunique()} recordings; session 1: {d[d.session == 1].subject_id.nunique()} subjects, "
        f"session 2: {d[d.session == 2].subject_id.nunique()} subjects.",
        f"- Split by subject: **{len(result['train_subjects'])} training** subjects "
        f"({(d.split == 'train').sum()} windows) and **{len(result['test_subjects'])} independent test** subjects "
        f"({(d.split == 'test').sum()} windows), stratified by mean SBP.",
        f"- Label range (all usable windows): SBP {d.sbp.min():.0f}–{d.sbp.max():.0f} mmHg (mean {d.sbp.mean():.1f}), "
        f"DBP {d.dbp.min():.0f}–{d.dbp.max():.0f} mmHg (mean {d.dbp.mean():.1f}).",
        "- Activities: " + ", ".join(f"{k} {v}" for k, v in d.activity.value_counts().items()) + ".",
        "",
    ]
    if "cleaning_log" in ctx:
        L += ["**Cleaning steps** (each row is cumulative):", "", _md_table(ctx["cleaning_log"]), ""]
    if "validation" in ctx:
        L += ["**Integrity checks:**", "", _md_table(ctx["validation"]), ""]
    if "reproducibility" in ctx and ctx["reproducibility"] is not None:
        r = ctx["reproducibility"]
        L += [
            f"**Feature reproducibility:** features recomputed by the production extractor from the published raw "
            f"waveforms for {int(r.n_windows.max())} windows; median Spearman agreement across features "
            f"ρ = {r.spearman.median():.3f} (full table: `feature_reproducibility.csv`). Small differences come from "
            "the stored waveforms being 100 Hz copies of the 1 kHz originals the features were computed from.",
            "",
        ]
    L += [f"![dataset]({rel(figures['dataset'])})", ""]

    # ---- protocol
    L += [
        "## 3. Protocol",
        "",
        "- Subject-wise split; the test subjects are used only once, for the final numbers.",
        f"- Feature selection on training subjects only: {len(result['features'])} of "
        f"{len(result['feature_selection'])} candidate features kept (missing > 20 %, constant, or |Spearman ρ| > "
        "0.95 with a higher-priority feature are dropped). All features are computable by the production extractor "
        "from a single PPG channel, so the model runs on MAX30102 IR input.",
        "- Each pipeline: winsorise (0.5/99.5 % training quantiles) → median imputation → standardisation → model, "
        "fitted on training folds only.",
        f"- Hyper-parameters: exhaustive grid search with {result['config']['cv_folds']}-fold GroupKFold over "
        f"training subjects, separately for SBP and DBP (at most {result['config']['max_fit_windows_per_subject']} "
        "non-overlapping windows per subject).",
        "- Selection: lowest cross-validated MAE. The test set is **not** used for selection, so the test numbers "
        "are not optimistically biased; the cross-validation numbers are (they were used for tuning).",
        "- Cross-session: the selected configuration refitted on session 1 of the training subjects and scored on "
        "session 2 of the test subjects (new people on a different day).",
        "- Seeds fixed (`seed=0`). Ridge, SVR and random forest results repeat exactly for the same data and package "
        "versions; the multithreaded gradient-boosting libraries can differ in the second decimal between runs.",
        "- Neural networks were not retrained: the project's raw-waveform benchmark (CNN, GRU, CNN-LSTM, TCN, "
        "Transformer; `results/deep_benchmark/`) did not beat these feature models on the same data.",
        "",
    ]

    # ---- results
    L += ["## 4. Results", "", "### 4.1 Model comparison", ""]
    for t in TARGETS:
        c = comp[comp.target == t][
            [
                "label",
                "cv_mae",
                "train_mae",
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_bias",
                "test_sd_error",
                "fit_seconds",
            ]
        ]
        c.columns = [
            "Model",
            "CV MAE",
            "Train MAE",
            "Test MAE",
            "Test RMSE",
            "Test R²",
            "Test bias",
            "Test SD",
            "Fit+tune s",
        ]
        L += [f"**{t.upper()}** (mmHg)", "", _md_table(c), ""]
    L += [f"![model comparison]({rel(figures['comparison'])})", ""]
    L += ["### 4.2 Selected models, all metrics", ""]
    ms = metrics_summary(result)
    ms = ms[
        [
            "target",
            "model",
            "evaluation",
            "mae",
            "rmse",
            "r2",
            "bias",
            "sd_error",
            "loa_low",
            "loa_high",
            "n_windows",
            "n_subjects",
        ]
    ]
    ms.columns = [
        "Target",
        "Model",
        "Evaluation",
        "MAE",
        "RMSE",
        "R²",
        "Bias",
        "SD",
        "LoA low",
        "LoA high",
        "Windows",
        "Subjects",
    ]
    L += [_md_table(ms), "", "Bias = mean(predicted − reference). LoA = Bland–Altman 95 % limits of agreement.", ""]
    L += ["### 4.3 Best model and parameters", ""]
    for t in TARGETS:
        L += [
            f"- **{t.upper()}: {MODEL_LABELS[sel.loc[t, 'model']]}**, parameters `{json.dumps(sel.loc[t, 'params'])}`; "
            f"90 % interval half-width ±{sel.loc[t, 'halfwidth_90']:.1f} mmHg (split-conformal from out-of-fold "
            f"residuals), observed test coverage {sel.loc[t, 'test_coverage_90']:.0%}."
        ]
    L += [
        "",
        f"![actual vs predicted]({rel(figures['scatter'])})",
        "",
        f"![residuals]({rel(figures['residuals'])})",
        "",
    ]

    # ---- artefacts and limitations
    L += ["## 5. Exported model", ""]
    L += [
        f"- Bundle: `{ctx['model_path']}` (version `{ctx['model_version']}`), loadable with "
        "`rakshpulse.inference.model_loader.load_bundle` and used by `rakshpulse.inference.api.BPEstimator`.",
        "- Contract: `experimental` for Auricle earlobe data; `demonstration` for `max30102` and the simulator; "
        "`validated` for nothing.",
        "",
    ]
    L += [
        "## 6. Known limitations",
        "",
        "- **No hardware validation.** The model has never seen MAX30102 data. Earlobe transmissive PPG (Nonin) and "
        "behind-the-ear reflective PPG (MAX30102) differ in optics, site and signal shape.",
        f"- **Small cohort.** {d.subject_id.nunique()} subjects; the independent test set has "
        f"{len(result['test_subjects'])}, so test metrics have wide uncertainty (see the bootstrap CIs).",
        "- **Reference.** CNAP is a calibrated continuous finger-cuff estimate, not an intra-arterial line.",
        "- **Pseudo-replication.** Windows from one person are correlated; the subject is the unit of analysis for "
        "the CIs.",
        "- **No personal calibration.** These are calibration-free population models. The project's research shows "
        "that per-person cuff calibration carries most of the usable information (docs/RESEARCH_FINDINGS.md).",
        "- **Not a medical device.** No clinical validation (ISO 81060-2 / IEEE 1708) has been performed.",
        "",
    ]
    L += [
        "## 7. Files",
        "",
        "All tables are in this folder as CSV: "
        + ", ".join(f"`{p.name}`" for p in sorted(out.parent.glob("*.csv")))
        + ".",
        "",
    ]
    out.write_text("\n".join(L), encoding="utf-8")
    return out
