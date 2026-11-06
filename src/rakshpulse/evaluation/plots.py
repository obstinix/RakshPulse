"""Plotting helpers: waveforms, beat templates, signal quality and BP-error diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..features.morphological import mean_beat  # noqa: E402
from ..preprocessing.segmentation import detect_peaks  # noqa: E402

IR_COLOR = "#b2182b"
RED_COLOR = "#ef8a62"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_window(raw: dict[str, np.ndarray], fs_raw: float, processed: dict, path, title: str = ""):
    """Raw ADC trace (top) and filtered, oriented signal with detected peaks (bottom), per channel."""
    chans = [c for c in ("ir", "red") if c in processed]
    fig, axes = plt.subplots(2, len(chans), figsize=(6 * len(chans), 5), squeeze=False)
    for j, c in enumerate(chans):
        t_raw = np.arange(raw[c].size) / fs_raw
        axes[0, j].plot(t_raw, raw[c], lw=0.8, color=IR_COLOR if c == "ir" else RED_COLOR)
        axes[0, j].set_title(f"{c.upper()} raw (ADC counts)")
        z = processed[c].filtered
        t = np.arange(z.size) / 100.0
        pk = detect_peaks(z, 100.0)
        axes[1, j].plot(t, z, lw=0.9, color=IR_COLOR if c == "ir" else RED_COLOR)
        axes[1, j].plot(t[pk], z[pk], "kx", ms=5)
        s = processed[c].sqi
        axes[1, j].set_title(
            f"{c.upper()} 0.5–10 Hz, z-scored | SQI {'OK' if s.acceptable else 'REJECT'} (tmpl r={s.template_corr:.2f})"
        )
        axes[1, j].set_xlabel("time (s)")
    fig.suptitle(title)
    return _save(fig, path)


def plot_mean_beats(groups: dict[str, list[list[np.ndarray]]], path, title: str = ""):
    """Ensemble-average normalised beats for several groups (e.g. datasets, BP tertiles)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, beat_lists in groups.items():
        mbs = np.array([mean_beat(b) for b in beat_lists if b])
        if mbs.size == 0:
            continue
        m, s = np.nanmean(mbs, 0), np.nanstd(mbs, 0)
        x = np.linspace(0, 1, m.size)
        ax.plot(x, m, label=f"{name} (n={len(mbs)})")
        ax.fill_between(x, m - s, m + s, alpha=0.15)
    ax.set_xlabel("normalised beat time")
    ax.set_ylabel("normalised amplitude")
    ax.legend(fontsize=8)
    ax.set_title(title)
    return _save(fig, path)


def bland_altman(y_true, y_pred, path, label: str):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true
    md, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(mean, diff, s=10, alpha=0.6)
    for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        ax.axhline(v, color="k", ls=ls, lw=1)
    ax.set_xlabel(f"mean of reference and estimate {label} (mmHg)")
    ax.set_ylabel("estimate − reference (mmHg)")
    ax.set_title(f"Bland–Altman {label}: bias {md:+.1f}, LoA ±{1.96 * sd:.1f}")
    return _save(fig, path)


def pred_vs_true(y_true, y_pred, path, label: str):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"reference {label} (mmHg)")
    ax.set_ylabel(f"estimated {label} (mmHg)")
    return _save(fig, path)


def error_hist(errors: dict[str, np.ndarray], path, label: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(-40, 40, 41)
    for name, e in errors.items():
        ax.hist(e, bins=bins, alpha=0.45, label=name)
    ax.set_xlabel(f"{label} error (mmHg)")
    ax.legend(fontsize=8)
    return _save(fig, path)
