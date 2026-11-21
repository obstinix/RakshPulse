"""Graphs for the operator manual, generated from real project data (docs/manual/figures/*.png).

    python docs/manual/make_figures.py

Data sources, stated on every image:
* REAL DATASET: Auricle (PhysioNet), earlobe PPG from a Nonin transmissive sensor with CNAP reference BP. This is
  real human data, but it is NOT MAX30102 data;
* SIMULATED: the RakshPulse MAX30102 simulator (synthetic signals with no blood-pressure meaning);
* MODEL RESULTS: reports/*.csv written by scripts/run_complete_pipeline.py;
* HARDWARE TEAM VIDEO: one frame of the team's ESP32-C3 + MAX30102 test video (crop of the laptop screen only).
The raw Auricle waveform arrays are downloaded from the GitHub release if they are missing (~350 MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.acquisition.simulator import SimConfig, simulate  # noqa: E402
from rakshpulse.features.feature_pipeline import preprocess_channel  # noqa: E402
from rakshpulse.inference.api import ANALOG_INPUT  # noqa: E402
from rakshpulse.ml import DatasetPaths, load_ml_dataset  # noqa: E402
from rakshpulse.preprocessing.pipeline import PipelineConfig, process_recording  # noqa: E402
from rakshpulse.preprocessing.segmentation import detect_peaks  # noqa: E402

OUT = HERE / "figures"
NAVY, TEAL, ORANGE, RED, GREY, BLUE = "#1B2A4A", "#0F8B8D", "#D9731A", "#C0392B", "#9AA3AE", "#2F6DB5"
SRC_REAL = "REAL DATASET · Auricle earlobe PPG (Nonin sensor), not MAX30102"
SRC_SIM = "SIMULATED · RakshPulse MAX30102 simulator, no blood-pressure meaning"
SRC_MODEL = "MODEL RESULTS · reports/ from run_complete_pipeline.py (offline, earlobe data)"
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10})


def tag(fig, text: str, colour: str = NAVY) -> None:
    fig.text(0.995, 0.004, text, ha="right", va="bottom", fontsize=9, color="white", weight="bold",
             bbox={"boxstyle": "round,pad=0.35", "fc": colour, "ec": colour})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


def auricle():
    paths = DatasetPaths()
    npz = paths.auricle / "waveforms" / "auricle_waveforms_session1.npz"
    if not npz.exists():
        from rakshpulse.data.release import fetch

        fetch("waveforms")
    wav = np.load(npz, allow_pickle=False)
    win = pd.read_csv(paths.auricle / "windows.csv", low_memory=False)
    return wav, win, {w: k for k, w in enumerate(wav["window_id"])}


def long_signal(wav, pos, win: pd.DataFrame, record: str, t0: float, seconds: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate consecutive non-overlapping 10 s windows of one recording (stored at 100 Hz)."""
    g = win[(win.record_id == record) & (win.t_start_s >= t0) & (win.t_start_s < t0 + seconds)
            & ((win.t_start_s - t0) % 10 == 0)].sort_values("t_start_s")
    return np.concatenate([wav["raw_ear"][pos[w]].astype(float) for w in g.window_id]), g


# ------------------------------------------------------------------------------------------------ figures
def fig_raw_red_ir() -> None:
    rec = simulate(SimConfig(fs=100.0, hr_bpm=72, seed=4), 8.0)
    t = np.arange(rec.ir.size) / 100
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    ax[0].plot(t, rec.red, color=RED, lw=1.2)
    ax[0].set(title="Raw RED channel (2nd number): a large steady level (DC) with a small pulse on top", ylabel="ADC counts")
    ax[1].plot(t, rec.ir, color=NAVY, lw=1.2)
    ax[1].set(title="Raw IR channel (3rd number): RakshPulse uses this one", ylabel="ADC counts", xlabel="time (s)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tag(fig, SRC_SIM, ORANGE)
    save(fig, "raw_red_ir_simulated.png")


def fig_peaks(wav, win, pos) -> None:
    row = win[win.ear_sqi_ok & win.label_valid & (win.activity_name == "rest") & (win.session == 1)].iloc[120]
    pc = preprocess_channel(wav["raw_ear"][pos[row.window_id]].astype(float), 100.0, has_dc=False)
    x = pc.filtered
    t = np.arange(x.size) / 100
    peaks = detect_peaks(x, 100.0)
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.plot(t, x, color=BLUE, lw=1.8, label="filtered PPG (IR-type optical signal)")
    ax.plot(peaks / 100, x[peaks], "o", color=RED, ms=8, label="detected heartbeat peak")
    if peaks.size >= 3:
        a, b = peaks[1] / 100, peaks[2] / 100
        y = x[peaks].max() + 0.45
        ax.add_patch(FancyArrowPatch((a, y), (b, y), arrowstyle="<->", mutation_scale=14, color=NAVY, lw=1.5))
        ibi = b - a
        ax.text((a + b) / 2, y + 0.12, f"interval {ibi:.2f} s  →  60 / {ibi:.2f} = {60 / ibi:.0f} beats per minute",
                ha="center", fontsize=10.5, color=NAVY, weight="bold")
        ax.annotate("a red dot marks WHEN a beat\npeaks; its height is NOT pressure",
                    xy=(peaks[3] / 100, x[peaks[3]]), xytext=(peaks[3] / 100 + 0.6, x[peaks[3]] + 1.2),
                    fontsize=10, color=RED, arrowprops={"arrowstyle": "->", "color": RED,
                                                         "connectionstyle": "arc3,rad=-0.25"})
    ax.set(title="How heartbeats are found in the PPG", xlabel="time (seconds)",
           ylabel="filtered amplitude (no units; NOT mmHg)", ylim=(x.min() - 0.6, x.max() + 1.5))
    hr = 60 / np.median(np.diff(peaks) / 100) if peaks.size > 1 else float("nan")
    ax.legend(loc="lower right", frameon=True)
    ax.text(0.01, 0.97, f"{peaks.size} peaks in 10 s · median heart rate {hr:.0f} bpm", transform=ax.transAxes,
            fontsize=10.5, va="top", color=NAVY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tag(fig, SRC_REAL)
    save(fig, "ppg_peak_detection.png")


def fig_ppg_vs_bp(wav, win, pos) -> None:
    rec = win[win.ear_sqi_ok & win.label_valid & (win.session == 1)].record_id.value_counts().index[0]
    g = win[(win.record_id == rec) & win.label_valid].sort_values("t_start_s")
    t0 = float(g.t_start_s.iloc[len(g) // 3])
    sig, gg = long_signal(wav, pos, win, rec, t0, 30)
    pc = preprocess_channel(sig, 100.0, has_dc=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].plot(np.arange(pc.filtered.size) / 100, pc.filtered, color=BLUE, lw=1.2)
    ax[0].set(title="PPG WAVEFORM = optical signal over time", xlabel="time (s)", ylabel="filtered amplitude (no units)")
    ax[0].text(0.5, -0.27, "shows heartbeats, their timing and their shape", transform=ax[0].transAxes,
               ha="center", fontsize=10, color=BLUE)
    sel = g[(g.t_start_s >= t0 - 300) & (g.t_start_s <= t0 + 300)]
    ax[1].plot((sel.t_start_s - t0) / 60, sel.sbp_mmHg, color=RED, lw=2, label="SBP (systolic)")
    ax[1].plot((sel.t_start_s - t0) / 60, sel.dbp_mmHg, color=TEAL, lw=2, label="DBP (diastolic)")
    ax[1].axvspan(0, 0.5, color=GREY, alpha=0.25)
    ax[1].text(0.25, ax[1].get_ylim()[0] + 2, "the 30 s\non the left", ha="center", fontsize=8.5, color=NAVY)
    ax[1].set(title="BLOOD PRESSURE = pressure in mmHg", xlabel="minutes", ylabel="mmHg (reference CNAP device)")
    ax[1].legend(loc="upper right", frameon=True)
    ax[1].text(0.5, -0.27, "two numbers per reading; measured here by a separate BP device", transform=ax[1].transAxes,
               ha="center", fontsize=10, color=RED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    tag(fig, SRC_REAL)
    save(fig, "ppg_vs_blood_pressure.png")


def fig_raw_vs_filtered(wav, win, pos) -> None:
    rec = win[win.ear_sqi_ok & (win.session == 1)].record_id.value_counts().index[3]
    t0 = float(win[(win.record_id == rec) & win.ear_sqi_ok].t_start_s.iloc[20])
    t0 -= t0 % 10
    sig, _ = long_signal(wav, pos, win, rec, t0, 20)
    pc = preprocess_channel(sig, 100.0, has_dc=False)
    t = np.arange(sig.size) / 100
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True)
    ax[0].plot(t, sig, color=GREY, lw=1)
    ax[0].set(title="BEFORE: raw optical signal (baseline drift, slow wander, noise)", ylabel="raw units")
    ax[1].plot(t[: pc.filtered.size], pc.filtered, color=TEAL, lw=1.3)
    ax[1].set(title="AFTER: preprocessed (drift removed, 0.5–10 Hz band-pass, scaled)", xlabel="time (s)",
              ylabel="z-score")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tag(fig, SRC_REAL)
    save(fig, "raw_vs_filtered.png")


def fig_quality_and_hr(wav, win, pos) -> None:
    # a 5-minute stretch (session 1) in which 40-80 % of windows pass the gate, so both outcomes are visible
    choice = None
    for rec, g in win[win.session == 1].sort_values("t_start_s").groupby("record_id"):
        g = g[g.t_start_s % 10 == 0]
        for t0 in g.t_start_s[::6]:
            seg = g[(g.t_start_s >= t0) & (g.t_start_s < t0 + 300)]
            if len(seg) == 30 and 0.4 <= seg.ear_sqi_ok.mean() <= 0.8:
                choice = (rec, float(t0))
                break
        if choice:
            break
    rec, t0 = choice
    sig, _ = long_signal(wav, pos, win, rec, t0, 300)
    raw = pd.DataFrame({"timestamp": np.arange(sig.size) / 100, "sample_index": np.arange(sig.size), "red": np.nan,
                        "infrared": sig})
    w, _ = process_recording(raw, PipelineConfig(fs=100.0, channels=("ir",), **ANALOG_INPUT))
    colours = {"usable": TEAL, "degraded": ORANGE, "unusable": RED, "missing": GREY}
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.2), gridspec_kw={"height_ratios": [1, 1, 0.6]})
    ok = w.status.isin(["usable", "degraded"])
    ax[0].plot(w.t_start_s[ok] + 5, w.ir_hr[ok], "o", color=NAVY, ms=5, label="accepted window: trustworthy")
    ax[0].plot(w.t_start_s[~ok] + 5, w.ir_hr[~ok], "x", color=RED, ms=6, label="rejected window: ignore")
    ax[0].set(title="Heart rate per 10 s window (from the peak intervals)", ylabel="beats per minute")
    ax[0].legend(loc="lower right", frameon=True)
    ax[1].bar(w.t_start_s + 5, w.quality_score, width=4.5, color=[colours.get(s, GREY) for s in w.status])
    ax[1].set(title="Signal-quality score per window (0–100); colour = quality class", ylabel="score",
              xlabel="time (s)")
    ax[0].set_xlim(ax[1].get_xlim())
    for k, (s, c) in enumerate(colours.items()):
        n = int((w.status == s).sum())
        ax[2].barh(0, n, left=int(sum((w.status == q).sum() for q in list(colours)[:k])), color=c,
                   label=f"{s}: {n}")
    ax[2].set(title=f"Accepted (usable + degraded) vs rejected windows: {len(w)} windows in 5 minutes",
              yticks=[], xlabel="number of windows")
    ax[2].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.55), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    tag(fig, SRC_REAL + " · processed by the RakshPulse pipeline")
    save(fig, "quality_and_heart_rate.png")


def fig_features() -> None:
    d = load_ml_dataset()
    d["SBP group"] = pd.qcut(d.groupby("subject_id").sbp.transform("mean"), 3, labels=["lower", "middle", "higher"])
    feats = [("ear_hr", "heart rate (bpm)"), ("ear_sys_ratio", "systolic-time ratio"),
             ("ear_w50_n", "pulse width at 50 % / beat length")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, (f, label) in zip(ax, feats, strict=True):
        for grp, c in (("lower", TEAL), ("middle", GREY), ("higher", RED)):
            v = d.loc[d["SBP group"] == grp, f].dropna()
            a.hist(v.clip(v.quantile(0.01), v.quantile(0.99)), bins=40, alpha=0.5, color=c, density=True,
                   label=f"{grp} SBP people")
        a.set(title=label, xlabel=f, ylabel="share of windows")
    ax[0].legend(frameon=False)
    fig.suptitle("Three of the 40 features the model uses: the groups overlap heavily", y=1.02, fontsize=12,
                 weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    tag(fig, SRC_REAL)
    save(fig, "feature_distributions.png")


def fig_model_results() -> None:
    rep = ROOT / "reports"
    p = pd.read_csv(rep / "test_predictions.csv")
    sel = pd.read_csv(rep / "selected_models.csv").set_index("target")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for a, t in zip(ax, ("sbp", "dbp"), strict=True):
        m = sel.loc[t, "model"]
        q = p[(p.target == t) & (p.model == m)]
        lo, hi = np.percentile(np.r_[q.y, q.pred], [1, 99])
        a.scatter(q.y, q.pred, s=7, alpha=0.35, color=NAVY, edgecolors="none", label="one 10 s window")
        a.plot([lo, hi], [lo, hi], color=RED, lw=1.5, label="perfect prediction")
        a.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10], color=RED, alpha=0.08, label="±10 mmHg")
        mae = np.mean(np.abs(q.pred - q.y))
        a.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"reference {t.upper()} (mmHg)",
              ylabel=f"model {t.upper()} (mmHg)", title=f"{t.upper()}: {m}, 8 unseen people, MAE {mae:.1f} mmHg")
        a.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tag(fig, SRC_MODEL)
    save(fig, "actual_vs_predicted.png")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for a, t in zip(ax, ("sbp", "dbp"), strict=True):
        m = sel.loc[t, "model"]
        q = p[(p.target == t) & (p.model == m)]
        e = q.pred - q.y
        a.hist(e, bins=50, color=TEAL, alpha=0.85)
        a.axvline(0, color=RED, lw=1.5)
        for v in (-10, 10):
            a.axvline(v, color=ORANGE, ls="--", lw=1)
        within = np.mean(np.abs(e) <= 10)
        a.set(title=f"{t.upper()} error: {within:.0%} of windows within ±10 mmHg", xlabel="model − reference (mmHg)",
              ylabel="windows")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tag(fig, SRC_MODEL)
    save(fig, "error_distribution.png")

    comp = pd.read_csv(rep / "model_comparison.csv")
    nest = pd.read_csv(rep / "nested_cv_summary.csv")
    names = {"mean": "baseline:\nguess the\naverage", "ridge": "ridge", "svr": "SVR", "random_forest": "random\nforest",
             "hist_gb": "gradient\nboosting", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, t in zip(ax, ("sbp", "dbp"), strict=True):
        c = comp[comp.target == t].reset_index(drop=True)
        cols = [GREY if m == "mean" else (TEAL if m == sel.loc[t, "model"] else NAVY) for m in c.model]
        a.bar(range(len(c)), c.test_mae, color=cols)
        base = float(c.loc[c.model == "mean", "test_mae"].iloc[0])
        a.axhline(base, color=RED, ls="--", lw=1.3, label=f"baseline {base:.2f}")
        for i, v in enumerate(c.test_mae):
            a.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=9)
        n = nest[(nest.target == t) & (nest.model == "nested-selected")].iloc[0]
        nb = nest[(nest.target == t) & (nest.model == "mean")].iloc[0]
        a.set_xticks(range(len(c)), [names[m] for m in c.model], fontsize=9)
        a.set(title=f"{t.upper()}: error on unseen people (lower is better)", ylabel="MAE (mmHg)",
              ylim=(0, max(c.test_mae) * 1.25))
        a.text(0.02, 0.97, f"teal = selected model\nall 40 people (nested CV): {n.mae:.1f} vs baseline {nb.mae:.1f}",
               transform=a.transAxes, va="top", fontsize=9, color=NAVY)
        a.legend(loc="upper right", frameon=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tag(fig, SRC_MODEL)
    save(fig, "model_vs_baseline.png")


def fig_video_frame() -> None:
    """Crop of the laptop screen from the hardware team's video (no people), if the extracted frame exists."""
    src = HERE / "media" / "video_plotter_crop.png"  # laptop-screen crop of the frame at 2.6 s; nobody visible
    if not src.exists():
        print("skipped video frame (docs/manual/media/video_plotter_crop.png not present)")
        return
    from PIL import Image

    crop = Image.open(src).convert("RGB").resize((928, 620))
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.imshow(crop)
    ax.axis("off")
    notes = [((150, 40), "three series: 'value 1', 'value 2', 'value 3'", (230, -70)),
             ((600, 70), "Arduino IDE Serial Plotter (not RakshPulse)", (700, -20)),
             ((620, 560), "115200 baud", (760, 650))]
    for xy, text, xyt in notes:
        ax.annotate(text, xy=xy, xytext=xyt, fontsize=10.5, color=RED, weight="bold", ha="center",
                    arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.6, "connectionstyle": "arc3,rad=0.2"},
                    annotation_clip=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    tag(fig, "HARDWARE TEAM VIDEO · ESP32-C3 + optical sensor, 21 Nov 2026 (screen crop)", BLUE)
    save(fig, "hardware_video_plotter.png")


def main() -> int:
    wav, win, pos = auricle()
    fig_raw_red_ir()
    fig_peaks(wav, win, pos)
    fig_ppg_vs_bp(wav, win, pos)
    fig_raw_vs_filtered(wav, win, pos)
    fig_quality_and_hr(wav, win, pos)
    fig_features()
    fig_model_results()
    fig_video_frame()
    return 0


if __name__ == "__main__":
    sys.exit(main())
