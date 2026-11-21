"""Generate the documentation images in docs/assets/ from real outputs of the implemented software.

    python scripts/run_complete_pipeline.py      # first: trains, evaluates, writes reports/figures
    python scripts/make_figures.py               # then: docs/assets/*.png

Every image is labelled with its data source:
* RECORDED research data: Auricle earlobe PPG with CNAP reference (real, public dataset);
* SIMULATED sensor input: the RakshPulse MAX30102 simulator (synthetic signals, no BP meaning);
* no image shows MAX30102 hardware measurements, because none exist yet.
The raw Auricle waveform arrays (~350 MB) are downloaded from the GitHub release if missing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.features.feature_pipeline import preprocess_channel  # noqa: E402
from rakshpulse.inference.api import ANALOG_INPUT, BPEstimator  # noqa: E402
from rakshpulse.inference.dashboard import render  # noqa: E402
from rakshpulse.ml import DatasetPaths, load_ml_dataset  # noqa: E402
from rakshpulse.preprocessing.pipeline import PipelineConfig, process_recording  # noqa: E402
from rakshpulse.preprocessing.segmentation import detect_feet, detect_peaks  # noqa: E402

OUT = ROOT / "docs" / "assets"
NAVY, TEAL, ORANGE, RED, GREY = "#1B2A4A", "#0F8B8D", "#D9731A", "#C0392B", "#9AA3AE"
REAL = "RECORDED research data (Auricle earlobe PPG, public dataset)"
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)


def save(fig, name: str) -> Path:
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.relative_to(ROOT))
    return p


def label(fig, text: str, colour: str = NAVY) -> None:
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=8, color=colour, style="italic")


def waveforms(paths: DatasetPaths):
    npz = paths.auricle / "waveforms" / "auricle_waveforms_session1.npz"
    if not npz.exists():
        from rakshpulse.data.release import fetch

        fetch("waveforms")
    return np.load(npz, allow_pickle=False)


def fig_ppg(wav, win: pd.DataFrame) -> None:
    ids = wav["window_id"]
    row = win[win.ear_sqi_ok & win.label_valid & (win.activity_name == "rest")].iloc[200]
    i = int(np.flatnonzero(ids == row.window_id)[0])
    raw = wav["raw_ear"][i].astype(float)
    pc = preprocess_channel(raw, 100.0, has_dc=False)
    t = np.arange(raw.size) / 100
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    ax[0].plot(t, raw, color=NAVY, lw=0.9)
    ax[0].set(title="Raw earlobe PPG (one 10 s window)", ylabel="mV")
    ax[1].plot(t[: pc.filtered.size], pc.filtered, color=TEAL, lw=1)
    feet = detect_feet(pc.filtered, detect_peaks(pc.filtered, 100.0))
    ax[1].plot(feet / 100, pc.filtered[feet], "v", color=ORANGE, ms=6, label="beat onset (foot)")
    peaks = detect_peaks(pc.filtered, 100.0)
    ax[1].plot(peaks / 100, pc.filtered[peaks], "^", color=RED, ms=6, label="systolic peak")
    ax[1].set(
        title="Preprocessed: detrend, Hampel despike, 0.5–10 Hz zero-phase band-pass, orientation, z-score",
        xlabel="time (s)",
        ylabel="z-score",
    )
    ax[1].legend(frameon=False, loc="upper right", ncol=2)
    fig.suptitle(
        f"Window {row.window_id}: CNAP reference {row.sbp_mmHg:.0f}/{row.dbp_mmHg:.0f} mmHg, "
        f"quality gate passed ({pc.sqi.n_beats} beats, template r={pc.sqi.template_corr:.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    label(fig, REAL)
    save(fig, "ppg_raw_vs_filtered.png")


def fig_quality(wav, win: pd.DataFrame) -> None:
    ids = wav["window_id"]
    reasons = win.loc[~win.ear_sqi_ok.astype(bool), "ear_sqi_reasons"].fillna("").str.split("|").explode()
    reasons = reasons[reasons != ""].value_counts().head(8)
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    a0 = fig.add_subplot(gs[:, 0])
    a0.barh(reasons.index[::-1], reasons.values[::-1], color=RED, alpha=0.85)
    ok = win.ear_sqi_ok.astype(bool).mean()
    a0.set(title=f"Quality gate on all {len(win):,} ear windows: {ok:.0%} pass", xlabel="rejected windows")
    rej = win[~win.ear_sqi_ok.astype(bool) & win.ear_sqi_reasons.fillna("").str.contains("template")]
    good = win[win.ear_sqi_ok.astype(bool) & (win.activity_name == "rest")]
    for k, (title, r, c) in enumerate(
        (
            ("Accepted window", good.iloc[50], TEAL),
            ("Rejected window (" + str(rej.iloc[10].ear_sqi_reasons).replace("|", ", ") + ")", rej.iloc[10], RED),
        )
    ):
        a = fig.add_subplot(gs[k, 1])
        i = np.flatnonzero(ids == r.window_id)
        if i.size == 0:
            continue
        pc = preprocess_channel(wav["raw_ear"][int(i[0])].astype(float), 100.0, has_dc=False)
        a.plot(np.arange(pc.filtered.size) / 100, pc.filtered, color=c, lw=0.9)
        a.set(title=f"{title}: template r = {pc.sqi.template_corr:.2f}", ylabel="z", xlabel="time (s)")
    fig.tight_layout()
    label(fig, REAL)
    save(fig, "signal_quality.png")


def fig_features(wav, win: pd.DataFrame, d: pd.DataFrame) -> None:
    ids = wav["window_id"]
    w1 = win[(win.session == 1) & win.ear_sqi_ok & win.label_valid & (win.activity_name == "rest")]
    subj_sbp = w1.groupby("subject_id").sbp_mmHg.mean()
    tert = pd.qcut(subj_sbp, 3, labels=["low", "middle", "high"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    colours = {"low": TEAL, "middle": GREY, "high": RED}
    pos = {w: k for k, w in enumerate(ids)}
    for grp, c in colours.items():
        beats = []
        for _, r in w1[w1.subject_id.isin(tert[tert == grp].index)].groupby("subject_id").head(15).iterrows():
            pc = preprocess_channel(wav["raw_ear"][pos[r.window_id]].astype(float), 100.0, has_dc=False)
            for b in pc.beats:
                beats.append(np.interp(np.linspace(0, len(b) - 1, 100), np.arange(len(b)), (b - b.min()) / np.ptp(b)))
        if beats:
            m = np.mean(beats, axis=0)
            ax[0].plot(
                np.linspace(0, 1, 100),
                m,
                color=c,
                lw=2,
                label=f"{grp} SBP tertile (n={int((tert == grp).sum())} subjects, {len(beats)} beats)",
            )
    ax[0].set(
        title="Ensemble-average beat by subject SBP tertile (rest, session 1)",
        xlabel="fraction of beat",
        ylabel="normalised amplitude",
    )
    ax[0].legend(frameon=False, fontsize=8)
    feats = [c for c in d.columns if c.startswith("ear_")]
    subj = d.groupby("subject_id")[feats + ["sbp", "dbp"]].median()
    rho = subj[feats].corrwith(subj.sbp, method="spearman").dropna().sort_values()
    top = pd.concat([rho.head(6), rho.tail(6)])
    ax[1].barh([f.replace("ear_", "") for f in top.index], top.values, color=[RED if v < 0 else TEAL for v in top])
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set(
        title=f"Subject-level Spearman ρ of features with SBP ({len(subj)} subjects)", xlabel="ρ", xlim=(-0.6, 0.6)
    )
    fig.tight_layout()
    label(fig, REAL)
    save(fig, "pulse_features.png")


def fig_inference_recorded(wav, win: pd.DataFrame, d: pd.DataFrame) -> None:
    """The hardware API on 30 s of recorded earlobe PPG from a held-out test subject."""
    split = pd.read_csv(ROOT / "reports" / "subject_split.csv")
    test_s = set(split[split.split == "test"].subject_id)
    ids = wav["window_id"]
    pos = {w: k for k, w in enumerate(ids)}
    w1 = win[win.session == 1].set_index("window_id").reindex(ids).reset_index()
    ok_ids = set(d[d.subject_id.isin(test_s) & (d.session == 1)].window_id)
    for _, g in w1[w1.subject_id.isin(test_s)].groupby("record_id"):
        g = g.sort_values("t_start_s")
        for t0 in g[g.window_id.isin(ok_ids)].t_start_s:
            chunk = g[g.t_start_s.isin([t0, t0 + 10, t0 + 20])]
            if len(chunk) == 3 and chunk.window_id.isin(ok_ids).all():
                sig = np.concatenate([wav["raw_ear"][pos[w]].astype(float) for w in chunk.window_id])
                est = BPEstimator(sensor="auricle_earlobe", pipeline_overrides=ANALOG_INPUT)
                r = est.estimate(
                    sig,
                    100.0,
                    sensor_status={"source": "dataset", "connection": "file", "units": "mV"},
                    metadata={"reference_cnap": [round(chunk.sbp_mmHg.mean(), 1), round(chunk.dbp_mmHg.mean(), 1)]},
                )
                r.message += (
                    f"; CNAP reference over the same 30 s: {chunk.sbp_mmHg.mean():.0f}/{chunk.dbp_mmHg.mean():.0f} mmHg"
                )
                raw = pd.DataFrame(
                    {
                        "timestamp": np.arange(sig.size) / 100,
                        "sample_index": np.arange(sig.size),
                        "red": np.nan,
                        "infrared": sig,
                    }
                )
                windows, _ = process_recording(raw, PipelineConfig(fs=100.0, channels=("ir",), **ANALOG_INPUT))
                render(sig, 100.0, r, OUT / "inference_dashboard_recorded.png", windows)
                print("wrote docs/assets/inference_dashboard_recorded.png")
                return


def terminal_png(cmd: list[str], name: str, title: str, note: str) -> None:
    """Run a real CLI command and render its output as an image (no editing of the text)."""
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore"},
    )
    lines = ["$ " + " ".join(["rakshpulse"] + cmd[3:])] + p.stdout.rstrip().splitlines()
    lines = [x for ln in lines for x in (textwrap.wrap(ln, 118, subsequent_indent="      ") or [""])]
    fig = plt.figure(figsize=(12.5, 0.24 * len(lines) + 0.9))
    fig.patch.set_facecolor("#1E2329")
    fig.text(0.01, 0.985, title, color="#8FA1B3", fontsize=9, va="top", family="monospace", weight="bold")
    for k, ln in enumerate(lines):
        col = (
            "#7FD1B9" if ln.startswith("$") else ("#F2A65A" if "DEMONSTRATION" in ln or "warning" in ln else "#E6E6E6")
        )
        fig.text(0.01, 0.985 - (k + 1.3) / (len(lines) + 2), ln, color=col, fontsize=8.6, va="top", family="monospace")
    fig.text(0.99, 0.01, note, color="#8FA1B3", fontsize=8, ha="right", style="italic")
    save(fig, name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = ROOT / "reports" / "figures"
    if not (rep / "subject_wise.png").exists():
        print("run `python scripts/run_complete_pipeline.py` first")
        return 2
    for f in ("actual_vs_predicted", "residuals", "model_comparison", "subject_wise", "calibration", "dataset"):
        shutil.copy2(rep / f"{f}.png", OUT / f"{f}.png")
        print("copied", f"docs/assets/{f}.png")
    paths = DatasetPaths()
    win = pd.read_csv(paths.auricle / "windows.csv", low_memory=False)
    d = load_ml_dataset(paths)
    wav = waveforms(paths)
    fig_ppg(wav, win)
    fig_quality(wav, win)
    fig_features(wav, win, d)
    fig_inference_recorded(wav, win, d)
    exe = [sys.executable, "-m", "rakshpulse.cli"]
    terminal_png(
        exe
        + ["estimate", "--source", "sim", "--seconds", "30", "--plot", "docs/assets/inference_dashboard_simulated.png"],
        "cli_estimate_simulated.png",
        "rakshpulse estimate (simulated MAX30102 input)",
        "SIMULATED sensor input: demonstration output, not a blood-pressure measurement",
    )
    terminal_png(exe + ["doctor"], "cli_doctor.png", "rakshpulse doctor", "real output on the development PC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
