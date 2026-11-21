"""One-page PNG status dashboard for a BP estimate (used by ``rakshpulse estimate --plot``).

It shows the signal the estimate was computed from (raw and preprocessed), the quality class of every analysis
window, and the structured result. The banner states where the samples came from (simulated, recorded file or live
sensor) and the inference mode, so a demonstration result can never be mistaken for a measurement.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..features.feature_pipeline import preprocess_channel  # noqa: E402
from .api import BPEstimate  # noqa: E402

COLOURS = {"usable": "#0F8B8D", "degraded": "#D9731A", "unusable": "#C0392B", "missing": "#9AA3AE"}
SOURCE_LABEL = {
    "simulator": "SIMULATED SENSOR INPUT",
    "replay": "RECORDED DATA (replay)",
    "serial": "LIVE SENSOR",
    "dataset": "RECORDED RESEARCH DATA",
}


def banner(r: BPEstimate) -> tuple[str, str]:
    """(text, colour) describing the input source and how far the number can be trusted."""
    src = SOURCE_LABEL.get(r.sensor_status.get("source", ""), "INPUT")
    if r.status != "ok":
        return f"{src} · NO ESTIMATE ({r.status.replace('_', ' ')})", "#C0392B"
    if r.inference_mode == "demonstration":
        return f"{src} · DEMONSTRATION MODE: NOT A BLOOD-PRESSURE MEASUREMENT", "#C0392B"
    if r.inference_mode == "experimental":
        return f"{src} · EXPERIMENTAL MODEL: NOT CLINICALLY VALIDATED", "#D9731A"
    return f"{src} · {str(r.inference_mode).upper()}", "#1B2A4A"


def render(ir: np.ndarray, fs: float, r: BPEstimate, out: Path, windows=None) -> Path:
    ir = np.asarray(ir, float)
    t = np.arange(ir.size) / fs
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.1, 0.35], width_ratios=[1, 1, 0.95], hspace=0.55, wspace=0.3)
    text, colour = banner(r)
    fig.suptitle(text, fontsize=12, fontweight="bold", color="white", backgroundcolor=colour, y=0.985)

    a0 = fig.add_subplot(gs[0, :2])
    a0.plot(t, ir, color="#1B2A4A", lw=0.7)
    a0.set(title="Raw IR PPG", xlabel="time (s)", ylabel=r.sensor_status.get("units", "ADC counts"))
    a1 = fig.add_subplot(gs[1, :2], sharex=a0)
    if ir.size >= 2 * fs and np.nanstd(ir) > 0:
        pc = preprocess_channel(np.nan_to_num(ir, nan=float(np.nanmedian(ir))), fs, has_dc=True)
        a1.plot(np.arange(pc.filtered.size) / 100.0, pc.filtered, color="#0F8B8D", lw=0.8)
    a1.set(title="Preprocessed (0.5–10 Hz band-pass, z-scored, 100 Hz)", xlabel="time (s)", ylabel="z")

    a2 = fig.add_subplot(gs[2, :2], sharex=a0)
    if windows is not None and len(windows):
        for _, w in windows.iterrows():
            a2.axvspan(w.t_start_s, w.t_end_s, ymin=0.15, ymax=0.85, color=COLOURS.get(w.status, "#9AA3AE"), alpha=0.55)
        for k, c in COLOURS.items():
            a2.plot([], [], color=c, lw=6, label=k)
        a2.legend(ncol=4, frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    a2.set_yticks([])
    a2.set_title("Window quality (10 s, 5 s hop)", loc="left", fontsize=10)
    a2.set_xlabel("time (s)")

    a3 = fig.add_subplot(gs[:, 2])
    a3.axis("off")
    q = r.signal_quality or {}

    def fmt(v, unit=""):
        return "—" if v is None else f"{v}{unit}"

    lines = [
        ("Status", r.status),
        ("Estimated SBP", fmt(r.sbp, " mmHg")),
        ("Estimated DBP", fmt(r.dbp, " mmHg")),
        (
            "90 % interval",
            "not valid for this sensor"
            if r.sbp_interval_90 is None
            else f"{r.sbp_interval_90[0]:.0f}–{r.sbp_interval_90[1]:.0f} / "
            f"{r.dbp_interval_90[0]:.0f}–{r.dbp_interval_90[1]:.0f}",
        ),
        ("Heart rate", fmt(r.heart_rate_bpm, " bpm")),
        (
            "Signal quality",
            f"{q.get('windows_usable', 0)} usable, {q.get('windows_degraded', 0)} degraded, "
            f"{q.get('windows_unusable', 0) + q.get('windows_missing', 0)} rejected",
        ),
        ("Input validation", r.input_validation or "—"),
        ("Calibration", r.calibration_status),
        ("Operating mode", str(r.inference_mode)),
        ("Sensor", f"{r.sensor} ({r.sensor_status.get('connection', 'n/a')})"),
        ("Sampling rate", fmt(r.fs_hz, " Hz")),
        ("Model", f"{r.model_id}\n{r.model_version}"),
        ("Latency", f"{r.latency_ms:.0f} ms"),
    ]
    y = 0.99
    for k, v in lines:
        big = k in ("Estimated SBP", "Estimated DBP")
        a3.text(0.0, y, k, fontsize=9, color="#6B7280", va="top", transform=a3.transAxes)
        a3.text(
            0.40,
            y,
            str(v),
            fontsize=14 if big else 9,
            fontweight="bold" if big else "normal",
            color="#1B2A4A",
            va="top",
            transform=a3.transAxes,
        )
        y -= (0.07 if big else 0.052) + (0.03 if chr(10) in str(v) else 0.0)
    notes = [r.message] + [f"• {w}" for w in r.warnings]
    wrapped = chr(10).join(textwrap.fill(n, 62) for n in notes if n)
    if wrapped:
        a3.text(
            0.0,
            y - 0.02,
            wrapped,
            fontsize=7.5,
            color="#C0392B" if r.status != "ok" else "#33373D",
            va="top",
            transform=a3.transAxes,
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
