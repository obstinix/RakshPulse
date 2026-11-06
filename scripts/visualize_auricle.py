"""Ear vs finger PPG figures from verified Auricle records (healthy cohort, session 1).

fig1: representative simultaneous raw traces + ensemble beats (same subjects, rest windows)
fig2: signal-quality comparison (pass rate by activity; accepted vs rejected example per site)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.data.dataset_loader import auricle_records, load_auricle_demographics, read_auricle  # noqa: E402
from rakshpulse.features.feature_pipeline import extract_features  # noqa: E402
from rakshpulse.features.morphological import mean_beat  # noqa: E402

FIG = ROOT / "docs" / "figures" / "cross_day"
COL = {"ear": "#c0392b", "finger": "#2c7fb8"}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    demo = load_auricle_demographics()
    beats = {"ear": [], "finger": []}
    example = None
    examples_q: dict = {}
    for rid in [r for r in auricle_records(cohorts=("HC",)) if r.endswith("_1")]:
        try:
            rec = read_auricle(rid, demo=demo)
        except ValueError:
            continue
        fs, w = rec.fs, int(10 * rec.fs)
        starts = [i for i in range(0, rec.ear.size - w, w) if rec.cnap_valid[i : i + w].all()]
        rest = [i for i in starts if (rec.activity[i : i + w] == 0).all()]
        if not rest:
            continue
        a = rest[len(rest) // 2]
        ok = {}
        for site, sig in (("ear", rec.ear), ("finger", rec.finger)):
            _, proc = extract_features({"ir": sig[a : a + w]}, fs, has_dc=False)
            ok[site] = proc["ir"].sqi.acceptable
            if ok[site]:
                beats[site].append(proc["ir"].beats)
        if example is None and all(ok.values()):
            example = (rid, a, rec)
        for i in starts[::3]:  # collect one accepted and one rejected example per site
            for site, sig in (("ear", rec.ear), ("finger", rec.finger)):
                _, proc = extract_features({"ir": sig[i : i + w]}, fs, has_dc=False)
                key = (site, proc["ir"].sqi.acceptable)
                if key not in examples_q:
                    examples_q[key] = (rid, i, proc["ir"])
    # ---- fig 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 2.2, 1.6]})
    rid, a, rec = example
    for k, (site, sig) in enumerate((("finger", rec.finger), ("ear", rec.ear))):
        seg = sig[a : a + int(5 * rec.fs)]
        t = np.arange(seg.size) / rec.fs
        ax[k].plot(t, (seg - seg.mean()) / seg.std(), color=COL[site], lw=1)
        ax[k].set_title(f"{site} PPG, {rid} (raw, standardised)")
        ax[k].set_xlabel("time (s)")
        ax[k].set_ylabel("amplitude (z-score)")
    for site in ("finger", "ear"):
        mb = np.array([mean_beat(b) for b in beats[site] if b])
        x = np.linspace(0, 1, mb.shape[1])
        m, s = mb.mean(0), mb.std(0)
        ax[2].plot(x, m, color=COL[site], label=f"{site} (n={len(mb)} subjects)")
        ax[2].fill_between(x, m - s, m + s, color=COL[site], alpha=0.15)
        ax[2].axvline(x[np.argmax(m)], color=COL[site], ls=":", lw=1)
    ax[2].set_xlabel("normalised beat time")
    ax[2].set_ylabel("normalised amplitude")
    ax[2].set_title("Ensemble beat ± SD (dotted: peak)")
    ax[2].legend(fontsize=7)
    fig.suptitle("Fig 1. Simultaneous finger and earlobe PPG (Auricle healthy cohort, rest)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ear_vs_finger_waveforms.png", dpi=150)
    plt.close(fig)
    # ---- fig 2
    feats = pd.read_csv(PROCESSED / "auricle_features.csv")
    feats = feats[feats.cohort == "HC"]
    names = {0: "rest", 1: "breath-hold", 2: "Valsalva", 3: "exercise", 4: "cold pressor"}
    rate = feats[feats.activity.isin(names)].groupby("activity")[["ear_sqi_ok", "finger_sqi_ok"]].mean() * 100
    fig = plt.figure(figsize=(13, 3.8))
    g = fig.add_gridspec(1, 3, width_ratios=[1.6, 2, 2])
    ax0 = fig.add_subplot(g[0])
    x = np.arange(len(rate))
    ax0.bar(x - 0.2, rate.ear_sqi_ok, 0.4, color=COL["ear"], label="ear")
    ax0.bar(x + 0.2, rate.finger_sqi_ok, 0.4, color=COL["finger"], label="finger")
    ax0.set_xticks(x, [names[i] for i in rate.index], rotation=25)
    ax0.set_ylabel("windows passing SQI (%)")
    ax0.set_title("SQI pass rate by activity")
    ax0.legend(fontsize=7)
    for k, site in enumerate(("ear", "finger")):
        axk = fig.add_subplot(g[k + 1])
        for acc, ls in ((True, "-"), (False, "-")):
            if (site, acc) in examples_q:
                rid, i, pc = examples_q[(site, acc)]
                z = pc.filtered
                axk.plot(
                    np.arange(z.size) / 100,
                    z + (0 if acc else -6),
                    ls,
                    color=COL[site] if acc else "grey",
                    lw=0.9,
                    label=f"{'accepted' if acc else 'rejected: ' + ','.join(pc.sqi.reasons[:2])} ({rid})",
                )
        axk.set_title(f"{site}: accepted vs rejected 10 s window")
        axk.set_xlabel("time (s)")
        axk.set_yticks([])
        axk.legend(fontsize=6, loc="upper right")
    fig.suptitle("Fig 2. Signal-quality comparison, earlobe vs finger")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_signal_quality.png", dpi=150)
    plt.close(fig)
    print({k: len(v) for k, v in beats.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
