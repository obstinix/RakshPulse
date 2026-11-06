"""Ear vs finger PPG morphology in the same Auricle subjects (healthy cohort, rest windows)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.data.dataset_loader import auricle_records, load_auricle_demographics, read_auricle  # noqa: E402
from rakshpulse.evaluation import plots  # noqa: E402
from rakshpulse.features.feature_pipeline import extract_features  # noqa: E402


def main() -> int:
    demo = load_auricle_demographics()
    groups = {"finger (Nonin, transmissive)": [], "earlobe (Nonin, transmissive)": []}
    for rid in [r for r in auricle_records(cohorts=("HC",)) if r.endswith("_1")]:
        try:
            rec = read_auricle(rid, demo=demo)
        except ValueError:
            continue
        fs = rec.fs
        idx = [
            i
            for i in range(0, rec.ear.size - int(10 * fs), int(10 * fs))
            if rec.cnap_valid[i : i + int(10 * fs)].all() and (rec.activity[i : i + int(10 * fs)] == 0).all()
        ]
        if not idx:
            continue
        a = idx[len(idx) // 2]
        for site, sig in (("finger (Nonin, transmissive)", rec.finger), ("earlobe (Nonin, transmissive)", rec.ear)):
            _, proc = extract_features({"ir": sig[a : a + int(10 * fs)]}, fs, has_dc=False)
            if proc["ir"].sqi.acceptable:
                groups[site].append(proc["ir"].beats)
    plots.plot_mean_beats(
        groups,
        ROOT / "docs" / "figures" / "auricle_ear_vs_finger_beat.png",
        "Auricle healthy cohort: ensemble beat, same subjects, rest",
    )
    print({k: len(v) for k, v in groups.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
