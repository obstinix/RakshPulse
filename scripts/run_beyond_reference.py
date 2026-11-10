"""RakshPulse 3.0 core experiment: does PPG add BP information beyond the last reference reading?

Datasets
  auricle  references = simulated intermittent readings (nearest usable window on a 2/5/10-min grid, CNAP
           value); truth = CNAP; sites ear and finger evaluated separately; plus a cross-day variant (day-1
           rest reference -> all day-2 windows).
  vitaldb  references = the patients' REAL intermittent cuff readings (NIBP); truth = arterial line.

Feature sets (Δ = change since the reference window):
  hr_only, shape_only, hr_shape, ppg_all (the 2.0 per-site set), ppg_wavelet (+ Haar energies),
  ecg_pat_only / hr_shape_pat (ECG-assisted; not available on the MAX30102),
  ear_finger_ptt / hr_shape_ptt (two optical sensors; Auricle only).
Baselines: last_reference, historical_mean (personal history), population_mean (training subjects),
and rakshpulse2_offset (2.0 ridge population model, offset-calibrated at each reference; Auricle only).

Outputs: results/beyond_reference/{metrics.csv, paired.csv, gap_strata.csv} + a provenance run in results/runs/.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation import auricle_study as st  # noqa: E402
from rakshpulse.evaluation.beyond_reference import evaluate, make_pairs  # noqa: E402
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.features.advanced import PAT_COLUMNS, WAVELET_COLUMNS  # noqa: E402
from rakshpulse.features.feature_pipeline import PER_CHANNEL  # noqa: E402

OUT = ROOT / "results" / "beyond_reference"
PRED = ROOT / "results" / "predictions"
AUR, VDB = ROOT / "data" / "auricle", ROOT / "data" / "vitaldb"
SEED, N_BOOT = 0, 2000
GAP_BINS = [0, 60, 180, 300, 600, 1e9]
log = logging.getLogger("beyond_reference")


def feature_sets(site: str, ecg: bool, ptt: bool) -> dict[str, list[str]]:
    shape = [f"{site}_{f}" for f in st.SHAPE_FEATURES]
    hr = [f"{site}_hr"]
    fs = {
        "hr_only": hr,
        "shape_only": shape,
        "hr_shape": hr + shape,
        "ppg_all": [f"{site}_{n}" for n in PER_CHANNEL],
        "ppg_wavelet": hr + shape + [f"{site}_{w}" for w in WAVELET_COLUMNS],
    }
    if ecg:
        fs["ecg_pat_only"] = [f"{site}_{c}" for c in PAT_COLUMNS[:2]]
        fs["hr_shape_pat"] = hr + shape + [f"{site}_{c}" for c in PAT_COLUMNS[:2]]
    if ptt:
        fs["ear_finger_ptt"] = ["ear_finger_ptt_ms"]
        fs["hr_shape_ptt"] = hr + shape + ["ear_finger_ptt_ms"]
    return fs


def grid_refs(d: pd.DataFrame, every_s: float, cols: list[str], truth: str) -> pd.DataFrame:
    rows = []
    for (sid, sess), g in d.groupby(["subject_id", "session"]):
        t = g.t_start_s.to_numpy()
        for x in np.arange(t.min(), t.max() + 1e-9, every_s):
            r = g.iloc[int(np.abs(t - x).argmin())]
            rows.append(
                {
                    "subject_id": sid,
                    "session": sess,
                    "t_ref_s": r.t_start_s,
                    "ref_reported": r[truth],
                    "ref_truth": r[truth],
                    **{f"ref_{c}": r[c] for c in cols},
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["subject_id", "session", "t_ref_s"])


def strata(pred: pd.DataFrame, methods: list[str], **keys) -> list[dict]:
    rows = []
    b = pd.cut(pred.gap_s, GAP_BINS, right=False)
    for interval, g in pred.groupby(b, observed=True):
        for m in methods:
            e = (g[m] - g.y).abs()
            rows.append(
                {**keys, "gap_bin_s": str(interval), "method": m, "mae": float(e.mean()), "n": int(e.notna().sum())}
            )
    return rows


def run_auricle(run: ExperimentRun) -> tuple[list, list, list]:
    w = pd.read_csv(AUR / "windows.csv", low_memory=False)
    f = pd.read_csv(AUR / "features.csv").drop(columns=["subject_id", "record_id", "session"])
    a = pd.read_csv(AUR / "features_advanced.csv").drop(columns=["subject_id", "record_id", "session"])
    d = w.merge(f, on="window_id").merge(a, on="window_id")
    d = d[(d.cohort == "HC") & d.label_valid].rename(columns={"activity_code": "activity"}).reset_index(drop=True)
    for s in ("ear", "finger"):
        st.add_normalised_shape(d, s)
    met, par, strat = [], [], []
    for site in ("ear", "finger"):
        u = d[d[f"{site}_sqi_ok"]].reset_index(drop=True)
        fsets = feature_sets(site, ecg=True, ptt=True)
        cols = sorted({c for v in fsets.values() for c in v})
        for target in ("sbp", "dbp"):
            y = f"{target}_mmHg"
            pop = pd.read_csv(PRED / f"personalization_pop_{site}_ridge_{target}.csv.gz").set_index("window_id").pred
            u["pop2"] = u.window_id.map(pop)
            truth_by_subject = u.groupby("subject_id")[y].mean()
            for every in (120.0, 300.0, 600.0):
                refs = grid_refs(u, every, cols + ["pop2"], y)
                pairs = make_pairs(u, refs, cols + ["pop2"], y)
                extra = {"rakshpulse2_offset": (pairs.ref_reported + pairs.d_pop2).to_numpy()}
                m, p, pred = evaluate(pairs, fsets, truth_by_subject, n_boot=N_BOOT, seed=SEED, extra_methods=extra)
                keys = {
                    "dataset": "auricle",
                    "site": site,
                    "target": target,
                    "scenario": f"within_day_ref_every_{int(every)}s",
                }
                met += [{**keys, **r} for r in m.to_dict("records")]
                par += [{**keys, **r} for r in p.to_dict("records")]
                strat += strata(pred, ["last_reference", "mean_change", "delta_hr_shape", "delta_hr_shape_pat"], **keys)
                log.info("auricle %s %s every %ds done", site, target, every)
            # cross-day: one day-1 reference (mean of first 3 rest windows) applied to all day-2 windows
            refs = []
            for sid, g in u.groupby("subject_id"):
                c1 = g[(g.session == 1) & g[f"{site}_calibration_eligible"]].sort_values("t_start_s").iloc[:3]
                if len(c1) < 3 or not (g.session == 2).any():
                    continue
                refs.append(
                    {
                        "subject_id": sid,
                        "session": 2,
                        "t_ref_s": -1e9,
                        "ref_reported": c1[y].mean(),
                        "ref_truth": c1[y].mean(),
                        **{f"ref_{c}": c1[c].mean() for c in cols + ["pop2"]},
                    }
                )
            pairs = make_pairs(u, pd.DataFrame(refs), cols + ["pop2"], y)
            extra = {"rakshpulse2_offset": (pairs.ref_reported + pairs.d_pop2).to_numpy()}
            m, p, _ = evaluate(pairs, fsets, truth_by_subject, n_boot=N_BOOT, seed=SEED, extra_methods=extra)
            keys = {"dataset": "auricle", "site": site, "target": target, "scenario": "cross_day_day1_rest_reference"}
            met += [{**keys, **r} for r in m.to_dict("records")]
            par += [{**keys, **r} for r in p.to_dict("records")]
    run.splits.update({s: k for s, k in st.subject_folds(d.subject_id, d.sbp_mmHg).items()})
    return met, par, strat


def run_vitaldb(run: ExperimentRun) -> tuple[list, list, list]:
    w = pd.read_csv(VDB / "windows.csv")
    f = pd.read_csv(VDB / "features.csv").drop(columns=["subject_id"])
    d = w.merge(f, on="window_id")
    d = d[d.label_valid & d.finger_sqi_ok].reset_index(drop=True)
    d["session"], d["activity"] = 1, -1
    st.add_normalised_shape(d, "finger")
    d = d.rename(columns={c: f"finger_{c}" for c in list(PAT_COLUMNS) + list(WAVELET_COLUMNS)})
    nib = pd.read_csv(VDB / "nibp_readings.csv")
    fsets = feature_sets("finger", ecg=True, ptt=False)
    cols = sorted({c for v in fsets.values() for c in v})
    met, par, strat = [], [], []
    for target, nib_col, y in (("sbp", "nibp_sbp", "art_sbp_mmHg"), ("dbp", "nibp_dbp", "art_dbp_mmHg")):
        refs = []
        for r in nib.itertuples():
            g = d[(d.caseid == r.caseid) & ((d.t_start_s - r.t_s).abs() <= 60)]
            if g.empty:
                continue
            x = g.iloc[int((g.t_start_s - r.t_s).abs().argmin())]
            refs.append(
                {
                    "subject_id": x.subject_id,
                    "session": 1,
                    "t_ref_s": r.t_s,
                    "ref_reported": getattr(r, nib_col),
                    "ref_truth": x[y],
                    **{f"ref_{c}": x[c] for c in cols},
                }
            )
        refs = pd.DataFrame(refs)
        pairs = make_pairs(d, refs, cols, y, min_gap_s=10.0, max_gap_s=1800.0)
        truth_by_subject = d.groupby("subject_id")[y].mean()
        m, p, pred = evaluate(pairs, fsets, truth_by_subject, n_boot=N_BOOT, seed=SEED)
        keys = {"dataset": "vitaldb", "site": "finger", "target": target, "scenario": "real_nibp_references"}
        met += [{**keys, "n_reference_readings": len(refs), **r} for r in m.to_dict("records")]
        par += [{**keys, **r} for r in p.to_dict("records")]
        strat += strata(pred, ["last_reference", "mean_change", "delta_hr_shape", "delta_hr_shape_pat"], **keys)
        log.info("vitaldb %s done (%d cuff references)", target, len(refs))
    run.splits.update({s: k for s, k in st.subject_folds(d.subject_id, d.art_sbp_mmHg).items()})
    return met, par, strat


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = [AUR / "windows.csv", AUR / "features.csv", AUR / "features_advanced.csv"]
    has_vdb = (VDB / "windows.csv").exists()
    if has_vdb:
        datasets += [VDB / "windows.csv", VDB / "features.csv", VDB / "nibp_readings.csv"]
    run = ExperimentRun(
        "beyond_reference",
        datasets,
        {
            "model": "RidgeCV(alphas=logspace(-1,4,12)) on feature changes; median-free zero imputation; standardised",
            "folds": "5 subject-grouped folds (subject_folds, balanced by mean truth)",
            "n_boot": N_BOOT,
            "auricle_reference_intervals_s": "120;300;600",
            "min_gap_s": 10,
            "vitaldb_max_gap_s": 1800,
        },
        seed=SEED,
    )
    met, par, strat = run_auricle(run)
    if has_vdb:
        m2, p2, s2 = run_vitaldb(run)
        met, par, strat = met + m2, par + p2, strat + s2
    pd.DataFrame(met).to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(par).to_csv(OUT / "paired.csv", index=False, float_format="%.4f")
    pd.DataFrame(strat).to_csv(OUT / "gap_strata.csv", index=False, float_format="%.4f")
    run.add_metrics(met)
    out = run.finish()
    log.info("done in %.0fs -> %s", time.time() - t0, out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
