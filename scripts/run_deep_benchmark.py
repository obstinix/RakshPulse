"""RakshPulse 3.0, Phase 4: raw-waveform deep models vs. simple baselines on identical subject-held-out folds.

Data: Auricle ear (or finger) processed waveforms (100 Hz, 10 s, z-scored), label-valid and SQI-accepted windows.
Protocol: for every fold (5 subject-grouped folds, ``subject_folds`` balanced by mean SBP), models are trained on
SESSION-1 windows of the training subjects only. They are evaluated on the held-out subjects' session 1
(same-day) and session 2 (other day).

Methods:
* ``training_mean``: mean SBP/DBP of the training windows (no signal);
* ``ridge_hr_shape``: the RakshPulse 2.0 population model family (ridge on HR + pulse-shape features), same folds;
* ``cnn``, ``gru``, ``cnn_lstm``, ``tcn`` and ``transformer``: compact networks sharing ONE training loop
  (``CNNRegressor``: subject-held-out early stopping, amplitude/shift augmentation, multi-task SBP+DBP).

Each method is reported twice:
* absolute (zero-shot);
* offset-calibrated: per subject and session, the mean residual of the first 3 accepted windows is removed and
  scoring uses the remaining windows. ``reference_only`` (the mean truth of those 3 windows carried forward) is the
  matching no-signal baseline.

Outputs: results/deep_benchmark/{metrics,paired}.csv, results/predictions/deep_benchmark_ear.csv.gz (git-ignored)
and a provenance run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation.auricle_study import (  # noqa: E402
    add_normalised_shape,
    feature_set,
    paired_subject_diff,
    point_metrics,
    subject_folds,
)
from rakshpulse.experiments.registry import ExperimentRun  # noqa: E402
from rakshpulse.models.cnn import CNNConfig, CNNRegressor, n_params  # noqa: E402
from rakshpulse.models.sequence import ARCHITECTURES  # noqa: E402

AUR = ROOT / "data" / "auricle"
OUT = ROOT / "results" / "deep_benchmark"
PRED_OUT = ROOT / "results" / "predictions"
SITE = "ear"
SEED, N_BOOT, N_FOLDS = 0, 2000, 5
TRAIN_STRIDE = 2  # every 2nd session-1 window (20 s apart) for training, limiting CPU time; evaluation uses all
CFG = CNNConfig(epochs=40, patience=8, batch_size=64, lr=1e-3, weight_decay=1e-3, width=16, seed=SEED)
TARGETS = ("sbp_mmHg", "dbp_mmHg")
log = logging.getLogger("deep")


def load() -> tuple[pd.DataFrame, np.ndarray]:
    w = pd.read_csv(AUR / "windows.csv")
    f = pd.read_csv(AUR / "features.csv")
    d = w.merge(f.drop(columns=[c for c in f.columns if c in w.columns and c != "window_id"]), on="window_id")
    d = d[d.label_valid & d[f"{SITE}_sqi_ok"]].copy()
    add_normalised_shape(d, SITE)
    X = np.zeros((len(d), 1, 1000), np.float32)
    for sess in (1, 2):
        z = np.load(AUR / "waveforms" / f"auricle_waveforms_session{sess}.npz")
        pos = {wid: i for i, wid in enumerate(z["window_id"])}
        rows = np.flatnonzero(d.session.to_numpy() == sess)
        idx = np.array([pos[x] for x in d.window_id.iloc[rows]])
        X[rows, 0] = z[f"proc_{SITE}"][idx]
    return d.reset_index(drop=True), X


def calibrated(t: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Offset-calibrate per subject and session on the first k windows; return (calibrated, reference_only)."""
    cal, ref = [], []
    for _, g in t.sort_values("t_start_s").groupby(["subject_id", "session"]):
        if len(g) <= k:
            continue
        head, rest = g.iloc[:k], g.iloc[k:].copy()
        r = rest.copy()
        rest["yhat"] = rest.yhat - (head.yhat - head.y).mean()
        r["yhat"] = head.y.mean()
        cal.append(rest)
        ref.append(r)
    return pd.concat(cal), pd.concat(ref)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d, X = load()
    s1 = d.session == 1
    folds = subject_folds(d.subject_id[s1], d.sbp_mmHg[s1], n=N_FOLDS)
    keep = d.subject_id.isin(folds).to_numpy()  # subjects without labelled session-1 windows cannot be folded
    d, X = d[keep].reset_index(drop=True), X[keep]
    fold = d.subject_id.map(folds).to_numpy()
    feats = feature_set("hr_shape", SITE, [])
    Y = d[list(TARGETS)].to_numpy(float)
    preds = {m: np.full((len(d), 2), np.nan) for m in ["training_mean", "ridge_hr_shape", *ARCHITECTURES]}
    log.info(
        "%d windows, %d subjects; params: %s",
        len(d),
        d.subject_id.nunique(),
        {k: n_params(f(1, CFG.width)) for k, f in ARCHITECTURES.items()},
    )
    for k in range(N_FOLDS):
        te = fold == k
        tr_all = (fold != k) & (d.session == 1).to_numpy()
        tr = np.flatnonzero(tr_all)[::TRAIN_STRIDE]
        preds["training_mean"][te] = Y[tr_all].mean(0)
        for j in range(2):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(np.logspace(-2, 4, 13)))
            m.fit(d.loc[tr_all, feats].to_numpy(float), Y[tr_all, j])
            preds["ridge_hr_shape"][te, j] = m.predict(d.loc[te, feats].to_numpy(float))
        for arch in ARCHITECTURES:
            t1 = time.time()
            net = CNNRegressor(1, CFG, arch=arch).fit(X[tr], Y[tr], d.subject_id.to_numpy()[tr])
            preds[arch][te] = net.predict(X[te])
            log.info("fold %d %-11s %.0fs", k, arch, time.time() - t1)
    rows, paired = [], []
    pdf = d[["window_id", "subject_id", "session", "t_start_s", "activity_code", *TARGETS]].copy()
    for j, target in enumerate(TARGETS):
        base = {}
        for meth, P in preds.items():
            pdf[f"{meth}_{target}"] = P[:, j]
            for sess, scope in ((1, "same_day_heldout_subjects"), (2, "other_day_heldout_subjects")):
                m = (d.session == sess).to_numpy()
                t = pd.DataFrame(
                    {
                        "subject_id": d.subject_id[m],
                        "session": sess,
                        "t_start_s": d.t_start_s[m],
                        "activity": d.activity_code[m],
                        "y": Y[m, j],
                        "yhat": P[m, j],
                    }
                )
                c, r = calibrated(t)
                for mode, tt in (("absolute", t), ("offset_calibrated", c)):
                    rows.append({"target": target, "scope": scope, "mode": mode, "method": meth, **point_metrics(tt)})
                    base[(meth, scope, mode)] = tt
                if meth == "training_mean":
                    rows.append(
                        {
                            "target": target,
                            "scope": scope,
                            "mode": "offset_calibrated",
                            "method": "reference_only",
                            **point_metrics(r),
                        }
                    )
                    base[("reference_only", scope, "offset_calibrated")] = r
        for (meth, scope, mode), tt in base.items():
            ref = "training_mean" if mode == "absolute" else "reference_only"
            if meth in (ref, "training_mean", "reference_only"):
                continue
            diff, lo, hi = paired_subject_diff(tt, base[(ref, scope, mode)], N_BOOT, SEED)
            paired.append(
                {
                    "target": target,
                    "scope": scope,
                    "mode": mode,
                    "comparison": f"{meth} vs {ref}",
                    "mae_diff": diff,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(hi < 0 or lo > 0),
                }
            )
    met, par = pd.DataFrame(rows), pd.DataFrame(paired)
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    par.to_csv(OUT / "paired.csv", index=False, float_format="%.4f")
    PRED_OUT.mkdir(parents=True, exist_ok=True)  # per-window, git-ignored like the 2.0 prediction caches
    pdf.to_csv(PRED_OUT / "deep_benchmark_ear.csv.gz", index=False, float_format="%.3f")
    run = ExperimentRun(
        "deep_waveform_benchmark",
        [AUR / "windows.csv", AUR / "features.csv", *sorted((AUR / "waveforms").glob("*.npz"))],
        {
            "site": SITE,
            "train": "session 1 of training subjects",
            "train_stride": TRAIN_STRIDE,
            "cnn_config": str(CFG),
            "architectures": ";".join(ARCHITECTURES),
            "n_boot": N_BOOT,
            "calibration": "offset, first 3 accepted windows per subject-session",
        },
        seed=SEED,
        t0=t0,
    )
    run.splits = dict(folds)
    run.add_metrics(met)
    run.finish()
    log.info("done in %.0fs", time.time() - t0)
    view = met[met.target == "sbp_mmHg"].pivot_table(index="method", columns=["scope", "mode"], values="mae")
    print(view.round(2).to_string())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
