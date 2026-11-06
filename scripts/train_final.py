"""Fit the selected model on all Guilin subjects and save an inference bundle.

The bundle carries the *subject-wise CV* error statistics of the same configuration
(from results/feature_experiments.json) so the predictor can report honest uncertainty.
It also measures footprint (serialised size, per-window feature + model latency).

Usage: python scripts/train_final.py --feature-set ppg_ir_red --model ridge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiments import FEATURE_SETS, gate  # noqa: E402

from rakshpulse.data.build import PROCESSED  # noqa: E402
from rakshpulse.inference.model_loader import ModelBundle  # noqa: E402
from rakshpulse.inference.predictor import BPPredictor  # noqa: E402
from rakshpulse.models.model_factory import make_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="ppg_ir_red")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--out", default=str(ROOT / "models" / "rakshpulse_bp_v0.joblib"))
    a = ap.parse_args()

    g = gate(pd.read_csv(PROCESSED / "guilin_dualwave_features.csv"), a.feature_set)
    feats = FEATURE_SETS[a.feature_set]
    exp = json.loads((ROOT / "results" / "feature_experiments.json").read_text())
    val = {}
    fitted = {}
    for target in ("sbp", "dbp"):
        row = next(
            r
            for r in exp["A_model_comparison"]
            if r["feature_set"] == a.feature_set and r["model"] == a.model and r["target"] == target
        )
        val[target] = {k.replace("subj_", ""): row[k] for k in ("subj_mae", "subj_sd_error", "subj_mean_error")}
        val[target]["n_subjects"] = row["n_subjects"]
        est, grid = make_model(a.model, 0)
        X, y, grp = g[feats].to_numpy(float), g[target].to_numpy(float), g["subject_id"].to_numpy()
        if grid:
            est = GridSearchCV(est, grid, cv=GroupKFold(5), scoring="neg_mean_absolute_error").fit(X, y, groups=grp)
            est = est.best_estimator_
        else:
            est.fit(X, y)
        fitted[target] = est
    channels = ("ir", "red") if "ir_red" in a.feature_set else (("ir",) if "ir" in a.feature_set else ("red",))
    bundle = ModelBundle(
        sbp=fitted["sbp"],
        dbp=fitted["dbp"],
        features=feats,
        channels=channels,
        uses_demographics="demo" in a.feature_set,
        validation=val,
        version=f"0.1.0-{a.feature_set}-{a.model}",
        trained_on="guilin_dualwave (Mendeley 10.17632/pz2zzr8vhm.1), 148 subjects, rest, fingertip(?)",
    )
    path = bundle.save(a.out)

    # footprint on a real 30 s Guilin recording slice
    from rakshpulse.data.dataset_loader import iter_guilin

    rec = next(iter_guilin())
    s = slice(int(20 * rec.fs), int(50 * rec.fs))
    pred = BPPredictor(bundle)
    demo = {"age": 24, "sex": 1, "height_cm": 176, "weight_kg": 72} if bundle.uses_demographics else None
    t0 = time.perf_counter()
    r = pred.predict(
        red_signal=rec.channels["red"][s], ir_signal=rec.channels["ir"][s], sampling_rate=rec.fs, demographics=demo
    )
    dt = time.perf_counter() - t0
    info = {
        "bundle": str(Path(path).relative_to(ROOT)).replace("\\", "/"),
        "version": bundle.version,
        "size_bytes": Path(path).stat().st_size,
        "n_features": len(feats),
        "validation": val,
        "demo_prediction_30s_record": {
            "accepted": r.accepted,
            "sbp": r.systolic_bp,
            "dbp": r.diastolic_bp,
            "windows": r.signal_quality.get("windows_total"),
        },
        "latency_30s_record_ms_desktop_cpu": round(dt * 1000, 1),
    }
    if hasattr(fitted["sbp"].named_steps["model"], "coef_"):
        info["n_parameters"] = int(np.size(fitted["sbp"].named_steps["model"].coef_) + 1) * 2
    (ROOT / "results" / "final_model.json").write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(info, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
