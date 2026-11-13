"""Write contract-carrying versions of the existing model bundles (RakshPulse 5.1). Originals are left untouched.

* ``models/auricle_population_ear_ridge.joblib`` → ``models/auricle_ear_ridge_v2.joblib`` (tracked in git).
  Ridge on 28 ear HR + shape features, trained on Auricle earlobe PPG (Nonin, transmissive), session 1, CNAP
  labels. On unseen people it is no better than predicting the population mean (RESEARCH_FINDINGS P1). It is
  marked **demonstration-only** for every sensor: it runs the full MAX30102 pipeline, but its output is not a BP
  measurement.
  - Required features: never missing in the training windows. Imputable: the rest, filled with the model's own
    imputer medians.
  - Uncertainty: 90 % cross-subject half-width from the 2.0 subject-held-out out-of-fold residuals of the same model
    family (``results/predictions/personalization_pop_ear_ridge_*.csv.gz``). Valid for Auricle earlobe data only, so
    for MAX30102 input it is reported as unavailable.
* ``models/rakshpulse_bp_v0.joblib`` (1.0 Guilin finger red/IR + demographics; git-ignored) →
  ``models/guilin_finger_ridge_v2.joblib`` (also git-ignored; written only if the source exists). Demographics are
  required. No uncertainty (no out-of-fold residuals were kept). Demonstration-only.

Run: ``python scripts/build_model_contracts.py``. The output is deterministic for given inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rakshpulse.evaluation.auricle_study import add_normalised_shape  # noqa: E402
from rakshpulse.inference.contract import CONTRACT_VERSION, DEFAULT_PROCESSING, validate_bundle  # noqa: E402
from rakshpulse.inference.model_loader import ModelBundle, load_bundle  # noqa: E402

MODELS, AUR, PRED = ROOT / "models", ROOT / "data" / "auricle", ROOT / "results" / "predictions"
DEMO_SENSORS = ["auricle_earlobe", "max30102", "simulator"]


def _imputation_cap(frame: pd.DataFrame, cols: list[str]) -> float:
    """Allowed share of imputed features per window: the 99th percentile seen in training (at least 10 %)."""
    per_window = frame[cols].isna().mean(axis=1)
    return float(max(0.1, np.quantile(per_window, 0.99)))


def _fills(bundle) -> list[float]:
    return [float(v) for v in bundle.sbp.named_steps["impute"].statistics_]


def ear_contract(b) -> dict:
    w = pd.read_csv(AUR / "windows.csv", usecols=["window_id", "session", "ear_sqi_ok", "label_valid", "cohort"])
    f = pd.read_csv(AUR / "features.csv").drop(columns=["session"])
    d = w.merge(f, on="window_id")
    d = d[(d.session == 1) & d.ear_sqi_ok & d.label_valid & (d.cohort == "HC")].copy()
    add_normalised_shape(d, "ear")
    nan_rate = d[b.features].isna().mean()
    feats = [
        {
            "name": n,
            "source": "ir_" + n[len("ear_") :],
            "kind": "signal",
            "required": bool(nan_rate[n] == 0),
            "fill": fill,
            "training_missing_rate": float(nan_rate[n]),
        }
        for n, fill in zip(b.features, _fills(b), strict=True)
    ]
    hw = {}
    for t in ("sbp", "dbp"):
        p = pd.read_csv(PRED / f"personalization_pop_ear_ridge_{t}.csv.gz")
        y = w[["window_id"]].merge(pd.read_csv(AUR / "windows.csv", usecols=["window_id", f"{t}_mmHg"]), on="window_id")
        e = p.merge(y, on="window_id").dropna()
        hw[t] = float(np.quantile(np.abs(e.pred - e[f"{t}_mmHg"]), 0.9))
    v = b.validation
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": "auricle-ear-ridge",
        "features": feats,
        "channels": ["ir"],
        "demographics": [],
        "fs_range_hz": [50.0, 1000.0],
        "processing": dict(DEFAULT_PROCESSING),
        "max_imputed_fraction": _imputation_cap(d, b.features),
        "validated_for": [],
        "experimental_for": [],
        "demonstration_for": DEMO_SENSORS,
        "uncertainty": {
            "method": "cross_subject_cv_residual_quantile_90",
            "halfwidth_90": hw,
            "valid_for": ["auricle_earlobe"],
            "source": "2.0 subject-held-out out-of-fold residuals of the same model family",
        },
        "calibration": {"supported": True, "required_for_tracking": True},
        "performance": {
            "unseen_subjects_mae_sbp": v["sbp"]["mae"],
            "unseen_subjects_mae_dbp": v["dbp"]["mae"],
            "predict_the_mean_mae_sbp": 15.29,
            "predict_the_mean_mae_dbp": 10.72,
            "n_subjects": 39,
        },
        "limitations": [
            "trained on Auricle earlobe transmissive PPG (Nonin), not on MAX30102 reflective PPG",
            "on unseen people no better than predicting the population mean (MAE 15.0 vs 15.3 mmHg SBP)",
            "never tested on MAX30102 recordings: outputs demonstrate the pipeline, they are not BP measurements",
            "uncertainty interval only valid on Auricle earlobe data",
        ],
    }


def guilin_contract(b) -> dict:
    g = pd.read_csv(ROOT / "data" / "processed" / "guilin_dualwave_features.csv")
    sig = [n for n in b.features if not n.startswith("demo_")]
    nan_rate = g[sig].isna().mean() if set(sig) <= set(g.columns) else pd.Series(0.0, index=sig)
    feats = []
    for n, fill in zip(b.features, _fills(b), strict=True):
        if n.startswith("demo_"):
            feats.append({"name": n, "source": n, "kind": "demographic", "required": True, "fill": None})
        else:
            feats.append(
                {
                    "name": n,
                    "source": n,
                    "kind": "signal",
                    "required": bool(nan_rate[n] == 0),
                    "fill": fill,
                    "training_missing_rate": float(nan_rate[n]),
                }
            )
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": "guilin-finger-ridge-demographics",
        "features": feats,
        "channels": ["ir", "red"],
        "demographics": ["age", "sex", "height_cm", "weight_kg"],
        "fs_range_hz": [50.0, 1000.0],
        "processing": dict(DEFAULT_PROCESSING),
        "max_imputed_fraction": _imputation_cap(g, sig) if set(sig) <= set(g.columns) else 0.1,
        "validated_for": [],
        "experimental_for": [],
        "demonstration_for": ["guilin_finger", "max30102", "simulator"],
        "uncertainty": {"method": "none", "halfwidth_90": {}, "valid_for": []},
        "calibration": {"supported": True},
        "performance": {
            "cv_mae_sbp": b.validation["sbp"]["mae"],
            "cv_mae_dbp": b.validation["dbp"]["mae"],
            "demographics_only_mae_sbp": 9.21,
            "demographics_only_mae_dbp": 7.28,
            "n_subjects": 148,
        },
        "limitations": [
            "trained on Guilin finger red/IR (sensor not stated by the dataset authors), one cuff reading per subject",
            "most SBP information comes from demographics; PPG adds no significant SBP information",
            "never tested on MAX30102 recordings: outputs demonstrate the pipeline, they are not BP measurements",
            "no uncertainty estimate available",
        ],
    }


def write(src: Path, dst: Path, make) -> None:
    b = load_bundle(src)
    nb = ModelBundle(
        b.sbp,
        b.dbp,
        list(b.features),
        tuple(b.channels),
        b.uses_demographics,
        b.validation,
        version=f"{b.version}+contract{CONTRACT_VERSION}",
        trained_on=b.trained_on,
        contract=make(b),
        created_utc=b.created_utc,
    )
    validate_bundle(nb)
    nb.save(dst)
    print(
        f"wrote {dst.relative_to(ROOT)} ({nb.contract['model_id']}, "
        f"demonstration_for={nb.contract['demonstration_for']})"
    )


if __name__ == "__main__":
    write(MODELS / "auricle_population_ear_ridge.joblib", MODELS / "auricle_ear_ridge_v2.joblib", ear_contract)
    if (MODELS / "rakshpulse_bp_v0.joblib").exists():
        write(MODELS / "rakshpulse_bp_v0.joblib", MODELS / "guilin_finger_ridge_v2.joblib", guilin_contract)
