"""Hardware-independent real-time estimation interface (research use).

    est = RealtimeEstimator.from_file("models/auricle_population_ear_ridge.joblib")
    out = est.estimate(ear_ppg=samples, sampling_rate=100, timestamp=t,
                       engine=user_engine, baseline=user_baseline)   # engine/baseline optional

The population model consumes **one optical site** (the bundle's ``channels[0]``, normally ``"ear"``),
using the same windowing, preprocessing, SQI gate and HR + pulse-shape features as the Auricle study.
A finger signal can be passed and is quality-assessed and reported, but it is not used by an ear model.

The output keeps four quantities apart: population estimate, personalised (calibrated) estimate,
deviation from the user's own reference baseline, and calibration status. No estimate is returned for an
input that fails validation or signal quality. ``uncertainty`` reports the population model's subject-wise
validation error and the calibration residual SD. These are research error statistics, **not** clinically
validated confidence intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..evaluation.auricle_study import add_normalised_shape
from ..features.feature_pipeline import PER_CHANNEL, extract_features
from ..personalization.baseline import BaselineTracker
from ..personalization.engine import PersonalizationEngine
from ..preprocessing.filtering import validate_sampling_rate
from ..preprocessing.segmentation import sliding_windows
from .model_loader import ModelBundle, load_bundle
from .predictor import DISCLAIMER, InputError

WINDOW_S, STEP_S = 10.0, 5.0
MIN_ACCEPTED_FRACTION = 0.5


@dataclass
class RealtimeResult:
    accepted: bool
    population_sbp: float | None = None
    population_dbp: float | None = None
    sbp: float | None = None  # personalised if a calibrated engine was supplied, else population
    dbp: float | None = None
    estimate_type: str = "none"  # "population" | "personalised" | "none"
    signal_quality: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    baseline_deviation: dict = field(default_factory=dict)
    recalibration_recommended: bool = False
    recalibration_reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    model_version: str = ""
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def _signal(x, name):
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be numeric") from exc
    if a.ndim != 1:
        raise InputError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def site_window_features(sig: np.ndarray, fs: float, site: str) -> tuple[list[dict], dict]:
    """Per-window features (prefixed by site) for windows passing SQI, plus quality statistics."""
    wins = sliding_windows(sig, fs, WINDOW_S, STEP_S)
    rows, reasons = [], {}
    for w in wins:
        if not np.isfinite(w).all():
            reasons["non_finite_samples"] = reasons.get("non_finite_samples", 0) + 1
            continue
        feats, proc = extract_features({"ir": w}, fs, has_dc=False)
        sqi = proc["ir"].sqi
        for r in sqi.reasons:
            reasons[r] = reasons.get(r, 0) + 1
        if sqi.acceptable:
            rows.append({f"{site}_{k}": feats[f"ir_{k}"] for k in PER_CHANNEL})
    q = {
        "windows_total": int(len(wins)),
        "windows_accepted": len(rows),
        "accepted_fraction": len(rows) / max(len(wins), 1),
        "rejection_counts": reasons,
    }
    return rows, q


class RealtimeEstimator:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.site = bundle.channels[0]

    @classmethod
    def from_file(cls, path) -> RealtimeEstimator:
        return cls(load_bundle(path))

    def estimate(
        self,
        ear_ppg=None,
        sampling_rate: float | None = None,
        finger_ppg=None,
        timestamp: float | None = None,
        engine: PersonalizationEngine | None = None,
        baseline: BaselineTracker | None = None,
    ) -> RealtimeResult:
        if sampling_rate is None:
            raise InputError("sampling_rate is required")
        try:
            fs = validate_sampling_rate(sampling_rate)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        sig = {"ear": _signal(ear_ppg, "ear_ppg"), "finger": _signal(finger_ppg, "finger_ppg")}
        b = self.bundle
        res = RealtimeResult(accepted=False, model_version=b.version)
        main = sig.get(self.site)
        if main is None:
            res.rejection_reasons = [f"missing_required_signal:{self.site}"]
            return res
        if main.size / fs < WINDOW_S:
            res.rejection_reasons = [f"too_short:{main.size / fs:.1f}s<{WINDOW_S:.0f}s"]
            return res
        rows, q = site_window_features(main, fs, self.site)
        res.signal_quality = {self.site: q}
        other = "finger" if self.site == "ear" else "ear"
        if sig.get(other) is not None and sig[other].size / fs >= WINDOW_S:
            res.signal_quality[other] = site_window_features(sig[other], fs, other)[1] | {"used_by_model": False}
        if not rows or q["accepted_fraction"] < MIN_ACCEPTED_FRACTION:
            res.rejection_reasons = ["insufficient_signal_quality"]
            return res
        X = add_normalised_shape(pd.DataFrame(rows), self.site).reindex(columns=b.features).to_numpy(float)
        ps, pdb = float(np.median(b.sbp.predict(X))), float(np.median(b.dbp.predict(X)))
        res.accepted = True
        res.population_sbp, res.population_dbp = ps, pdb
        res.sbp, res.dbp, res.estimate_type = ps, pdb, "population"
        res.uncertainty = {
            "population_validation_mae_sbp": b.validation.get("sbp", {}).get("mae"),
            "population_validation_mae_dbp": b.validation.get("dbp", {}).get("mae"),
            "note": "subject-wise cross-validation error on the Auricle healthy cohort (CNAP reference); "
            "research statistic, not a clinical confidence interval",
        }
        now = timestamp if timestamp is not None else 0.0
        if engine is not None:
            st = engine.status(now)
            res.calibration = st
            if st["calibrated"]:
                pers = engine.apply(ps, pdb, timestamp)
                res.sbp, res.dbp, res.estimate_type = pers["sbp"], pers["dbp"], "personalised"
                res.uncertainty["calibration_residual_sd_sbp"] = st["residual_sd_sbp"]
                res.uncertainty["calibration_residual_sd_dbp"] = st["residual_sd_dbp"]
            res.recalibration_recommended = st["recalibration_recommended"]
            res.recalibration_reasons = st["reasons"]
        else:
            res.calibration = {"calibrated": False}
            res.recalibration_recommended = True
            res.recalibration_reasons = ["no personal calibration supplied"]
        if baseline is not None:
            res.baseline_deviation = baseline.deviation(res.sbp, res.dbp, upto=timestamp)
        return res
