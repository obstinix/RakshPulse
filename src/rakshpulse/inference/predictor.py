"""Inference interface proposed for the RakshPulse hardware team.

    predictor = BPPredictor.from_file("models/rakshpulse_bp_v0.joblib")
    result = predictor.predict(red_signal=red, ir_signal=ir, sampling_rate=100)
    if result.accepted:
        print(result.systolic_bp, result.diastolic_bp)
    else:
        print("re-measure:", result.rejection_reasons)

This is a *proposed software interface* for research use. It has **not** been validated on
behind-ear MAX30102 data, and its output is not a diagnostic measurement (see
docs/HARDWARE_INTEGRATION.md and docs/MODEL_EVALUATION.md).

Behaviour:
* validates array types, lengths, finiteness and the sampling rate;
* splits the input into 10 s windows (5 s step), applies the same preprocessing and
  signal-quality gate used in training, and predicts only from accepted windows;
* rejects the measurement (``accepted=False``, BP = None) instead of guessing when
  signal quality is insufficient or a required channel is missing;
* reports uncertainty as the subject-wise cross-validation error SD of the model (the
  honest population-level error) plus the spread of the per-window estimates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..features.feature_pipeline import extract_features
from ..preprocessing.filtering import validate_sampling_rate
from ..preprocessing.segmentation import sliding_windows
from .model_loader import ModelBundle, load_bundle

WINDOW_S = 10.0
STEP_S = 5.0
MIN_DURATION_S = WINDOW_S
MIN_ACCEPTED_FRACTION = 0.5
MIN_ACCEPTED_WINDOWS = 1
DISCLAIMER = (
    "Research prototype. Not clinically validated. Not validated for behind-ear placement. "
    "Do not use for diagnosis or treatment decisions."
)


class InputError(ValueError):
    """Raised for malformed input (wrong types, lengths, sampling rate)."""


@dataclass
class BPResult:
    accepted: bool
    systolic_bp: float | None
    diastolic_bp: float | None
    signal_quality: dict
    uncertainty: dict
    rejection_reasons: list[str] = field(default_factory=list)
    model_version: str = ""
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def _as_signal(x, name: str) -> np.ndarray | None:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a numeric 1-D sequence") from exc
    if a.ndim != 1:
        raise InputError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.isfinite(a).any():
        raise InputError(f"{name} contains no finite samples")
    return a


class BPPredictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @classmethod
    def from_file(cls, path) -> BPPredictor:
        return cls(load_bundle(path))

    def predict(
        self,
        red_signal=None,
        ir_signal=None,
        sampling_rate: float | None = None,
        demographics: dict | None = None,
        raw_counts: bool = True,
    ) -> BPResult:
        """Estimate SBP/DBP from red/IR PPG.

        ``raw_counts=True`` means the arrays are unfiltered sensor readings that still contain the
        DC level (MAX30102 FIFO output). Set False for already band-passed signals.
        """
        b = self.bundle
        if sampling_rate is None:
            raise InputError("sampling_rate is required")
        try:
            fs = validate_sampling_rate(sampling_rate)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        sig = {"red": _as_signal(red_signal, "red_signal"), "ir": _as_signal(ir_signal, "ir_signal")}
        present = {k for k, v in sig.items() if v is not None}
        if sig["red"] is not None and sig["ir"] is not None and sig["red"].size != sig["ir"].size:
            raise InputError(f"red ({sig['red'].size}) and ir ({sig['ir'].size}) lengths differ")

        missing = [c for c in b.channels if c not in present]
        if missing:
            return self._reject([f"missing_required_channel:{c}" for c in missing], {})
        n = min(sig[c].size for c in b.channels)
        if n / fs < MIN_DURATION_S:
            return self._reject([f"too_short:{n / fs:.1f}s<{MIN_DURATION_S:.0f}s"], {})
        demo = {}
        if b.uses_demographics:
            needed = ("age", "sex", "height_cm", "weight_kg")
            if not demographics or any(k not in demographics for k in needed):
                raise InputError(f"this model requires demographics {needed}")
            demo = {f"demo_{k}": float(demographics[k]) for k in needed}

        stack = np.stack([sig[c][:n] for c in b.channels])
        windows = sliding_windows(stack, fs, WINDOW_S, STEP_S)
        rows, reasons = [], {}
        for w in windows:
            feats, proc = extract_features({c: w[i] for i, c in enumerate(b.channels)}, fs, has_dc=raw_counts)
            ok = all(proc[c].sqi.acceptable for c in b.channels)
            for c in b.channels:
                for r in proc[c].sqi.reasons:
                    reasons[f"{c}:{r}"] = reasons.get(f"{c}:{r}", 0) + 1
            if ok:
                rows.append({**feats, **demo})
        q = {
            "windows_total": len(windows),
            "windows_accepted": len(rows),
            "accepted_fraction": len(rows) / max(len(windows), 1),
            "rejection_counts": reasons,
        }
        if len(rows) < MIN_ACCEPTED_WINDOWS or q["accepted_fraction"] < MIN_ACCEPTED_FRACTION:
            return self._reject(["insufficient_signal_quality"], q)

        X = pd.DataFrame(rows).reindex(columns=b.features).to_numpy(dtype=float)
        s_w = b.sbp.predict(X)
        d_w = b.dbp.predict(X)
        unc = {
            "validation_sd_error_sbp": b.validation.get("sbp", {}).get("sd_error"),
            "validation_sd_error_dbp": b.validation.get("dbp", {}).get("sd_error"),
            "validation_mae_sbp": b.validation.get("sbp", {}).get("mae"),
            "validation_mae_dbp": b.validation.get("dbp", {}).get("mae"),
            "window_iqr_sbp": float(np.subtract(*np.percentile(s_w, [75, 25]))) if len(s_w) > 1 else 0.0,
            "window_iqr_dbp": float(np.subtract(*np.percentile(d_w, [75, 25]))) if len(d_w) > 1 else 0.0,
            "note": "validation_* are population-level subject-wise CV errors on fingertip data, "
            "not a per-reading confidence interval.",
        }
        return BPResult(True, float(np.median(s_w)), float(np.median(d_w)), q, unc, [], b.version)

    def _reject(self, reasons: list[str], q: dict) -> BPResult:
        return BPResult(False, None, None, q, {}, reasons, self.bundle.version)
