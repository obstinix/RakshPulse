"""Simple, hardware-facing BP inference: PPG samples in, a structured SBP/DBP result out.

    from rakshpulse.inference.api import BPEstimator

    est = BPEstimator()                                   # default: models/rakshpulse_ear_bp.joblib
    r = est.estimate(ir=ir_counts, fs=100.0, red=red_counts, metadata={"subject": "P001"})
    r.status, r.sbp, r.dbp, r.signal_quality, r.inference_mode, r.model_version
    r.to_dict()                                           # JSON-ready

    stream = StreamingBPEstimator(est, fs=100.0, buffer_s=30)
    stream.push(ir_chunk, red_chunk)                      # any chunk size, e.g. one FIFO read
    r = stream.estimate()                                 # estimate on the last ``buffer_s`` seconds

Everything goes through the production path used for recorded sessions: ``preprocessing.pipeline``
(windows, quality gate, features) and ``inference.engine.predict_windows`` (model contract, sensor and
sampling-rate checks, inference mode). The SBP/DBP result is the median over windows that passed the quality gate.

Status values:

* ``ok``: enough windows passed the quality gate; ``sbp``/``dbp`` are set.
* ``poor_signal``: the input was valid but too few windows passed the quality gate. No BP is returned.
* ``invalid_input``: the samples or sampling rate are unusable (``message`` says why). No BP is returned.
* ``incompatible_model``: the model does not support this sensor, rate or channel set. No BP is returned.

``inference_mode`` is copied from the model contract. For ``max30102`` input every current model is
``demonstration``: the numbers show the pipeline works and are **not** blood-pressure measurements. The 90 %
interval is only reported when the contract says it is valid for the given sensor.

Input expectations (MAX30102): raw, unfiltered ADC counts (0–262143 at 18 bit) and the *effective* sampling rate
(sample rate ÷ averaging). Other sources can override the pipeline's ADC limits with ``pipeline_overrides``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.download import REPO_ROOT
from ..preprocessing.pipeline import PipelineConfig, process_recording
from .contract import IncompatibleModelError, validate_bundle
from .engine import predict_windows
from .model_loader import ModelBundle, load_bundle

DEFAULT_MODEL = REPO_ROOT / "models" / "rakshpulse_ear_bp.joblib"
# pipeline_overrides for signals that are not raw ADC counts (e.g. research recordings in mV): no rail or
# flat-line checks on ADC limits, which are meaningless for such units
ANALOG_INPUT = {"adc_max": float("inf"), "rail_margin_counts": float("-inf"), "flat_sd_counts": 1e-9}
MIN_SECONDS = 10.0  # one analysis window
STATUSES = ("ok", "poor_signal", "invalid_input", "incompatible_model")


@dataclass
class BPEstimate:
    status: str
    sbp: float | None = None
    dbp: float | None = None
    sbp_interval_90: tuple[float, float] | None = None
    dbp_interval_90: tuple[float, float] | None = None
    signal_quality: dict = field(default_factory=dict)
    inference_mode: str | None = None
    model_id: str = ""
    model_version: str = ""
    sensor: str = ""
    fs_hz: float | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("sbp_interval_90", "dbp_interval_90"):
            d[k] = list(d[k]) if d[k] is not None else None
        return d


def _signal(x, name: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of numbers") from None
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if np.isinf(a).any():
        raise ValueError(f"{name} contains infinite values")
    if np.isnan(a).mean() > 0.01:
        raise ValueError(f"{name} has {np.isnan(a).mean():.1%} missing (NaN) samples; at most 1 % is accepted")
    return a


class BPEstimator:
    """Stateless estimator: one call analyses one block of samples (at least 10 s)."""

    def __init__(
        self,
        model: str | Path | ModelBundle = DEFAULT_MODEL,
        sensor: str = "max30102",
        min_usable_fraction: float = 0.5,
        pipeline_overrides: dict | None = None,
    ):
        self.bundle = model if isinstance(model, ModelBundle) else load_bundle(model)
        self.contract = validate_bundle(self.bundle)
        self.sensor = sensor
        self.min_usable_fraction = float(min_usable_fraction)
        self.pipeline_overrides = dict(pipeline_overrides or {})

    @property
    def model_id(self) -> str:
        return str(self.contract.get("model_id", ""))

    def describe(self) -> dict:
        c = self.contract
        return {
            "model_id": self.model_id,
            "model_version": self.bundle.version,
            "channels": list(c["channels"]),
            "fs_range_hz": c["fs_range_hz"],
            "sensor": self.sensor,
            "features": len(self.bundle.features),
            "validated_for": c.get("validated_for", []),
            "experimental_for": c.get("experimental_for", []),
            "demonstration_for": c.get("demonstration_for", []),
            "limitations": c.get("limitations", []),
        }

    def estimate(self, ir, fs: float, red=None, metadata: dict | None = None) -> BPEstimate:
        t0 = time.perf_counter()
        base = {
            "model_id": self.model_id,
            "model_version": self.bundle.version,
            "sensor": self.sensor,
            "metadata": dict(metadata or {}),
        }
        try:
            fs = float(fs)
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"sampling rate must be a positive number, got {fs!r}")
            ir_a = _signal(ir, "ir")
            red_a = _signal(red, "red") if red is not None else None
            if red_a is not None and red_a.size != ir_a.size:
                raise ValueError(f"red ({red_a.size}) and ir ({ir_a.size}) must have the same length")
            if ir_a.size < MIN_SECONDS * fs:
                raise ValueError(
                    f"need at least {MIN_SECONDS:g} s of samples ({int(MIN_SECONDS * fs)} at "
                    f"{fs:g} Hz), got {ir_a.size}"
                )
        except ValueError as exc:
            return self._done(BPEstimate("invalid_input", fs_hz=_num(fs), message=str(exc), **base), t0)

        n = ir_a.size
        raw = pd.DataFrame(
            {
                "timestamp": np.arange(n) / fs,
                "sample_index": np.arange(n),
                "red": red_a if red_a is not None else np.full(n, np.nan),
                "infrared": ir_a,
            }
        )
        cfg = PipelineConfig(fs=fs, channels=("ir", "red") if red_a is not None else ("ir",), **self.pipeline_overrides)
        try:
            windows, _ = process_recording(raw, cfg)
            preds, compat = predict_windows(self.bundle, windows, self.sensor, fs, pipeline=cfg)
        except IncompatibleModelError as exc:
            return self._done(BPEstimate("incompatible_model", fs_hz=fs, message=str(exc), **base), t0)
        except ValueError as exc:
            return self._done(BPEstimate("invalid_input", fs_hz=fs, message=str(exc), **base), t0)

        quality = _quality(windows, preds)
        out = BPEstimate("poor_signal", fs_hz=fs, inference_mode=compat["mode"], signal_quality=quality, **base)
        out.warnings = _mode_warnings(compat["mode"])
        ok = preds[np.isfinite(preds.sbp) & np.isfinite(preds.dbp)]
        frac = len(ok) / max(len(preds), 1)
        if len(ok) == 0 or frac < self.min_usable_fraction:
            out.message = (
                f"{len(ok)}/{len(preds)} windows passed the quality and input checks "
                f"(need {self.min_usable_fraction:.0%}); re-measure with a steady sensor"
            )
            return self._done(out, t0)
        out.status = "ok"
        out.sbp, out.dbp = round(float(ok.sbp.median()), 1), round(float(ok.dbp.median()), 1)
        unc = self.contract.get("uncertainty", {})
        if self.sensor in unc.get("valid_for", []):
            hw = unc.get("halfwidth_90", {})
            out.sbp_interval_90 = (round(out.sbp - hw["sbp"], 1), round(out.sbp + hw["sbp"], 1))
            out.dbp_interval_90 = (round(out.dbp - hw["dbp"], 1), round(out.dbp + hw["dbp"], 1))
        else:
            out.warnings.append(
                f"no uncertainty interval: the model's interval is not valid for sensor '{self.sensor}'"
            )
        out.message = f"median of {len(ok)} accepted 10 s windows"
        return self._done(out, t0)

    @staticmethod
    def _done(r: BPEstimate, t0: float) -> BPEstimate:
        r.latency_ms = round(1000 * (time.perf_counter() - t0), 1)
        return r


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _quality(windows: pd.DataFrame, preds: pd.DataFrame) -> dict:
    status = windows["status"].value_counts().to_dict() if len(windows) else {}
    reasons: dict[str, int] = {}
    for r in windows.get("reasons", pd.Series(dtype=str)).dropna():
        for x in str(r).split("|"):
            if x:
                reasons[x] = reasons.get(x, 0) + 1
    q = windows["quality_score"] if "quality_score" in windows else pd.Series(dtype=float)
    return {
        "windows_total": int(len(windows)),
        "windows_usable": int(status.get("usable", 0)),
        "windows_degraded": int(status.get("degraded", 0)),
        "windows_unusable": int(status.get("unusable", 0)),
        "windows_missing": int(status.get("missing", 0)),
        "windows_predicted": int(np.isfinite(preds.sbp).sum()) if len(preds) else 0,
        "median_quality_score": float(q.median()) if q.notna().any() else None,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8]),
    }


def _mode_warnings(mode: str | None) -> list[str]:
    if mode == "demonstration":
        return [
            "DEMONSTRATION MODE: this model has not been validated for this sensor; the numbers show that the "
            "pipeline runs and are NOT blood-pressure measurements"
        ]
    if mode == "experimental":
        return [
            "experimental model: cross-validated and tested on held-out subjects of its training population "
            "only; not a clinically validated measurement"
        ]
    return []


class StreamingBPEstimator:
    """Ring buffer in front of :class:`BPEstimator` for chunked input (e.g. MAX30102 FIFO reads)."""

    def __init__(self, estimator: BPEstimator, fs: float, buffer_s: float = 30.0, with_red: bool = True):
        if buffer_s < MIN_SECONDS:
            raise ValueError(f"buffer_s must be at least {MIN_SECONDS:g} s")
        self.estimator, self.fs, self.with_red = estimator, float(fs), with_red
        n = int(round(buffer_s * self.fs))
        self._ir: deque[float] = deque(maxlen=n)
        self._red: deque[float] = deque(maxlen=n)
        self.samples_seen = 0

    def push(self, ir, red=None) -> int:
        ir_a = _signal(ir, "ir")
        if self.with_red:
            if red is None:
                raise ValueError("this stream was created with_red=True: pass red samples too")
            red_a = _signal(red, "red")
            if red_a.size != ir_a.size:
                raise ValueError("red and ir chunks must have the same length")
            self._red.extend(red_a.tolist())
        self._ir.extend(ir_a.tolist())
        self.samples_seen += ir_a.size
        return len(self._ir)

    @property
    def buffered_s(self) -> float:
        return len(self._ir) / self.fs

    def reset(self) -> None:
        self._ir.clear()
        self._red.clear()

    def estimate(self, metadata: dict | None = None) -> BPEstimate:
        red = np.fromiter(self._red, float) if self.with_red else None
        return self.estimator.estimate(np.fromiter(self._ir, float), self.fs, red=red, metadata=metadata)


__all__ = ["ANALOG_INPUT", "BPEstimate", "BPEstimator", "DEFAULT_MODEL", "STATUSES", "StreamingBPEstimator"]
