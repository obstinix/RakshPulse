"""Hardware-independent inference for red/IR PPG: streaming (:class:`InferenceEngine`) and recorded sessions
(:func:`predict_windows`), both behind the model contract (``inference/contract.py``).

Before any sample is accepted, the engine:

* validates the bundle (schema, feature order, input dimension, contract version);
* checks the model may run on this input: the sensor (→ mode ``validated`` / ``experimental`` /
  ``demonstration``, or refused), the raw sampling rate, channels, preprocessing (window, band, target rate,
  feature-set version) and required demographics. Problems raise :class:`IncompatibleModelError` at construction;
  nothing is predicted.

Every ``hop_s`` of new data it then:

1. takes the newest ``window_s`` of samples from ring buffers. A counter gap inside the window is filled if it is
   ≤ ``max_fill_samples``, otherwise the window is rejected (``gap_in_window``);
2. computes quality and features with **the same function as the offline pipeline** (``window_features``);
3. gates on quality (``statuses``) and builds the model input through ``build_feature_vector``:
   * required features must be finite;
   * imputable features are filled with the training median (bounded, and listed in ``imputed``);
   * Inf/NaN never reach the model, and a window failing these checks gets status ``invalid_input``;
4. predicts SBP and DBP, then aggregates accepted windows (median over ``aggregate_s``);
5. attaches a 90 % interval **only** if the model has one valid for this sensor (``uncertainty_note`` otherwise),
   and a calibration from plausible cuff readings (:meth:`add_reference`) with policy-based expiry;
6. records the inference mode, status, reasons, timestamps and latency. An unexpected exception in one window
   becomes an ``error`` event; the stream continues.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..evaluation.auricle_study import add_normalised_shape
from ..personalization.hierarchical import HierarchicalCalibrator, HierParams
from ..personalization.strategies import Z90, CalibrationPolicy, calibration_status
from ..preprocessing.pipeline import STATUS, PipelineConfig, window_features
from .contract import (
    IncompatibleModelError,
    InvalidInputError,
    build_feature_vector,
    check_compatibility,
    validate_bundle,
)
from .model_loader import ModelBundle
from .streaming import RingBuffer

SBP_RANGE, DBP_RANGE = (50.0, 260.0), (25.0, 160.0)


@dataclass
class EngineConfig:
    fs: float  # raw sampling rate of the incoming samples (Hz); never assumed
    sensor: str = "max30102"  # input sensor key matched against the model contract
    window_s: float = 10.0
    hop_s: float = 5.0
    buffer_s: float = 30.0
    aggregate_s: float = 30.0
    statuses: tuple[str, ...] = ("usable", "degraded")
    max_fill_samples: int = 2
    feature_prefix_map: dict = field(default_factory=dict)  # only for legacy bundles without a contract
    session_id: str = "live"
    demographics: dict | None = None
    policy: CalibrationPolicy = field(default_factory=CalibrationPolicy)


@dataclass
class Prediction:
    timestamp: float  # end of the evaluated window (session time, s)
    window_start: float
    status: str  # usable | degraded | unusable | missing | invalid_input | warming_up | error
    reasons: str = ""
    inference_mode: str = ""  # validated | experimental | demonstration
    quality_score: float = float("nan")
    sbp_window: float = float("nan")
    dbp_window: float = float("nan")
    sbp: float = float("nan")  # aggregated (and calibrated, if valid) estimate
    dbp: float = float("nan")
    sbp_lo: float = float("nan")
    sbp_hi: float = float("nan")
    dbp_lo: float = float("nan")
    dbp_hi: float = float("nan")
    estimate_type: str = "none"  # none | population | calibrated
    imputed: str = ""  # model inputs filled with training medians (window)
    calibration_valid: bool = False
    calibration_reason: str = "no_readings"
    n_aggregated: int = 0
    latency_ms: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def window_feature_dict(f: dict) -> dict:
    """Add the beat-duration-normalised shape features that model contracts reference."""
    df = pd.DataFrame([f])
    if "ir_beat_T" in df:
        add_normalised_shape(df, "ir")
    return df.iloc[0].to_dict()


class InferenceEngine:
    def __init__(
        self,
        bundle: ModelBundle,
        cfg: EngineConfig,
        pipeline: PipelineConfig | None = None,
        hier_params: dict[str, HierParams] | None = None,
    ):
        c = self.cfg = cfg
        if c.fs is None or not np.isfinite(c.fs) or c.fs <= 0:
            raise ValueError("EngineConfig.fs (raw sampling rate) must be given explicitly")
        if c.hop_s <= 0 or c.window_s <= 0 or c.buffer_s < c.window_s:
            raise ValueError("need hop_s > 0, window_s > 0 and buffer_s >= window_s")
        self.bundle = bundle
        self.pipeline = pipeline or PipelineConfig(fs=c.fs, window_s=c.window_s)
        if abs(self.pipeline.fs - c.fs) > 1e-9:
            raise ValueError(f"pipeline fs {self.pipeline.fs} != engine fs {c.fs}")
        self.contract = validate_bundle(bundle, c.feature_prefix_map)
        compat = check_compatibility(
            self.contract,
            c.sensor,
            c.fs,
            {"window_s": self.pipeline.window_s, "band_hz": list(self.pipeline.band)},
            tuple(self.pipeline.channels),
            c.demographics,
        )
        self.mode = compat["mode"]
        self.demographics = compat["demographics"]
        self.halfwidth = compat["uncertainty"]
        self.uncertainty_note = compat["uncertainty_note"]
        n = int(round(c.buffer_s * c.fs))
        self.win_n, self.hop_n = int(round(c.window_s * c.fs)), int(round(c.hop_s * c.fs))
        self.buf = {k: RingBuffer(n) for k in ("t", "idx", "red", "ir")}
        self.since = 0
        self.last_t: float | None = None
        self.recent: deque = deque()
        self.history: list[Prediction] = []
        self.readings: list[dict] = []
        self.cal = {t: HierarchicalCalibrator(p) for t, p in (hier_params or {}).items()}

    def describe(self) -> dict:
        return {
            "model_id": self.contract.get("model_id"),
            "version": self.bundle.version,
            "mode": self.mode,
            "sensor": self.cfg.sensor,
            "fs": self.cfg.fs,
            "uncertainty": self.halfwidth or self.uncertainty_note,
            "limitations": self.contract.get("limitations", []),
        }

    # ------------------------------------------------------------------------------------------ input
    def push(self, rows: pd.DataFrame) -> list[Prediction]:
        """Add samples (timestamps must not go backwards); returns the predictions due."""
        if rows.empty:
            return []
        t = rows.timestamp.to_numpy(float)
        if (self.last_t is not None and t[0] < self.last_t) or np.any(np.diff(t) < 0):
            raise ValueError("timestamps must not go backwards")
        out = []
        start = 0
        while start < len(rows):
            take = min(self.hop_n - self.since, len(rows) - start)
            sl = slice(start, start + take)
            for k, col in (("t", "timestamp"), ("idx", "sample_index"), ("red", "red"), ("ir", "infrared")):
                self.buf[k].write(rows[col].to_numpy(float)[sl])
            self.since += take
            start += take
            if self.since >= self.hop_n:
                self.since = 0
                out.append(self._evaluate())
        self.last_t = float(t[-1])
        return out

    __call__ = push  # Recorder sink

    def add_reference(self, timestamp: float, sbp: float, dbp: float, session_id: str | None = None) -> dict:
        """Register a cuff reading of this session. Implausible values, readings from another session and readings
        without an accepted window within ``window_s`` are refused (never used for calibration)."""
        try:
            sbp, dbp, timestamp = float(sbp), float(dbp), float(timestamp)
        except (TypeError, ValueError):
            return {"accepted": False, "reason": "reading values must be numeric"}
        if not (SBP_RANGE[0] <= sbp <= SBP_RANGE[1] and DBP_RANGE[0] <= dbp <= DBP_RANGE[1] and sbp > dbp):
            return {"accepted": False, "reason": f"implausible reading SBP={sbp} DBP={dbp}"}
        if session_id is not None and session_id != self.cfg.session_id:
            return {"accepted": False, "reason": f"reading belongs to session {session_id}, not {self.cfg.session_id}"}
        cands = [
            p
            for p in self.history
            if p.status in self.cfg.statuses
            and np.isfinite(p.sbp_window)
            and abs(p.timestamp - timestamp) <= self.cfg.window_s
        ]
        if not cands:
            return {"accepted": False, "reason": "no accepted window within window_s of the reading"}
        p = min(cands, key=lambda q: abs(q.timestamp - timestamp))
        self.readings.append(
            {
                "t": timestamp,
                "session_id": self.cfg.session_id,
                "sbp": sbp,
                "dbp": dbp,
                "pred_sbp": p.sbp_window,
                "pred_dbp": p.dbp_window,
            }
        )
        return {"accepted": True, "residual_sbp": sbp - p.sbp_window, "residual_dbp": dbp - p.dbp_window}

    # ------------------------------------------------------------------------------------------ core
    def _evaluate(self) -> Prediction:
        t0 = time.perf_counter()
        b = self.buf
        if b["ir"].filled < self.win_n:
            return self._emit(
                Prediction(
                    timestamp=float(b["t"].last(1)[0]),
                    window_start=float("nan"),
                    status="warming_up",
                    reasons="buffer_filling",
                    inference_mode=self.mode,
                )
            )
        tt, idx = b["t"].last(self.win_n), b["idx"].last(self.win_n).astype(np.int64)
        red, ir = b["red"].last(self.win_n), b["ir"].last(self.win_n)
        pred = Prediction(
            timestamp=float(tt[-1] + 1 / self.cfg.fs),
            window_start=float(tt[0]),
            status="error",
            inference_mode=self.mode,
        )
        try:
            notes = []
            jumps = np.diff(idx)
            if np.any(jumps < 1):
                raise ValueError("sample counter not increasing inside the window")
            missing = int((jumps - 1).sum())
            if np.any(jumps - 1 > self.cfg.max_fill_samples):
                pred.status, pred.reasons = "missing", "gap_in_window"
                return self._finish(pred, t0)
            if missing:
                full = np.arange(idx[0], idx[-1] + 1)
                red, ir = np.interp(full, idx, red), np.interp(full, idx, ir)
                notes.append(f"filled_{missing}_missing_samples")
            f = window_features(red, ir, self.cfg.fs, self.pipeline)
            pred.status = STATUS[f["quality_class"]]
            pred.quality_score = float(f["quality_score"])
            pred.reasons = "|".join(x for x in (f.get("reasons", ""), *notes) if x)
            if pred.status in self.cfg.statuses:
                try:
                    X, imputed = build_feature_vector(self.contract, window_feature_dict(f), self.demographics)
                except InvalidInputError as exc:
                    pred.status = "invalid_input"
                    pred.reasons = "|".join(x for x in (pred.reasons, f"invalid_input: {exc}") if x)
                    return self._finish(pred, t0)
                pred.imputed = ",".join(imputed)
                pred.sbp_window = float(np.ravel(self.bundle.sbp.predict(X))[0])
                pred.dbp_window = float(np.ravel(self.bundle.dbp.predict(X))[0])
                if not (np.isfinite(pred.sbp_window) and np.isfinite(pred.dbp_window)):
                    pred.status, pred.reasons = "error", "model returned a non-finite value"
                    pred.sbp_window = pred.dbp_window = float("nan")
                else:
                    self.recent.append((pred.timestamp, pred.sbp_window, pred.dbp_window))
        except Exception as exc:  # keep streaming: report the failure on this window
            pred.status, pred.reasons = "error", f"{type(exc).__name__}: {exc}"
        return self._finish(pred, t0)

    def _finish(self, pred: Prediction, t0: float) -> Prediction:
        while self.recent and self.recent[0][0] < pred.timestamp - self.cfg.aggregate_s:
            self.recent.popleft()
        if self.recent:
            s = float(np.median([r[1] for r in self.recent]))
            d = float(np.median([r[2] for r in self.recent]))
            pred.n_aggregated, pred.estimate_type = len(self.recent), "population"
            pred.sbp, pred.dbp = s, d
            for tgt, v in (("sbp", s), ("dbp", d)):
                hw = self.halfwidth.get(tgt)
                if hw is not None:  # only when the model has an interval valid for this sensor
                    setattr(pred, f"{tgt}_lo", v - hw)
                    setattr(pred, f"{tgt}_hi", v + hw)
            self._calibrate(pred)
        pred.latency_ms = (time.perf_counter() - t0) * 1000
        return self._emit(pred)

    def _calibrate(self, pred: Prediction) -> None:
        if not self.readings:
            return
        r = pd.DataFrame(self.readings)
        valid, reason = calibration_status(r, pred.timestamp, self.cfg.session_id, self.cfg.policy)
        pred.calibration_valid, pred.calibration_reason = valid, reason
        if not valid:
            return
        cur = (r.session_id == self.cfg.session_id).to_numpy()
        dt_last = pred.timestamp - r.t[cur].max()
        for tgt in ("sbp", "dbp"):
            base = getattr(pred, tgt)
            res = (r[tgt] - r[f"pred_{tgt}"]).to_numpy()
            if tgt in self.cal:
                p = self.cal[tgt].predict(np.array([base]), np.array([dt_last]), res, cur)
                v, sd = float(p.yhat.iloc[0]), float(p.sd.iloc[0])
                lo, hi = v - Z90 * sd, v + Z90 * sd
            else:
                v, lo, hi = base + float(res[cur].mean()), float("nan"), float("nan")
            setattr(pred, tgt, v)
            setattr(pred, f"{tgt}_lo", lo)
            setattr(pred, f"{tgt}_hi", hi)
        pred.estimate_type = "calibrated"

    def _emit(self, pred: Prediction) -> Prediction:
        self.history.append(pred)
        return pred

    def predictions(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.history], columns=list(Prediction.__dataclass_fields__))


def predict_windows(
    bundle: ModelBundle,
    windows: pd.DataFrame,
    sensor: str,
    fs: float,
    demographics: dict | None = None,
    statuses: tuple[str, ...] = ("usable", "degraded"),
    pipeline: PipelineConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Predict every window of a processed recording (``derived/windows.csv``) through the same contract checks
    as streaming. Returns (per-window predictions, compatibility info). Raises IncompatibleModelError."""
    pipeline = pipeline or PipelineConfig(fs=fs)
    contract = validate_bundle(bundle)
    compat = check_compatibility(
        contract,
        sensor,
        fs,
        {"window_s": pipeline.window_s, "band_hz": list(pipeline.band)},
        tuple(pipeline.channels),
        demographics,
    )
    rows = []
    for _, w in windows.iterrows():
        r = {
            "window_id": w.get("window_id"),
            "t_start_s": w.get("t_start_s"),
            "t_end_s": w.get("t_end_s"),
            "status": w.get("status"),
            "reasons": w.get("reasons") if isinstance(w.get("reasons"), str) else "",
            "inference_mode": compat["mode"],
            "sbp": np.nan,
            "dbp": np.nan,
            "imputed": "",
        }
        if r["status"] in statuses:
            try:
                X, imputed = build_feature_vector(contract, w.to_dict(), compat["demographics"])
                r["sbp"] = float(np.ravel(bundle.sbp.predict(X))[0])
                r["dbp"] = float(np.ravel(bundle.dbp.predict(X))[0])
                r["imputed"] = ",".join(imputed)
            except InvalidInputError as exc:
                r["status"], r["reasons"] = "invalid_input", "|".join(x for x in (r["reasons"], str(exc)) if x)
        rows.append(r)
    return pd.DataFrame(rows), {**compat, "contract": contract}


__all__ = [
    "EngineConfig",
    "IncompatibleModelError",
    "InferenceEngine",
    "InvalidInputError",
    "Prediction",
    "predict_windows",
]
