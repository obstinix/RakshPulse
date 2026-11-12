"""Hardware-independent streaming inference engine for red/IR PPG (RakshPulse 5.0).

``InferenceEngine`` is a sink for :class:`~rakshpulse.acquisition.ingest.Recorder` rows (``timestamp,
sample_index, red, infrared``). Every ``hop_s`` of new data it:

1. takes the newest ``window_s`` of samples from ring buffers. A sample-counter gap inside the window is filled by
   interpolation if it is ≤ ``max_fill_samples``; otherwise the window is rejected (``gap_in_window``);
2. computes quality and features with **the same function as the offline pipeline**
   (``preprocessing.pipeline.window_features``), so live and recorded data are processed identically;
3. gates on quality (``statuses``, default usable + degraded) and predicts SBP/DBP with the model bundle.
   ``feature_prefix_map`` maps pipeline names to bundle names (e.g. ``{"ir_": "ear_"}`` to run the 4.0
   Auricle ear bundle on MAX30102 IR data; this crosses a sensor/site domain and is **not** validated);
4. aggregates accepted windows (median over ``aggregate_s``) and attaches uncertainty:
   * uncalibrated: the model's cross-subject conformal 90 % half-width (stored at training time), if present;
   * calibrated: after :meth:`add_reference`, the hierarchical Bayesian posterior (``hier_params``) or, without
     parameters, a plain offset correction with no interval. :class:`CalibrationPolicy` flags expiry;
5. records timestamps, status, reasons and processing latency. Any exception while processing a window becomes an
   ``error`` event with the message; the stream continues.
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
from .model_loader import ModelBundle
from .streaming import RingBuffer


@dataclass
class EngineConfig:
    fs: float = 100.0
    window_s: float = 10.0
    hop_s: float = 5.0
    buffer_s: float = 30.0
    aggregate_s: float = 30.0
    statuses: tuple[str, ...] = ("usable", "degraded")
    max_fill_samples: int = 2
    feature_prefix_map: dict = field(default_factory=dict)
    session_id: str = "live"
    policy: CalibrationPolicy = field(default_factory=CalibrationPolicy)


@dataclass
class Prediction:
    timestamp: float  # end of the evaluated window (session time, s)
    window_start: float
    status: str  # usable | degraded | unusable | missing | warming_up | error
    reasons: str = ""
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
    calibration_valid: bool = False
    calibration_reason: str = "no_readings"
    n_aggregated: int = 0
    latency_ms: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceEngine:
    def __init__(
        self,
        bundle: ModelBundle,
        cfg: EngineConfig | None = None,
        pipeline: PipelineConfig | None = None,
        hier_params: dict[str, HierParams] | None = None,
    ):
        self.bundle, self.cfg = bundle, cfg or EngineConfig()
        c = self.cfg
        if c.hop_s <= 0 or c.window_s <= 0 or c.buffer_s < c.window_s:
            raise ValueError("need hop_s > 0, window_s > 0 and buffer_s >= window_s")
        self.pipeline = pipeline or PipelineConfig(fs=c.fs, window_s=c.window_s)
        n = int(round(c.buffer_s * c.fs))
        self.win_n, self.hop_n = int(round(c.window_s * c.fs)), int(round(c.hop_s * c.fs))
        self.buf = {k: RingBuffer(n) for k in ("t", "idx", "red", "ir")}
        self.since = 0
        self.last_t: float | None = None
        self.recent: deque = deque()
        self.history: list[Prediction] = []
        self.readings: list[dict] = []
        self.cal = {t: HierarchicalCalibrator(p) for t, p in (hier_params or {}).items()}
        self.conformal = {t: bundle.validation.get(f"conformal_90_halfwidth_{t}", float("nan")) for t in ("sbp", "dbp")}

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
        """Register a cuff reading. Its residual is taken against the nearest accepted window (± window_s)."""
        cands = [
            p
            for p in self.history
            if p.status in self.cfg.statuses and abs(p.timestamp - timestamp) <= self.cfg.window_s
        ]
        if not cands:
            return {"accepted": False, "reason": "no accepted window within window_s of the reading"}
        p = min(cands, key=lambda q: abs(q.timestamp - timestamp))
        self.readings.append(
            {
                "t": float(timestamp),
                "session_id": session_id or self.cfg.session_id,
                "sbp": float(sbp),
                "dbp": float(dbp),
                "pred_sbp": p.sbp_window,
                "pred_dbp": p.dbp_window,
            }
        )
        return {"accepted": True, "residual_sbp": sbp - p.sbp_window, "residual_dbp": dbp - p.dbp_window}

    # ------------------------------------------------------------------------------------------ core
    def _features_to_bundle(self, f: dict) -> np.ndarray:
        df = pd.DataFrame([f])
        if "ir_beat_T" in df:
            add_normalised_shape(df, "ir")
        for a, b in self.cfg.feature_prefix_map.items():
            df = df.rename(columns={c: b + c[len(a) :] for c in df.columns if c.startswith(a)})
        return df.reindex(columns=self.bundle.features).to_numpy(float)

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
                )
            )
        tt, idx = b["t"].last(self.win_n), b["idx"].last(self.win_n).astype(np.int64)
        red, ir = b["red"].last(self.win_n), b["ir"].last(self.win_n)
        pred = Prediction(timestamp=float(tt[-1] + 1 / self.cfg.fs), window_start=float(tt[0]), status="error")
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
                X = self._features_to_bundle(f)
                pred.sbp_window = float(np.ravel(self.bundle.sbp.predict(X))[0])
                pred.dbp_window = float(np.ravel(self.bundle.dbp.predict(X))[0])
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
                hw = self.conformal[tgt]
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
        return pd.DataFrame([p.to_dict() for p in self.history])
