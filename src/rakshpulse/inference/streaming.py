"""Streaming (sample-by-sample or chunked) BP estimation on top of :class:`RealtimeEstimator`'s model.

    s = StreamingEstimator(bundle, fs=100, engine=engine, baseline=baseline, log_path="infer.jsonl")
    for chunk_ts, ear_chunk in sensor_stream():        # any chunk size
        for event in s.push(ear_chunk, timestamp=chunk_ts):
            ...                                          # one event per evaluated window (every hop_s)

Design:
* ring buffer of the last ``buffer_s`` seconds; a 10 s window ending at the newest sample is evaluated every
  ``hop_s`` seconds of new data;
* each window passes the same SQI gate (plus a flat-signal check) and feature pipeline as training;
* rejected windows produce events with ``accepted=False`` and reasons. After ``lost_after`` consecutive
  rejections the stream status becomes ``signal_lost``;
* the reported estimate is the median of accepted per-window estimates in the last ``aggregate_s`` seconds,
  personalised by the optional :class:`PersonalizationEngine`;
* every event records feature-extraction and model latency. Events are kept in a bounded history and optionally
  appended as JSON lines.
Hardware-independent: it consumes plain arrays and timestamps (e.g. MAX30102 FIFO reads).
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.auricle_study import add_normalised_shape
from ..features.feature_pipeline import PER_CHANNEL, extract_features
from ..personalization.baseline import BaselineTracker
from ..personalization.engine import PersonalizationEngine
from ..preprocessing.filtering import validate_sampling_rate
from .model_loader import ModelBundle
from .predictor import InputError

WINDOW_S = 10.0


class RingBuffer:
    """Fixed-size circular sample buffer. ``write`` is O(chunk) (no whole-buffer shift as with ``np.roll``);
    ``last(n)`` returns the newest ``n`` samples in time order (one copy, at evaluation time only)."""

    def __init__(self, size: int):
        self.data = np.zeros(size)
        self.pos = 0  # next write index
        self.filled = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, float)
        n, size = x.size, self.data.size
        if n >= size:
            self.data[:] = x[-size:]
            self.pos, self.filled = 0, size
            return
        end = self.pos + n
        if end <= size:
            self.data[self.pos : end] = x
        else:
            k = size - self.pos
            self.data[self.pos :] = x[:k]
            self.data[: n - k] = x[k:]
        self.pos = end % size
        self.filled = min(self.filled + n, size)

    def last(self, n: int) -> np.ndarray:
        start = (self.pos - n) % self.data.size
        if start < self.pos:
            return self.data[start : self.pos].copy()
        return np.concatenate([self.data[start:], self.data[: self.pos]])


FLAT_STD = 1e-9  # relative flat-signal check is done on the window's own scale below


@dataclass
class StreamEvent:
    timestamp: float  # time of the newest sample in the evaluated window
    accepted: bool
    window_sbp: float | None = None
    window_dbp: float | None = None
    sbp: float | None = None  # aggregated (and personalised if calibrated) estimate
    dbp: float | None = None
    estimate_type: str = "none"
    n_windows_aggregated: int = 0
    status: str = "ok"  # ok | warming_up | poor_signal | signal_lost
    rejection_reasons: list[str] = field(default_factory=list)
    accepted_fraction_recent: float = 0.0
    recalibration_recommended: bool | None = None
    baseline_deviation: dict = field(default_factory=dict)
    feature_ms: float = 0.0
    model_ms: float = 0.0
    quality_class: str | None = None  # 4.0 graded quality (only when grade_quality=True)
    quality_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingEstimator:
    def __init__(
        self,
        bundle: ModelBundle,
        fs: float,
        engine: PersonalizationEngine | None = None,
        baseline: BaselineTracker | None = None,
        hop_s: float = 5.0,
        buffer_s: float = 30.0,
        aggregate_s: float = 30.0,
        lost_after: int = 3,
        history: int = 1000,
        log_path: str | Path | None = None,
        grade_quality: bool = False,
    ):
        try:
            self.fs = validate_sampling_rate(fs)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        if hop_s <= 0 or buffer_s < WINDOW_S:
            raise InputError("hop_s must be > 0 and buffer_s >= 10 s")
        self.bundle, self.site = bundle, bundle.channels[0]
        self.engine, self.baseline = engine, baseline
        self.hop_n = int(round(hop_s * self.fs))
        self.win_n = int(round(WINDOW_S * self.fs))
        self.ring = RingBuffer(int(round(buffer_s * self.fs)))
        self.grade_quality = grade_quality
        self._quality: tuple[str, float] | None = None
        self.since_last = 0
        self.t_last: float | None = None
        self.aggregate_s, self.lost_after = aggregate_s, lost_after
        self.recent: deque = deque()  # (timestamp, sbp, dbp) of accepted windows
        self.quality: deque = deque(maxlen=12)
        self.consecutive_rejects = 0
        self.history: deque = deque(maxlen=history)
        self.log_path = Path(log_path) if log_path else None

    # ------------------------------------------------------------------------------------------ input
    def push(self, samples, timestamp: float | None = None) -> list[StreamEvent]:
        """Append samples; ``timestamp`` is the time of the **last** sample in the chunk (seconds)."""
        x = np.asarray(samples, dtype=float).ravel()
        if x.size == 0:
            return []
        if timestamp is not None and self.t_last is not None and timestamp < self.t_last:
            raise InputError("timestamps must be non-decreasing")
        t_end = timestamp if timestamp is not None else (self.t_last or 0.0) + x.size / self.fs
        events = []
        # process in hop-sized pieces so a large chunk yields every due window
        start = 0
        while start < x.size:
            take = min(self.hop_n - self.since_last, x.size - start)
            piece = x[start : start + take]
            self.ring.write(piece)
            self.since_last += piece.size
            start += take
            if self.since_last >= self.hop_n:
                self.since_last = 0
                t_piece = t_end - (x.size - start) / self.fs
                events.append(self._evaluate(t_piece))
        self.t_last = t_end
        return events

    # ------------------------------------------------------------------------------------------ core
    def _evaluate(self, t: float) -> StreamEvent:
        if self.ring.filled < self.win_n:
            ev = StreamEvent(timestamp=t, accepted=False, status="warming_up", rejection_reasons=["buffer_filling"])
            return self._emit(ev)
        w = self.ring.last(self.win_n)
        t0 = time.perf_counter()
        reasons: list[str] = []
        if not np.isfinite(w).all():
            reasons.append("non_finite_samples")
        elif np.std(w) <= max(FLAT_STD, 1e-6 * (np.abs(w).mean() + 1e-12)):
            reasons.append("no_signal_flatline")
        feats = None
        self._quality = ("missing", 0.0) if reasons else None
        if not reasons:
            f, proc = extract_features({"ir": w}, self.fs, has_dc=False)
            if self.grade_quality:
                from ..preprocessing.filtering import TARGET_FS, resample
                from ..preprocessing.quality_score import score_window

                raw100 = resample(w, self.fs, TARGET_FS) if self.fs != TARGET_FS else w
                # flat_sd=0: the stream's own flat-line check above already decided "missing" in device units
                q = score_window(proc["ir"].filtered, raw100, TARGET_FS, flat_sd=0.0)
                self._quality = (q.quality_class, q.score)
            if proc["ir"].sqi.acceptable:
                feats = {f"{self.site}_{k}": f[f"ir_{k}"] for k in PER_CHANNEL}
            else:
                reasons += proc["ir"].sqi.reasons
        feature_ms = (time.perf_counter() - t0) * 1000
        self.quality.append(feats is not None)
        acc_frac = float(np.mean(self.quality))
        if feats is None:
            self.consecutive_rejects += 1
            status = "signal_lost" if self.consecutive_rejects >= self.lost_after else "poor_signal"
            ev = StreamEvent(
                timestamp=t,
                accepted=False,
                status=status,
                rejection_reasons=reasons,
                accepted_fraction_recent=acc_frac,
                feature_ms=feature_ms,
            )
            return self._emit(self._attach_aggregate(ev, t))
        self.consecutive_rejects = 0
        t1 = time.perf_counter()
        X = add_normalised_shape(pd.DataFrame([feats]), self.site).reindex(columns=self.bundle.features)
        Xv = X.to_numpy(float)
        ws, wd = float(self.bundle.sbp.predict(Xv)[0]), float(self.bundle.dbp.predict(Xv)[0])
        model_ms = (time.perf_counter() - t1) * 1000
        self.recent.append((t, ws, wd))
        ev = StreamEvent(
            timestamp=t,
            accepted=True,
            window_sbp=ws,
            window_dbp=wd,
            accepted_fraction_recent=acc_frac,
            feature_ms=feature_ms,
            model_ms=model_ms,
        )
        return self._emit(self._attach_aggregate(ev, t))

    def _attach_aggregate(self, ev: StreamEvent, t: float) -> StreamEvent:
        while self.recent and self.recent[0][0] < t - self.aggregate_s:
            self.recent.popleft()
        if not self.recent:
            return ev
        s = float(np.median([r[1] for r in self.recent]))
        d = float(np.median([r[2] for r in self.recent]))
        ev.n_windows_aggregated, ev.sbp, ev.dbp, ev.estimate_type = len(self.recent), s, d, "population"
        if self.engine is not None:
            st = self.engine.status(t)
            ev.recalibration_recommended = st["recalibration_recommended"]
            if st["calibrated"]:
                p = self.engine.apply(s, d, t)
                ev.sbp, ev.dbp, ev.estimate_type = p["sbp"], p["dbp"], "personalised"
        if self.baseline is not None and ev.sbp is not None:
            ev.baseline_deviation = self.baseline.deviation(ev.sbp, ev.dbp, upto=t)
        return ev

    def _emit(self, ev: StreamEvent) -> StreamEvent:
        if self.grade_quality and ev.status != "warming_up" and getattr(self, "_quality", None):
            ev.quality_class, ev.quality_score = self._quality
        self.history.append(ev)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev.to_dict(), default=float) + "\n")
        return ev

    # ------------------------------------------------------------------------------------------ stats
    def latency_summary(self) -> dict:
        f = np.array([e.feature_ms for e in self.history if e.status != "warming_up"])
        m = np.array([e.model_ms for e in self.history if e.accepted])
        q = lambda a, p: float(np.percentile(a, p)) if a.size else float("nan")  # noqa: E731
        return {
            "windows": int(f.size),
            "feature_ms_p50": q(f, 50),
            "feature_ms_p95": q(f, 95),
            "model_ms_p50": q(m, 50),
            "model_ms_p95": q(m, 95),
        }
