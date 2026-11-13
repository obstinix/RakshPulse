"""Real-time ingestion: timestamping, missing-sample detection and recording control (RakshPulse 5.0).

:class:`TimestampSync` turns :class:`SampleBlock` objects into per-sample rows with session timestamps:

* device timestamps, if the firmware sends them, are used directly (they must increase);
* otherwise a device sample counter gives ``t = t0 + counter / fs``. Counter jumps > 1 are **missing samples**
  (logged as gap events, never interpolated here). Counter wrap-around (``counter_modulus``) is handled;
* otherwise, with no counter, samples are assumed contiguous at ``fs``. A gap is inferred only when a block
  arrives later than ``gap_tolerance_s`` after its expected time (``n_missing = round(lateness · fs)``).

The host arrival times are kept, and :meth:`TimestampSync.drift_ppm` estimates the device clock error against the
host clock (useful to verify that the configured ``fs`` matches the hardware).

:class:`Recorder` pulls blocks from a source, synchronises them, forwards rows to sinks (a CSV session writer,
a streaming estimator, ...) and supports start/stop from another thread or by duration / sample limits.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sources import SampleBlock, SensorSource


@dataclass
class GapEvent:
    timestamp: float  # session time of the first missing sample
    first_missing_index: int
    n_missing: int
    detected_by: str  # "counter" | "host_time" | "device_time"


@dataclass
class TimestampSync:
    fs: float
    counter_modulus: int | None = None  # e.g. 2**16 if the firmware counter wraps
    gap_tolerance_s: float = 0.5
    t0_host: float | None = None
    _last_index: int | None = None
    _wraps: int = 0
    _next_t: float | None = None
    _first_counter: int | None = None
    _host: list = field(default_factory=list)  # (sample_index, host_time) per block for drift estimation
    gaps: list = field(default_factory=list)
    n_samples: int = 0
    n_missing: int = 0
    n_duplicates: int = 0  # repeated sample counters (dropped, never written twice)
    n_resets: int = 0  # counter restarted (device reboot): recorded as a discontinuity
    allow_counter_reset: bool = True
    _reset_offset: int = 0

    def process(self, b: SampleBlock) -> pd.DataFrame:
        n = len(b)
        if n == 0:
            return pd.DataFrame(columns=["timestamp", "sample_index", "red", "infrared"])
        if self.t0_host is None:
            self.t0_host = b.host_time_s - (n - 1) / self.fs
        if b.device_time_s is not None:
            t = np.asarray(b.device_time_s, float)
            if self._next_t is None:
                self._dev0 = t[0]
            t = t - self._dev0
            if np.any(np.diff(t) <= 0) or (self._next_t is not None and t[0] <= self._next_t - 1 / self.fs):
                raise ValueError("device timestamps must strictly increase")
            dt = np.diff(np.r_[self._next_t - 1 / self.fs if self._next_t is not None else t[0] - 1 / self.fs, t])
            idx_f = np.round(t * self.fs).astype(np.int64)
            for j in np.flatnonzero(dt > 1.5 / self.fs):
                miss = int(round(dt[j] * self.fs)) - 1
                self._gap(t[j] - miss / self.fs, int(idx_f[j]) - miss, miss, "device_time")
            idx = idx_f
        elif b.sample_index is not None:
            raw = np.asarray(b.sample_index, np.int64)
            uw = self._unwrap(raw)
            if self._first_counter is None:
                self._first_counter = int(uw[0])
            prev = self._last_index
            idx_l, keep = [], np.ones(n, bool)
            for j in range(n):
                v = int(uw[j]) - self._first_counter + self._reset_offset  # offset may change at a reset
                if prev is not None and v == prev:  # duplicate sample (re-sent line): drop
                    self.n_duplicates += 1
                    keep[j] = False
                    continue
                if prev is not None and v < prev:
                    prev_raw = prev - self._reset_offset + self._first_counter
                    rebooted = uw[j] < prev_raw / 2 and prev_raw >= 2 * self.fs  # restarted near zero after >= 2 s
                    if not (self.allow_counter_reset and rebooted):
                        raise ValueError(
                            "sample counter went backwards (corrupted or out-of-order data; configure "
                            "counter_modulus if the firmware counter wraps)"
                        )
                    # the device restarted: continue after a discontinuity of one second so no window spans it
                    new_v = prev + 1 + max(int(round(self.fs)), 3)
                    self._reset_offset += new_v - v
                    self.n_resets += 1
                    self.gaps.append(GapEvent(float(prev + 1) / self.fs, int(prev + 1), 0, "counter_reset"))
                    v = new_v
                elif prev is not None and v - prev > 1:
                    self._gap((prev + 1) / self.fs, prev + 1, v - prev - 1, "counter")
                idx_l.append(v)
                prev = v
            if not idx_l:
                return pd.DataFrame(columns=["timestamp", "sample_index", "red", "infrared"])
            idx = np.asarray(idx_l, np.int64)
            b = SampleBlock(
                red=np.asarray(b.red)[keep],
                ir=np.asarray(b.ir)[keep],
                host_time_s=b.host_time_s,
                sample_index=idx,
                temperature_c=None if b.temperature_c is None else np.asarray(b.temperature_c)[keep],
            )
            n = idx.size
            t = idx / self.fs
        else:
            start = 0 if self._last_index is None else self._last_index + 1
            if self._next_t is not None:
                expected_end = self.t0_host + (start + n - 1) / self.fs
                late = b.host_time_s - expected_end
                if late > self.gap_tolerance_s:
                    miss = int(round(late * self.fs))
                    self._gap(start / self.fs, start, miss, "host_time")
                    start += miss
            idx = np.arange(start, start + n)
            t = idx / self.fs
        self._last_index = int(idx[-1])
        self._next_t = float(t[-1]) + 1 / self.fs
        self._host.append((int(idx[-1]), float(b.host_time_s)))
        self.n_samples += n
        out = pd.DataFrame({"timestamp": t, "sample_index": idx, "red": b.red, "infrared": b.ir})
        if b.temperature_c is not None:
            out["temperature_c"] = b.temperature_c
        return out

    def _unwrap(self, raw: np.ndarray) -> np.ndarray:
        if self.counter_modulus is None:
            return raw
        out = np.empty_like(raw)
        last = (
            None if self._last_index is None else (self._last_index + (self._first_counter or 0)) % self.counter_modulus
        )
        for i, v in enumerate(raw):
            if last is not None and v < last:
                self._wraps += 1
            out[i] = v + self._wraps * self.counter_modulus
            last = v
        return out

    def _gap(self, t: float, first: int, n: int, how: str) -> None:
        self.gaps.append(GapEvent(float(t), int(first), int(n), how))
        self.n_missing += int(n)

    def drift_ppm(self) -> float:
        """Device-vs-host clock rate error (ppm) from block arrival times; NaN with < 3 blocks."""
        if len(self._host) < 3:
            return float("nan")
        idx, host = np.array(self._host, float).T
        slope = np.polyfit(idx, host, 1)[0]
        return float((slope * self.fs - 1.0) * 1e6)

    def gap_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [g.__dict__ for g in self.gaps], columns=["timestamp", "first_missing_index", "n_missing", "detected_by"]
        )


Sink = Callable[[pd.DataFrame], None]


class Recorder:
    """Pull blocks from ``source`` → :class:`TimestampSync` → every sink. Stop with :meth:`stop` (thread-safe),
    ``duration_s`` (session time) or ``max_samples``."""

    def __init__(self, source: SensorSource, sinks: list[Sink], sync: TimestampSync | None = None):
        self.source, self.sinks = source, list(sinks)
        self.sync = sync or TimestampSync(source.fs)
        self._stop = threading.Event()
        self.started_utc: float | None = None
        self.stopped_utc: float | None = None
        self.state = "idle"

    def stop(self) -> None:
        self._stop.set()

    def run(self, duration_s: float | None = None, max_samples: int | None = None) -> dict:
        if self.state == "running":
            raise RuntimeError("recorder already running")
        self.state, self.started_utc = "running", time.time()
        try:
            for block in self.source.blocks():
                rows = self.sync.process(block)
                if duration_s is not None:
                    rows = rows[rows.timestamp < duration_s]
                if max_samples is not None:
                    rows = rows.iloc[: max(0, max_samples - (self.sync.n_samples - len(block)))]
                if len(rows):
                    for s in self.sinks:
                        s(rows)
                done_t = duration_s is not None and self.sync._next_t is not None and self.sync._next_t >= duration_s
                done_n = max_samples is not None and self.sync.n_samples >= max_samples
                if self._stop.is_set() or done_t or done_n:
                    break
        finally:
            self.source.close()
            self.state, self.stopped_utc = "stopped", time.time()
        return {
            "samples": int(self.sync.n_samples),
            "missing_samples": int(self.sync.n_missing),
            "gaps": len(self.sync.gaps),
            "drift_ppm": self.sync.drift_ppm(),
            "stopped_by": "stop()" if self._stop.is_set() else "limit_or_end",
        }
