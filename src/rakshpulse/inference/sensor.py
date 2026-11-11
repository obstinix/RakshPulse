"""Hardware-independent sensor interface for the streaming estimator (RakshPulse 4.0).

The inference core never imports a driver. A device integration implements :class:`PPGSource` (for example a
MAX30102 FIFO reader on an ESP32 bridge or a Linux I²C service), and :func:`run_stream` feeds it to a
:class:`~rakshpulse.inference.streaming.StreamingEstimator`.

MAX30102 notes for the integrator (from the datasheet; not tested here, because no device was available):
* use the IR channel (the bundle is single-channel), with a sample rate configuration of 100 sps (or 50–400;
  ``fs`` must match the configured rate);
* the 32-sample FIFO should be drained at least every ~300 ms at 100 sps. Pass each burst with the timestamp of
  its last sample;
* raw counts carry a large DC and an inverted pulse. The pipeline removes DC and fixes the orientation, so no
  sensor-side filtering is required;
* behind-ear MAX30102 data have never been validated with these models (docs/DOMAIN_ADAPTATION.md).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from .streaming import StreamEvent, StreamingEstimator


@runtime_checkable
class PPGSource(Protocol):
    """Anything that yields ``(samples, timestamp_of_last_sample_s)`` chunks at a fixed sampling rate."""

    fs: float

    def chunks(self) -> Iterator[tuple[np.ndarray, float]]: ...


class ArraySource:
    """Replays a recorded array in fixed-size chunks (e.g. 32-sample FIFO bursts). Used for tests and replay."""

    def __init__(self, x: np.ndarray, fs: float, chunk: int = 32, t0: float = 0.0):
        self.x, self.fs, self.chunk, self.t0 = np.asarray(x, float), float(fs), int(chunk), float(t0)

    def chunks(self) -> Iterator[tuple[np.ndarray, float]]:
        for i in range(0, self.x.size, self.chunk):
            c = self.x[i : i + self.chunk]
            yield c, self.t0 + (i + c.size) / self.fs


def run_stream(source: PPGSource, estimator: StreamingEstimator) -> Iterator[StreamEvent]:
    """Drive ``estimator`` from ``source``, yielding every evaluated-window event."""
    if abs(source.fs - estimator.fs) > 1e-9:
        raise ValueError(f"source fs {source.fs} != estimator fs {estimator.fs}")
    for samples, ts in source.chunks():
        yield from estimator.push(samples, timestamp=ts)
