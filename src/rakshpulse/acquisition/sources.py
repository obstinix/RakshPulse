"""Sensor sources (RakshPulse 5.0): one interface for simulated, replayed and real (serial/USB) PPG input.

Every source yields :class:`SampleBlock` objects: consecutive samples with a device sample counter (if the
device provides one), red and IR counts, and the host time the block arrived. Timestamps are assigned later by
:class:`~rakshpulse.acquisition.ingest.TimestampSync`, so the rest of the pipeline never depends on the transport.

**No hardware protocol is assumed.** :class:`LineFormat` describes the text lines the microcontroller firmware
emits (delimiter, column order, whether a sample counter / device time / die temperature is present). The
hardware team fixes the firmware output and configures ``LineFormat`` to match. See
docs/HARDWARE_INTEGRATION.md ("Connecting a MAX30102").

:class:`MAX30102Config` only *validates and records* the sensor settings that determine the data. It accepts
the register options the datasheet lists (sample rate, pulse width/ADC resolution, ADC range, sample averaging,
LED currents). The firmware applies them; this software does not talk I²C.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, TextIO, runtime_checkable

import numpy as np
import pandas as pd

from .simulator import SIMULATED_FLAG, SimConfig, simulate

# MAX30102 register options (datasheet): SpO2 sample rates, LED pulse widths (with ADC resolution), ADC full-scale
# ranges and FIFO sample averaging.
MAX30102_SAMPLE_RATES = (50, 100, 200, 400, 800, 1000, 1600, 3200)
MAX30102_PULSE_WIDTHS_US = {69: 15, 118: 16, 215: 17, 411: 18}  # µs → ADC bits
MAX30102_ADC_RANGES_NA = (2048, 4096, 8192, 16384)
MAX30102_SAMPLE_AVERAGING = (1, 2, 4, 8, 16, 32)
MAX30102_FIFO_DEPTH = 32


@dataclass
class MAX30102Config:
    sample_rate_hz: int = 100
    sample_averaging: int = 1
    pulse_width_us: int = 411
    adc_range_na: int = 16384
    led_current_red_ma: float = 7.0
    led_current_ir_ma: float = 7.0
    mode: str = "spo2"  # red + IR

    def validate(self) -> list[str]:
        """Raise on values the datasheet does not offer; return warnings for settings needing a datasheet check."""
        errs = []
        if self.sample_rate_hz not in MAX30102_SAMPLE_RATES:
            errs.append(f"sample_rate_hz must be one of {MAX30102_SAMPLE_RATES}")
        if self.pulse_width_us not in MAX30102_PULSE_WIDTHS_US:
            errs.append(f"pulse_width_us must be one of {sorted(MAX30102_PULSE_WIDTHS_US)}")
        if self.adc_range_na not in MAX30102_ADC_RANGES_NA:
            errs.append(f"adc_range_na must be one of {MAX30102_ADC_RANGES_NA}")
        if self.sample_averaging not in MAX30102_SAMPLE_AVERAGING:
            errs.append(f"sample_averaging must be one of {MAX30102_SAMPLE_AVERAGING}")
        for name in ("led_current_red_ma", "led_current_ir_ma"):
            if not 0.0 <= getattr(self, name) <= 51.0:
                errs.append(f"{name} must be within 0–51 mA")
        if self.mode != "spo2":
            errs.append("mode must be 'spo2' (red + IR); heart-rate mode records IR only")
        if errs:
            raise ValueError("; ".join(errs))
        warnings = []
        if self.pulse_width_us == 411 and self.sample_rate_hz > 400:
            warnings.append("411 µs pulse width is limited to ≤ 400 sps in SpO2 mode (datasheet table)")
        if self.sample_rate_hz > 400:
            warnings.append("check the datasheet's allowed sample-rate / pulse-width combinations for this rate")
        return warnings

    @property
    def output_rate_hz(self) -> float:
        """Rate at which samples leave the FIFO (sample rate / averaging)."""
        return self.sample_rate_hz / self.sample_averaging

    @property
    def adc_bits(self) -> int:
        return MAX30102_PULSE_WIDTHS_US[self.pulse_width_us]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleBlock:
    red: np.ndarray
    ir: np.ndarray
    host_time_s: float  # host clock when the block arrived (time.time() for live sources)
    sample_index: np.ndarray | None = None  # device counter, if the firmware provides one
    device_time_s: np.ndarray | None = None  # device timestamps, if provided
    temperature_c: np.ndarray | None = None  # MAX30102 die temperature, if reported (NOT skin temperature)
    flags: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(np.asarray(self.ir).size)


@runtime_checkable
class SensorSource(Protocol):
    fs: float
    sensor_type: str

    def blocks(self) -> Iterator[SampleBlock]: ...

    def close(self) -> None: ...


# ------------------------------------------------------------------------------------------------ simulated
class SimulatedSensor:
    """Streams a simulated recording in FIFO-sized blocks (default 32 samples, as a MAX30102 FIFO).

    ``realtime=True`` paces the blocks with the wall clock; otherwise blocks are produced as fast as possible with
    synthetic host times (deterministic, for tests and offline recording)."""

    sensor_type = "simulator"

    def __init__(self, cfg: SimConfig, duration_s: float, block: int = MAX30102_FIFO_DEPTH, realtime: bool = False):
        self.cfg, self.fs, self.block, self.realtime = cfg, float(cfg.fs), int(block), realtime
        self.rec = simulate(cfg, duration_s)
        self._closed = False

    def blocks(self) -> Iterator[SampleBlock]:
        r = self.rec
        t0 = time.time()
        for i in range(0, r.ir.size, self.block):
            if self._closed:
                return
            sl = slice(i, i + self.block)
            last_t = float(r.t[sl][-1])
            if self.realtime:
                time.sleep(max(0.0, t0 + last_t - time.time()))
            yield SampleBlock(
                red=r.red[sl],
                ir=r.ir[sl],
                host_time_s=(time.time() if self.realtime else last_t),
                sample_index=r.sample_index[sl],
                flags={"source": SIMULATED_FLAG},
            )

    def close(self) -> None:
        self._closed = True


# --------------------------------------------------------------------------------------------------- replay
class ReplaySource:
    """Replays a recorded ``raw_ppg.csv`` (dataset schema) in blocks, keeping its timestamps and counter."""

    sensor_type = "replay"

    def __init__(self, path: str | Path, fs: float, block: int = MAX30102_FIFO_DEPTH, chunksize: int = 100_000):
        self.path, self.fs, self.block, self.chunksize = Path(path), float(fs), int(block), int(chunksize)

    def blocks(self) -> Iterator[SampleBlock]:
        for chunk in pd.read_csv(self.path, chunksize=self.chunksize):
            for i in range(0, len(chunk), self.block):
                c = chunk.iloc[i : i + self.block]
                yield SampleBlock(
                    red=c["red"].to_numpy(float),
                    ir=c["infrared"].to_numpy(float),
                    host_time_s=float(c["timestamp"].iloc[-1]),
                    sample_index=c["sample_index"].to_numpy(np.int64),
                    device_time_s=c["timestamp"].to_numpy(float),
                    temperature_c=c["temperature_c"].to_numpy(float) if "temperature_c" in c else None,
                )

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------ text / serial
@dataclass
class LineFormat:
    """Describes one text line per sample emitted by the firmware, e.g. ``"1234,95321,118204"``.

    ``fields`` lists the column names in order. Recognised names: ``sample_index``, ``device_time_us``,
    ``device_time_ms``, ``red``, ``ir``, ``temperature_c``. Others (e.g. ``ignore``) are skipped. ``red`` and ``ir``
    are required. Lines starting with ``comment_prefix`` or failing to parse are counted, not raised.
    """

    fields: tuple[str, ...] = ("sample_index", "red", "ir")
    delimiter: str = ","
    comment_prefix: str = "#"

    def parse(self, line: str) -> dict | None:
        line = line.strip()
        if not line or line.startswith(self.comment_prefix):
            return None
        parts = line.split(self.delimiter)
        if len(parts) != len(self.fields):
            return None
        try:
            vals = {f: float(p) for f, p in zip(self.fields, parts, strict=True) if f != "ignore"}
        except ValueError:
            return None
        if "red" not in vals or "ir" not in vals:
            return None
        return vals


class TextStreamSource:
    """Reads samples from any text stream (a serial port wrapper, a pipe, a file or ``io.StringIO``)."""

    sensor_type = "max30102_text_stream"

    def __init__(self, stream: TextIO | io.TextIOBase, fs: float, fmt: LineFormat | None = None, block: int = 32):
        self.stream, self.fs, self.fmt, self.block = stream, float(fs), fmt or LineFormat(), int(block)
        self.bad_lines = 0
        self._closed = False

    def _emit(self, rows: list[dict]) -> SampleBlock:
        get = lambda k: np.array([r[k] for r in rows], float) if k in rows[0] else None  # noqa: E731
        dev = get("device_time_us")
        dev = dev / 1e6 if dev is not None else (get("device_time_ms") / 1e3 if "device_time_ms" in rows[0] else None)
        idx = get("sample_index")
        return SampleBlock(
            red=get("red"),
            ir=get("ir"),
            host_time_s=time.time(),
            sample_index=idx.astype(np.int64) if idx is not None else None,
            device_time_s=dev,
            temperature_c=get("temperature_c"),
        )

    def blocks(self) -> Iterator[SampleBlock]:
        rows: list[dict] = []
        for line in self.stream:
            if self._closed:
                break
            v = self.fmt.parse(line if isinstance(line, str) else line.decode("ascii", "replace"))
            if v is None:
                if line.strip() and not str(line).lstrip().startswith(self.fmt.comment_prefix):
                    self.bad_lines += 1
                continue
            if rows and set(v) != set(rows[0]):
                self.bad_lines += 1
                continue
            rows.append(v)
            if len(rows) >= self.block:
                yield self._emit(rows)
                rows = []
        if rows:
            yield self._emit(rows)

    def close(self) -> None:
        self._closed = True


class SerialSource(TextStreamSource):
    """USB-serial input (e.g. an ESP32/Arduino forwarding MAX30102 FIFO reads as text lines).

    Needs the optional ``pyserial`` package (``pip install pyserial``). The port, baud rate and line format must
    match the firmware. Nothing is assumed about the device beyond :class:`LineFormat`."""

    sensor_type = "max30102_serial"

    def __init__(self, port: str, baudrate: int, fs: float, fmt: LineFormat | None = None, timeout_s: float = 1.0):
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the optional package
            raise RuntimeError("SerialSource needs the optional 'pyserial' package: pip install pyserial") from exc
        self._port = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
        wrapper = io.TextIOWrapper(io.BufferedReader(self._port), encoding="ascii", errors="replace", newline=None)
        super().__init__(wrapper, fs, fmt)

    def close(self) -> None:  # pragma: no cover - needs a device
        super().close()
        self._port.close()
