"""Sensor sources (RakshPulse 5.0): one interface for simulated, replayed and real (serial/USB) PPG input.

Every source yields :class:`SampleBlock` objects: consecutive samples with a device sample counter (if the
device provides one), red and IR counts, and the host time the block arrived. Timestamps are assigned later by
:class:`~rakshpulse.acquisition.ingest.TimestampSync`, so the rest of the pipeline never depends on the transport.

**The serial contract is fixed in docs/HARDWARE_HANDOFF.md** (default line ``sample_index,red,ir``; the reference
firmware is ``hardware/firmware/rakshpulse_max30102``). :class:`LineFormat` describes the text lines the firmware
emits (delimiter, column order, whether a sample counter / device time / die temperature is present). The
hardware team fixes the firmware output and configures ``LineFormat`` to match.

:class:`MAX30102Config` only *validates and records* the sensor settings that determine the data. It accepts
the register options the datasheet lists (sample rate, pulse width/ADC resolution, ADC range, sample averaging,
LED currents). The firmware applies them; this software does not talk I²C.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

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
    ``device_time_ms``, ``red``, ``ir`` (or ``infrared``), ``temperature_c``. Others (e.g. ``ignore``) are
    skipped. ``red`` and ``ir`` are required. Lines starting with ``comment_prefix`` or failing to parse are
    counted, not raised.
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
        if "infrared" in vals:  # accept the documented long name as an alias of "ir"
            vals["ir"] = vals.pop("infrared")
        if "red" not in vals or "ir" not in vals:
            return None
        return vals


class AcquisitionError(RuntimeError):
    """An acquisition problem the user can act on (no port, no data, wrong rate...). The message says what to do."""


def parse_banner(line: str) -> dict:
    """``# rakshpulse key=value key=value`` → dict (the firmware's self-description; other comments → {})."""
    body = line.strip().lstrip("#").strip()
    if not body.startswith("rakshpulse"):
        return {}
    out = {}
    for tok in body.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class TextStreamSource:
    """Reads samples from any text stream (a serial port wrapper, a pipe, a file or ``io.StringIO``).

    Comment lines (``#``) are skipped. A ``# rakshpulse key=value ...`` banner is parsed into ``device_info`` (the
    firmware declares e.g. ``fs_hz=100``). Malformed lines, non-finite values, and lines whose field set differs from
    the stream's first data line are counted in ``bad_lines`` and never emitted."""

    sensor_type = "max30102"
    transport = "text_stream"

    def __init__(self, stream, fs: float, fmt: LineFormat | None = None, block: int = 32):
        self.stream, self.fs, self.fmt, self.block = stream, float(fs), fmt or LineFormat(), int(block)
        self.bad_lines = 0
        self.good_lines = 0
        self.device_info: dict = {}
        self._closed = False

    def _emit(self, rows: list[dict]) -> SampleBlock:
        def get(k):
            return np.array([r[k] for r in rows], float) if k in rows[0] else None

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
            flags=dict(self.device_info),
        )

    def _lines(self) -> Iterator[str]:
        for line in self.stream:
            yield line if isinstance(line, str) else line.decode("ascii", "replace")

    def _consume(self, line: str, rows: list[dict]) -> bool:
        """Parse one line into ``rows``; returns True if a sample was added."""
        if line.lstrip().startswith(self.fmt.comment_prefix):
            info = parse_banner(line)
            if info:
                self.device_info.update(info)
            return False
        v = self.fmt.parse(line)
        if v is None:
            if line.strip():
                self.bad_lines += 1
            return False
        if rows and set(v) != set(rows[0]):
            self.bad_lines += 1
            return False
        if not (np.isfinite(v["red"]) and np.isfinite(v["ir"])):
            self.bad_lines += 1
            return False
        rows.append(v)
        self.good_lines += 1
        return True

    def blocks(self) -> Iterator[SampleBlock]:
        rows: list[dict] = []
        try:
            for line in self._lines():
                if self._closed:
                    break
                self._consume(line, rows)
                if len(rows) >= self.block:
                    yield self._emit(rows)
                    rows = []
        except AcquisitionError:
            if rows:  # deliver samples already received before reporting the problem
                yield self._emit(rows)
            raise
        if rows:
            yield self._emit(rows)

    def close(self) -> None:
        self._closed = True


def _open_serial(port: str, baudrate: int, timeout_s: float):
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AcquisitionError("serial input needs the optional 'pyserial' package: pip install -e .[serial]") from exc
    try:
        if "://" in port:  # loop://, socket://host:port, rfc2217:// (virtual or networked ports)
            return serial.serial_for_url(port, baudrate=baudrate, timeout=timeout_s)
        return serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
    except (serial.SerialException, OSError, ValueError) as exc:
        raise AcquisitionError(
            f"cannot open serial port {port!r} at {baudrate} baud: {exc}. Check the port name (rakshpulse ports), "
            "that no other program (e.g. the Arduino Serial Monitor) has it open, and the USB cable."
        ) from exc


class SerialSource(TextStreamSource):
    """USB-serial input from an ESP32/Arduino that forwards MAX30102 samples as text lines (see
    docs/HARDWARE_HANDOFF.md and ``hardware/firmware``).

    * Opening needs the optional ``pyserial`` (``pip install -e .[serial]``). A missing package or a busy/absent port
      raises :class:`AcquisitionError` with instructions.
    * Each read waits at most ``timeout_s``. If no valid sample arrives for ``no_data_timeout_s`` the source raises
      :class:`AcquisitionError` (wrong baud rate, firmware not running, sensor not detected).
    * On a port error (cable pulled) it reopens up to ``reconnect_attempts`` times. Samples lost meanwhile appear as
      counter gaps downstream, never silently stitched together.
    * ``port_factory(port, baudrate, timeout_s)`` can replace pyserial (tests, other transports)."""

    transport = "serial"

    def __init__(
        self,
        port: str,
        baudrate: int,
        fs: float,
        fmt: LineFormat | None = None,
        timeout_s: float = 0.5,
        no_data_timeout_s: float = 5.0,
        reconnect_attempts: int = 3,
        reconnect_wait_s: float = 1.0,
        block: int = 32,
        port_factory=None,
    ):
        if baudrate <= 0:
            raise AcquisitionError("baud rate must be positive")
        if not fs or fs <= 0:
            raise AcquisitionError("the effective sampling rate (--fs) must be given and positive")
        self.port_name, self.baudrate, self.timeout_s = port, int(baudrate), float(timeout_s)
        self.no_data_timeout_s, self.reconnect_attempts = float(no_data_timeout_s), int(reconnect_attempts)
        self.reconnect_wait_s = float(reconnect_wait_s)
        self._factory = port_factory or _open_serial
        self._port = self._factory(port, self.baudrate, self.timeout_s)
        self.reconnects = 0
        super().__init__(None, fs, fmt, block)

    def _reopen(self, exc: Exception) -> None:
        for _ in range(self.reconnect_attempts):
            time.sleep(self.reconnect_wait_s)
            try:
                self._port.close()
            except Exception:  # the old handle may already be invalid
                pass
            try:
                self._port = self._factory(self.port_name, self.baudrate, self.timeout_s)
                self.reconnects += 1
                return
            except AcquisitionError:
                continue
        raise AcquisitionError(
            f"serial port {self.port_name!r} was lost and could not be reopened after "
            f"{self.reconnect_attempts} attempts ({exc})"
        ) from exc

    def _lines(self) -> Iterator[str]:
        last_data = time.monotonic()
        while not self._closed:
            try:
                raw = self._port.readline()
            except AcquisitionError:
                raise
            except Exception as exc:  # serial.SerialException / OSError: try to reopen
                self._reopen(exc)
                continue
            if raw:
                before = self.good_lines
                yield raw.decode("ascii", "replace") if isinstance(raw, bytes) else str(raw)
                if self.good_lines > before:
                    last_data = time.monotonic()
                    continue
            if time.monotonic() - last_data > self.no_data_timeout_s:
                raise AcquisitionError(
                    f"no valid samples from {self.port_name!r} for {self.no_data_timeout_s:.0f} s "
                    f"({self.bad_lines} malformed lines). Check the baud rate ({self.baudrate}), that the firmware is "
                    "running and printing 'sample_index,red,ir' lines, and that the sensor was detected."
                )

    def close(self) -> None:
        super().close()
        try:
            self._port.close()
        except Exception:  # closing twice or after a lost port must not raise
            pass


def list_ports() -> list[dict]:
    """Serial ports visible to the OS (needs pyserial)."""
    try:
        from serial.tools import list_ports as lp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AcquisitionError("listing ports needs the optional 'pyserial' package: pip install -e .[serial]") from exc
    return [{"device": p.device, "description": p.description, "hwid": p.hwid} for p in lp.comports()]


def check_serial(source: TextStreamSource, seconds: float = 5.0, rate_tolerance: float = 0.05) -> dict:
    """Read ``seconds`` of data and report whether the stream matches the RakshPulse serial contract."""
    from .ingest import TimestampSync

    sync = TimestampSync(source.fs)
    t0 = time.monotonic()
    n, red, ir = 0, [], []
    err = None
    first_t = None
    try:
        for b in source.blocks():
            if first_t is None:
                first_t = time.monotonic()
            rows = sync.process(b)
            n += len(rows)
            red.append(rows.red.to_numpy())
            ir.append(rows.infrared.to_numpy())
            if time.monotonic() - t0 >= seconds:
                break
    except (AcquisitionError, ValueError) as exc:
        err = str(exc)
    finally:
        source.close()
    elapsed = time.monotonic() - (first_t or t0)
    rate = (n + sync.n_missing) / elapsed if elapsed > 0 and n else float("nan")
    declared = float(source.device_info["fs_hz"]) if "fs_hz" in source.device_info else None
    problems = []
    if err:
        problems.append(err)
    if n == 0:
        problems.append("no valid samples received")
    if declared is not None and abs(declared - source.fs) > 1e-6:
        problems.append(f"firmware declares fs_hz={declared:g} but --fs is {source.fs:g}")
    if n and np.isfinite(rate) and elapsed >= 2 and abs(rate - source.fs) / source.fs > rate_tolerance:
        problems.append(
            f"measured rate {rate:.1f} Hz differs from --fs {source.fs:g} Hz by more than {rate_tolerance:.0%}"
        )
    allr = np.concatenate(red) if red else np.array([])
    alli = np.concatenate(ir) if ir else np.array([])
    if alli.size > 1 and (np.ptp(alli) == 0 or np.ptp(allr) == 0):
        problems.append("a channel is constant: the sensor is not reading (check wiring and skin contact)")
    return {
        "samples": n,
        "malformed_lines": source.bad_lines,
        "duplicates": sync.n_duplicates,
        "missing_samples": sync.n_missing,
        "gaps": len(sync.gaps),
        "counter_resets": sync.n_resets,
        "measured_rate_hz": round(rate, 2) if np.isfinite(rate) else None,
        "declared_fs_hz": declared,
        "device_info": dict(source.device_info),
        "red_range": [float(allr.min()), float(allr.max())] if allr.size else None,
        "ir_range": [float(alli.min()), float(alli.max())] if alli.size else None,
        "ok": not problems,
        "problems": problems,
    }
