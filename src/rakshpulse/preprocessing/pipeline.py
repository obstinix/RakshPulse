"""Configurable recording → windows → quality → features pipeline for raw red/IR recordings (RakshPulse 5.0).

Input is a raw recording in the dataset schema (``timestamp, sample_index, red, infrared``), as a DataFrame or
an iterator of chunks, so long recordings are processed without loading them whole. Steps per window:

1. **Gaps.** The recording is split into contiguous segments wherever the sample counter jumps (after
   filling gaps of ≤ ``max_fill_samples``) or the time step exceeds ``max_time_step_s``. Windows never span a gap.
   Samples too short to form a window are reported in the segment summary, never silently dropped.
2. **Irregular sampling.** If timestamps jitter by more than ``jitter_tolerance`` of the nominal period inside a
   segment, the window is linearly resampled onto a uniform grid, and ``resampled_irregular`` is noted.
3. **Saturation / flat line** are judged on the raw counts: % samples at the ADC rails, and raw SD below
   ``flat_sd_counts``.
4. **Preprocessing (existing, unchanged):** DC removal → resampling to 100 Hz → Hampel despiking → band-pass
   (``band``) → orientation → z-score (``features.feature_pipeline.preprocess_channel``). The baseline wander
   and detrending are handled by the band-pass high edge and the linear detrend.
5. **Quality grading** (``quality_score``, 4.0): good / degraded / unreliable / missing → status ``usable`` /
   ``degraded`` / ``unusable`` / ``missing``. Every reason is kept in ``reasons``.
6. **Features:** the existing ``ir_*`` / ``red_*`` per-channel features and cross-channel features (definitions
   unchanged from 4.0), beat-duration-normalised shape features, and the 4.0 contour features
   (``ir_notch_t`` …). Features are computed for every window that has a signal, so downstream code can choose
   its own gate. ``status`` says which windows the default gate accepts.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..evaluation.auricle_study import add_normalised_shape
from ..features.feature_pipeline import extract_features
from ..features.physiological import PHYSIO_FEATURES, physio_features
from . import filtering
from .quality_score import score_window

WINDOW_COLUMNS = [
    "subject_id",
    "session_id",
    "window_id",
    "t_start_s",
    "t_end_s",
    "first_sample_index",
    "n_samples",
    "status",
    "reasons",
    "quality_class",
    "quality_score",
]
STATUS = {"good": "usable", "degraded": "degraded", "unreliable": "unusable", "missing": "missing"}


@dataclass
class PipelineConfig:
    fs: float = 100.0  # nominal sampling rate of the raw recording
    window_s: float = 10.0
    step_s: float = 5.0
    band: tuple[float, float] = filtering.DEFAULT_BAND
    max_time_step_s: float = 0.5  # a counter jump, or a time step longer than this, splits segments
    jitter_tolerance: float = 0.2  # fraction of 1/fs
    adc_max: float = 2**18 - 1
    rail_margin_counts: float = 1.0
    rail_max_pct: float = 1.0
    flat_sd_counts: float = 2.0
    max_fill_samples: int = 2  # counter gaps up to this many missing samples are interpolated (and noted)
    channels: tuple[str, ...] = ("ir", "red")
    contour_features: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _segments(t: np.ndarray, idx: np.ndarray, cfg: PipelineConfig) -> list[tuple[int, int]]:
    """[start, stop) index ranges of contiguous samples."""
    if t.size == 0:
        return []
    # The device counter decides continuity; timestamp jitter alone never splits (it is resampled per window).
    # Only a genuinely large time jump (e.g. a clock reset with a contiguous counter) splits as well.
    br = np.flatnonzero((np.diff(idx) != 1) | (np.diff(t) > cfg.max_time_step_s)) + 1
    edges = np.r_[0, br, t.size]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _fill_short_gaps(data: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Linearly interpolate counter gaps of ≤ ``max_fill_samples`` missing samples; mark them ``filled``."""
    idx = data.sample_index.to_numpy(np.int64)
    jump = np.diff(idx)
    small = np.flatnonzero((jump > 1) & (jump <= cfg.max_fill_samples + 1))
    if cfg.max_fill_samples <= 0 or small.size == 0:
        return data.assign(filled=data.get("filled", False))
    full = np.arange(idx[0], idx[-1] + 1)
    keep = np.ones(full.size, bool)
    big = np.flatnonzero(jump > cfg.max_fill_samples + 1)
    for j in big:  # do not fill long gaps: drop their indices again
        keep[(full > idx[j]) & (full < idx[j + 1])] = False
    full = full[keep]
    out = pd.DataFrame({"sample_index": full})
    for c in ("timestamp", "red", "infrared"):
        out[c] = np.interp(full, idx, data[c].to_numpy(float))
    out["filled"] = ~np.isin(full, idx)
    if "filled" in data:
        out.loc[np.isin(full, idx[data.filled.to_numpy(bool)]), "filled"] = True
    return out


def _iter_raw(raw: pd.DataFrame | Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    if isinstance(raw, pd.DataFrame):
        yield raw
    else:
        yield from raw


def window_features(red: np.ndarray | None, ir: np.ndarray, fs: float, cfg: PipelineConfig) -> dict:
    """Quality + features of one raw window (counts). Returns a flat dict."""
    out: dict = {}
    reasons: list[str] = []
    rails = []
    for name, x in (("ir", ir), ("red", red)):
        if x is None:
            continue
        rail = float(np.mean((x >= cfg.adc_max - cfg.rail_margin_counts) | (x <= cfg.rail_margin_counts)) * 100)
        out[f"{name}_rail_pct"] = rail
        rails.append(rail)
    if not np.isfinite(ir).all() and np.isfinite(ir).mean() < 0.95:
        return {**out, "quality_class": "missing", "quality_score": 0.0, "reasons": "non_finite_samples"}
    if np.nanstd(ir) < cfg.flat_sd_counts:
        return {**out, "quality_class": "missing", "quality_score": 0.0, "reasons": "no_signal_flatline"}
    chans = {c: (ir if c == "ir" else red) for c in cfg.channels if (ir if c == "ir" else red) is not None}
    feats, proc = extract_features(chans, fs, has_dc=True)
    out.update(feats)
    raw100 = (
        filtering.resample(ir - np.median(ir), fs, filtering.TARGET_FS)
        if fs != filtering.TARGET_FS
        else ir - np.median(ir)
    )
    q = score_window(
        proc["ir"].filtered,
        raw100,
        filtering.TARGET_FS,
        flat_sd=0.0,
        rail_pct=max(rails),
        rail_max_pct=cfg.rail_max_pct,
    )
    reasons += q.reasons
    out.update({"quality_class": q.quality_class, "quality_score": q.score})
    if cfg.contour_features:
        pf, _ = physio_features(proc["ir"].filtered, filtering.TARGET_FS)
        out.update({f"ir_{k}": v for k, v in pf.items()})
    out["reasons"] = "|".join(reasons)
    return out


def process_recording(
    raw: pd.DataFrame | Iterable[pd.DataFrame],
    cfg: PipelineConfig | None = None,
    subject_id: str = "",
    session_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (windows table, segment summary). Raw input can be a DataFrame or an iterator of chunks."""
    cfg = cfg or PipelineConfig()
    fs, win, step = cfg.fs, int(round(cfg.window_s * cfg.fs)), int(round(cfg.step_s * cfg.fs))
    rows, segs = [], []
    buf: pd.DataFrame | None = None  # unconsumed samples of the open segment (starts at the next window start)
    cur: dict | None = None  # running summary of the open segment

    def emit(seg: pd.DataFrame) -> int:
        """Emit every complete window of ``seg``; return the number of samples consumed (next window start)."""
        n_win = 0 if len(seg) < win else (len(seg) - win) // step + 1
        for k in range(n_win):
            w = seg.iloc[k * step : k * step + win]
            tt = w.timestamp.to_numpy(float)
            ir = w.infrared.to_numpy(float)
            red = w.red.to_numpy(float) if "red" in cfg.channels else None
            notes = []
            dt = np.diff(tt)
            if dt.size and np.max(np.abs(dt - 1 / fs)) > cfg.jitter_tolerance / fs:
                grid = tt[0] + np.arange(win) / fs
                ir = np.interp(grid, tt, ir)
                red = np.interp(grid, tt, red) if red is not None else None
                notes.append("resampled_irregular")
            n_filled = int(w.filled.sum()) if "filled" in w else 0
            if n_filled:
                notes.append(f"filled_{n_filled}_missing_samples")
            f = window_features(red, ir, fs, cfg)
            reasons = "|".join(x for x in (f.pop("reasons", ""), *notes) if x)
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "window_id": f"{subject_id}_{session_id}_w{len(rows):06d}",
                    "t_start_s": float(tt[0]),
                    "t_end_s": float(tt[-1] + 1 / fs),
                    "first_sample_index": int(w.sample_index.iloc[0]),
                    "n_samples": int(len(w)),
                    "status": STATUS[f["quality_class"]],
                    "reasons": reasons,
                    **f,
                }
            )
        cur["n_windows"] += n_win
        if n_win:
            cur["covered_to"] = float(seg.timestamp.iloc[(n_win - 1) * step + win - 1])
        return n_win * step

    def close_segment(tail: pd.DataFrame | None) -> None:
        t_end = cur["t_end_s"]
        covered = cur.get("covered_to")
        left = cur["n_samples"] if covered is None else int(0 if tail is None else (tail.timestamp > covered).sum())
        segs.append(
            {
                "t_start_s": cur["t_start_s"],
                "t_end_s": t_end,
                "n_samples": cur["n_samples"],
                "n_windows": cur["n_windows"],
                "samples_not_in_any_window": left,
            }
        )

    for chunk in _iter_raw(raw):
        chunk = chunk[["timestamp", "sample_index", "red", "infrared"]].assign(filled=False)
        n_buf = 0 if buf is None else len(buf)
        data = chunk if buf is None else pd.concat([buf, chunk], ignore_index=True)
        data = _fill_short_gaps(data, cfg)
        parts = _segments(data.timestamp.to_numpy(float), data.sample_index.to_numpy(np.int64), cfg)
        for j, (a, b) in enumerate(parts):
            seg = data.iloc[a:b].reset_index(drop=True)
            if j == 0 and buf is not None:  # continuation of the open segment
                cur["n_samples"] += len(seg) - n_buf
            else:
                if cur is not None:
                    close_segment(buf)
                cur = {"t_start_s": float(seg.timestamp.iloc[0]), "n_samples": len(seg), "n_windows": 0}
            cur["t_end_s"] = float(seg.timestamp.iloc[-1])
            consumed = emit(seg)
            buf = seg.iloc[consumed:].reset_index(drop=True)
            if j < len(parts) - 1:  # a gap follows: the segment is complete
                close_segment(buf)
                cur, buf = None, None
    if cur is not None:
        close_segment(buf)
    w = pd.DataFrame(rows, columns=None if rows else WINDOW_COLUMNS)
    if len(w) and "ir_beat_T" in w:
        add_normalised_shape(w, "ir")
    return w, pd.DataFrame(segs)


CONTOUR_COLUMNS = [f"ir_{c}" for c in PHYSIO_FEATURES]
