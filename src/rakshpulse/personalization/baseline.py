"""Individual BP baseline tracker (reference measurements only).

Keeps one user's history of *reference* BP readings (e.g. validated cuff) and summarises them. The four
quantities that the inference output keeps apart are:

1. **population reference**: what the population model predicts (not stored here);
2. **individual historical baseline**: statistics of this user's own reference readings (this class);
3. **current estimate**: the (personalised) model output for the current measurement;
4. **reference category of a single reading**: :func:`reference_category`, a lookup of the 2017
   ACC/AHA adult office-BP categories for *one number*. It is not a diagnosis, and it is never applied
   automatically to the historical average. A diagnosis requires repeated, properly measured readings
   interpreted by a clinician.

Rolling windows (5/10/20 readings by default) are analytical summaries, not clinical thresholds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Reading:
    timestamp: float  # seconds (any epoch, but consistent per user)
    sbp: float
    dbp: float
    source: str = "reference"


@dataclass
class BaselineTracker:
    user_id: str
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    readings: list[Reading] = field(default_factory=list)

    # ------------------------------------------------------------------ updates
    def add(self, timestamp: float, sbp: float, dbp: float, source: str = "reference") -> None:
        if not (np.isfinite(sbp) and np.isfinite(dbp)):
            raise ValueError("reference SBP/DBP must be finite numbers")
        if not (40 <= sbp <= 300 and 20 <= dbp <= 200 and sbp > dbp):
            raise ValueError(f"implausible reference reading {sbp}/{dbp} mmHg")
        if self.readings and timestamp < self.readings[-1].timestamp:
            raise ValueError("readings must be added in time order (no back-dating)")
        self.readings.append(Reading(float(timestamp), float(sbp), float(dbp), source))

    # ------------------------------------------------------------------ summaries
    def _arr(self, key: str, upto: float | None = None) -> np.ndarray:
        return np.array([getattr(r, key) for r in self.readings if upto is None or r.timestamp <= upto], float)

    def stats(self, upto: float | None = None) -> dict:
        """Summary of readings with ``timestamp <= upto`` (all readings if None)."""
        ts = self._arr("timestamp", upto)
        out: dict = {"user_id": self.user_id, "n": int(ts.size)}
        if ts.size == 0:
            return out
        out["last_timestamp"] = float(ts[-1])
        for k in ("sbp", "dbp"):
            v = self._arr(k, upto)
            out[f"{k}_mean"] = float(v.mean())
            out[f"{k}_median"] = float(np.median(v))
            out[f"{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
            for w in self.rolling_windows:
                out[f"{k}_rolling{w}_mean"] = float(v[-w:].mean()) if v.size >= w else float("nan")
            span = ts[-1] - ts[0]
            # trend in mmHg per day; only reported with >= 3 readings spread over time
            out[f"{k}_trend_per_day"] = (
                float(np.polyfit((ts - ts[0]) / 86400.0, v, 1)[0]) if v.size >= 3 and span > 0 else float("nan")
            )
        return out

    def deviation(self, sbp: float, dbp: float, upto: float | None = None) -> dict:
        """Deviation of a value from this user's own historical baseline (median and robust z)."""
        s = self.stats(upto)
        if s["n"] == 0:
            return {"available": False, "reason": "no reference readings"}
        out = {"available": True, "n_reference": s["n"]}
        for k, val in (("sbp", sbp), ("dbp", dbp)):
            diff = val - s[f"{k}_median"]
            sd = s[f"{k}_sd"]
            out[f"{k}_minus_baseline"] = float(diff)
            out[f"{k}_z"] = float(diff / sd) if s["n"] >= 3 and np.isfinite(sd) and sd > 0 else float("nan")
        return out

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "rolling_windows": list(self.rolling_windows),
            "readings": [asdict(r) for r in self.readings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> BaselineTracker:
        t = cls(d["user_id"], tuple(d.get("rolling_windows", (5, 10, 20))))
        t.readings = [Reading(**r) for r in d.get("readings", [])]
        return t

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaselineTracker:
        return cls.from_dict(json.loads(Path(path).read_text()))


ACC_AHA_2017 = "Whelton PK et al. 2017 ACC/AHA high blood pressure guideline (Hypertension 2018;71:e13-e115)"


def reference_category(sbp: float, dbp: float) -> str:
    """2017 ACC/AHA adult category of a single reading. Reference lookup only, not a diagnosis."""
    if sbp >= 140 or dbp >= 90:
        return "stage 2 range"
    if sbp >= 130 or dbp >= 80:
        return "stage 1 range"
    if sbp >= 120:
        return "elevated range"
    return "normal range"
