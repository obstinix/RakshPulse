"""Personalisation engine: per-user calibration of population BP predictions.

The population model is never changed. For each user the engine keeps a separate, versioned
:class:`CalibrationState` that maps the population prediction to a personal estimate:

    personal = a + b * population_prediction      (independently for SBP and DBP)

Methods (``method``):

* ``offset``: b = 1, a = recency-weighted mean of (reference − prediction). Works from one reading.
* ``linear``: regularised slope. b = 1 + β, where β is ridge-penalised towards 0 (λ in units of
  prediction variance), so with few or clustered readings it falls back to ``offset``. Needs
  ``min_points_linear`` readings spanning at least ``min_pred_range`` mmHg of predictions.

Adaptation: :meth:`PersonalizationEngine.update` adds a new *reference pair* (population prediction made from
PPG at that time, and the trusted reference BP at the same time). Before refitting it records the
**prequential residual** (the error the current calibration would have made), which feeds the drift monitor.
Refits use exponential recency weights (``half_life_s``), so old calibrations fade. A refit that moves the
offset by more than ``max_offset_jump`` mmHg or pushes the slope outside ``slope_bounds`` is flagged
*unstable*. It is kept but marked, and the previous version remains available via :meth:`rollback`.

Recalibration is **recommended** (never automatic) when:
* there is no calibration;
* calibration age exceeds ``max_age_s``. The default is 12 h, because in the Auricle cross-day study a
  day-1 calibration did not hold on day 2;
* recent prequential residuals exceed ``residual_alarm`` mmHg in mean absolute value;
* the last update was unstable.
PPG alone is never used to decide that a calibration has failed. That requires reference readings.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EngineConfig:
    method: str = "offset"  # "offset" | "linear"
    half_life_s: float = 6 * 3600.0
    max_age_s: float = 12 * 3600.0
    min_points_linear: int = 4
    min_pred_range: float = 10.0
    ridge_lambda: float = 1.0
    slope_bounds: tuple[float, float] = (0.25, 2.0)
    max_offset_jump: float = 25.0
    residual_alarm: float = 12.0
    residual_window: int = 3
    max_points: int = 50


@dataclass
class TargetFit:
    a: float = 0.0
    b: float = 1.0
    residual_sd: float = float("nan")


@dataclass
class CalibrationState:
    user_id: str
    population_model: str = ""
    version: int = 0
    created_at: float | None = None  # timestamp of the most recent reference used
    first_calibrated_at: float | None = None
    n_points: int = 0
    method_used: dict = field(default_factory=lambda: {"sbp": "none", "dbp": "none"})
    sbp: TargetFit = field(default_factory=TargetFit)
    dbp: TargetFit = field(default_factory=TargetFit)
    unstable: bool = False
    unstable_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationState:
        d = dict(d)
        d["sbp"], d["dbp"] = TargetFit(**d["sbp"]), TargetFit(**d["dbp"])
        return cls(**d)


@dataclass
class RefPoint:
    timestamp: float
    pred_sbp: float
    pred_dbp: float
    ref_sbp: float
    ref_dbp: float


class PersonalizationEngine:
    """Per-user calibration with history, drift monitoring and rollback. One instance per user."""

    def __init__(self, user_id: str, config: EngineConfig | None = None, population_model: str = ""):
        self.config = config or EngineConfig()
        self.state = CalibrationState(user_id=user_id, population_model=population_model)
        self.points: list[RefPoint] = []
        self.history: list[dict] = []  # previous states (for rollback)
        self.prequential: list[dict] = []  # residuals of the *pre-update* calibration on each new reference

    # ----------------------------------------------------------------------------------- fitting
    def _weights(self, ts: np.ndarray, now: float) -> np.ndarray:
        return 0.5 ** ((now - ts) / self.config.half_life_s)

    def _fit_target(self, pred: np.ndarray, ref: np.ndarray, w: np.ndarray) -> tuple[TargetFit, str]:
        c = self.config
        r = ref - pred
        wn = w / w.sum()
        a_off = float(np.sum(wn * r))
        use_linear = c.method == "linear" and pred.size >= c.min_points_linear and np.ptp(pred) >= c.min_pred_range
        if not use_linear:
            res = r - a_off
            sd = float(np.sqrt(np.sum(wn * res**2) * pred.size / max(pred.size - 1, 1))) if pred.size > 1 else np.nan
            return TargetFit(a_off, 1.0, sd), "offset"
        pm = float(np.sum(wn * pred))
        xc = pred - pm
        var = float(np.sum(wn * xc**2))
        # residual slope shrunk towards 0 (i.e. personal slope towards 1); lambda is relative to var(pred)
        beta = float(np.sum(wn * xc * (r - a_off)) / (var * (1.0 + c.ridge_lambda)))
        b = 1.0 + beta
        a = (pm + a_off) - b * pm  # keeps the weighted-mean calibration point: a + b*pm = pm + a_off
        res = ref - (a + b * pred)
        sd = float(np.sqrt(np.sum(wn * res**2) * pred.size / max(pred.size - 2, 1)))
        return TargetFit(a, b, sd), "linear"

    def _refit(self, now: float) -> CalibrationState:
        pts = self.points[-self.config.max_points :]
        ts = np.array([p.timestamp for p in pts])
        w = self._weights(ts, now)
        new = copy.deepcopy(self.state)
        for k in ("sbp", "dbp"):
            pred = np.array([getattr(p, f"pred_{k}") for p in pts])
            ref = np.array([getattr(p, f"ref_{k}") for p in pts])
            fit, used = self._fit_target(pred, ref, w)
            setattr(new, k, fit)
            new.method_used[k] = used
        new.version += 1
        new.created_at = now
        new.first_calibrated_at = new.first_calibrated_at if new.first_calibrated_at is not None else now
        new.n_points = len(self.points)
        return new

    def _check_stability(self, old: CalibrationState, new: CalibrationState) -> tuple[bool, str]:
        c = self.config
        reasons = []
        for k in ("sbp", "dbp"):
            o, n = getattr(old, k), getattr(new, k)
            if old.n_points and abs((n.a + n.b * 120) - (o.a + o.b * 120)) > c.max_offset_jump:
                reasons.append(f"{k} calibration moved > {c.max_offset_jump} mmHg")
            if not c.slope_bounds[0] <= n.b <= c.slope_bounds[1]:
                reasons.append(f"{k} slope {n.b:.2f} outside {c.slope_bounds}")
        return (not reasons), "; ".join(reasons)

    # ----------------------------------------------------------------------------------- public
    def calibrate(self, points: list[RefPoint]) -> CalibrationState:
        """Initial calibration from one or more reference pairs (replaces any previous points)."""
        if not points:
            raise ValueError("calibration needs at least one reference measurement")
        self.points = []
        for p in sorted(points, key=lambda q: q.timestamp):
            self._validate(p)
            self.points.append(p)
        self._commit(self._refit(self.points[-1].timestamp), stable=True, reason="")
        return self.state

    def update(self, point: RefPoint) -> dict:
        """Adaptive update with a new trusted reference pair. Returns the prequential residual record."""
        self._validate(point)
        if self.points and point.timestamp < self.points[-1].timestamp:
            raise ValueError("reference points must arrive in time order (prevents future-data leakage)")
        rec = {"timestamp": point.timestamp}
        if self.state.n_points:
            est = self.apply(point.pred_sbp, point.pred_dbp, point.timestamp)
            rec.update(residual_sbp=point.ref_sbp - est["sbp"], residual_dbp=point.ref_dbp - est["dbp"])
            self.prequential.append(rec)
        self.points.append(point)
        new = self._refit(point.timestamp)
        ok, why = self._check_stability(self.state, new) if self.state.n_points else (True, "")
        self._commit(new, ok, why)
        rec["stable"], rec["version"] = ok, self.state.version
        return rec

    def _commit(self, new: CalibrationState, stable: bool, reason: str) -> None:
        if self.state.n_points:
            self.history.append({"state": self.state.to_dict(), "points": [asdict(p) for p in self.points[:-1]]})
        new.unstable, new.unstable_reason = (not stable), reason
        self.state = new

    def rollback(self, steps: int = 1) -> CalibrationState:
        """Restore the calibration from ``steps`` updates ago (points are restored too)."""
        if steps < 1 or steps > len(self.history):
            raise ValueError(f"cannot roll back {steps} step(s); {len(self.history)} available")
        snap = self.history[-steps]
        self.history = self.history[:-steps]
        self.state = CalibrationState.from_dict(snap["state"])
        self.points = [RefPoint(**p) for p in snap["points"]]
        return self.state

    def apply(self, pred_sbp: float, pred_dbp: float, timestamp: float | None = None) -> dict:
        """Personalised estimate for a population prediction made at ``timestamp``."""
        s = self.state
        if timestamp is not None and s.created_at is not None and timestamp < s.created_at - 1e-9:
            raise ValueError("prediction time precedes the calibration in use (would use future reference data)")
        if not s.n_points:
            return {"sbp": float(pred_sbp), "dbp": float(pred_dbp), "calibrated": False}
        return {
            "sbp": float(s.sbp.a + s.sbp.b * pred_sbp),
            "dbp": float(s.dbp.a + s.dbp.b * pred_dbp),
            "calibrated": True,
            "version": s.version,
        }

    def status(self, now: float) -> dict:
        """Calibration status and recalibration recommendation (based on reference evidence only)."""
        c, s = self.config, self.state
        reasons = []
        if not s.n_points:
            return {"calibrated": False, "recalibration_recommended": True, "reasons": ["no calibration"]}
        age = now - s.created_at
        if age > c.max_age_s:
            reasons.append(f"calibration age {age / 3600:.1f} h > {c.max_age_s / 3600:.0f} h")
        recent = self.prequential[-c.residual_window :]
        if recent:
            mae = float(np.mean([abs(r["residual_sbp"]) for r in recent]))
            if len(recent) >= min(2, c.residual_window) and mae > c.residual_alarm:
                reasons.append(f"recent SBP residual {mae:.1f} mmHg > {c.residual_alarm}")
        if s.unstable:
            reasons.append(f"last update unstable: {s.unstable_reason}")
        return {
            "calibrated": True,
            "version": s.version,
            "n_reference_points": s.n_points,
            "age_s": float(age),
            "method": dict(s.method_used),
            "residual_sd_sbp": s.sbp.residual_sd,
            "residual_sd_dbp": s.dbp.residual_sd,
            "unstable": s.unstable,
            "recalibration_recommended": bool(reasons),
            "reasons": reasons,
        }

    @staticmethod
    def _validate(p: RefPoint) -> None:
        vals = (p.timestamp, p.pred_sbp, p.pred_dbp, p.ref_sbp, p.ref_dbp)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("reference point contains non-finite values")
        if not (40 <= p.ref_sbp <= 300 and 20 <= p.ref_dbp <= 200 and p.ref_sbp > p.ref_dbp):
            raise ValueError(f"implausible reference {p.ref_sbp}/{p.ref_dbp} mmHg")

    # ----------------------------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        cfg = asdict(self.config)
        cfg["slope_bounds"] = list(cfg["slope_bounds"])
        return {
            "config": cfg,
            "state": self.state.to_dict(),
            "points": [asdict(p) for p in self.points],
            "history": self.history,
            "prequential": self.prequential,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PersonalizationEngine:
        cfg = dict(d["config"])
        cfg["slope_bounds"] = tuple(cfg["slope_bounds"])
        e = cls(d["state"]["user_id"], EngineConfig(**cfg))
        e.state = CalibrationState.from_dict(d["state"])
        e.points = [RefPoint(**p) for p in d["points"]]
        e.history, e.prequential = d["history"], d["prequential"]
        return e

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> PersonalizationEngine:
        return cls.from_dict(json.loads(Path(path).read_text()))
