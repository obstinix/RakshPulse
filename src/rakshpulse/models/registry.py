"""Train, evaluate, version, save and load feature models on RakshPulse window tables (5.0).

``train_model(frame, cfg)``:

1. Keeps windows whose ``status`` is in ``cfg.statuses`` and that have a reference for every target
   (``y_sbp`` / ``y_dbp`` columns, typically from ``datastore.association``).
2. Runs leakage-checked cross-validation (``evaluation.splits``; subject, session or cross-day) and collects
   out-of-fold predictions and per-fold metrics. Hyper-parameters come from ``cfg.hyperparams`` or, if
   ``cfg.tune`` is set, from an inner ``GroupKFold`` grid search over **training subjects only**.
3. Fits the final model on all eligible windows. It stores the cross-subject conformal 90 % half-width (the
   90th percentile of out-of-fold |error|) per target, so inference can report an honest interval, and records
   the configuration, a SHA-256 of the training table, the git commit and the CV metrics.

Bundles are ``inference.model_loader.ModelBundle`` files (joblib; load only files you created). Versions are
``<name>-<UTC>-<data sha8>``, and every save appends a row to ``<models_dir>/registry.csv``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from ..evaluation.splits import SplitSpec, make_splits
from ..inference.model_loader import ModelBundle, load_bundle
from .model_factory import make_model

TARGETS = ("sbp", "dbp")
EXCLUDE = {
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
    "y_sbp",
    "y_dbp",
    "ref_sbp",
    "ref_dbp",
    "ref_timestamp",
    "ref_offset_s",
    "ir_rail_pct",
    "red_rail_pct",
}


@dataclass
class TrainConfig:
    name: str = "rakshpulse"
    model: str = "ridge"
    features: list[str] | str = "ir_core"  # explicit list, "ir_core" (HR + shape) or "all"
    split: SplitSpec = field(default_factory=SplitSpec)
    statuses: tuple[str, ...] = ("usable", "degraded")
    hyperparams: dict = field(default_factory=dict)
    tune: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["split"] = asdict(self.split)
        return d


def resolve_features(frame: pd.DataFrame, spec: list[str] | str) -> list[str]:
    if isinstance(spec, list):
        missing = [c for c in spec if c not in frame]
        if missing:
            raise KeyError(f"features not in the table: {missing[:5]}")
        return spec
    if spec == "ir_core":
        from ..evaluation.auricle_study import SHAPE_FEATURES

        cols = ["ir_hr"] + [f"ir_{c}" for c in SHAPE_FEATURES]
        return [c for c in cols if c in frame]
    if spec == "all":
        return [c for c in frame.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(frame[c])]
    raise KeyError(f"unknown feature set {spec!r}")


def _metrics(y: np.ndarray, p: np.ndarray, subj: np.ndarray) -> dict:
    e = p - y
    per = pd.Series(np.abs(e)).groupby(subj).mean()
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mean_error": float(e.mean()),
        "sd_error": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
        "pearson_r": r,
        "subject_mae": float(per.mean()),
        "n_subjects": int(per.size),
        "n_windows": int(e.size),
    }


def _fit(cfg: TrainConfig, X, y, groups):
    est, grid = make_model(cfg.model, cfg.seed)
    if cfg.hyperparams:
        est.set_params(**{k if k.startswith("model__") else f"model__{k}": v for k, v in cfg.hyperparams.items()})
    elif cfg.tune and grid and len(np.unique(groups)) >= 3:
        gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="neg_mean_absolute_error")
        return gs.fit(X, y, groups=groups).best_estimator_
    return est.fit(X, y)


def train_model(frame: pd.DataFrame, cfg: TrainConfig) -> tuple[ModelBundle, pd.DataFrame, pd.DataFrame]:
    """Return (final bundle, per-fold metrics, out-of-fold predictions)."""
    d = frame[frame.status.isin(cfg.statuses)].dropna(subset=[f"y_{t}" for t in TARGETS]).reset_index(drop=True)
    if d.empty:
        raise ValueError("no eligible windows (check statuses and reference association)")
    feats = resolve_features(d, cfg.features)
    X = d[feats].to_numpy(float)
    groups = d.subject_id.astype(str).to_numpy()
    folds = make_splits(d, cfg.split)
    oof = d[["subject_id", "session_id", "window_id", "t_start_s"]].copy()
    rows = []
    for t in TARGETS:
        y = d[f"y_{t}"].to_numpy(float)
        oof[f"y_{t}"] = y
        oof[f"p_{t}"] = np.nan
        for k, (tr, te) in enumerate(folds):
            m = _fit(cfg, X[tr], y[tr], groups[tr])
            p = np.ravel(m.predict(X[te]))
            oof.loc[te, f"p_{t}"] = p
            rows.append({"target": t, "fold": k, **_metrics(y[te], p, groups[te])})
    scored = oof.dropna(subset=[f"p_{t}" for t in TARGETS])
    validation = {
        "split": asdict(cfg.split),
        "cv": pd.DataFrame(rows).groupby("target").mean(numeric_only=True).drop(columns="fold").to_dict("index"),
    }
    for t in TARGETS:
        err = np.abs(scored[f"p_{t}"] - scored[f"y_{t}"])
        validation[f"conformal_90_halfwidth_{t}"] = float(np.quantile(err, 0.9)) if len(err) else float("nan")
        validation[f"oof_{t}"] = _metrics(
            scored[f"y_{t}"].to_numpy(), scored[f"p_{t}"].to_numpy(), scored.subject_id.astype(str).to_numpy()
        )
    models = {t: _fit(cfg, X, d[f"y_{t}"].to_numpy(float), groups) for t in TARGETS}
    data_sha = hashlib.sha256(
        pd.util.hash_pandas_object(d[feats + ["y_sbp", "y_dbp"]], index=False).values.tobytes()
    ).hexdigest()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    channels = tuple(sorted({c.split("_")[0] for c in feats if c.split("_")[0] in ("ir", "red")})) or ("ir",)
    validation.update(
        {
            "config": cfg.to_dict(),
            "training_data_sha256": data_sha,
            "git_commit": _git(),
            "n_subjects": int(d.subject_id.nunique()),
            "n_windows": int(len(d)),
        }
    )
    bundle = ModelBundle(
        models["sbp"],
        models["dbp"],
        feats,
        channels,
        False,
        validation,
        version=f"{cfg.name}-{stamp}-{data_sha[:8]}",
        trained_on="rakshpulse window table",
    )
    return bundle, pd.DataFrame(rows), oof


def _git() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except OSError:
        return "unavailable"


def save_model(bundle: ModelBundle, models_dir: str | Path) -> Path:
    """Save ``<models_dir>/<version>.joblib`` (never overwriting) and append to ``registry.csv``."""
    models_dir = Path(models_dir)
    path = models_dir / f"{bundle.version}.joblib"
    if path.exists():
        raise FileExistsError(path)
    bundle.save(path)
    v = bundle.validation
    row = {
        "version": bundle.version,
        "path": path.name,
        "created_utc": bundle.created_utc,
        "model": v.get("config", {}).get("model"),
        "n_features": len(bundle.features),
        "n_subjects": v.get("n_subjects"),
        "split": v.get("split", {}).get("kind"),
        "cv_mae_sbp": v.get("cv", {}).get("sbp", {}).get("mae"),
        "cv_mae_dbp": v.get("cv", {}).get("dbp", {}).get("mae"),
        "conformal_90_sbp": v.get("conformal_90_halfwidth_sbp"),
        "training_data_sha256": v.get("training_data_sha256"),
        "git_commit": v.get("git_commit"),
    }
    reg = models_dir / "registry.csv"
    old = pd.read_csv(reg) if reg.exists() else None
    (pd.DataFrame([row]) if old is None else pd.concat([old, pd.DataFrame([row])], ignore_index=True)).to_csv(
        reg, index=False
    )
    return path


def load_model(models_dir: str | Path, version: str | None = None) -> ModelBundle:
    """Load a version from the registry (latest if ``version`` is None)."""
    reg = pd.read_csv(Path(models_dir) / "registry.csv")
    row = reg.iloc[-1] if version is None else reg[reg.version == version].iloc[-1]
    return load_bundle(Path(models_dir) / row.path)
