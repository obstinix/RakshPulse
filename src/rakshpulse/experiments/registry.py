"""Experiment provenance registry.

Each run writes ``results/runs/<run_id>/`` with:
  ``config.csv``   key/value: dataset files + SHA-256, feature and model configuration, hyper-parameters, seed,
                   git commit (+ dirty flag), software versions, start time and duration;
  ``splits.csv``   subject → fold assignment;
  ``metrics.csv``  all metric rows reported by the run;
and appends a summary row to ``results/runs/runs.csv``. Earlier runs are never overwritten. A run ID is
``<name>-<UTC timestamp>``, so RakshPulse 2.0 outputs and their provenance stay alongside newer results.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "results" / "runs"
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch", "wfdb", "vitaldb")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def git_state() -> dict[str, str]:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"

    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": str(bool(run("status", "--porcelain")))}


@dataclass
class ExperimentRun:
    name: str
    datasets: list[Path]
    config: dict
    seed: int = 0
    metrics: list[dict] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    t0: float | None = None  # time.time() when the experiment really started (if the run is created at the end)

    def __post_init__(self):
        self._t0 = self.t0 if self.t0 is not None else time.time()
        self.started = dt.datetime.fromtimestamp(self._t0, dt.timezone.utc)
        self.run_id = f"{self.name}-{self.started.strftime('%Y%m%dT%H%M%SZ')}"

    def add_metrics(self, rows: list[dict] | pd.DataFrame) -> None:
        self.metrics.extend(rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows)

    def finish(self) -> Path:
        out = RUNS / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "run_id": self.run_id,
            "name": self.name,
            "seed": self.seed,
            "started_utc": self.started.isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._t0, 1),
            **git_state(),
            **software_versions(),
        }
        for p in self.datasets:
            p = Path(p)
            rel = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
            cfg[f"dataset:{rel}"] = sha256(p)
        cfg.update({f"config:{k}": v for k, v in self.config.items()})
        pd.DataFrame(list(cfg.items()), columns=["key", "value"]).to_csv(out / "config.csv", index=False)
        if self.splits:
            pd.DataFrame(sorted(self.splits.items()), columns=["subject_id", "fold"]).to_csv(
                out / "splits.csv", index=False
            )
        pd.DataFrame(self.metrics).to_csv(out / "metrics.csv", index=False, float_format="%.5g")
        row = pd.DataFrame(
            [
                {
                    "run_id": self.run_id,
                    "name": self.name,
                    "started_utc": cfg["started_utc"],
                    "duration_s": cfg["duration_s"],
                    "git_commit": cfg["git_commit"][:12],
                    "git_dirty": cfg["git_dirty"],
                    "seed": self.seed,
                    "n_metric_rows": len(self.metrics),
                    "n_subjects_in_splits": len(self.splits),
                    "output_dir": out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out),
                }
            ]
        )
        idx = RUNS / "runs.csv"
        row.to_csv(idx, mode="a", header=not idx.exists(), index=False)
        return out
