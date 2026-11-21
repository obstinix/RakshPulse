"""The full train-and-evaluate sequence, shared by ``scripts/run_complete_pipeline.py`` and the master notebook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import report as R
from .dataset import candidate_features
from .evaluation import calibration_study, nested_cv
from .training import PipelineConfig, run_training


def train_and_evaluate(d: pd.DataFrame, cfg: PipelineConfig | None = None, outer_folds: int = 5, log=print) -> dict:
    """Held-out-test protocol (``run_training``), then nested subject-wise CV over all subjects and the per-person
    calibration study. Returns the ``run_training`` result with ``nested`` and ``calibration`` added."""
    cfg = cfg or PipelineConfig()
    result = run_training(d, cfg=cfg, log=log)
    log("nested subject-wise cross-validation over all subjects")
    result["nested"] = nested_cv(d, candidate_features(), cfg, outer_folds=outer_folds, log=log)
    log("per-person calibration study (held-out test subjects)")
    result["calibration"] = calibration_study(
        result["data"], result["train_subjects"], result["test_subjects"], result["models"], result["features"], cfg
    )
    return result


def make_figures(result: dict, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "dataset": R.plot_dataset(result["data"], out / "dataset.png"),
        "comparison": R.plot_model_comparison(result["comparison"], out / "model_comparison.png"),
        "scatter": R.plot_actual_vs_predicted(
            result["test_predictions"], result["selection"], out / "actual_vs_predicted.png"
        ),
        "residuals": R.plot_residuals(result["test_predictions"], result["selection"], out / "residuals.png"),
    }
    if "nested" in result:
        figs["subjects"] = R.plot_subject_wise(result["nested"], out / "subject_wise.png")
    if "calibration" in result:
        figs["calibration"] = R.plot_calibration(result["calibration"], out / "calibration.png")
    return figs
