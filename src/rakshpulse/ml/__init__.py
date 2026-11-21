"""The complete RakshPulse ML workflow: dataset audit and preparation, model training with subject-wise
evaluation, reporting and export.

``scripts/run_complete_pipeline.py`` and ``notebooks/RakshPulse_Complete_ML.ipynb`` both call these functions, so
the notebook reproduces exactly what the pipeline does. Features come from the production feature extractor
(``features/feature_pipeline.py``), and the exported bundle runs through the production inference engine.
"""

from .dataset import (
    DatasetPaths,
    audit_datasets,
    build_ear_dataset,
    candidate_features,
    load_ml_dataset,
    make_smoke_dataset,
)
from .training import PipelineConfig, export_bundle, run_training

__all__ = [
    "DatasetPaths",
    "PipelineConfig",
    "audit_datasets",
    "build_ear_dataset",
    "candidate_features",
    "export_bundle",
    "load_ml_dataset",
    "make_smoke_dataset",
    "run_training",
]
