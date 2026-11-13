"""Serialisation of trained BP models together with the metadata inference needs.

A model bundle is a joblib file with:
  * ``sbp`` / ``dbp``: fitted scikit-learn pipelines,
  * ``features``: ordered feature names expected by the pipelines,
  * ``channels``: optical channels the model requires (e.g. ("ir", "red")),
  * ``uses_demographics``: whether age/sex/height/weight inputs are required,
  * ``validation``: subject-wise CV error statistics (used to report uncertainty),
  * ``version``, ``trained_on``, ``created_utc``,
  * ``contract`` (5.1): feature schema, required inputs, supported sensors and modes, sampling rates,
    preprocessing, uncertainty capability and limitations (``inference/contract.py``). Bundles without one get a
    conservative demonstration-only legacy contract when loaded.

Bundles are only loaded from paths the caller supplies. joblib/pickle can execute code,
so only load bundles you created yourself.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib

REQUIRED_KEYS = {"sbp", "dbp", "features", "channels", "uses_demographics", "validation", "version"}


@dataclass
class ModelBundle:
    sbp: object
    dbp: object
    features: list[str]
    channels: tuple[str, ...]
    uses_demographics: bool
    validation: dict
    version: str
    trained_on: str = ""
    contract: dict = field(default_factory=dict)  # see inference/contract.py; {} for bundles saved before 5.1
    created_utc: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path, compress=3)
        return path


def load_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model bundle not found: {path}")
    d = joblib.load(path)
    missing = REQUIRED_KEYS - set(d)
    if missing:
        raise ValueError(f"model bundle {path} missing keys: {sorted(missing)}")
    d["channels"] = tuple(d["channels"])
    return ModelBundle(**d)
