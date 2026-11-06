"""Per-window normalisation.

All normalisation here is computed from the window itself (no dataset statistics),
so it cannot leak information between train and test subjects. Dataset-level
feature scaling is done inside scikit-learn pipelines fitted on training folds only.
"""

from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.maximum(sd, eps)


def minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, eps)


def orient_ppg(x: np.ndarray, fs: float) -> np.ndarray:
    """Make systolic peaks point upwards.

    Raw optical readings (e.g. MAX30102 ADC counts) fall when blood volume rises, so the
    pulse appears inverted. A PPG pulse rises quickly and decays slowly, which makes the
    first derivative positively skewed for a correctly oriented signal. If the skew is
    negative, the signal is flipped.
    """
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    if d.size < 3 or d.std() == 0:
        return x
    skew = np.mean(((d - d.mean()) / d.std()) ** 3)
    return -x if skew < 0 else x
