"""Unsupervised domain adaptation for feature models (RakshPulse 5.0).

Both transforms use **unlabelled** target features only, so they are allowed at test time:

* :class:`SessionStandardizer`: z-scores every feature within each subject × session (session-level feature
  normalisation). It removes per-session offsets and scales. 4.0 (W1) found that this can also remove genuine BP
  information, so it is an experiment, not a default.
* :class:`CORAL`: correlation alignment (Sun et al., AAAI 2016). It whitens target-domain features with their own
  covariance and re-colours them with the source covariance, aligning second-order statistics.

The fit/transform signatures follow scikit-learn; ``groups`` carries the subject × session key where needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SessionStandardizer:
    def __init__(self, min_windows: int = 5):
        self.min_windows = min_windows

    def fit(self, X, y=None, groups=None):
        return self

    def transform(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        out = np.full_like(X, np.nan)
        for g in pd.unique(np.asarray(groups)):
            m = np.asarray(groups) == g
            xg = X[m]
            if m.sum() < self.min_windows:
                out[m] = 0.0
                continue
            mu = np.nanmean(xg, 0)
            sd = np.nanstd(xg, 0)
            out[m] = (xg - mu) / np.where(sd > 1e-12, sd, 1.0)
        return out


def _cov(X: np.ndarray, eps: float) -> np.ndarray:
    Xc = X - np.nanmean(X, 0)
    Xc = np.nan_to_num(Xc)
    return Xc.T @ Xc / max(len(X) - 1, 1) + eps * np.eye(X.shape[1])


def _msqrt(C: np.ndarray, inv: bool = False) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


class CORAL:
    def __init__(self, eps: float = 1.0):
        self.eps = eps

    def fit(self, X_source: np.ndarray, y=None):
        Xs = np.asarray(X_source, float)
        self.mu_s = np.nanmean(Xs, 0)
        self.Cs_half = _msqrt(_cov(Xs, self.eps))
        return self

    def transform(self, X_target: np.ndarray) -> np.ndarray:
        """Align ``X_target`` (one target domain, unlabelled) to the source statistics."""
        Xt = np.asarray(X_target, float)
        mu_t = np.nanmean(Xt, 0)
        A = _msqrt(_cov(Xt, self.eps), inv=True) @ self.Cs_half
        return np.nan_to_num(Xt - mu_t) @ A + self.mu_s
