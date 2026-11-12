"""Leakage-checked data splits (RakshPulse 5.0).

* ``subject``: K folds over subjects; a subject is entirely in train or in test.
* ``session``: session-independent. Train on ``train_sessions`` of the training subjects, test on the other
  sessions of the test subjects. Optional ``same_subjects=True`` tests unseen sessions of *seen* subjects (the
  personalised cross-session setting). Otherwise test subjects are disjoint too.
* ``cross_day``: strict. Subject-disjoint, the model is trained on day-1 sessions only, and tested only on later
  sessions.

Every split is checked with :func:`check_no_leakage` before it is returned. A violation raises, so an
experiment cannot silently train on its test subjects or sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitSpec:
    kind: str = "subject"  # subject | session | cross_day
    n_folds: int = 5
    seed: int = 0
    train_sessions: tuple = ("1",)
    same_subjects: bool = False


def subject_folds(subjects, n_folds: int = 5, seed: int = 0) -> dict:
    """Seeded subject → fold assignment (balanced sizes)."""
    s = np.array(sorted({str(x) for x in subjects}))
    if s.size < n_folds:
        raise ValueError(f"need at least {n_folds} subjects for {n_folds} folds, got {s.size}")
    perm = np.random.default_rng(seed).permutation(s.size)
    return {s[i]: int(k % n_folds) for k, i in enumerate(perm)}


def check_no_leakage(frame: pd.DataFrame, tr: np.ndarray, te: np.ndarray, spec: SplitSpec) -> None:
    a, b = frame.iloc[tr], frame.iloc[te]
    if len(set(tr) & set(te)):
        raise AssertionError("leakage: a row is in both train and test")
    ta = set(a.subject_id.astype(str))
    tb = set(b.subject_id.astype(str))
    if spec.kind in ("subject", "cross_day") or (spec.kind == "session" and not spec.same_subjects):
        if ta & tb:
            raise AssertionError(f"leakage: subjects in train and test: {sorted(ta & tb)[:5]}")
    if spec.kind in ("session", "cross_day"):
        pairs_a = set(zip(a.subject_id.astype(str), a.session_id.astype(str), strict=True))
        pairs_b = set(zip(b.subject_id.astype(str), b.session_id.astype(str), strict=True))
        if pairs_a & pairs_b:
            raise AssertionError("leakage: a subject-session is in both train and test")
        if not set(a.session_id.astype(str)) <= {str(x) for x in spec.train_sessions}:
            raise AssertionError("leakage: training data from a non-training session")


def make_splits(frame: pd.DataFrame, spec: SplitSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (train positional indices, test positional indices), one per fold, leakage-checked."""
    sub = frame.subject_id.astype(str).to_numpy()
    ses = frame.session_id.astype(str).to_numpy()
    train_ses = np.isin(ses, [str(x) for x in spec.train_sessions])
    out = []
    if spec.kind == "session" and spec.same_subjects:
        tr, te = np.flatnonzero(train_ses), np.flatnonzero(~train_ses)
        check_no_leakage(frame, tr, te, spec)
        return [(tr, te)]
    folds = subject_folds(sub, spec.n_folds, spec.seed)
    f = np.array([folds[s] for s in sub])
    for k in range(spec.n_folds):
        if spec.kind == "subject":
            tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        elif spec.kind in ("session", "cross_day"):
            tr = np.flatnonzero((f != k) & train_ses)
            te = np.flatnonzero((f == k) & ~train_ses)
        else:
            raise ValueError(f"unknown split kind {spec.kind!r}")
        if len(te) == 0 or len(tr) == 0:
            continue
        check_no_leakage(frame, tr, te, spec)
        out.append((tr, te))
    if not out:
        raise ValueError("no usable folds (check sessions / subjects)")
    return out
