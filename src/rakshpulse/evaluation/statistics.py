"""Subject-level statistics for RakshPulse 4.0.

The subject is the unit of independent generalisation. Windows from one person are correlated pseudo-replicates,
so every test here first reduces to **one number per subject** (e.g. that subject's MAE) and then resamples or
permutes subjects:

* :func:`paired_test`: mean per-subject difference with a subject-bootstrap CI, a sign-flip permutation p-value
  (exact for n ≤ 16 subjects, Monte-Carlo otherwise), a Wilcoxon signed-rank p-value and Cohen's d_z;
* :func:`holm` and :func:`benjamini_hochberg`: family-wise and false-discovery-rate adjusted p-values;
* :func:`bland_altman_repeated`: bias and 95 % limits of agreement for repeated measurements per subject
  (Bland & Altman 2007, one-way random-effects variance components), with subject-bootstrap CIs;
* :func:`error_distribution`: subject-weighted error quantiles and the fraction of subjects' windows within
  5/10/15 mmHg;
* :func:`subject_errors`: per-subject (or per subject-session) MAE.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def subject_errors(t: pd.DataFrame, by: tuple[str, ...] = ("subject_id",)) -> pd.Series:
    """Per-unit MAE from a frame with ``y`` and ``yhat`` (unit = subject, or subject × session)."""
    t = t[np.isfinite(t.yhat) & np.isfinite(t.y)]
    return (t.yhat - t.y).abs().groupby([t[c] for c in by]).mean()


def paired_test(a: pd.Series, b: pd.Series, n_boot: int = 2000, n_perm: int = 10000, seed: int = 0) -> dict[str, float]:
    """Compare per-unit errors ``a`` vs ``b`` (aligned by index). Negative ``mean_diff`` favours ``a``."""
    d = (a - b).dropna().to_numpy(float)
    n = d.size
    out = {
        "n_units": n,
        "mean_diff": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "cohen_dz": float("nan"),
        "p_perm": float("nan"),
        "p_wilcoxon": float("nan"),
    }
    if n < 2:
        return out
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, (n_boot, n))].mean(1)
    obs = d.mean()
    if n <= 16:  # exact sign-flip distribution
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    else:
        signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    null = (signs * d).mean(1)
    p_perm = float(
        (np.sum(np.abs(null) >= abs(obs) - 1e-12) + (0 if n <= 16 else 1)) / (len(null) + (0 if n <= 16 else 1))
    )
    sd = d.std(ddof=1)
    try:
        p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = float("nan")
    out.update(
        mean_diff=float(obs),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        cohen_dz=float(obs / sd) if sd > 0 else float("nan"),
        p_perm=p_perm,
        p_wilcoxon=p_w,
    )
    return out


def holm(p: np.ndarray | list[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (family-wise error). NaNs are ignored and returned as NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not ok.size:
        return out
    order = ok[np.argsort(p[ok])]
    m = order.size
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(p: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (false discovery rate). NaNs are ignored and returned as NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not ok.size:
        return out
    order = ok[np.argsort(p[ok])]
    m = order.size
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def _ba_components(d: pd.Series, g: pd.Series) -> tuple[float, float, float]:
    """(bias, between-subject var, within-subject var) of differences d grouped by g (one-way random effects)."""
    df = pd.DataFrame({"d": d.to_numpy(float), "g": g.to_numpy()})
    grp = df.groupby("g").d
    m = grp.size().to_numpy(float)
    k, n = m.size, m.sum()
    means = grp.mean().to_numpy()
    grand = df.d.mean()
    ss_w = float(((df.d - grp.transform("mean")) ** 2).sum())
    ss_b = float((m * (means - grand) ** 2).sum())
    ms_w = ss_w / max(n - k, 1)
    ms_b = ss_b / max(k - 1, 1)
    divisor = (n**2 - (m**2).sum()) / ((k - 1) * n) if k > 1 else 1.0
    var_b = max((ms_b - ms_w) / divisor, 0.0)
    return float(grand), var_b, ms_w


def bland_altman_repeated(t: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Bias and 95 % LoA of ``yhat − y`` with several windows per subject (Bland & Altman 2007)."""
    t = t[np.isfinite(t.yhat) & np.isfinite(t.y)]
    d = t.yhat - t.y
    bias, vb, vw = _ba_components(d, t.subject_id)
    sd = np.sqrt(vb + vw)
    rng = np.random.default_rng(seed)
    subj = t.subject_id.unique()
    groups = {s: i for s, i in t.groupby("subject_id").groups.items()}
    lo_b, hi_b, bias_b = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(subj, subj.size, replace=True)
        idx = np.concatenate([groups[s] for s in pick])
        # relabel duplicates so a subject drawn twice counts as two subjects
        lab = np.concatenate([np.full(len(groups[s]), i) for i, s in enumerate(pick)])
        bb, b_vb, b_vw = _ba_components(d.loc[idx].reset_index(drop=True), pd.Series(lab))
        s_ = np.sqrt(b_vb + b_vw)
        bias_b.append(bb)
        lo_b.append(bb - 1.96 * s_)
        hi_b.append(bb + 1.96 * s_)
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return {
        "bias": bias,
        "bias_ci_low": pct(bias_b, 2.5),
        "bias_ci_high": pct(bias_b, 97.5),
        "sd_total": float(sd),
        "sd_between_subject": float(np.sqrt(vb)),
        "sd_within_subject": float(np.sqrt(vw)),
        "loa_low": float(bias - 1.96 * sd),
        "loa_low_ci_low": pct(lo_b, 2.5),
        "loa_low_ci_high": pct(lo_b, 97.5),
        "loa_high": float(bias + 1.96 * sd),
        "loa_high_ci_low": pct(hi_b, 2.5),
        "loa_high_ci_high": pct(hi_b, 97.5),
        "n_subjects": int(subj.size),
        "n_windows": int(len(t)),
    }


def error_distribution(t: pd.DataFrame) -> dict[str, float]:
    """Subject-weighted absolute-error distribution: each subject contributes equally."""
    t = t[np.isfinite(t.yhat) & np.isfinite(t.y)]
    ae = (t.yhat - t.y).abs()
    w = 1.0 / t.groupby("subject_id").subject_id.transform("size")
    order = np.argsort(ae.to_numpy())
    cw = np.cumsum(w.to_numpy()[order]) / w.sum()
    q = lambda p: float(ae.to_numpy()[order][np.searchsorted(cw, p)])  # noqa: E731
    per_subj = lambda thr: float(  # noqa: E731
        (ae <= thr).groupby(t.subject_id).mean().mean()
    )
    return {
        "ae_p50": q(0.5),
        "ae_p90": q(0.9),
        "ae_p95": q(0.95),
        "within_5": per_subj(5),
        "within_10": per_subj(10),
        "within_15": per_subj(15),
        "n_subjects": int(t.subject_id.nunique()),
    }


def icc31(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(3,1), two-way mixed, consistency, for two measurements per subject (NaN if < 5 subjects)."""
    x = np.c_[np.asarray(a, float), np.asarray(b, float)]
    x = x[np.isfinite(x).all(1)]
    n = len(x)
    if n < 5:
        return float("nan")
    gm = x.mean()
    ss_r = 2 * ((x.mean(1) - gm) ** 2).sum()
    ss_c = n * ((x.mean(0) - gm) ** 2).sum()
    ss_e = ((x - gm) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / (n - 1)
    return float((ms_r - ms_e) / (ms_r + ms_e)) if (ms_r + ms_e) > 0 else float("nan")
