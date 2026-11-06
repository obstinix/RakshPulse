"""BP estimation metrics.

The ISO 81060-2 and BHS statistics are reported **for orientation only**. Those protocols
prescribe specific populations, reference methods and sample sizes that public-dataset
experiments do not meet, so they say nothing about compliance.
"""

from __future__ import annotations

import numpy as np

BP_BINS = {"sbp": [0, 110, 120, 130, 140, 400], "dbp": [0, 60, 70, 80, 90, 300]}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e = y_pred - y_true
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_pred.std() > 0 and y_true.std() > 0 else float("nan")
    ae = np.abs(e)
    return {
        "n": int(e.size),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": r,
        "mean_error": float(e.mean()),
        "sd_error": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
        "loa_low": float(e.mean() - 1.96 * e.std(ddof=1)) if e.size > 1 else float("nan"),
        "loa_high": float(e.mean() + 1.96 * e.std(ddof=1)) if e.size > 1 else float("nan"),
        "pct_within_5": float(np.mean(ae <= 5) * 100),
        "pct_within_10": float(np.mean(ae <= 10) * 100),
        "pct_within_15": float(np.mean(ae <= 15) * 100),
    }


def iso81060_like(m: dict) -> bool:
    """Criterion 1 of ISO 81060-2 (|mean error| <= 5 mmHg and SD <= 8 mmHg). Orientation only."""
    return abs(m["mean_error"]) <= 5 and m["sd_error"] <= 8


def bhs_grade(m: dict) -> str:
    """BHS grade from cumulative % within 5/10/15 mmHg. Orientation only."""
    p5, p10, p15 = m["pct_within_5"], m["pct_within_10"], m["pct_within_15"]
    for g, (a, b, c) in {"A": (60, 85, 95), "B": (50, 75, 90), "C": (40, 65, 85)}.items():
        if p5 >= a and p10 >= b and p15 >= c:
            return g
    return "D"


def by_range(y_true, y_pred, target: str) -> list[dict]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    edges = BP_BINS[target]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0:
            continue
        e = y_pred[m] - y_true[m]
        out.append(
            {"range": f"[{lo},{hi})", "n": int(m.sum()), "mae": float(np.abs(e).mean()), "mean_error": float(e.mean())}
        )
    return out
