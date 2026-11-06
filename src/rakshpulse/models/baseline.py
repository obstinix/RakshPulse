"""Reference baselines that every PPG model must beat to show the PPG adds information.

* ``mean``: predicts the training-set mean BP. On a narrow-BP cohort this is
  surprisingly hard to beat, and any model that cannot beat it has learned nothing useful.
* ``demographics``: ridge regression on age, sex, height and weight only. If a PPG model is
  not better than this, its apparent skill comes from demographic correlates, not from the
  pulse wave.
"""

from __future__ import annotations

DEMOGRAPHIC_FEATURES = ["demo_age", "demo_sex", "demo_height_cm", "demo_weight_kg"]
