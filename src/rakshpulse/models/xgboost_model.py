"""Gradient-boosted trees (XGBoost) configuration."""

from __future__ import annotations

from xgboost import XGBRegressor


def make_xgb(seed: int = 0) -> XGBRegressor:
    # Shallow trees, strong subsampling and an L2 penalty: the label set is small (one cuff
    # value per subject), so variance control matters more than capacity.
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        min_child_weight=10,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        verbosity=0,
    )
