"""Random-forest regressor configuration."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def make_rf(seed: int = 0) -> RandomForestRegressor:
    # 300 trees: out-of-fold error had plateaued by ~200 trees in pilot runs. n_jobs=-1 for speed.
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=seed)
