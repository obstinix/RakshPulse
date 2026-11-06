"""Model registry. Every feature-based model is a scikit-learn Pipeline whose imputer and
scaler are fitted inside each training fold only, so they cannot leak test-set statistics.

Hyper-parameter grids are deliberately small. With 148 labelled subjects, large searches
mostly overfit the validation folds.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .xgboost_model import make_xgb


def _pipe(est, scale: bool = True) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps)


def make_model(name: str, seed: int = 0) -> tuple[Pipeline, dict]:
    """Return (estimator, param_grid). Grid keys use the ``model__`` prefix."""
    if name == "mean":
        return _pipe(DummyRegressor(strategy="mean"), scale=False), {}
    if name == "linear":
        return _pipe(LinearRegression()), {}
    if name == "ridge":
        return _pipe(Ridge()), {"model__alpha": [1.0, 10.0, 100.0, 1000.0]}
    if name == "svr":
        return _pipe(SVR(kernel="rbf")), {"model__C": [1.0, 10.0, 30.0], "model__epsilon": [1.0, 3.0]}
    if name == "random_forest":
        from .random_forest import make_rf

        return _pipe(make_rf(seed), scale=False), {
            "model__min_samples_leaf": [5, 20],
            "model__max_features": [0.3, 0.6],
        }
    if name == "xgboost":
        return _pipe(make_xgb(seed), scale=False), {"model__max_depth": [2, 4], "model__learning_rate": [0.03, 0.1]}
    raise KeyError(name)


FEATURE_MODELS = ("mean", "linear", "ridge", "svr", "random_forest", "xgboost")

# keep RandomForestRegressor importable from here for type checks
__all__ = ["make_model", "FEATURE_MODELS", "RandomForestRegressor"]
