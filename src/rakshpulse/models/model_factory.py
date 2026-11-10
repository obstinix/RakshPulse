"""Model registry. Every feature-based model is a scikit-learn Pipeline whose imputer and
scaler are fitted inside each training fold only, so they cannot leak test-set statistics.

Hyper-parameter grids are deliberately small. With 148 labelled subjects, large searches
mostly overfit the validation folds.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
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
    if name == "elasticnet":
        return _pipe(ElasticNet(max_iter=20_000)), {"model__alpha": [0.1, 1.0, 3.0], "model__l1_ratio": [0.2, 0.8]}
    if name == "extra_trees":
        et = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=seed)
        return _pipe(et, scale=False), {"model__min_samples_leaf": [10, 40]}
    if name == "hist_gb":
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=seed)
        return _pipe(hgb, scale=False), {"model__max_depth": [3, None], "model__min_samples_leaf": [20, 80]}
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        lgb = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            min_child_samples=30,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        return _pipe(lgb, scale=False), {"model__num_leaves": [7, 31], "model__learning_rate": [0.03, 0.1]}
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
EXTENDED_MODELS = ("ridge", "elasticnet", "svr", "random_forest", "extra_trees", "hist_gb", "xgboost")
PERSONALIZATION_MODELS = EXTENDED_MODELS + ("lightgbm",)

# keep RandomForestRegressor importable from here for type checks
__all__ = ["make_model", "FEATURE_MODELS", "RandomForestRegressor"]
