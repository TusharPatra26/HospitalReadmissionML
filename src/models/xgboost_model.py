"""Gradient Boosting models: XGBoost and HistGradientBoosting."""

from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


def get_xgboost(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBClassifier:
    """Returns configured XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_hist_gradient_boosting(
    max_iter: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """Returns Scikit-Learn HistGradientBoostingClassifier."""
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=random_state,
    )

