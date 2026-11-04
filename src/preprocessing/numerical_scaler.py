"""Numerical scaling strategies for linear and distance-based estimators."""

from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class PassThroughScaler(BaseEstimator, TransformerMixin):
    """Identity transformer that passes numerical features through unchanged (for tree models)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def get_numerical_scaler(scaler_type: str = "standard") -> BaseEstimator:
    """Returns a numerical scaler based on configuration.

    Args:
        scaler_type: 'standard', 'robust', 'minmax', or 'passthrough' (for trees)

    Returns:
        Fitted Scikit-Learn transformer
    """
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type in ["passthrough", "none"]:
        return PassThroughScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose standard, robust, minmax, or passthrough.")

