"""Categorical encoding strategies handling nominal healthcare categories and unseen levels."""

from sklearn.preprocessing import OneHotEncoder


def get_categorical_encoder(sparse_output: bool = False, handle_unknown: str = "ignore") -> OneHotEncoder:
    """Returns a configured OneHotEncoder that gracefully handles unseen categories at inference.

    Args:
        sparse_output: Whether to return sparse matrix or dense numpy array.
        handle_unknown: 'ignore' to output all zeros for unseen categories at inference time.

    Returns:
        Scikit-Learn OneHotEncoder
    """
    return OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=sparse_output,
        drop=None,  # Retain all categories for symmetric SHAP explanations
    )

