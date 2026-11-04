"""Baseline linear and tree classifiers."""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_logistic_regression(
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Returns configured regularized Logistic Regression estimator."""
    return LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )


def get_decision_tree(
    max_depth: int = 8,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Returns configured Decision Tree baseline classifier."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )

