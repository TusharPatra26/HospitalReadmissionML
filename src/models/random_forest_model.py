"""Random Forest and Extra Trees ensemble estimators."""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def get_random_forest(
    n_estimators: int = 150,
    max_depth: int = 12,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Returns configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_extra_trees(
    n_estimators: int = 150,
    max_depth: int = 12,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> ExtraTreesClassifier:
    """Returns configured Extra Trees ensemble classifier."""
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

