"""End-to-end preprocessing pipeline and patient-aware dataset partitioner."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer

from src.preprocessing.column_transformer import build_column_transformer
from src.feature_engineering.feature_builder import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_patient_aware_data(
    df: pd.DataFrame,
    group_col: str = "patient_nbr",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions dataset into Train, Validation, and Test splits grouped strictly by patient.

    Guarantees 0% patient overlap between all partitions to eliminate clinical data leakage.

    Args:
        df: Input DataFrame containing patient encounters.
        group_col: Patient identifier column ('patient_nbr').
        test_size: Proportion for test partition (e.g., 0.15).
        val_size: Proportion for validation partition (e.g., 0.15).
        random_state: Reproducible random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("Executing patient-aware splitting on group '%s'...", group_col)
    if group_col not in df.columns:
        raise KeyError(f"Patient group column '{group_col}' missing from DataFrame.")

    # Stage 1: Split into Train+Val and Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=df[group_col]))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Stage 2: Split Train+Val into Train and Validation
    val_size_adjusted = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df[group_col]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Integrity verification
    s_train = set(train_df[group_col])
    s_val = set(val_df[group_col])
    s_test = set(test_df[group_col])

    overlap_tv = len(s_train.intersection(s_val))
    overlap_tt = len(s_train.intersection(s_test))
    overlap_vt = len(s_val.intersection(s_test))

    if overlap_tv > 0 or overlap_tt > 0 or overlap_vt > 0:
        raise ValueError(
            f"Leakage detected! Patient overlap found: TV={overlap_tv}, TT={overlap_tt}, VT={overlap_vt}"
        )

    logger.info(
        "Patient-aware split successful: Train=%d (%.1f%%, %d patients), Val=%d (%.1f%%, %d patients), Test=%d (%.1f%%, %d patients). Zero patient overlap verified.",
        len(train_df),
        len(train_df) / len(df) * 100,
        len(s_train),
        len(val_df),
        len(val_df) / len(df) * 100,
        len(s_val),
        len(test_df),
        len(test_df) / len(df) * 100,
        len(s_test),
    )
    return train_df, val_df, test_df


class PreprocessingPipeline:
    """Encapsulates fitted ColumnTransformer and feature metadata for training and inference."""

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.num_cols, self.cat_cols = get_feature_columns()
        self.transformer: ColumnTransformer = build_column_transformer(
            numerical_cols=self.num_cols,
            categorical_cols=self.cat_cols,
            scaler_type=self.scaler_type,
        )
        self.is_fitted = False
        self.feature_names_out: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "PreprocessingPipeline":
        """Fits the ColumnTransformer strictly on the training partition."""
        logger.info("Fitting PreprocessingPipeline strictly on training partition (%d rows)...", len(X_train))
        self.transformer.fit(X_train)
        self.is_fitted = True

        # Extract transformed feature names
        try:
            self.feature_names_out = list(self.transformer.get_feature_names_out())
        except Exception:
            # Fallback feature naming
            self.feature_names_out = [f"feat_{i}" for i in range(self.transformer.transform(X_train.head(2)).shape[1])]

        logger.info("PreprocessingPipeline fitted: %d transformed features generated.", len(self.feature_names_out))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame into preprocessed numpy feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted before calling transform().")
        return self.transformer.transform(X)

    def fit_transform(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> np.ndarray:
        """Fits on training data and returns transformed feature matrix."""
        return self.fit(X_train, y_train).transform(X_train)

