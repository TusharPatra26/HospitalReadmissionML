"""Missing value handling and imputation strategies."""

from typing import List, Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns with excessive missingness (>90%) that should be excluded
DEFAULT_DROP_COLS = ["weight"]


def clean_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes missing markers ('?', '', 'Unknown/Invalid') across the dataset.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized missing markers.
    """
    df = df.copy()
    # Replace '?' with NaN or explicit Missing
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].replace(["?", "", "None"], np.nan)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Handles missing values based on clinical and statistical principles.

    Args:
        df: Input DataFrame
        drop_cols: List of columns to drop due to extreme missingness (default ['weight'])

    Returns:
        DataFrame with imputed/handled missing values.
    """
    df = df.copy()
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    # 1. Drop columns with excessive missingness
    for col in drop_cols:
        if col in df.columns:
            logger.info("Dropping high-missingness column: %s", col)
            df = df.drop(columns=[col])

    # 2. Categorical features with missingness: assign explicit 'Missing/Unknown'
    cat_impute_cols = ["race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]
    for col in cat_impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Missing/Unknown")

    # 3. Lab test fields: 'None' signifies test was not ordered
    for col in ["max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df

