"""Healthcare prior utilization feature engineering."""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds aggregate utilization indexes and high-risk utilization flags."""
    df = df.copy()

    util_cols = ["number_inpatient", "number_emergency", "number_outpatient"]
    for col in util_cols:
        if col not in df.columns:
            df[col] = 0

    # 1. Total prior healthcare encounters in preceding 12 months
    df["total_prior_visits"] = df["number_inpatient"] + df["number_emergency"] + df["number_outpatient"]

    # 2. Binary presence indicators
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["has_prior_outpatient"] = (df["number_outpatient"] > 0).astype(int)

    # 3. High utilizer flag (>=2 inpatient admissions or >=3 total encounters)
    df["high_prior_utilizer"] = (
        (df["number_inpatient"] >= 2) | (df["total_prior_visits"] >= 3)
    ).astype(int)

    return df

