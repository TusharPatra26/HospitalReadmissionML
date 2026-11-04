"""Feature engineering for patient age categories and continuous surrogates."""

from typing import Dict
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

AGE_MIDPOINT_MAP: Dict[str, float] = {
    "[0-10)": 5.0,
    "[10-20)": 15.0,
    "[20-30)": 25.0,
    "[30-40)": 35.0,
    "[40-50)": 45.0,
    "[50-60)": 55.0,
    "[60-70)": 65.0,
    "[70-80)": 75.0,
    "[80-90)": 85.0,
    "[90-100)": 95.0,
}


def build_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds age midpoint, ordinal rank, and age category features.

    Note: The age midpoint is an analytical surrogate for modeling, not an exact
    measured biological age.
    """
    df = df.copy()
    if "age" not in df.columns:
        return df

    # 1. Age midpoint
    df["age_midpoint"] = df["age"].map(AGE_MIDPOINT_MAP).fillna(65.0)

    # 2. Senior citizen indicator (>= 65 years)
    df["is_senior"] = (df["age_midpoint"] >= 65.0).astype(int)

    # 3. Clinically interpretable age group
    def _categorize_age(val: float) -> str:
        if val < 30:
            return "Young Adult (<30)"
        elif val < 60:
            return "Adult (30-59)"
        elif val < 75:
            return "Older Adult (60-74)"
        else:
            return "Elderly (75+)"

    df["age_group"] = df["age_midpoint"].apply(_categorize_age)

    return df

