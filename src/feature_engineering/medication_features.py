"""Medication utilization, polypharmacy, and titration feature engineering."""

from typing import List
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALL_DIABETES_MEDS: List[str] = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]


def build_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates active diabetes medication counts, titration changes, and insulin features."""
    df = df.copy()

    # Find which diabetes med columns exist in df
    present_meds = [m for m in ALL_DIABETES_MEDS if m in df.columns]

    if present_meds:
        # Count medications prescribed (value != 'No')
        df["num_diabetes_meds"] = df[present_meds].apply(
            lambda row: sum(str(v).strip() != "No" for v in row), axis=1
        )

        # Count medications that were adjusted/titrated (value is 'Up' or 'Down')
        df["num_med_changes"] = df[present_meds].apply(
            lambda row: sum(str(v).strip() in ["Up", "Down"] for v in row), axis=1
        )

    # Insulin status features
    if "insulin" in df.columns:
        df["on_insulin"] = (df["insulin"] != "No").astype(int)
        df["insulin_titration"] = df["insulin"].isin(["Up", "Down"]).astype(int)

    # General medication flags
    if "change" in df.columns:
        df["is_med_changed"] = (df["change"] == "Ch").astype(int)

    if "diabetesMed" in df.columns:
        df["is_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

    return df

