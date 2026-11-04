"""Duplicate encounter auditing and deduplication module."""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_and_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Audits and removes exact duplicate rows while strictly preserving repeat patient encounters.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with exact duplicates removed.
    """
    initial_len = len(df)
    # Check exact row duplicates across all features
    exact_duplicates = df.duplicated().sum()
    if exact_duplicates > 0:
        logger.warning("Found %d exact duplicate rows. Removing duplicates.", exact_duplicates)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("Audit confirmed: 0 exact duplicate rows found.")

    # Audit repeat patients (DO NOT DROP)
    if "patient_nbr" in df.columns:
        unique_patients = df["patient_nbr"].nunique()
        repeat_encounters = len(df) - unique_patients
        logger.info(
            "Patient encounter audit: %d total encounters across %d unique patients (%d repeat visits preserved).",
            len(df),
            unique_patients,
            repeat_encounters,
        )

    return df

