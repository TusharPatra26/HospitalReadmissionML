"""Unified data cleaning pipeline executing clinical and data-quality transformations."""

from typing import Dict, List, Optional
import pandas as pd
from src.cleansing.missing_values import clean_missing_markers, handle_missing_values
from src.cleansing.duplicate_handler import check_and_remove_duplicates
from src.cleansing.outlier_handler import cap_utilization_outliers
from src.data_ingestion.data_loader import parse_ids_mapping, get_dataset_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Terminal disposition codes indicating patient expired or transferred to hospice
TERMINAL_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]


def clean_encounter_data(
    df: pd.DataFrame,
    ids_mapping: Optional[Dict[str, Dict[int, str]]] = None,
    filter_terminal_discharges: bool = True,
) -> pd.DataFrame:
    """Executes the complete clinical data cleaning protocol.

    Args:
        df: Raw encounters DataFrame
        ids_mapping: Parsed mappings dict from IDS_mapping.csv
        filter_terminal_discharges: Whether to remove expired/hospice encounters

    Returns:
        Cleaned pd.DataFrame ready for feature engineering.
    """
    logger.info("Starting clinical data cleaning on %d encounters...", len(df))
    df = df.copy()

    # 1. Deduplication audit
    df = check_and_remove_duplicates(df)

    # 2. Exclude terminal dispositions (expired / hospice) if requested
    if filter_terminal_discharges and "discharge_disposition_id" in df.columns:
        initial_count = len(df)
        terminal_mask = df["discharge_disposition_id"].isin(TERMINAL_DISPOSITION_IDS)
        excluded_count = int(terminal_mask.sum())
        df = df[~terminal_mask].reset_index(drop=True)
        logger.info(
            "Filtered %d encounters with terminal discharge disposition (expired/hospice): %d remaining.",
            excluded_count,
            len(df),
        )

    # 3. Filter invalid gender categories ('Unknown/Invalid' has only 3 rows)
    if "gender" in df.columns:
        invalid_gender_count = (df["gender"] == "Unknown/Invalid").sum()
        if invalid_gender_count > 0:
            logger.info("Filtering %d encounters with 'Unknown/Invalid' gender.", invalid_gender_count)
            df = df[df["gender"] != "Unknown/Invalid"].reset_index(drop=True)

    # 4. Standardize missing markers and impute
    df = clean_missing_markers(df)
    df = handle_missing_values(df)

    # 5. Outlier capping on utilization
    df = cap_utilization_outliers(df)

    # 6. Map ID columns using IDS_mapping if provided
    if ids_mapping is None:
        _, ids_path = get_dataset_paths()
        ids_mapping = parse_ids_mapping(ids_path)

    if ids_mapping:
        if "admission_type_id" in df.columns and "admission_type" not in df.columns:
            df["admission_type"] = (
                df["admission_type_id"].map(ids_mapping.get("admission_type", {})).fillna("Unknown")
            )
        if "discharge_disposition_id" in df.columns and "discharge_disposition" not in df.columns:
            df["discharge_disposition"] = (
                df["discharge_disposition_id"].map(ids_mapping.get("discharge_disposition", {})).fillna("Unknown")
            )
        if "admission_source_id" in df.columns and "admission_source" not in df.columns:
            df["admission_source"] = (
                df["admission_source_id"].map(ids_mapping.get("admission_source", {})).fillna("Unknown")
            )

    logger.info("Cleaning complete: %d encounters retained.", len(df))
    return df

