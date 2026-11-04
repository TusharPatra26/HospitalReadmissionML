"""ICD-9 diagnosis classification and comorbidity feature engineering."""

from typing import Any, List
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def map_icd9_to_group(code: Any) -> str:
    """Maps a raw ICD-9 diagnosis code into standard clinical disease categories.

    Clinical Mapping Taxonomy (adapted from Strack et al., 2014 & HCUP CCS):
    - Circulatory: 390-459, 785
    - Respiratory: 460-519, 786
    - Digestive: 520-579, 787
    - Diabetes: 250.xx
    - Injury/Poisoning: 800-999
    - Musculoskeletal: 710-739
    - Genitourinary: 580-629, 788
    - Neoplasms: 140-239
    - Other: V-codes, E-codes, all other diagnostic numbers, or Missing
    """
    if pd.isna(code) or code in ["?", "", "Missing/Unknown", "None"]:
        return "Other/Unknown"

    code_str = str(code).strip().upper()

    # V or E codes
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Other/Unknown"

    # Diabetes check (250 or 250.xx)
    if code_str.startswith("250"):
        return "Diabetes"

    try:
        num_code = float(code_str.split(".")[0])
    except ValueError:
        return "Other/Unknown"

    if (390 <= num_code <= 459) or num_code == 785:
        return "Circulatory"
    elif (460 <= num_code <= 519) or num_code == 786:
        return "Respiratory"
    elif (520 <= num_code <= 579) or num_code == 787:
        return "Digestive"
    elif (800 <= num_code <= 999):
        return "Injury"
    elif (710 <= num_code <= 739):
        return "Musculoskeletal"
    elif (580 <= num_code <= 629) or num_code == 788:
        return "Genitourinary"
    elif (140 <= num_code <= 239):
        return "Neoplasms"
    else:
        return "Other/Unknown"


def build_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts clinical diagnosis groups and comorbidity complexity metrics."""
    df = df.copy()

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(map_icd9_to_group)

    # Primary condition indicators
    if "diag_1_group" in df.columns:
        df["has_diabetes_primary"] = (df["diag_1_group"] == "Diabetes").astype(int)
        df["has_circulatory_primary"] = (df["diag_1_group"] == "Circulatory").astype(int)

    # Comorbidity cross-checks
    group_cols = [f"{c}_group" for c in ["diag_1", "diag_2", "diag_3"] if f"{c}_group" in df.columns]
    if group_cols:
        df["has_diabetes_any"] = (
            df[group_cols].apply(lambda row: any(v == "Diabetes" for v in row), axis=1).astype(int)
        )
        df["num_distinct_diag_groups"] = df[group_cols].apply(
            lambda row: len(set(v for v in row if v != "Other/Unknown")), axis=1
        )

    return df

