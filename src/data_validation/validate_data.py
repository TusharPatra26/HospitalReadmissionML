"""Data validation module ensuring dataset integrity and schema adherence."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
    "readmitted"
]


def validate_raw_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates that a DataFrame conforms to the expected UCI Diabetes schema.

    Args:
        df: pd.DataFrame to validate

    Returns:
        Dict detailing validation status, issues, and statistics.
    """
    logger.info("Validating dataset schema and dimensions...")
    report: Dict[str, Any] = {
        "valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_expected_columns": [],
        "extra_columns": [],
        "issues": [],
    }

    # Check columns
    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        report["valid"] = False
        report["missing_expected_columns"] = missing_cols
        report["issues"].append(f"Missing expected columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in EXPECTED_RAW_COLUMNS and c != "readmitted_30_days"]
    if extra_cols:
        report["extra_columns"] = extra_cols

    # Check target
    if "readmitted" in df.columns:
        valid_targets = {"<30", ">30", "NO"}
        found_targets = set(df["readmitted"].dropna().unique())
        if not found_targets.issubset(valid_targets):
            report["valid"] = False
            report["issues"].append(f"Unexpected target values: {found_targets - valid_targets}")
    else:
        report["valid"] = False
        report["issues"].append("Target column 'readmitted' missing.")

    # Check identifier columns
    if "encounter_id" in df.columns:
        unique_enc = df["encounter_id"].nunique()
        report["unique_encounters"] = unique_enc
        if unique_enc != len(df):
            report["issues"].append(f"Encounter ID not unique for all rows: {unique_enc} vs {len(df)}")
    if "patient_nbr" in df.columns:
        report["unique_patients"] = df["patient_nbr"].nunique()

    # Numerical sanity checks
    num_sanity = {
        "time_in_hospital": (1, 14),
        "num_lab_procedures": (1, 150),
        "num_medications": (1, 100),
    }
    for col, (min_val, max_val) in num_sanity.items():
        if col in df.columns:
            actual_min = pd.to_numeric(df[col], errors="coerce").min()
            actual_max = pd.to_numeric(df[col], errors="coerce").max()
            if actual_min < min_val or actual_max > max_val:
                report["issues"].append(
                    f"Warning: {col} values out of typical range [{min_val}, {max_val}]: [{actual_min}, {actual_max}]"
                )

    if report["valid"]:
        logger.info("Dataset validation passed successfully.")
    else:
        logger.error("Dataset validation failed with issues: %s", report["issues"])

    return report

