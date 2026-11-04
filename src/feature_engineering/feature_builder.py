"""Centralized feature pipeline assembling demographic, clinical, and utilization features."""

from typing import List, Optional, Tuple
import pandas as pd
from src.feature_engineering.age_features import build_age_features
from src.feature_engineering.diagnosis_features import build_diagnosis_features
from src.feature_engineering.medication_features import build_medication_features
from src.feature_engineering.utilization_features import build_utilization_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_SPECIALTIES = [
    "InternalMedicine",
    "Emergency/Trauma",
    "Family/GeneralPractice",
    "Cardiology",
    "Surgery",
]


def bin_medical_specialty(specialty: str) -> str:
    """Bins medical specialty into high-level categories to manage cardinality."""
    if pd.isna(specialty) or specialty in ["?", "", "Missing/Unknown", "None"]:
        return "Missing/Unknown"
    spec = str(specialty).replace("-", "").replace(" ", "").lower()
    if "internalmedicine" in spec:
        return "InternalMedicine"
    elif "emergency" in spec or "trauma" in spec:
        return "Emergency/Trauma"
    elif "family" in spec or "generalpractice" in spec:
        return "Family/GeneralPractice"
    elif "cardio" in spec:
        return "Cardiology"
    elif "surg" in spec:
        return "Surgery"
    elif "orthoped" in spec:
        return "Orthopedics"
    elif "nephrol" in spec:
        return "Nephrology"
    else:
        return "Other_Specialty"


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all modular feature engineering functions sequentially.

    Args:
        df: Cleaned encounters DataFrame

    Returns:
        DataFrame with full set of engineered features.
    """
    logger.info("Building engineered features for %d encounters...", len(df))
    df = df.copy()

    # 1. Demographic & age features
    df = build_age_features(df)

    # 2. Medical specialty binning
    if "medical_specialty" in df.columns:
        df["medical_specialty_binned"] = df["medical_specialty"].apply(bin_medical_specialty)
    else:
        df["medical_specialty_binned"] = "Missing/Unknown"

    # 3. Diagnosis ICD-9 grouping & complexity
    df = build_diagnosis_features(df)

    # 4. Medication, polypharmacy & titration features
    df = build_medication_features(df)

    # 5. Prior healthcare utilization indices
    df = build_utilization_features(df)

    logger.info("Feature engineering complete: %d features available.", len(df.columns))
    return df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """Returns canonical lists of numerical and categorical feature names expected by the model."""
    numerical_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "age_midpoint",
        "total_prior_visits",
        "num_diabetes_meds",
        "num_med_changes",
        "num_distinct_diag_groups",
        "is_senior",
        "has_prior_inpatient",
        "has_prior_emergency",
        "high_prior_utilizer",
        "has_diabetes_primary",
        "has_circulatory_primary",
        "on_insulin",
        "insulin_titration",
        "is_med_changed",
        "is_diabetes_med",
    ]

    categorical_features = [
        "race",
        "gender",
        "age_group",
        "admission_type",
        "discharge_disposition",
        "admission_source",
        "medical_specialty_binned",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
    ]

    return numerical_features, categorical_features

