"""Unit tests for feature engineering and cleansing modules."""

import pytest
import pandas as pd
from src.feature_engineering.age_features import build_age_features, AGE_MIDPOINT_MAP
from src.feature_engineering.diagnosis_features import map_icd9_to_group, build_diagnosis_features
from src.feature_engineering.medication_features import build_medication_features
from src.feature_engineering.utilization_features import build_utilization_features
from src.feature_engineering.feature_builder import build_all_features, get_feature_columns
from src.cleansing.data_cleaner import clean_encounter_data


def test_age_features():
    df = pd.DataFrame({"age": ["[50-60)", "[70-80)", "[10-20)"]})
    df_feat = build_age_features(df)
    assert df_feat["age_midpoint"].tolist() == [55.0, 75.0, 15.0]
    assert df_feat["is_senior"].tolist() == [0, 1, 0]
    assert df_feat["age_group"].tolist() == ["Adult (30-59)", "Elderly (75+)", "Young Adult (<30)"]


def test_map_icd9_to_group():
    assert map_icd9_to_group("250.02") == "Diabetes"
    assert map_icd9_to_group("410") == "Circulatory"
    assert map_icd9_to_group("428") == "Circulatory"
    assert map_icd9_to_group("493") == "Respiratory"
    assert map_icd9_to_group("584") == "Genitourinary"
    assert map_icd9_to_group("530") == "Digestive"
    assert map_icd9_to_group("820") == "Injury"
    assert map_icd9_to_group("715") == "Musculoskeletal"
    assert map_icd9_to_group("174") == "Neoplasms"
    assert map_icd9_to_group("V45") == "Other/Unknown"
    assert map_icd9_to_group("?") == "Other/Unknown"


def test_diagnosis_features():
    df = pd.DataFrame({
        "diag_1": ["250.01", "428"],
        "diag_2": ["428", "250.02"],
        "diag_3": ["486", "715"],
    })
    df_feat = build_diagnosis_features(df)
    assert df_feat["diag_1_group"].tolist() == ["Diabetes", "Circulatory"]
    assert df_feat["has_diabetes_primary"].tolist() == [1, 0]
    assert df_feat["has_diabetes_any"].tolist() == [1, 1]
    assert df_feat["num_distinct_diag_groups"].tolist() == [3, 3]


def test_medication_features():
    df = pd.DataFrame({
        "insulin": ["Up", "No"],
        "metformin": ["Steady", "No"],
        "glipizide": ["No", "Down"],
        "change": ["Ch", "No"],
        "diabetesMed": ["Yes", "No"],
    })
    df_feat = build_medication_features(df)
    assert df_feat["num_diabetes_meds"].tolist() == [2, 1]
    assert df_feat["num_med_changes"].tolist() == [1, 1]
    assert df_feat["on_insulin"].tolist() == [1, 0]
    assert df_feat["insulin_titration"].tolist() == [1, 0]
    assert df_feat["is_med_changed"].tolist() == [1, 0]


def test_utilization_features():
    df = pd.DataFrame({
        "number_inpatient": [0, 2, 0],
        "number_emergency": [0, 1, 3],
        "number_outpatient": [1, 0, 0],
    })
    df_feat = build_utilization_features(df)
    assert df_feat["total_prior_visits"].tolist() == [1, 3, 3]
    assert df_feat["has_prior_inpatient"].tolist() == [0, 1, 0]
    assert df_feat["high_prior_utilizer"].tolist() == [0, 1, 1]


def test_feature_canonical_lists():
    num_cols, cat_cols = get_feature_columns()
    assert len(num_cols) >= 20
    assert len(cat_cols) >= 12
    # Ensure no overlap
    assert len(set(num_cols).intersection(set(cat_cols))) == 0

