"""Unit tests for preprocessing transformers, scalers, encoders, and patient-aware splits."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.numerical_scaler import get_numerical_scaler, PassThroughScaler
from src.preprocessing.categorical_encoder import get_categorical_encoder
from src.preprocessing.column_transformer import build_column_transformer
from src.preprocessing.preprocessing_pipeline import split_patient_aware_data, PreprocessingPipeline


def test_numerical_scalers():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    std_scaler = get_numerical_scaler("standard")
    std_scaler.fit(X)
    X_std = std_scaler.transform(X)
    assert np.isclose(X_std.mean(), 0.0, atol=1e-5)
    assert np.isclose(X_std.std(), 1.0, atol=1e-5)

    pt_scaler = get_numerical_scaler("passthrough")
    assert isinstance(pt_scaler, PassThroughScaler)
    assert np.allclose(pt_scaler.fit_transform(X), X)


def test_categorical_encoder_unseen_categories():
    df_train = pd.DataFrame({"race": ["Caucasian", "AfricanAmerican", "Caucasian"]})
    df_test = pd.DataFrame({"race": ["Caucasian", "Asian", "UnknownCategory"]})

    encoder = get_categorical_encoder()
    X_tr = encoder.fit_transform(df_train)
    assert X_tr.shape[1] == 2  # Caucasian, AfricanAmerican

    # Transforming unseen categories should NOT crash and should output 0s
    X_te = encoder.transform(df_test)
    assert X_te.shape[1] == 2
    # The unseen categories should have all 0s
    assert X_te[1].sum() == 0.0
    assert X_te[2].sum() == 0.0


def test_split_patient_aware_data_no_leakage():
    # Create dataset with repeat patients
    data = {
        "encounter_id": list(range(100)),
        "patient_nbr": [i // 2 for i in range(100)],  # 50 unique patients, 2 encounters each
        "val": np.random.randn(100),
    }
    df = pd.DataFrame(data)

    train_df, val_df, test_df = split_patient_aware_data(df, group_col="patient_nbr", test_size=0.2, val_size=0.2)

    s_train = set(train_df["patient_nbr"])
    s_val = set(val_df["patient_nbr"])
    s_test = set(test_df["patient_nbr"])

    # Strict zero overlap
    assert len(s_train.intersection(s_val)) == 0
    assert len(s_train.intersection(s_test)) == 0
    assert len(s_val.intersection(s_test)) == 0
    assert len(train_df) + len(val_df) + len(test_df) == 100


def test_preprocessing_pipeline_fit_and_names():
    from src.feature_engineering.feature_builder import get_feature_columns

    num_cols, cat_cols = get_feature_columns()
    # Mock row
    mock_data = {col: [1.0] for col in num_cols}
    mock_data.update({col: ["CategoryA"] for col in cat_cols})
    df_mock = pd.DataFrame(mock_data)

    pipeline = PreprocessingPipeline(scaler_type="standard")
    X_out = pipeline.fit_transform(df_mock)

    assert X_out.shape[0] == 1
    assert len(pipeline.feature_names_out) == X_out.shape[1]
    assert pipeline.is_fitted is True

