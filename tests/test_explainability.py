"""Unit tests for SHAP explainer, consistency verification, and local patient explanation."""

import pytest
import numpy as np
from src.explainability.shap_explainer import HospitalShapExplainer
from src.explainability.local_explanation import explain_single_patient, humanize_feature_name
from src.utils.file_handler import load_joblib


def test_shap_explainer_initialization_and_consistency():
    explainer = HospitalShapExplainer("artifacts/models")
    assert explainer.model is not None
    assert explainer.base_value is not None
    assert len(explainer.feature_names) > 50

    # Load 5 sample instances from background
    bg_data = load_joblib("artifacts/explainability/shap_background_data.pkl")
    X_samples = bg_data[:5]

    is_consistent = explainer.verify_consistency(X_samples)
    assert is_consistent is True


def test_explain_single_patient():
    explainer = HospitalShapExplainer("artifacts/models")
    bg_data = load_joblib("artifacts/explainability/shap_background_data.pkl")
    single_sample = bg_data[0]

    result = explain_single_patient(single_sample, explainer=explainer, top_n=5)

    assert "predicted_probability" in result
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert "top_positive_factors" in result
    assert "top_negative_factors" in result
    assert "summary_narrative" in result
    assert "disclaimer" in result
    assert len(result["top_positive_factors"]) <= 5
    assert len(result["top_negative_factors"]) <= 5


def test_humanize_feature_name():
    assert "Inpatient" in humanize_feature_name("num__number_inpatient")
    assert "Circulatory" in humanize_feature_name("cat__diag_1_group_Circulatory")

