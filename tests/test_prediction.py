"""Unit and integration tests for framework-independent prediction service."""

import pytest
from src.prediction.prediction_service import PredictionService, predict_readmission
from src.prediction.input_validator import validate_patient_payload


@pytest.fixture
def sample_valid_patient():
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "time_in_hospital": 4,
        "admission_type": "Emergency",
        "discharge_disposition": "Discharged to home",
        "admission_source": "Emergency Room",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 8,
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Respiratory",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes",
    }


def test_validate_patient_payload_valid(sample_valid_patient):
    is_valid, sanitized, errors = validate_patient_payload(sample_valid_patient)
    assert is_valid is True
    assert len(errors) == 0
    assert sanitized["time_in_hospital"] == 4


def test_validate_patient_payload_invalid():
    invalid_data = {
        "time_in_hospital": 99,  # Exceeds max 14
        "gender": "InvalidGender",
    }
    is_valid, sanitized, errors = validate_patient_payload(invalid_data)
    assert is_valid is False
    assert len(errors) > 0


def test_predict_service_valid_patient(sample_valid_patient):
    result = predict_readmission(sample_valid_patient)
    assert result["success"] is True
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "prediction" in result
    assert result["prediction"] in [0, 1]
    assert "risk_interpretation" in result
    assert "local_explanation" in result
    assert len(result["local_explanation"]["top_positive_factors"]) > 0


def test_high_vs_low_risk_contrast():
    low_risk_patient = {
        "race": "Caucasian",
        "gender": "Male",
        "age": "[20-30)",
        "time_in_hospital": 1,
        "admission_type": "Elective",
        "discharge_disposition": "Discharged to home",
        "admission_source": "Physician Referral",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 15,
        "num_procedures": 0,
        "num_medications": 4,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 2,
        "diag_1_group": "Other/Unknown",
        "diag_2_group": "Other/Unknown",
        "diag_3_group": "Other/Unknown",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "insulin": "No",
        "change": "No",
        "diabetesMed": "No",
    }

    high_risk_patient = {
        "race": "AfricanAmerican",
        "gender": "Female",
        "age": "[70-80)",
        "time_in_hospital": 9,
        "admission_type": "Emergency",
        "discharge_disposition": "Discharged/transferred to SNF",
        "admission_source": "Emergency Room",
        "medical_specialty": "Emergency/Trauma",
        "num_lab_procedures": 75,
        "num_procedures": 3,
        "num_medications": 28,
        "number_outpatient": 2,
        "number_emergency": 2,
        "number_inpatient": 4,  # High prior utilization!
        "number_diagnoses": 9,
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Respiratory",
        "max_glu_serum": ">300",
        "A1Cresult": ">8",
        "insulin": "Up",
        "change": "Ch",
        "diabetesMed": "Yes",
    }

    res_low = predict_readmission(low_risk_patient)
    res_high = predict_readmission(high_risk_patient)

    assert res_low["success"] is True
    assert res_high["success"] is True
    assert res_high["probability"] > res_low["probability"]
    assert res_high["prediction"] == 1

