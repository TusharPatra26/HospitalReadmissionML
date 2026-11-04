"""Integration tests for Flask application routes and API endpoints."""

import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_redirect(client):
    response = client.get("/")
    assert response.status_code in [302, 200]


def test_dashboard_route(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Hospital Readmission Predictor" in response.data
    assert b"101,766" in response.data


def test_predict_page_route(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Patient 30-Day Readmission Risk" in response.data


def test_api_predict_endpoint(client):
    payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "time_in_hospital": 3,
        "admission_type": "Emergency",
        "discharge_disposition": "Discharged to home",
        "admission_source": "Emergency Room",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 14,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 7,
        "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes",
        "diag_3_group": "Respiratory",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "insulin": "Steady",
        "change": "No",
        "diabetesMed": "Yes",
    }
    response = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "probability" in data
    assert "risk_interpretation" in data
    assert "local_explanation" in data


def test_explanation_route(client):
    response = client.get("/explanation")
    assert response.status_code == 200
    assert b"Explainable AI" in response.data


def test_analytics_route(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"Population Analytics" in response.data


def test_performance_route(client):
    response = client.get("/model-performance")
    assert response.status_code == 200
    assert b"Model Performance" in response.data


def test_about_route(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About the Hospital Readmission" in response.data

