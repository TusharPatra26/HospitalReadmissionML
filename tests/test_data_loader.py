"""Unit tests for data ingestion, IDS mapping parser, and schema validation."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_ingestion.data_loader import load_raw_data, parse_ids_mapping, get_dataset_paths
from src.data_validation.validate_data import validate_raw_schema, EXPECTED_RAW_COLUMNS


def test_get_dataset_paths():
    data_path, ids_path = get_dataset_paths()
    assert data_path.is_file(), f"diabetic_data.csv not found at {data_path}"
    assert ids_path.is_file(), f"IDS_mapping.csv not found at {ids_path}"


def test_load_raw_data_and_schema():
    df = load_raw_data(derive_target=True)
    assert len(df) == 101766
    assert "readmitted_30_days" in df.columns
    assert df["readmitted_30_days"].isin([0, 1]).all()
    # Check positive target count (<30)
    assert df["readmitted_30_days"].sum() == 11357


def test_validate_raw_schema():
    df = load_raw_data(derive_target=False)
    report = validate_raw_schema(df)
    assert report["valid"] is True
    assert len(report["missing_expected_columns"]) == 0
    assert report["unique_encounters"] == 101766
    assert report["unique_patients"] == 71518


def test_parse_ids_mapping():
    _, ids_path = get_dataset_paths()
    mappings = parse_ids_mapping(ids_path)
    assert "admission_type" in mappings
    assert "discharge_disposition" in mappings
    assert "admission_source" in mappings
    assert len(mappings["admission_type"]) >= 8
    assert len(mappings["discharge_disposition"]) >= 25
    assert len(mappings["admission_source"]) >= 20
    assert mappings["admission_type"][1] == "Emergency"
    assert mappings["discharge_disposition"][1] == "Discharged to home"

