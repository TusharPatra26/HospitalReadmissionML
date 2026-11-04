"""Standalone evaluation script verifying saved model artifacts on test data."""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "logs" / ".matplotlib_cache")

import numpy as np
from src.data_ingestion.data_loader import load_raw_data
from src.cleansing.data_cleaner import clean_encounter_data
from src.feature_engineering.feature_builder import build_all_features
from src.preprocessing.preprocessing_pipeline import split_patient_aware_data
from src.models.model_registry import load_model_artifacts
from src.evaluation.classification_metrics import compute_all_metrics

def main():
    print("=" * 70)
    print("Hospital Readmission Risk Predictor — Standalone Evaluation")
    print("=" * 70)

    # Load artifacts
    artifacts = load_model_artifacts("artifacts/models")
    model = artifacts["model"]
    pipeline = artifacts["pipeline"]
    metadata = artifacts["metadata"]

    best_threshold = metadata.get("best_threshold", 0.130)
    print(f"Loaded Model: {metadata.get('model_name')}")
    print(f"Optimal Threshold: {best_threshold:.3f}")

    # Re-generate test partition
    raw_df = load_raw_data(derive_target=True)
    clean_df = clean_encounter_data(raw_df, filter_terminal_discharges=True)
    feat_df = build_all_features(clean_df)
    _, _, test_df = split_patient_aware_data(feat_df, group_col="patient_nbr", test_size=0.15, val_size=0.15, random_state=42)

    X_test_proc = pipeline.transform(test_df)
    y_test = test_df["readmitted_30_days"].values

    y_test_proba = model.predict_proba(X_test_proc)[:, 1]
    metrics = compute_all_metrics(y_test, y_test_proba, threshold=best_threshold)

    print("\nVerified Test Set Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20}: {v}")
    print("=" * 70)

if __name__ == "__main__":
    main()

