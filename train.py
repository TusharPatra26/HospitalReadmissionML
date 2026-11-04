"""End-to-end training execution script."""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "logs" / ".matplotlib_cache")

from src.models.model_training import run_model_experiments

if __name__ == "__main__":
    print("=" * 70)
    print("Hospital Readmission Risk Predictor — Model Training & Evaluation")
    print("=" * 70)
    results = run_model_experiments()
    print("\nTraining workflow completed successfully!")
    print(f"Winner Model Selected: {results['best_model_name']}")
    print(f"Optimal Decision Threshold: {results['best_threshold']:.3f}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"Test PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
    print(f"Test F1-Score: {results['test_metrics']['f1']:.4f}")
    print(f"Test Recall: {results['test_metrics']['recall']:.4f}")
    print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
    print("=" * 70)
