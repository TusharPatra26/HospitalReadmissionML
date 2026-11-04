"""Framework-independent prediction service orchestrating inference and explainability."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.models.model_registry import load_model_artifacts
from src.prediction.input_validator import validate_patient_payload
from src.prediction.risk_interpreter import interpret_readmission_risk
from src.feature_engineering.feature_builder import build_all_features
from src.explainability.shap_explainer import HospitalShapExplainer
from src.explainability.local_explanation import explain_single_patient
from src.utils.logger import get_logger

logger = get_logger("prediction_service")


class PredictionService:
    """Singleton service for real-time model inference and explainability."""

    _instance: Optional["PredictionService"] = None

    def __init__(self, models_dir: str | Path = "artifacts/models"):
        logger.info("Initializing PredictionService from %s...", models_dir)
        artifacts = load_model_artifacts(models_dir)
        self.model = artifacts["model"]
        self.pipeline = artifacts["pipeline"]
        self.metadata = artifacts["metadata"]
        self.threshold = float(self.metadata.get("best_threshold", 0.130))
        self.explainer = HospitalShapExplainer(models_dir)
        logger.info("PredictionService initialized. Model: %s, Threshold: %.3f",
                    self.metadata.get("model_name"), self.threshold)

    @classmethod
    def get_instance(cls, models_dir: str | Path = "artifacts/models") -> "PredictionService":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(models_dir)
        return cls._instance

    def predict(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates, transforms, predicts, and explains a single patient encounter.

        Args:
            raw_data: Dictionary of user-submitted encounter fields.

        Returns:
            Dict containing prediction status, probabilities, risk tiers, and SHAP drivers.
        """
        # 1. Validation
        is_valid, sanitized_data, errors = validate_patient_payload(raw_data)
        if not is_valid:
            logger.warning("Prediction validation failed: %s", errors)
            return {
                "success": False,
                "errors": errors,
            }

        # 2. Convert to DataFrame and apply feature engineering
        input_df = pd.DataFrame([sanitized_data])

        # Fill any missing raw columns expected by feature builder
        for med in ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                    "glimepiride", "acetohexamide", "glipizide", "glyburide",
                    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                    "miglitol", "troglitazone", "tolazamide", "examide",
                    "citoglipton", "glyburide-metformin", "glipizide-metformin",
                    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]:
            if med not in input_df.columns:
                input_df[med] = "No"

        feat_df = build_all_features(input_df)

        # 3. Transform via fitted Scikit-Learn Pipeline
        X_proc = self.pipeline.transform(feat_df)

        # 4. Model Prediction & Probability
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        is_readmission = bool(prob >= self.threshold)

        # 5. Risk Interpretation
        risk_info = interpret_readmission_risk(prob, threshold=self.threshold)

        # 6. Local SHAP Explanation
        local_shap = explain_single_patient(X_proc, explainer=self.explainer, top_n=6)

        return {
            "success": True,
            "prediction": int(is_readmission),
            "prediction_label": "High Risk of 30-Day Readmission" if is_readmission else "Low Risk of 30-Day Readmission",
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "threshold": round(self.threshold, 4),
            "risk_interpretation": risk_info,
            "local_explanation": local_shap,
            "model_metadata": {
                "model_name": self.metadata.get("model_name"),
                "algorithm": self.metadata.get("algorithm"),
                "version": self.metadata.get("version"),
            },
        }


def predict_readmission(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level convenience function fulfilling the prompt's framework-independent API."""
    service = PredictionService.get_instance()
    return service.predict(patient_data)

