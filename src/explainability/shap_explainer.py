"""Core SHAP explainer wrapper for Tree-based models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import shap

from src.models.model_registry import load_model_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HospitalShapExplainer:
    """Encapsulates SHAP TreeExplainer with feature name mapping and additive verification."""

    def __init__(self, models_dir: str | Path = "artifacts/models"):
        artifacts = load_model_artifacts(models_dir)
        self.model = artifacts["model"]
        self.pipeline = artifacts["pipeline"]
        self.metadata = artifacts["metadata"]
        self.feature_names = self.metadata.get("transformed_feature_names", [])

        # Initialize TreeExplainer
        logger.info("Initializing SHAP TreeExplainer for %s...", self.metadata.get("model_name"))
        self.explainer = shap.TreeExplainer(self.model)

        # Baseline expected value
        try:
            sample_exp = self.explainer(np.zeros((1, len(self.feature_names))))
            self.base_value = float(sample_exp.base_values[0])
        except Exception:
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                self.base_value = float(self.explainer.expected_value[0])
            else:
                self.base_value = float(self.explainer.expected_value)

        logger.info("SHAP Explainer ready. Base value (log-odds margin): %.4f", self.base_value)

    def explain_instance(self, X_sample: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculates SHAP values for a single sample or batch.

        Returns:
            (shap_values, base_value)
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        exp_result = self.explainer(X_sample)
        shap_vals = exp_result.values
        if hasattr(exp_result, "base_values"):
            base_val = float(exp_result.base_values[0])
        else:
            base_val = self.base_value

        return shap_vals, base_val

    def verify_consistency(self, X_sample: np.ndarray, atol: float = 1e-4) -> bool:
        """Verifies additive consistency: base_value + sum(shap_values) == margin."""
        shap_vals, base_val = self.explain_instance(X_sample)
        raw_margin = self.model.predict(X_sample, output_margin=True)
        if raw_margin.ndim > 1:
            raw_margin = raw_margin[:, 0]

        computed = base_val + np.sum(shap_vals, axis=1)
        diff = np.abs(computed - raw_margin)
        max_diff = float(np.max(diff))
        is_consistent = bool(max_diff < atol)
        if not is_consistent:
            logger.warning("SHAP consistency warning: max diff = %e (limit: %e)", max_diff, atol)
        else:
            logger.info("SHAP additive consistency verified (max diff = %e).", max_diff)
        return is_consistent
