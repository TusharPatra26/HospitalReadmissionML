"""Local patient encounter explainability and narrative synthesis."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.explainability.shap_explainer import HospitalShapExplainer
from src.utils.visualization import save_plot, set_plot_theme
from src.utils.logger import get_logger

logger = get_logger(__name__)


def humanize_feature_name(name: str) -> str:
    """Translates technical transformer feature names into clear clinical labels."""
    clean = name.replace("num__", "").replace("cat__", "")
    if "_" in clean and any(term in clean for term in ["group", "admission", "discharge", "specialty", "insulin"]):
        parts = clean.split("_")
        return " ".join(parts).title()
    return clean.replace("_", " ").title()


def explain_single_patient(
    X_sample: np.ndarray,
    explainer: Optional[HospitalShapExplainer] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Generates structured local SHAP explanation for an individual patient encounter.

    Args:
        X_sample: 1D or 2D transformed numerical vector (shape: [1, n_features])
        explainer: HospitalShapExplainer instance
        top_n: Number of top positive and negative features to extract

    Returns:
        Structured dictionary containing contributions, feature labels, base value, and narrative.
    """
    if explainer is None:
        explainer = HospitalShapExplainer()

    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    shap_vals, base_val = explainer.explain_instance(X_sample)
    shap_vals = shap_vals[0]  # First instance
    feature_names = explainer.feature_names

    raw_margin = float(base_val + np.sum(shap_vals))
    prob = 1.0 / (1.0 + np.exp(-raw_margin))

    # Pair features with contributions
    contributions = []
    for i, val in enumerate(shap_vals):
        contributions.append({
            "feature_raw": feature_names[i],
            "feature_label": humanize_feature_name(feature_names[i]),
            "shap_value": float(val),
            "abs_impact": abs(float(val)),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        })

    # Sort positive (increasing risk) and negative (reducing risk)
    pos_factors = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_n]
    neg_factors = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_n]

    # Narrative explanation synthesis
    pos_narrative = ", ".join([f"{item['feature_label']} (+{item['shap_value']:.2f})" for item in pos_factors[:3]])
    neg_narrative = ", ".join([f"{item['feature_label']} ({item['shap_value']:.2f})" for item in neg_factors[:3]])

    summary_text = (
        f"This patient has an estimated 30-day readmission probability of {prob * 100:.1f}%. "
        f"Key factors driving the model estimate upward include: {pos_narrative if pos_narrative else 'None significant'}. "
        f"Counterbalancing factors attenuating the estimate include: {neg_narrative if neg_narrative else 'None significant'}."
    )

    disclaimer = (
        "Medical Disclaimer: Feature attributions represent statistical model weighting in log-odds space. "
        "They do not establish biological causation, disease diagnosis, or treatment efficacy. "
        "Clinical judgment by a licensed medical provider must always supersede algorithmic outputs."
    )

    return {
        "base_value": float(base_val),
        "total_shap_sum": float(np.sum(shap_vals)),
        "model_output_log_odds": float(raw_margin),
        "predicted_probability": float(prob),
        "top_positive_factors": pos_factors,
        "top_negative_factors": neg_factors,
        "summary_narrative": summary_text,
        "disclaimer": disclaimer,
    }


def plot_local_waterfall(
    explanation: Dict[str, Any],
    output_path: str | Path = "artifacts/plots/shap/local_waterfall_example.png",
) -> None:
    """Generates and saves a clean local waterfall/contribution bar chart."""
    set_plot_theme()
    pos = explanation["top_positive_factors"][:5]
    neg = explanation["top_negative_factors"][:5]
    all_factors = pos + neg
    all_factors = sorted(all_factors, key=lambda x: x["shap_value"])

    labels = [item["feature_label"] for item in all_factors]
    values = [item["shap_value"] for item in all_factors]
    colors = ["#EF4444" if v > 0 else "#0D9488" for v in values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("SHAP Contribution (Log-Odds Impact on Readmission)")
    ax.set_title("Local Explanation: Key Drivers for Patient Readmission Risk", fontweight="bold")

    save_plot(fig, output_path)
    logger.info("Saved local explanation plot to %s", output_path)

