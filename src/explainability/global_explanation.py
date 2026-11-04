"""Global model explainability via SHAP summary, importance bar, and dependence plots."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.explainability.shap_explainer import HospitalShapExplainer
from src.utils.file_handler import load_joblib
from src.utils.visualization import save_plot, set_plot_theme
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_global_shap_plots(
    explainer: Optional[HospitalShapExplainer] = None,
    output_dir: str | Path = "artifacts/plots/shap",
    n_samples: int = 400,
) -> None:
    """Computes global SHAP values and saves summary, bar, and dependence plots."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if explainer is None:
        explainer = HospitalShapExplainer()

    # Load background/validation sample
    bg_data = load_joblib("artifacts/explainability/shap_background_data.pkl")
    X_sample = bg_data[:min(n_samples, len(bg_data))]

    logger.info("Computing global SHAP values for %d samples across %d features...", len(X_sample), X_sample.shape[1])
    shap_vals, base_val = explainer.explain_instance(X_sample)

    feature_names = explainer.feature_names
    # Clean up feature names for display
    clean_names = [f.replace("cat__", "").replace("num__", "").replace("_", " ").title() for f in feature_names]

    # --- 1. SHAP Feature Importance Bar Chart ---
    logger.info("Generating SHAP importance bar chart...")
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:]

    set_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_indices)), mean_abs_shap[top_indices], color="#1E3A8A")
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([clean_names[i] for i in top_indices], fontsize=10)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Log-Odds)")
    ax.set_title("Global Feature Importance — Top 15 Predictors (SHAP)", fontweight="bold")
    save_plot(fig, dest / "shap_importance_bar.png")

    # --- 2. SHAP Summary / Beeswarm Plot ---
    logger.info("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=clean_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Global Summary (Feature Impact Direction)", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(dest / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 3. Dependence Plot: Prior Inpatient Encounters ---
    logger.info("Generating SHAP dependence plots...")
    # Find index of number_inpatient
    inp_idx = next((i for i, name in enumerate(feature_names) if "number_inpatient" in name), None)
    if inp_idx is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_sample[:, inp_idx], shap_vals[:, inp_idx], color="#0D9488", alpha=0.6, edgecolors="none")
        ax.set_xlabel("Standardized Prior Inpatient Visits")
        ax.set_ylabel("SHAP Value for Inpatient Visits (Log-Odds)")
        ax.set_title("SHAP Dependence: Prior Inpatient Admissions vs Risk Contribution", fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        save_plot(fig, dest / "shap_dependence_inpatient.png")

    # --- 4. Dependence Plot: Time in Hospital ---
    los_idx = next((i for i, name in enumerate(feature_names) if "time_in_hospital" in name), None)
    if los_idx is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_sample[:, los_idx], shap_vals[:, los_idx], color="#3B82F6", alpha=0.6, edgecolors="none")
        ax.set_xlabel("Standardized Length of Stay (Days)")
        ax.set_ylabel("SHAP Value for Length of Stay (Log-Odds)")
        ax.set_title("SHAP Dependence: Length of Stay vs Risk Contribution", fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        save_plot(fig, dest / "shap_dependence_hospital_time.png")

    logger.info("All global SHAP plots generated successfully in %s", dest)


if __name__ == "__main__":
    generate_global_shap_plots()

