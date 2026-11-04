"""ROC, Precision-Recall, and Probability Calibration evaluation plots."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

from src.utils.visualization import save_plot, set_plot_theme
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Selected Model",
    output_path: str | Path = "artifacts/plots/training/roc_curve.png",
) -> float:
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    set_plot_theme()
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1E3A8A", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="No-Skill Baseline (AUC = 0.500)")

    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")

    save_plot(fig, output_path)
    logger.info("Saved ROC curve to %s", output_path)
    return float(roc_auc)


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Selected Model",
    output_path: str | Path = "artifacts/plots/training/pr_curve.png",
) -> float:
    """Plots and saves the Precision-Recall (PR) curve."""
    set_plot_theme()
    prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(rec, prec)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color="#0D9488", lw=2.5, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, color="#EF4444", lw=1.5, linestyle="--", label=f"Baseline Prevalence ({baseline:.3f})")

    ax.set_title(f"Precision-Recall Curve — {model_name}", fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")

    save_plot(fig, output_path)
    logger.info("Saved PR curve to %s", output_path)
    return float(pr_auc)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Selected Model",
    n_bins: int = 10,
    output_path: str | Path = "artifacts/plots/training/calibration_curve.png",
) -> float:
    """Plots reliability curve / probability calibration diagram."""
    set_plot_theme()
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, marker="o", color="#3B82F6", lw=2, label=f"{model_name} (Brier = {brier:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", lw=1.5, label="Perfectly Calibrated")

    ax.set_title(f"Probability Calibration (Reliability Diagram) — {model_name}", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Readmission Proportion")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper left")

    save_plot(fig, output_path)
    logger.info("Saved calibration curve to %s", output_path)
    return float(brier)

