"""Decision threshold exploration and optimization on validation data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.classification_metrics import compute_all_metrics
from src.utils.visualization import save_plot, set_plot_theme
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_threshold_range(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluates classification performance across a fine-grained grid of decision thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.90, 86)

    records = []
    for t in thresholds:
        metrics = compute_all_metrics(y_true, y_pred_proba, threshold=t)
        records.append({
            "threshold": t,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
        })

    return pd.DataFrame(records)


def find_optimal_threshold(
    thresh_df: pd.DataFrame,
    criterion: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """Selects the best threshold based on a specified validation objective.

    Criteria:
    - 'f1': Maximizes Harmonic Mean of Precision and Recall
    - 'balanced': Minimizes difference between Sensitivity and Specificity
    - 'clinical_screening': High recall (target recall >= 0.50) while maximizing precision
    """
    if criterion == "f1":
        best_idx = thresh_df["f1"].idxmax()
    elif criterion == "balanced":
        diff = (thresh_df["sensitivity"] - thresh_df["specificity"]).abs()
        best_idx = diff.idxmin()
    elif criterion == "clinical_screening":
        eligible = thresh_df[thresh_df["recall"] >= 0.50]
        if not eligible.empty:
            best_idx = eligible["precision"].idxmax()
        else:
            best_idx = thresh_df["f1"].idxmax()
    else:
        best_idx = thresh_df["f1"].idxmax()

    best_row = thresh_df.loc[best_idx].to_dict()
    best_threshold = float(best_row["threshold"])
    logger.info("Optimal threshold chosen via criterion '%s': %.3f (F1: %.3f, Recall: %.3f, Precision: %.3f)",
                criterion, best_threshold, best_row["f1"], best_row["recall"], best_row["precision"])
    return best_threshold, best_row


def plot_threshold_curve(
    thresh_df: pd.DataFrame,
    best_threshold: float,
    output_path: str | Path = "artifacts/plots/training/threshold_analysis.png",
) -> None:
    """Plots precision, recall, F1, and specificity as a function of decision threshold."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color="#0D9488", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall / Sensitivity", color="#EF4444", lw=2)
    ax.plot(thresh_df["threshold"], thresh_df["f1"], label="F1-Score", color="#3B82F6", lw=2.5)
    ax.plot(thresh_df["threshold"], thresh_df["specificity"], label="Specificity", color="#6B7280", lw=1.5, ls="--")

    ax.axvline(best_threshold, color="#1E3A8A", lw=2, ls=":", label=f"Selected Threshold ({best_threshold:.3f})")
    ax.set_title("Decision Threshold Optimization Curve (Validation Data)", fontweight="bold")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best")

    save_plot(fig, output_path)
    logger.info("Saved threshold optimization curve to %s", output_path)
