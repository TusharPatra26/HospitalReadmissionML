"""Confusion matrix visualization with absolute counts and percentages."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.visualization import save_plot, set_plot_theme
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    output_path: str | Path = "artifacts/plots/training/confusion_matrix.png",
) -> np.ndarray:
    """Generates and saves a detailed annotated healthcare confusion matrix heatmap."""
    set_plot_theme()
    if labels is None:
        labels = ["No Readmit / >30d", "<30d Readmitted"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = np.sum(cm)

    # Format annotations: Count + (Percentage)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total) * 100
            annot[i, j] = f"{count:,}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix (Optimized Threshold)", fontweight="bold")
    ax.set_xlabel("Predicted Clinical Outcome", fontweight="bold")
    ax.set_ylabel("Actual Clinical Outcome", fontweight="bold")

    save_plot(fig, output_path)
    logger.info("Saved confusion matrix heatmap to %s", output_path)
    return cm

