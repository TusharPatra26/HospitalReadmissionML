"""Comprehensive classification metrics computation module."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes full suite of classification metrics including sensitivity, specificity, and PR-AUC.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class (1)
        threshold: Decision threshold for converting probabilities to binary classes

    Returns:
        Dict of computed evaluation metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba).ravel()
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion Matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_pred_proba))
    except Exception:
        pr_auc = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, y_pred_proba))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": round(threshold, 4),
    }

