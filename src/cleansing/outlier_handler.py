"""Outlier detection and robust capping utilities."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

NUMERICAL_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Calculates IQR outlier boundaries for numerical features."""
    if columns is None:
        columns = [c for c in NUMERICAL_COLS if c in df.columns]

    boundaries = {}
    for col in columns:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower = max(0, q25 - 1.5 * iqr)
        upper = q75 + 1.5 * iqr
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        boundaries[col] = {
            "q25": float(q25),
            "q75": float(q75),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(outlier_count),
            "outlier_pct": float(outlier_count / len(df) * 100),
        }
    return boundaries


def cap_utilization_outliers(df: pd.DataFrame, caps: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Caps extreme healthcare utilization features at clinical thresholds.

    Extreme repeat visit counts (e.g., 40 emergency visits) are capped to prevent
    high-leverage outliers from distorting linear models while retaining order.
    """
    df = df.copy()
    if caps is None:
        # Clinically defensible 99th+ percentile caps
        caps = {
            "number_inpatient": 15.0,
            "number_emergency": 20.0,
            "number_outpatient": 20.0,
        }

    for col, cap_val in caps.items():
        if col in df.columns:
            capped_count = (df[col] > cap_val).sum()
            if capped_count > 0:
                logger.info("Capping %s at %.1f (%d values affected)", col, cap_val, capped_count)
                df[col] = df[col].clip(upper=cap_val)

    return df

