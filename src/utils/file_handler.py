"""File I/O utilities for models, datasets, and metrics."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_joblib(obj: Any, file_path: str | Path) -> None:
    """Safely saves an object using joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved joblib artifact to %s", path)


def load_joblib(file_path: str | Path) -> Any:
    """Safely loads an object using joblib."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    logger.info("Loading joblib artifact from %s", path)
    return joblib.load(path)


def save_json(data: Dict[str, Any], file_path: str | Path, indent: int = 4) -> None:
    """Saves a dictionary as formatted JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info("Saved JSON file to %s", path)


def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Loads a JSON file as dictionary."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, file_path: str | Path, index: bool = False) -> None:
    """Saves a pandas DataFrame to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Saved DataFrame (%d rows, %d cols) to %s", len(df), len(df.columns), path)

