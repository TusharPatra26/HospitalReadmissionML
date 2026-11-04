"""Data acquisition and ingestion module for UCI Diabetes 130-US Hospitals dataset."""

import os
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_dataset_paths(raw_dir: str | Path = "data/raw") -> Tuple[Path, Path]:
    """Resolves and returns paths for raw dataset files, handling case variations."""
    raw_path = Path(raw_dir)
    data_path = raw_path / "diabetic_data.csv"

    # Handle case sensitivity variations on Windows/Linux (IDS_mapping vs IDs_mapping)
    ids_candidates = [
        raw_path / "IDS_mapping.csv",
        raw_path / "IDs_mapping.csv",
        raw_path / "ids_mapping.csv",
    ]
    ids_path = None
    for cand in ids_candidates:
        if cand.is_file():
            ids_path = cand
            break

    if ids_path is None:
        ids_path = raw_path / "IDS_mapping.csv"

    return data_path, ids_path


def parse_ids_mapping(file_path: Path | str) -> Dict[str, Dict[int, str]]:
    """Parses multi-table IDS_mapping.csv into structured dictionaries.

    Args:
        file_path: Path to IDS_mapping.csv

    Returns:
        Dict with keys 'admission_type', 'discharge_disposition', 'admission_source'
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning("IDs mapping file not found at %s", path)
        return {"admission_type": {}, "discharge_disposition": {}, "admission_source": {}}

    mappings: Dict[str, Dict[int, str]] = {
        "admission_type": {},
        "discharge_disposition": {},
        "admission_source": {},
    }
    current_key = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line == ",":
                continue
            if "admission_type_id,description" in clean_line:
                current_key = "admission_type"
                continue
            elif "discharge_disposition_id,description" in clean_line:
                current_key = "discharge_disposition"
                continue
            elif "admission_source_id,description" in clean_line:
                current_key = "admission_source"
                continue

            if current_key and "," in clean_line:
                parts = clean_line.split(",", 1)
                try:
                    code_id = int(parts[0].strip())
                    desc = parts[1].strip().strip('"')
                    mappings[current_key][code_id] = desc
                except ValueError:
                    continue

    logger.info(
        "Parsed IDS mapping: %d admission types, %d discharge dispositions, %d admission sources",
        len(mappings["admission_type"]),
        len(mappings["discharge_disposition"]),
        len(mappings["admission_source"]),
    )
    return mappings


def load_raw_data(
    data_path: str | Path = "data/raw/diabetic_data.csv",
    derive_target: bool = True,
) -> pd.DataFrame:
    """Loads raw diabetic_data.csv into a pandas DataFrame.

    Args:
        data_path: Path to diabetic_data.csv
        derive_target: Whether to derive the binary 30-day readmission target

    Returns:
        pd.DataFrame containing the dataset
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Authentic dataset file not found at {path}. "
            "Please ensure diabetic_data.csv is placed in data/raw/."
        )

    logger.info("Loading raw dataset from %s ...", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Dataset loaded successfully: %d encounters, %d columns", len(df), len(df.columns))

    if derive_target:
        if "readmitted" not in df.columns:
            raise KeyError("Expected 'readmitted' column in raw dataset.")
        
        # 1: readmitted within 30 days (<30)
        # 0: not readmitted within 30 days (NO or >30)
        df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)
        pos = df["readmitted_30_days"].sum()
        total = len(df)
        logger.info(
            "Derived binary target 'readmitted_30_days': %d positive (%.2f%%), %d negative (%.2f%%)",
            pos,
            (pos / total) * 100,
            total - pos,
            ((total - pos) / total) * 100,
        )

    return df

