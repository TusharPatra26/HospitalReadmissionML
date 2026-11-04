"""Configuration loader and schema definition."""

from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """Loads configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Dict containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

