"""Standardized logging configuration."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "hospital_readmission",
    log_file: Optional[str] = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Creates or retrieves a standardized logger.

    Args:
        name: Logger name.
        log_file: Path to log file destination. If None, logs only to stream.
        level: Logging level (default INFO).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        os.makedirs(log_path.parent, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

