"""Model persistence, artifact registration, and metadata serialization."""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib

from src.utils.file_handler import save_joblib, load_joblib, save_json, load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model_artifacts(
    model: Any,
    pipeline: Any,
    metadata: Dict[str, Any],
    models_dir: str | Path = "artifacts/models",
) -> None:
    """Saves the trained model, preprocessing pipeline, and JSON metadata.

    Args:
        model: Trained classifier (e.g. XGBClassifier)
        pipeline: Fitted PreprocessingPipeline
        metadata: Dict containing version, metrics, threshold, features
        models_dir: Destination directory
    """
    dest = Path(models_dir)
    dest.mkdir(parents=True, exist_ok=True)

    model_path = dest / "best_model.pkl"
    pipeline_path = dest / "preprocessing_pipeline.pkl"
    meta_path = dest / "model_metadata.json"

    save_joblib(model, model_path)
    save_joblib(pipeline, pipeline_path)
    save_json(metadata, meta_path)

    logger.info("Successfully persisted all model artifacts to %s", dest)


def load_model_artifacts(
    models_dir: str | Path = "artifacts/models",
) -> Dict[str, Any]:
    """Loads all model artifacts required for production inference.

    Returns:
        Dict with keys: 'model', 'pipeline', 'metadata'
    """
    src = Path(models_dir)
    model_path = src / "best_model.pkl"
    pipeline_path = src / "preprocessing_pipeline.pkl"
    meta_path = src / "model_metadata.json"

    if not model_path.is_file() or not pipeline_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing model artifacts in {src}. Please ensure training pipeline has executed."
        )

    model = load_joblib(model_path)
    pipeline = load_joblib(pipeline_path)
    metadata = load_json(meta_path)

    logger.info("Loaded model artifacts: %s (v%s)", metadata.get("model_name"), metadata.get("version"))
    return {
        "model": model,
        "pipeline": pipeline,
        "metadata": metadata,
    }

