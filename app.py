"""Production Flask Web Application for Hospital Readmission Risk Predictor."""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "logs" / ".matplotlib_cache")

import pandas as pd
from src.prediction.prediction_service import PredictionService, predict_readmission
from src.utils.file_handler import load_json
from src.utils.logger import get_logger

logger = get_logger("flask_app")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# Pre-load prediction service on application startup
try:
    prediction_service = PredictionService.get_instance()
    logger.info("PredictionService pre-loaded successfully.")
except Exception as e:
    prediction_service = None
    logger.warning("Could not pre-load PredictionService on startup: %s", e)


def get_cached_metadata():
    """Loads saved model metadata and evaluation metrics for dashboard display."""
    meta_path = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"
    eval_path = PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_metrics.json"
    comp_path = PROJECT_ROOT / "artifacts" / "evaluation" / "model_comparison.csv"

    metadata = {}
    metrics = {}
    comparison = []

    if meta_path.is_file():
        try:
            metadata = load_json(meta_path)
        except Exception as e:
            logger.error("Failed loading model_metadata.json: %s", e)

    if eval_path.is_file():
        try:
            metrics = load_json(eval_path)
        except Exception as e:
            logger.error("Failed loading evaluation_metrics.json: %s", e)

    if comp_path.is_file():
        try:
            comp_df = pd.read_csv(comp_path)
            comparison = comp_df.to_dict(orient="records")
        except Exception as e:
            logger.error("Failed loading model_comparison.csv: %s", e)

    return metadata, metrics, comparison


# --- ROUTE HANDLERS ---

@app.route("/")
def index():
    """Redirects root to dashboard."""
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    """Renders clinical overview dashboard."""
    metadata, metrics, _ = get_cached_metadata()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        metadata=metadata,
        metrics=metrics,
    )


@app.route("/predict")
def predict_page():
    """Renders interactive patient risk estimation form."""
    metadata, _, _ = get_cached_metadata()
    return render_template(
        "prediction.html",
        active_page="predict",
        metadata=metadata,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for real-time patient readmission estimation."""
    if not request.is_json:
        return jsonify({"success": False, "errors": ["Request must be JSON."]}), 400

    data = request.get_json()
    try:
        result = predict_readmission(data)
        return jsonify(result)
    except Exception as e:
        logger.error("Prediction API error: %s", e, exc_info=True)
        return jsonify({"success": False, "errors": ["Internal server error during prediction."]}), 500


@app.route("/explanation")
def explanation_page():
    """Renders explainable AI and SHAP methodology page."""
    metadata, _, _ = get_cached_metadata()
    return render_template(
        "explanation.html",
        active_page="explanation",
        metadata=metadata,
    )


@app.route("/analytics")
def analytics_page():
    """Renders population analytics and demographic exploratory visualizations."""
    return render_template(
        "analytics.html",
        active_page="analytics",
    )


@app.route("/model-performance")
def performance_page():
    """Renders model benchmark, experiments, and evaluation curves."""
    metadata, metrics, comparison = get_cached_metadata()
    return render_template(
        "model_insights.html",
        active_page="performance",
        metadata=metadata,
        metrics=metrics,
        comparison_data=comparison,
    )


@app.route("/feature-insights")
def insights_page():
    """Renders feature importance and dependence insights."""
    return render_template(
        "explanation.html",
        active_page="insights",
    )


@app.route("/about")
def about_page():
    """Renders project documentation, research guidelines, and ethical limitations."""
    return render_template(
        "about.html",
        active_page="about",
    )


@app.route("/artifacts/plots/<path:filename>")
def serve_artifacts_plot(filename):
    """Serves generated artifact plots (eda, training, shap)."""
    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    return send_from_directory(plots_dir, filename)


# Error Handlers
@app.errorhandler(404)
def not_found(e):
    return render_template("dashboard.html", active_page="dashboard", error="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("dashboard.html", active_page="dashboard", error="An internal error occurred."), 500


if __name__ == "__main__":
    logger.info("Starting Flask application on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)

