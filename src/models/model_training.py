"""Comprehensive model experimentation, class imbalance comparison, and training pipeline."""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.data_ingestion.data_loader import load_raw_data
from src.cleansing.data_cleaner import clean_encounter_data
from src.feature_engineering.feature_builder import build_all_features, get_feature_columns
from src.preprocessing.preprocessing_pipeline import split_patient_aware_data, PreprocessingPipeline
from src.models.baseline_models import get_logistic_regression, get_decision_tree
from src.models.random_forest_model import get_random_forest, get_extra_trees
from src.models.xgboost_model import get_xgboost, get_hist_gradient_boosting
from src.models.model_registry import save_model_artifacts
from src.evaluation.classification_metrics import compute_all_metrics
from src.evaluation.threshold_analysis import evaluate_threshold_range, find_optimal_threshold, plot_threshold_curve
from src.evaluation.roc_analysis import plot_roc_curve, plot_pr_curve, plot_calibration_curve
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.utils.file_handler import save_dataframe, save_json
from src.utils.logger import get_logger

logger = get_logger("model_training")


def run_model_experiments() -> Dict[str, Any]:
    """Executes end-to-end model experiments, class imbalance comparisons, and final model evaluation."""
    start_all = time.time()
    logger.info("Starting complete Machine Learning training workflow...")

    # 1. Ingestion & Preprocessing
    raw_df = load_raw_data(derive_target=True)
    clean_df = clean_encounter_data(raw_df, filter_terminal_discharges=True)
    feat_df = build_all_features(clean_df)

    # 2. Patient-Aware Split
    train_df, val_df, test_df = split_patient_aware_data(
        feat_df, group_col="patient_nbr", test_size=0.15, val_size=0.15, random_state=42
    )

    y_train = train_df["readmitted_30_days"].values
    y_val = val_df["readmitted_30_days"].values
    y_test = test_df["readmitted_30_days"].values

    # 3. Fit Preprocessing Pipeline strictly on Train partition
    pipeline = PreprocessingPipeline(scaler_type="standard")
    X_train_proc = pipeline.fit_transform(train_df)
    X_val_proc = pipeline.transform(val_df)
    X_test_proc = pipeline.transform(test_df)

    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(1, pos_count)
    logger.info("Training class ratio: %d Negatives / %d Positives (scale_pos_weight = %.2f)", neg_count, pos_count, scale_pos_weight)

    # 4. Define Candidate Model Experiments
    experiments: List[Dict[str, Any]] = [
        {
            "name": "Logistic Regression (Unweighted)",
            "estimator": get_logistic_regression(class_weight=None),
            "resampler": None,
            "strategy": "None (Unweighted)",
        },
        {
            "name": "Logistic Regression (Balanced Weights)",
            "estimator": get_logistic_regression(class_weight="balanced"),
            "resampler": None,
            "strategy": "Cost-Sensitive (class_weight='balanced')",
        },
        {
            "name": "Decision Tree (Balanced)",
            "estimator": get_decision_tree(max_depth=8, class_weight="balanced"),
            "resampler": None,
            "strategy": "Cost-Sensitive (class_weight='balanced')",
        },
        {
            "name": "Random Forest (Balanced Weights)",
            "estimator": get_random_forest(n_estimators=150, max_depth=12, class_weight="balanced"),
            "resampler": None,
            "strategy": "Cost-Sensitive (class_weight='balanced')",
        },
        {
            "name": "Extra Trees (Balanced)",
            "estimator": get_extra_trees(n_estimators=150, max_depth=12, class_weight="balanced"),
            "resampler": None,
            "strategy": "Cost-Sensitive (class_weight='balanced')",
        },
        {
            "name": "HistGradientBoosting (Balanced)",
            "estimator": get_hist_gradient_boosting(max_iter=150, max_depth=6, class_weight="balanced"),
            "resampler": None,
            "strategy": "Cost-Sensitive (class_weight='balanced')",
        },
        {
            "name": "XGBoost (Unweighted)",
            "estimator": get_xgboost(n_estimators=200, max_depth=5, scale_pos_weight=1.0),
            "resampler": None,
            "strategy": "None (Unweighted)",
        },
        {
            "name": "XGBoost (scale_pos_weight)",
            "estimator": get_xgboost(n_estimators=200, max_depth=5, scale_pos_weight=scale_pos_weight),
            "resampler": None,
            "strategy": f"Cost-Sensitive (scale_pos_weight={scale_pos_weight:.2f})",
        },
        {
            "name": "Random Forest + SMOTE",
            "estimator": get_random_forest(n_estimators=120, max_depth=12, class_weight=None),
            "resampler": SMOTE(random_state=42),
            "strategy": "Oversampling (SMOTE strictly on train)",
        },
        {
            "name": "Random Forest + RandomOverSampler",
            "estimator": get_random_forest(n_estimators=120, max_depth=12, class_weight=None),
            "resampler": RandomOverSampler(random_state=42),
            "strategy": "Oversampling (RandomOverSampler)",
        },
        {
            "name": "Random Forest + RandomUnderSampler",
            "estimator": get_random_forest(n_estimators=120, max_depth=12, class_weight=None),
            "resampler": RandomUnderSampler(random_state=42),
            "strategy": "Undersampling (RandomUnderSampler)",
        },
    ]

    results_table = []
    trained_estimators = {}
    val_predictions = {}

    # 5. Run Experiments on Validation Set
    for exp in experiments:
        name = exp["name"]
        clf = exp["estimator"]
        resampler = exp["resampler"]
        strategy = exp["strategy"]

        logger.info("Evaluating experiment: %s ...", name)
        t0 = time.time()

        # Resample strictly on training partition if applicable
        if resampler is not None:
            X_tr, y_tr = resampler.fit_resample(X_train_proc, y_train)
        else:
            X_tr, y_tr = X_train_proc, y_train

        clf.fit(X_tr, y_tr)
        train_time = time.time() - t0

        # Validation inference
        t_inf0 = time.time()
        if hasattr(clf, "predict_proba"):
            y_val_proba = clf.predict_proba(X_val_proc)[:, 1]
        else:
            # Fallback
            y_val_proba = clf.decision_function(X_val_proc)
        inf_time = (time.time() - t_inf0) / len(X_val_proc) * 1000  # ms per sample

        # Evaluate at standard 0.5 threshold for fair model comparison
        metrics = compute_all_metrics(y_val, y_val_proba, threshold=0.5)

        record = {
            "model_name": name,
            "imbalance_strategy": strategy,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
            "train_time_sec": round(train_time, 2),
            "inf_time_ms_per_sample": round(inf_time, 4),
        }
        results_table.append(record)
        trained_estimators[name] = clf
        val_predictions[name] = y_val_proba

    comparison_df = pd.DataFrame(results_table)
    save_dataframe(comparison_df, "artifacts/evaluation/model_comparison.csv", index=False)
    logger.info("Saved model comparison to artifacts/evaluation/model_comparison.csv")

    # 6. Model Selection based on PR-AUC & ROC-AUC Evidence
    # We rank primarily by PR-AUC (Average Precision) and ROC-AUC on the imbalanced validation set
    comparison_df["selection_score"] = comparison_df["pr_auc"] * 0.6 + comparison_df["roc_auc"] * 0.4
    best_row = comparison_df.sort_values(by="selection_score", ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    best_estimator = trained_estimators[best_model_name]
    best_val_proba = val_predictions[best_model_name]

    logger.info("FINAL WINNING MODEL SELECTED: '%s' (PR-AUC: %.4f, ROC-AUC: %.4f)",
                best_model_name, best_row["pr_auc"], best_row["roc_auc"])

    # 7. Threshold Optimization on Validation Set
    logger.info("Optimizing decision threshold on validation data for %s...", best_model_name)
    thresh_df = evaluate_threshold_range(y_val, best_val_proba)
    best_threshold, best_thresh_row = find_optimal_threshold(thresh_df, criterion="f1")
    plot_threshold_curve(thresh_df, best_threshold, output_path="artifacts/plots/training/threshold_analysis.png")

    # 8. Single Final Evaluation on untouched Test Set
    logger.info("Executing SINGLE final evaluation of frozen winner on untouched test partition...")
    t_test0 = time.time()
    y_test_proba = best_estimator.predict_proba(X_test_proc)[:, 1]
    test_inf_time = (time.time() - t_test0) / len(X_test_proc) * 1000

    test_metrics = compute_all_metrics(y_test, y_test_proba, threshold=best_threshold)
    test_metrics["inference_time_ms"] = round(test_inf_time, 4)
    logger.info(
        "TEST SET FINAL RESULTS for %s at threshold %.3f: ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f, Recall=%.4f, Precision=%.4f, Specificity=%.4f",
        best_model_name,
        best_threshold,
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["f1"],
        test_metrics["recall"],
        test_metrics["precision"],
        test_metrics["specificity"],
    )

    # 9. Generate Training Evaluation Plots on Test Set
    plot_roc_curve(y_test, y_test_proba, model_name=best_model_name, output_path="artifacts/plots/training/roc_curve.png")
    plot_pr_curve(y_test, y_test_proba, model_name=best_model_name, output_path="artifacts/plots/training/pr_curve.png")
    plot_calibration_curve(y_test, y_test_proba, model_name=best_model_name, output_path="artifacts/plots/training/calibration_curve.png")
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    plot_confusion_matrix(y_test, y_test_pred, output_path="artifacts/plots/training/confusion_matrix.png")

    # 10. Persist Artifacts and Metadata
    num_cols, cat_cols = get_feature_columns()
    model_metadata = {
        "model_name": best_model_name,
        "algorithm": best_estimator.__class__.__name__,
        "version": "1.0.0",
        "best_threshold": float(best_threshold),
        "val_metrics": best_row.to_dict(),
        "test_metrics": test_metrics,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "transformed_feature_names": pipeline.feature_names_out,
        "total_transformed_features": len(pipeline.feature_names_out),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

    save_model_artifacts(
        model=best_estimator,
        pipeline=pipeline,
        metadata=model_metadata,
        models_dir="artifacts/models",
    )
    save_json(test_metrics, "artifacts/evaluation/evaluation_metrics.json")

    # Also persist SHAP background sample (e.g. 200 transformed train rows) for explainability
    shap_bg_sample = X_train_proc[:200]
    from src.utils.file_handler import save_joblib
    save_joblib(shap_bg_sample, "artifacts/explainability/shap_background_data.pkl")

    total_time = time.time() - start_all
    logger.info("Complete training workflow executed successfully in %.1f seconds.", total_time)

    return {
        "comparison_table": results_table,
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    run_model_experiments()

