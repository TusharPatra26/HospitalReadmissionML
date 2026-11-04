# Hospital Readmission Risk Predictor
## Antigravity IDE — Master Project Prompt

> **Purpose:** Build a complete, modular, explainable machine-learning application that estimates whether a diabetic patient encounter will be followed by a hospital readmission within 30 days. The project is an academic/research prototype, not a clinically validated medical device.

---

# 1. Your role and overall objective

Act as a senior Machine Learning Engineer, Data Scientist, Python Backend Developer, and UI/UX Designer.

Build an end-to-end project named **Hospital Readmission Risk Predictor** using the authentic **UCI Diabetes 130-US Hospitals for Years 1999–2008** dataset.

Dataset landing page:  
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset contains approximately 101,766 hospital encounters from 130 US hospitals. A row represents an encounter, not necessarily a unique patient; a patient may appear in multiple rows.

The system should estimate the probability that a diabetic patient encounter is associated with readmission within 30 days, using information available at the intended prediction point (defined in this project as discharge).

This must be a serious end-to-end ML project—not a single notebook with one classifier. It must include extensive EDA, documented cleansing and feature engineering, fair model experimentation, class-imbalance analysis, model selection, evaluation, SHAP explainability, a polished Flask interface, tests, and deployment preparation for a later Streamlit Cloud version.

**Do not begin by assuming Random Forest or XGBoost must win.** Explore reasonable approaches, compare them on consistent data splits and metrics, and select the final model based on evidence.

---

# 2. Technology stack

## Data science and machine learning
- Python 3.10+
- Pandas, NumPy, SciPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib, Seaborn, Plotly

## Backend
- Flask
- Jinja2
- Joblib
- Python logging

## Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5 or an equivalent responsive framework
- Chart.js or Plotly.js
- Font Awesome or Lucide icons

## Deployment preparation
- Flask for the first working interface.
- A separate `streamlit_app.py` for future Streamlit Community Cloud deployment.
- A framework-independent prediction service shared by both interfaces.

**Architecture rule:** Flask routes and Streamlit UI code must not contain duplicated ML logic. Both must call the same prediction, validation, preprocessing, and explanation services. A Flask app is not itself a native Streamlit app; provide a separate Streamlit entry point.

---

# 3. Dataset acquisition and target definition

## Dataset acquisition

Use the real UCI dataset. Do not replace it with generated, simulated, or unrelated data.

Implement a data acquisition module that:
1. Downloads the dataset from the official UCI source when possible.
2. Supports manual placement of the downloaded CSV.
3. Extracts an archive if the downloaded source is compressed.
4. Retrieves and uses `IDs_mapping.csv` where available.
5. Validates expected files, columns, row counts, and data types.
6. Reports download or extraction failures clearly.
7. Never silently substitutes synthetic data.

Keep raw data unchanged in `data/raw/`. Store intermediate and processed data separately.

## Target

The original `readmitted` column has three values:
- `<30`: readmitted within 30 days.
- `>30`: readmitted after 30 days.
- `NO`: not readmitted.

Create the binary target:

```python
df["readmitted_30_days"] = (
    df["readmitted"] == "<30"
).astype(int)
```

Interpretation:
- `1`: readmitted within 30 days.
- `0`: not readmitted within 30 days (including encounters labelled `>30`).

Drop the original `readmitted` target column from model inputs after deriving the binary target. It must never be included as a feature.

Explain that this target predicts **30-day readmission**, not all future readmissions.

---

# 4. Required modular project structure

Use a structure inspired by the separation of concerns in this reference repository:  
https://github.com/TusharPatra26/studentperformanceML

Adapt it for this healthcare ML workflow. You may improve the structure, but do not put the entire project in one file.

```text
Hospital-Readmission-Predictor/
│
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv
│   │   └── IDs_mapping.csv
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_Dataset_Overview.ipynb
│   ├── 02_Extensive_EDA.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Experiments.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_SHAP_Explainability.ipynb
│
├── src/
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── data_validation/
│   │   ├── __init__.py
│   │   └── validate_data.py
│   ├── cleansing/
│   │   ├── __init__.py
│   │   ├── missing_values.py
│   │   ├── duplicate_handler.py
│   │   ├── outlier_handler.py
│   │   └── data_cleaner.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── categorical_encoder.py
│   │   ├── numerical_scaler.py
│   │   ├── preprocessing_pipeline.py
│   │   └── column_transformer.py
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── age_features.py
│   │   ├── diagnosis_features.py
│   │   ├── medication_features.py
│   │   ├── utilization_features.py
│   │   └── feature_builder.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   ├── logistic_regression_model.py
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── model_tuning.py
│   │   ├── model_training.py
│   │   ├── model_selection.py
│   │   └── model_registry.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── classification_metrics.py
│   │   ├── threshold_analysis.py
│   │   ├── confusion_matrix.py
│   │   ├── roc_analysis.py
│   │   └── model_comparison.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── global_explanation.py
│   │   └── local_explanation.py
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── prediction_service.py
│   │   ├── input_validator.py
│   │   └── risk_interpreter.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       ├── file_handler.py
│       └── visualization.py
│
├── artifacts/
│   ├── models/
│   │   ├── best_model.pkl
│   │   ├── preprocessing_pipeline.pkl
│   │   └── model_metadata.json
│   ├── explainability/
│   │   └── shap_background_data.pkl
│   ├── evaluation/
│   │   ├── model_comparison.csv
│   │   ├── classification_report.json
│   │   └── evaluation_metrics.json
│   └── plots/
│       ├── eda/
│       ├── training/
│       └── shap/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── explanation.html
│   ├── analytics.html
│   ├── model_insights.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   ├── style.css
│   │   ├── dashboard.css
│   │   └── components.css
│   ├── js/
│   │   ├── main.js
│   │   ├── charts.js
│   │   └── prediction.js
│   └── images/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_prediction.py
│
├── reports/
│   ├── EDA_REPORT.md
│   ├── FEATURE_ENGINEERING_REPORT.md
│   ├── MODEL_EXPERIMENTS.md
│   ├── MODEL_SELECTION.md
│   └── LIMITATIONS.md
│
├── app.py
├── streamlit_app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── config.yaml
├── README.md
└── .gitignore
```

Keep training scripts, model artifacts, inference services, and web routes separate. Avoid unnecessary abstractions: modularity should make the project easier to understand, not harder.

---

# 5. Extensive exploratory data analysis (EDA)

EDA is a major deliverable. Do not limit it to `describe()` and a few histograms.

Create `02_Extensive_EDA.ipynb`, save important plots to `artifacts/plots/eda/`, and write interpretations in `reports/EDA_REPORT.md`.

Every important plot must have:
- A meaningful title.
- Clearly labelled axes and legends.
- Readable text and appropriate formatting.
- A short interpretation.
- An explanation of why the finding may matter for modeling.

## A. Dataset overview
Investigate:
- Row and column counts.
- Data types and memory usage.
- Unique values and cardinality.
- Missing values and invalid markers.
- Exact duplicate rows.
- Unique patients and encounter identifiers.
- Numerical and categorical columns.

## B. Target distribution
- Plot the original three-class `readmitted` distribution.
- Plot the binary target distribution.
- Show counts, percentages, and class ratio.
- Calculate majority-class baseline accuracy.
- Explain why accuracy alone is misleading for an imbalanced target.

## C. Demographic analysis
Explore age, gender, and race distributions and their observed 30-day readmission rates. Use suitable count plots, rate plots, and confidence intervals where useful.

## D. Clinical analysis
Explore:
- Primary, secondary, and tertiary diagnoses.
- Number of diagnoses.
- Laboratory procedure count.
- Medication count.
- HbA1c test result.
- Glucose serum result.
- Insulin and diabetes medication fields.
- Medication changes.

## E. Encounter analysis
Explore:
- Length of stay.
- Admission type.
- Discharge disposition.
- Admission source.
- Prior inpatient, outpatient, and emergency visits.
- Number of procedures.

## F. Missingness
Investigate missing values and `?` markers, especially in:
- `weight`
- `payer_code`
- `medical_specialty`
- `race`
- `diag_1`, `diag_2`, `diag_3`
- `A1Cresult`
- `max_glu_serum`

Create missing-value counts, percentages, plots, and missingness-pattern analysis. Investigate whether the fact that a test was not performed could itself be informative. Do not automatically mean-impute or delete every column.

## G. Relationships and statistics
Use:
- Numerical correlation heatmaps for suitable numerical variables.
- Categorical readmission-rate comparisons.
- Diagnosis-group comparisons.
- Medication and utilization comparisons.
- Chi-square tests for categorical associations where assumptions are met.
- Mann–Whitney U tests for skewed numerical comparisons where appropriate.
- Effect sizes and confidence intervals where practical.

Do not interpret statistical association as causation. Note multiple-testing concerns when many hypotheses are tested.

## H. EDA conclusions
The report must summarize:
1. Dataset characteristics.
2. Target imbalance.
3. Important missingness patterns.
4. Potentially useful features.
5. Potential leakage risks.
6. Candidate transformations.
7. Limitations and modeling implications.

---

# 6. Data cleansing and preprocessing

Document every important cleaning decision in code comments and `FEATURE_ENGINEERING_REPORT.md`.

## Missing values
Investigate NaN, `?`, empty strings, and invalid categories.

Compare reasonable options:
- Numerical median imputation.
- Mean imputation only where justified.
- Categorical most-frequent imputation.
- Explicit `Unknown` category.
- Missingness indicators.
- Excluding a high-missingness feature if evidence supports exclusion.

For each decision, explain the data-quality and modeling rationale. Do not treat a missing test result as equivalent to a normal result.

## Duplicates and repeated patients
- Remove exact duplicate rows only when justified.
- Do not assume repeated `patient_nbr` values are duplicate encounters.
- A patient may have multiple legitimate encounters.
- Retain patient identifiers temporarily for group-aware splitting, then exclude identifiers from predictive features.

## Identifiers
Investigate `encounter_id` and `patient_nbr`.
- Do not use either as an ordinary predictive feature.
- Use `patient_nbr` only for grouping/splitting and audit checks.
- Do not let the same patient appear in both train and test partitions when using a patient-grouped evaluation design.

## Categorical mapping
Use `IDs_mapping.csv` to decode admission type, discharge disposition, and admission source where mappings are available and verified. Preserve unknown or unmapped values explicitly; do not invent meanings for codes.

## Diagnosis processing
Diagnosis fields contain ICD-9 codes and may include special or non-numeric values. Do not treat ICD codes as continuous measurements. Build a reusable diagnosis-grouping utility for all three diagnosis columns. If using broad ICD-9 groupings, document the mapping rules and their limitations.

---

# 7. Feature engineering

Create separate modules for age, diagnosis, medication, utilization, and general feature building.

For every engineered feature, document:
1. Original columns.
2. Transformation and formula.
3. Reason for the transformation.
4. Potential benefit.
5. Assumptions and limitations.

Explore the following candidates; retain them only if they are valid and useful.

## Age
The original `age` field uses ranges such as `[50-60)`. Compare:
- Range midpoint.
- Ordinal age encoding.
- Original age categories.

Do not imply that a range midpoint is an exact patient age.

## Prior utilization
Example:

```python
total_prior_visits = (
    number_outpatient
    + number_emergency
    + number_inpatient
)
```

Also investigate prior inpatient use, prior emergency use, and high-utilization indicators. Ensure that the underlying fields are available at the intended prediction time.

## Medication
Explore:
- Total medication count.
- Diabetes medication indicator.
- Insulin status.
- Medication-change indicator.
- Number of diabetes medications changed, if the source fields support a defensible definition.

Do not assign clinical meaning beyond what the source fields support.

## Diagnoses
Explore:
- Primary, secondary, and tertiary diagnosis groups.
- Number of distinct diagnosis groups.
- Diabetes-related diagnosis indicator, if reliably derived.
- Presence of multiple diagnosis categories.

## Laboratory fields
Explore:
- HbA1c result category.
- Glucose serum result category.
- Test-performed indicators.
- Missingness indicators.

These fields are categorical results. Do not fabricate numeric HbA1c or glucose values.

## Length of stay and encounter features
Explore raw length of stay, clinically interpretable bins, procedure count, lab count, diagnosis count, and medication count. Choose bins based on transparent domain reasoning or observed distributions, not on test-set performance.

## Prediction-time and leakage rule
Define the intended prediction point as **discharge**. Only use information that would genuinely be available at that time.

Review every candidate variable for temporal availability and leakage. Be especially careful with discharge disposition and any field that may encode an outcome or post-discharge event. Do not assume a field is safe merely because it appears in the dataset. Record included, excluded, and uncertain variables in the feature engineering report. If the intended deployment point changes, revisit the feature set.

---

# 8. Preprocessing strategies

Use Scikit-learn `Pipeline` and `ColumnTransformer` where practical.

Compare:
- StandardScaler.
- MinMaxScaler.
- RobustScaler.
- No scaling for tree-based models.
- OneHotEncoder for nominal categorical features.
- OrdinalEncoder only for genuinely ordered categories.
- Frequency encoding only as a documented experiment where suitable.

The pipeline must:
1. Fit transformations on training data only.
2. Apply identical transformations to validation and test data.
3. Handle unseen categories.
4. Handle missing values.
5. Preserve transformed feature names for explainability.
6. Be serializable and reusable at inference time.

Do not preprocess the full dataset before splitting. All learned preprocessing must be fitted inside the training workflow.

---

# 9. Class imbalance experiments

Treat imbalance as a central modeling challenge.

Compare:
1. Original class distribution without resampling.
2. Class weights (`class_weight="balanced"` where supported).
3. Tuned class weights.
4. XGBoost `scale_pos_weight`.
5. RandomOverSampler.
6. SMOTE, where appropriate.
7. SMOTENC for mixed numerical/categorical data, where appropriate.
8. Random undersampling as an efficiency/performance experiment.

Mandatory safeguards:
- Never resample before train/test splitting.
- Never resample validation or test data.
- Apply resampling only inside the training folds/pipeline.
- Compare strategies on the same untouched evaluation set.
- Report precision and recall together.
- Do not assume a more balanced training set is automatically better.

Create an imbalance experiment report that includes minority-class recall, precision, F1, PR-AUC, and false-positive/false-negative counts.

---

# 10. Model development and experimentation

Train and compare suitable algorithms. At minimum:
- Logistic Regression as a baseline.
- Decision Tree.
- Random Forest.
- XGBoost.
- Extra Trees.

Optionally investigate HistGradientBoosting or CatBoost if installation, runtime, and deployment compatibility are reasonable.

Explore suitable parameters, such as:
- Tree depth.
- Number of estimators.
- Minimum samples per split/leaf.
- Feature subsampling.
- Learning rate.
- Subsample and column sampling.
- Regularization.
- Class weights or positive-class weighting.

Use:
- `RandomizedSearchCV` for broader search spaces.
- `GridSearchCV` for smaller, targeted spaces.
- Group-aware validation when possible.

## Group-aware evaluation

Because patients may have multiple encounters, use a patient-aware split. Investigate `GroupShuffleSplit` or a suitable stratified group splitting method. If a chosen method cannot preserve class balance adequately, document the compromise and quantify the resulting class distributions.

Use a clear training/validation/test strategy:
- Keep a final test partition untouched until model and threshold decisions are complete.
- Use training-only cross-validation for tuning.
- Do not use test results to select features, tune hyperparameters, choose resampling, or choose the threshold.

If the project uses a temporal split as an additional robustness check, explain its purpose and do not mix it with the primary test evaluation without clearly labelling the results.

## Experiment tracking

Save each experiment's:
- Model and version/configuration.
- Feature set.
- Preprocessing configuration.
- Imbalance strategy.
- Hyperparameters.
- Cross-validation results.
- Validation metrics.
- Training and inference time.

Generate `artifacts/evaluation/model_comparison.csv` with:
- Model name.
- Accuracy.
- Precision.
- Recall/sensitivity.
- F1-score.
- Specificity.
- ROC-AUC.
- PR-AUC.
- Brier score if calculated.
- Training time.
- Inference time.
- Experiment identifier and configuration.

Do not hardcode a winner or fabricate results.

---

# 11. Evaluation, threshold analysis, and calibration

Accuracy must not be the only or primary selection criterion.

Report:
- Accuracy.
- Precision.
- Recall/sensitivity.
- F1-score.
- Specificity.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.
- Brier score and calibration analysis where appropriate.

Generate:
- Confusion matrix heatmap.
- ROC curves.
- Precision-recall curves.
- Model comparison charts.
- Classification reports.
- Threshold-versus-precision/recall/F1/specificity plots.
- Calibration curves.

## Threshold selection
Do not automatically use 0.5. Explore thresholds using validation data only. Show the trade-off between false negatives, false positives, recall, precision, specificity, and F1. Select a threshold according to a documented research objective, not by looking at test-set performance.

Do not call a threshold clinically optimal without clinical validation.

## Calibration
Investigate probability calibration using calibration curves and Brier score. Consider `CalibratedClassifierCV` if appropriate. Any calibration step must be fitted without test data leakage.

## Final model selection report
Answer with actual evidence:
- Which model performed best on the chosen objective?
- Which detected the most positive cases?
- Which achieved the strongest PR-AUC?
- Which was most efficient?
- Did resampling help?
- Was calibration useful?
- Why was the final model selected?

Distinguish validation-based model selection from final test-set reporting.

---

# 12. SHAP explainability

Explainability is a core feature, not an optional extra.

## Global explanations
Generate:
- SHAP summary/beeswarm plot.
- SHAP feature-importance bar chart.
- Dependence plots for important features.
- Interaction analysis only where computationally feasible and interpretable.

## Local explanations
For each prediction, show:
- Main features increasing the model output.
- Main features decreasing the model output.
- SHAP contribution values.
- Direction and relative magnitude.
- A concise plain-language summary.

Use a waterfall plot or a clear equivalent.

## Implementation
Create:
- `src/explainability/shap_explainer.py`
- `src/explainability/global_explanation.py`
- `src/explainability/local_explanation.py`

Use a suitable explainer, such as TreeExplainer for compatible tree models. Ensure that SHAP is calculated for the actual selected model and that transformed feature names map back to meaningful inputs wherever technically feasible.

Be precise about SHAP output:
- Explain the model output scale (raw score, log-odds, or probability) used by the explainer.
- Do not label SHAP values as probability percentage-point changes unless the explanation is actually on a probability scale.
- Check the positive-class output index for binary classifiers.
- Validate that the sum of contributions plus the base value corresponds to the explained model output within numerical tolerance.
- If one-hot encoded features are grouped for display, document how the grouping is performed.

SHAP indicates how features contribute to a model output. It does not prove causation, establish a biological mechanism, or show that changing a feature will change a patient's outcome. Do not provide treatment advice.

---

# 13. Model persistence and prediction service

Persist:
- Final trained model or complete fitted pipeline.
- Selected threshold.
- Feature schema.
- Model metadata/version.
- Evaluation metrics.
- SHAP background data or explainer assets, if needed.
- Training configuration.

Use Joblib where appropriate. Load artifacts once when the application starts, not on every request.

Create a framework-independent service:

```python
def predict_readmission(patient_data):
    """
    Validate patient input.
    Apply the saved feature-building and preprocessing workflow.
    Generate a positive-class probability.
    Apply the saved decision threshold.
    Generate a local explanation.
    Return structured results and model metadata.
    """
```

Return a structured result containing:
- Predicted class.
- Predicted probability.
- Selected threshold.
- Clearly defined display category, if used.
- Top positive SHAP contributions.
- Top negative SHAP contributions.
- Model version.

Any probability bands or labels such as “low”, “moderate”, or “high” must be explicitly described as project-defined display bands, not clinically validated categories.

The inference transformation must exactly match the training transformation. Avoid maintaining separate, drifting preprocessing code for training and prediction.

---

# 14. Flask UI/UX requirements

Build a polished healthcare analytics interface—not a generic college-project template.

## Visual direction
Use a modern clinical analytics aesthetic:
- Deep navy/dark blue.
- Teal/cyan accents.
- White and light-grey surfaces.
- Subtle gradients.
- Rounded cards.
- Soft shadows.
- Consistent spacing and typography.
- Responsive layouts.
- Subtle transitions, not distracting animation.

## Navigation
Provide:
1. Overview Dashboard.
2. Patient Risk Prediction.
3. Prediction Explanation.
4. Population Analytics.
5. Model Performance.
6. Feature Insights.
7. About the Project.

## Page 1 — Overview dashboard
Include:
- Project title and concise objective.
- Dataset description and source.
- Key dataset statistics.
- Target distribution.
- Model performance KPIs loaded from saved evaluation artifacts.
- A “Why 30-Day Readmission Prediction Matters” section.

Never hardcode example performance values as if they were measured results. If artifacts do not exist, show a clear “Train the model to populate this section” state.

## Page 2 — Patient prediction
Organize the form into cards or accordions.

### Demographics
- Age category.
- Gender.
- Race.

### Admission and encounter
- Admission type.
- Admission source.
- Discharge disposition.
- Length of stay.

### Clinical information
- Primary, secondary, tertiary diagnosis inputs or supported diagnosis groups.
- Number of diagnoses.
- Number of procedures.
- Number of laboratory procedures.

### Laboratory results
- HbA1c result category.
- Glucose serum result category.

### Medication
- Number of medications.
- Diabetes medication fields.
- Insulin status.
- Medication change.

### Previous healthcare utilization
- Prior inpatient visits.
- Prior outpatient visits.
- Prior emergency visits.

Only expose inputs that the trained model actually expects. The form schema must be generated from or validated against the saved feature schema. Do not create form fields that are silently ignored.

Requirements:
- Clear labels and tooltips.
- Valid ranges and category options based on the source dataset.
- Explicit “Unknown/not recorded” options where appropriate.
- Reset form.
- Demo patient using clearly labelled illustrative values—not a real patient.
- Validation and helpful errors.
- No default values that falsely imply normal clinical measurements.

Add a prominent **Predict 30-Day Readmission Risk** button.

On submission:
1. Validate input.
2. Build the expected input representation.
3. Run the saved inference pipeline.
4. Generate positive-class probability.
5. Apply the saved threshold.
6. Generate a local SHAP explanation.
7. Display results without a full-page refresh where practical.

Display:
- Predicted class.
- Probability percentage.
- Selected threshold.
- Project-defined display band, if used.
- Probability gauge.
- Factors increasing the model output.
- Factors decreasing the model output.

Label the result as an experimental model estimate, not a diagnosis or clinical decision.

## Page 3 — SHAP explanation
Show:
- Local waterfall plot.
- Top positive contributions.
- Top negative contributions.
- Contribution values and output scale.
- Plain-language interpretation.
- A short explanation of SHAP limitations.

Populate all content from the actual prediction result.

## Page 4 — Population analytics
Use the authentic dataset and correct denominators.

Include:
- Readmission distribution.
- Age-wise rates.
- Gender-wise rates.
- Race-wise rates.
- Admission type and discharge disposition.
- Length-of-stay analysis.
- Diagnosis-group analysis.
- Insulin and medication-change analysis.
- Prior utilization analysis.

Provide practical filters where possible. Display subgroup sample sizes and flag small groups. Do not imply that descriptive subgroup differences are causal.

## Page 5 — Model performance
Show:
- Selected model.
- Actual saved metrics.
- Confusion matrix.
- ROC curve.
- Precision-recall curve.
- Calibration curve.
- Threshold analysis.
- Model comparison across all completed experiments.
- “Why this model was selected” based on the saved selection report.

## Page 6 — Feature insights
Show:
- Global SHAP importance.
- Top features.
- Dependence plots.
- Relevant distributions.
- Clear distinction between predictive association and causation.

## Page 7 — About
Include:
- Problem statement.
- Dataset and target definition.
- Workflow.
- Models explored.
- Imbalance strategy.
- SHAP approach.
- Technology stack.
- Limitations and references.

Display this disclaimer prominently:

> This project is an academic machine-learning research prototype using historical, de-identified hospital encounter data from 1999–2008. Predictions are not clinically validated and must not be used for diagnosis, treatment, discharge decisions, or patient management. The model is intended for educational and research purposes only.

---

# 15. Flask backend

Keep routes lightweight and delegate logic to services.

Suggested routes:

```text
GET  /
GET  /dashboard
GET  /predict
POST /api/predict
GET  /explanation
GET  /analytics
GET  /model-performance
GET  /feature-insights
GET  /about
```

Requirements:
- JSON API response for prediction.
- Server-side input validation.
- Safe artifact loading.
- Clear error handling and logging.
- No stack traces or internal file paths in user-facing production errors.
- No duplicate inference logic in routes.
- Use appropriate request-size and input validation limits.
- Add CSRF protection for state-changing form submissions where applicable.
- Do not expose patient inputs in URLs or routine logs.

---

# 16. Streamlit Cloud compatibility

Create a separate `streamlit_app.py` that reuses:
- Prediction service.
- Input validator.
- Risk interpreter.
- Saved model artifacts.
- SHAP utilities.

Keep Flask and Streamlit presentation code separate. Do not fork or duplicate the ML pipeline.

Document:

```bash
# Flask
python app.py

# Streamlit
streamlit run streamlit_app.py
```

Use relative paths and environment-based configuration. Avoid relying on machine-specific absolute paths. Keep large raw datasets and unnecessary training artifacts out of the deployment bundle where possible. Explain any artifact hosting or repository-size constraints in the README.

---

# 17. Reports and documentation

Create a thorough `README.md` containing:
1. Project overview.
2. Problem statement.
3. Dataset source and description.
4. Target definition.
5. Architecture.
6. Setup and installation.
7. Dataset acquisition.
8. EDA summary.
9. Cleansing decisions.
10. Feature engineering.
11. Preprocessing.
12. Imbalance experiments.
13. Model experiments and tuning.
14. Evaluation and threshold selection.
15. SHAP explainability.
16. Flask execution.
17. Streamlit execution and deployment.
18. Tests.
19. Limitations.
20. Future improvements.
21. References.

Create:
- `EDA_REPORT.md`
- `FEATURE_ENGINEERING_REPORT.md`
- `MODEL_EXPERIMENTS.md`
- `MODEL_SELECTION.md`
- `LIMITATIONS.md`

Update reports from actual experiment outputs. Do not write conclusions or metrics before the experiments have run.

---

# 18. Software engineering requirements

- Use functions/classes where appropriate.
- Add docstrings and type hints where useful.
- Use centralized configuration.
- Use logging rather than excessive `print()` calls.
- Use reproducible random seeds.
- Avoid hardcoded absolute paths.
- Avoid duplicated transformations.
- Keep training separate from inference.
- Pin or constrain compatible dependencies in `requirements.txt`.
- Add tests for data loading, feature building, preprocessing, model loading, and prediction.
- Make artifacts reproducible.
- Handle missing datasets and model artifacts gracefully.
- Ensure a fresh environment can run the documented workflow.

---

# 19. Mandatory precautions and scientific safeguards

These requirements are **non-negotiable**.

## A. Dataset integrity
1. Use the authentic UCI dataset.
2. Never silently replace it with synthetic data.
3. Validate the downloaded file and expected schema.
4. Preserve the original raw file unchanged.
5. Record dataset source, retrieval date, and any version or checksum information available.

## B. Target and leakage
6. Only `<30` is positive; `>30` and `NO` are negative.
7. Drop the original `readmitted` field from model inputs after target creation.
8. Never use target-derived information as a feature.
9. Do not use `patient_nbr` or `encounter_id` as ordinary predictive features.
10. Use patient identifiers only for grouping/audit purposes, then exclude them from model inputs.
11. Review each feature for availability at the stated prediction time.
12. Investigate possible leakage from discharge-related and outcome-proxy fields. Document decisions instead of assuming all columns are safe.

## C. Splitting and validation
13. Use patient-aware splitting to reduce the risk of the same patient's encounters appearing in both training and test partitions.
14. Keep a final test set untouched until model selection and threshold selection are complete.
15. Fit imputers, encoders, scalers, feature selectors, and other learned transformations on training data only.
16. Perform hyperparameter tuning using training/validation data only.
17. If group-aware stratification is difficult, document the compromise and report class proportions in every partition.
18. Do not repeatedly inspect test performance and then modify the model.

## D. Imbalance handling
19. Never run SMOTE or oversampling before splitting.
20. Never resample validation or test data.
21. Put resampling inside training folds/pipelines.
22. Compare resampling against class-weighted and unmodified baselines.
23. Evaluate precision and recall together; high recall alone can come with an unacceptable false-positive rate.

## E. Evaluation and claims
24. Do not optimize or report accuracy alone.
25. Report confusion matrix, precision, recall, specificity, F1, ROC-AUC, and PR-AUC.
26. Use validation data—not the test set—to choose the classification threshold.
27. Investigate probability calibration before presenting probabilities as risk estimates.
28. Never fabricate, round selectively, or hardcode model metrics.
29. Clearly label validation metrics versus final test metrics.
30. Do not claim clinical utility, clinical validation, or improved patient outcomes based only on retrospective model metrics.

## F. Explainability
31. SHAP values are model explanations, not causal effects.
32. State whether SHAP explains raw scores, log-odds, or probabilities.
33. Verify the positive-class output index and contribution-sum consistency.
34. Do not describe a SHAP contribution as a percentage-point probability change unless it is genuinely on that scale.
35. Do not make treatment recommendations from feature contributions.
36. Do not claim that changing a feature will necessarily change the outcome.

## G. Healthcare, privacy, and fairness
37. Treat the model as a research prototype, not a diagnostic or clinical decision tool.
38. Do not use it to recommend treatment, deny care, determine discharge, or prioritize real patients.
39. Do not upload real patient-identifiable data to public services or external APIs.
40. Do not log submitted clinical values or expose them in URLs.
41. The source dataset is historical and may not represent current clinical practice or other hospitals.
42. Examine subgroup performance where sample sizes permit, including recall, false-positive rates, and calibration.
43. Discuss potential bias and limitations associated with race, gender, age, hospital practices, and historical data collection.
44. Do not interpret demographic feature importance as biological causation.
45. Display sample sizes and caution for small subgroups.
46. Do not present the interface as HIPAA-compliant or production-ready without a proper security, privacy, and compliance assessment.

## H. Engineering and reproducibility
47. Do not load untrusted pickle/joblib files. Only load artifacts produced and controlled by this project.
48. Keep secrets and credentials out of source control.
49. Use safe input validation and error handling.
50. Ensure inference uses exactly the same feature transformations as training.
51. Check for dependency compatibility, especially between SHAP, NumPy, Scikit-learn, and XGBoost.
52. Test unseen categories, missing values, invalid numeric ranges, and missing artifacts.
53. Keep the UI honest: if a model or evaluation artifact is missing, show an informative empty state rather than placeholder results.
54. Make all charts and displayed metrics derive from actual saved data or model artifacts.

---

# 20. Phased execution plan

Do not generate the entire project in one unstructured operation. Work phase by phase and verify each phase before proceeding.

## Phase 1 — Initialization
- Inspect the workspace before modifying it.
- Present a concise implementation plan.
- Create folders and configuration.
- Set up logging and dependencies.
- Implement data acquisition and validation.

**Deliverable:** Project skeleton and authentic dataset available.

## Phase 2 — EDA
- Build the comprehensive EDA notebook.
- Perform demographic, clinical, encounter, missingness, and target analysis.
- Save visualizations.
- Generate the EDA report.

**Deliverable:** Reproducible EDA and documented findings.

## Phase 3 — Cleansing and feature engineering
- Implement cleansing modules.
- Map categorical codes.
- Investigate duplicates.
- Build diagnosis, medication, utilization, age, and encounter features.
- Document every important decision.

**Deliverable:** Reusable and documented data preparation workflow.

## Phase 4 — Preprocessing and splitting
- Compare encoders and scalers.
- Build pipelines.
- Implement patient-aware splitting.
- Validate leakage safeguards.

**Deliverable:** Reusable training and inference preprocessing.

## Phase 5 — Model experiments
- Train baselines and candidate models.
- Compare imbalance strategies.
- Tune selected candidates.
- Save all configurations and results.

**Deliverable:** Experiment results and comparison report.

## Phase 6 — Evaluation and selection
- Evaluate validation performance.
- Select threshold using validation data.
- Investigate calibration.
- Freeze the final model.
- Evaluate once on the untouched test set.
- Write the selection report.

**Deliverable:** Final model and defensible evaluation.

## Phase 7 — SHAP
- Generate global explanations.
- Implement local explanations.
- Validate output scale and contribution consistency.
- Save plots and explanation utilities.

**Deliverable:** Working global and local explainability.

## Phase 8 — Flask application
- Build the responsive dashboard.
- Build the patient form.
- Integrate the shared prediction service.
- Add local SHAP explanations.
- Add population analytics and model performance pages.
- Test all routes and interactions.

**Deliverable:** Working Flask application.

## Phase 9 — Tests and Streamlit preparation
- Test preprocessing and predictions.
- Test edge cases and artifact loading.
- Build Streamlit UI reusing the same services.
- Complete deployment instructions.
- Run a clean-environment verification.

**Deliverable:** Tested project ready for demonstration and deployment preparation.

---

# 21. How Antigravity should operate

1. **Inspect before creating:** Inspect the current workspace and preserve useful existing files.
2. **Plan before implementation:** Present the architecture and phase plan before making broad changes.
3. **Implement incrementally:** Complete and verify one phase at a time.
4. **Do not create placeholder functionality:** Buttons, charts, routes, forms, and explanations must work.
5. **Execute the code:** Run scripts and tests where possible; fix errors rather than leaving broken code.
6. **Verify artifacts:** Load the saved model independently and test inference.
7. **Verify the UI:** Run the application and inspect each page, route, chart, and form.
8. **Maintain documentation:** Record decisions as they are made.
9. **Preserve reproducibility:** A fresh environment must be able to follow the README.
10. **Explain decisions:** Explain why each meaningful cleansing, encoding, feature, imbalance, tuning, and model-selection decision was made.
11. **Be transparent about limitations:** If compute or environment constraints prevent a requested experiment, state what was not run and why. Do not pretend it succeeded.
12. **Ask before destructive actions:** Do not delete existing project files, datasets, or artifacts without checking whether they are needed.

---

# 22. Final acceptance checklist

The project is complete only when:

- [ ] Authentic UCI dataset acquired and validated.
- [ ] Binary target correctly defined.
- [ ] Extensive EDA completed with saved plots.
- [ ] EDA report written from actual findings.
- [ ] Missing-value decisions documented.
- [ ] Duplicate and identifier handling documented.
- [ ] Leakage review completed.
- [ ] Feature engineering modularized and explained.
- [ ] Preprocessing pipeline implemented.
- [ ] Patient-aware split implemented and documented.
- [ ] Multiple models trained and compared.
- [ ] Imbalance strategies compared without leakage.
- [ ] Hyperparameter tuning completed.
- [ ] Threshold analysis completed using validation data.
- [ ] Calibration investigated.
- [ ] Final model selected using documented evidence.
- [ ] Final test metrics reported once on untouched test data.
- [ ] SHAP global explainability implemented.
- [ ] SHAP local explainability implemented and validated.
- [ ] Model and preprocessing artifacts saved.
- [ ] Prediction service works independently of Flask.
- [ ] Flask application runs successfully.
- [ ] Patient prediction form uses the real model schema.
- [ ] Prediction probability and threshold are displayed.
- [ ] Local SHAP explanation is displayed.
- [ ] Population analytics use real data and correct denominators.
- [ ] Model performance dashboard uses actual artifacts.
- [ ] Responsive UI verified.
- [ ] Streamlit interface reuses shared services.
- [ ] Tests pass.
- [ ] README and technical reports are complete.
- [ ] Clinical and research limitations are clearly displayed.

---

# 23. Final instruction

Build this as a serious, well-researched, end-to-end Machine Learning product rather than a typical academic classification assignment.

Prioritize:

1. **Data science quality:** extensive EDA, defensible cleansing, and meaningful feature engineering.
2. **ML rigor:** patient-aware evaluation, leakage prevention, fair experiments, class-imbalance analysis, calibration, and justified model selection.
3. **Explainable AI:** validated SHAP explanations for global behavior and individual predictions.
4. **Software engineering:** modular code, reusable services, reproducible artifacts, tests, and documentation.
5. **UI/UX quality:** an impressive, interactive, responsive healthcare analytics dashboard.

Do not sacrifice scientific correctness for attractive metrics or visual design. Do not stop after creating folders or placeholder pages. Implement, execute, debug, and verify the project.

**Start by inspecting the workspace, then present the implementation plan. Proceed phase by phase, reporting completed work, test results, and any limitations before moving to the next phase.**
