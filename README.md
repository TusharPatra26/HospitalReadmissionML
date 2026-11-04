# Hospital Readmission Risk Predictor
### Explainable Machine Learning System for 30-Day Hospital Readmission Risk in Diabetic Inpatients

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-teal.svg)](app.py)
[![XAI: SHAP TreeExplainer](https://img.shields.io/badge/XAI-SHAP%20Waterfall-purple.svg)](src/explainability)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](tests/)

---

> **Academic & Research Disclaimer:**  
> This project is an academic machine-learning research prototype utilizing historical, de-identified electronic health records (1999–2008). Predictions are **not clinically validated** and must not be used for diagnosis, medical treatment, discharge timing, or clinical triage. The system is intended solely for educational and research investigation.

---

## 1. Project Overview & Clinical Problem Statement

Hospital readmission within 30 days of discharge is a major benchmark of healthcare quality and clinical coordination. Under the **Hospital Readmissions Reduction Program (HRRP)** established by the Centers for Medicare & Medicaid Services (CMS), hospitals face severe financial reimbursement penalties for excessive 30-day readmissions.

In patients diagnosed with diabetes mellitus, readmission risks are heightened by:
- Acute inpatient medication adjustments and insulin titration.
- Complex comorbidity burdens (cardiovascular disease, renal impairment, diabetic ketoacidosis).
- Post-discharge care transition breakdowns and polypharmacy confusion.

This application provides an **end-to-end, modular, and explainable Machine Learning pipeline** that estimates the probability of a 30-day readmission at the **intended clinical prediction point (discharge)**, providing personalized **SHAP (SHapley Additive exPlanations)** feature attributions to support transitional care coordination.

---

## 2. Dataset Pedigree & Cardinality Audit

- **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Cohort Volume:** 101,766 hospital encounters across 130 US hospitals.
- **Unique Patients (`patient_nbr`):** 71,518 unique patients (mean of 1.423 encounters per patient; up to 40 repeat admissions).
- **Target Variable:** Derived binary target `readmitted_30_days`:
  - `1` (Readmitted $<30$ days): **11,357 encounters (11.16%)**
  - `0` (Not readmitted $<30$ days; includes `>30` and `NO`): **90,409 encounters (88.84%)**
- **Imbalance Dynamics:** ~8:1 negative-to-positive ratio (baseline majority-class accuracy: **88.84%**).

---

## 3. Project Structure

The Flask routes are separate from the reusable data, modeling, and prediction code:

```text
Hospital-Readmission-Predictor/
├── data/
│   ├── raw/                 # Source CSV files
│   ├── interim/             # Optional intermediate data
│   └── processed/           # Optional processed data
├── notebooks/
│   ├── 1. EDA HOSPITAL READMISSION.ipynb
│   └── 2. MODEL TRAINING.ipynb
├── src/
│   ├── data_ingestion/      # Dataset loading and ID mappings
│   ├── data_validation/     # Input schema validation
│   ├── cleansing/           # Missing values, duplicates, and outliers
│   ├── preprocessing/       # Encoders, scalers, and model pipeline
│   ├── feature_engineering/ # Clinical and utilization features
│   ├── models/              # Model training and artifact loading
│   ├── evaluation/          # Metrics and threshold analysis
│   ├── explainability/      # SHAP explanations
│   ├── prediction/          # Validation and inference service
│   └── utils/               # Configuration, logging, and file helpers
├── artifacts/
│   ├── models/              # Tracked inference model, preprocessing pipeline, and metadata
│   ├── evaluation/          # Evaluation metrics and model comparison
│   └── plots/               # EDA, training, and SHAP figures
├── reports/                 # EDA, model, and limitation reports
├── static/                  # CSS, JavaScript, and images
├── templates/               # Flask/Jinja templates
├── tests/                   # pytest suite
├── app.py                   # Flask application
├── train.py                 # Model training pipeline
├── evaluate.py              # Model evaluation
├── config.yaml              # Project configuration
└── requirements.txt         # Python dependencies
```

The local virtual environment, Python caches, other generated model binaries, and intermediate datasets are excluded from version control. The model and preprocessing pipeline required for inference are tracked so hosted deployments can load them.

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation (Windows PowerShell)
```powershell
git clone https://github.com/TusharPatra26/HospitalReadmissionML.git
cd HospitalReadmissionML
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, create and activate the environment with:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Run the Flask app
```bash
python app.py
```
Open <http://127.0.0.1:5000>.

### Run tests
```bash
python -m pytest tests/ -q
```

### Train or evaluate the model
```bash
python train.py
python evaluate.py
```

The prediction app and prediction tests require generated model files in `artifacts/models/`. If they are not present in your checkout, run `python train.py` first. The training workflow requires the raw CSV files in `data/raw/`.

### Deploying on Render

The `.python-version` file pins Render to Python 3.12, which is within the versions supported by the dependency constraints in `requirements.txt`. Render needs `artifacts/models/best_model.pkl` and `artifacts/models/preprocessing_pipeline.pkl` in the deployed checkout; keep these tracked files in Git when updating the model. If your Render service has a `PYTHON_VERSION` environment variable set, remove it or set it to a Python 3.12 patch release; that setting takes precedence over `.python-version`. Push the model files and code changes, then trigger a new deploy.

---

## 5. Exploratory Data Analysis (EDA) Highlights

An exhaustive analysis of all 101,766 encounters revealed:
1. **Prior Utilization is the Dominant Factor:** Readmitted patients had more than double the prior inpatient admissions (mean **1.22 vs 0.56 visits**, Mann-Whitney $U = 4.04 \times 10^8, p < 10^{-300}$). Encounters with $\ge 3$ prior inpatient admissions exhibit readmission rates exceeding **27.5%**.
2. **Insulin Titration Indication:** Patients with active insulin dose adjustments (`Up` or `Down`) had higher readmission rates (**12.5% vs 10.1%** without insulin, $\chi^2 = 190.86, p = 3.98 \times 10^{-41}$).
3. **Gender Invariance:** Readmission rates between females (**11.23%**) and males (**11.08%**) show no statistically significant difference ($\chi^2 = 1.24, p = 0.539$).
4. **Information-Bearing Missingness:** HbA1c was not ordered in 83.3% of encounters; unrecorded status reflects physician order intent and is preserved as a first-class categorical level.

*Full details available in [`reports/EDA_REPORT.md`](reports/EDA_REPORT.md).*

---

## 6. Data Cleansing & Leakage Prevention Safeguards

| Decision | Implementation | Clinical & Statistical Rationale |
| :--- | :--- | :--- |
| **Terminal Discharge Exclusion** | Excluded `discharge_disposition_id` $\in \{11, 13, 14, 19, 20, 21\}$ (Expired / Hospice). | Deceased and hospice patients cannot be acutely readmitted; retaining them introduces biological survivor bias and post-event outcome leakage. |
| **High Missingness Drop** | `weight` dropped completely (96.86% missing). | Imputing 97% of missing values introduces severe noise without clinical reliability. |
| **Informative Missingness** | `medical_specialty` & `payer_code` retained `'Missing/Unknown'`. | Unassigned specialty represents non-specialized community hospital admission. |
| **Patient-Aware Splitting** | `GroupShuffleSplit` on `patient_nbr` (70% train / 15% val / 15% test). | Prevents repeat patient encounters from appearing across both training and evaluation partitions (0% patient overlap). |
| **Resampling Isolation** | Oversampling/SMOTE applied **strictly inside training folds**. | Validation and test partitions remain untouched with natural class prevalence (~11.3%). |

*Full details available in [`reports/FEATURE_ENGINEERING_REPORT.md`](reports/FEATURE_ENGINEERING_REPORT.md).*

---

## 7. Model Experiments & Benchmark Comparison

Evaluated on the **Validation Partition (14,823 encounters, 11.32% positive prevalence)**:

| Model Name | Imbalance Treatment | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | Brier Score | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Unweighted)** | **None (Natural)** | **88.67%** | 46.15% | 0.72% | **99.89%** | 0.0141 | **0.6724** | **0.2294** | **0.0953** | 6.17s |
| **XGBoost (scale_pos_weight)** | Cost-Sensitive ($w=7.72$) | 66.24% | 18.34% | 57.39% | 67.37% | 0.2779 | 0.6713 | 0.2256 | 0.2146 | 2.06s |
| **HistGradientBoosting** | Cost-Sensitive (`balanced`) | 65.45% | 18.03% | 57.87% | 66.42% | 0.2750 | 0.6703 | 0.2265 | 0.2191 | 4.00s |
| **Random Forest (Balanced)** | Cost-Sensitive (`balanced`) | 71.99% | 19.59% | 47.50% | 75.12% | 0.2774 | 0.6647 | 0.2194 | 0.1995 | 3.89s |
| **Extra Trees (Balanced)** | Cost-Sensitive (`balanced`) | 67.31% | 18.32% | 54.59% | 68.93% | 0.2743 | 0.6657 | 0.2251 | 0.2143 | 6.01s |
| **Logistic Regression (Balanced)** | Cost-Sensitive (`balanced`) | 66.30% | 17.69% | 54.11% | 67.85% | 0.2666 | 0.6624 | 0.2147 | 0.2259 | 2.41s |
| **Logistic Regression (Baseline)** | None (Unweighted) | 88.69% | **51.92%** | 1.61% | 99.81% | 0.0312 | 0.6621 | 0.2154 | 0.0962 | 1.93s |
| **Random Forest + ROS** | RandomOverSampler | 69.63% | 19.26% | 52.74% | 71.78% | 0.2822 | 0.6675 | 0.2204 | 0.2067 | 7.43s |
| **Random Forest + RUS** | RandomUnderSampler | 63.08% | 17.19% | **59.24%** | 63.58% | 0.2665 | 0.6634 | 0.2173 | 0.2279 | 0.86s |
| **Decision Tree (Balanced)** | Cost-Sensitive (`balanced`) | 61.57% | 16.74% | 60.25% | 61.73% | 0.2620 | 0.6458 | 0.2081 | 0.2249 | 1.28s |
| **Random Forest + SMOTE** | SMOTE Resampling | 87.62% | 31.79% | 8.16% | 97.76% | 0.1299 | 0.6466 | 0.1957 | 0.1299 | 8.50s |

*Full analysis available in [`reports/MODEL_EXPERIMENTS.md`](reports/MODEL_EXPERIMENTS.md).*

---

## 8. Final Model Selection & Test Set Evaluation

**Selected Architecture:** `XGBoost (Unweighted)` with post-hoc validation threshold tuning ($T = 0.130$).

### Single Final Test Evaluation (Untouched Test Partition: 14,815 Encounters)

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.6859** | Substantial discrimination over random chance ($0.500$) |
| **PR-AUC** | **0.2463** | Over **2.21x higher precision** than baseline prevalence ($0.111$) |
| **Sensitivity (Recall)** | **52.37%** | Successfully flags **861 out of 1,644** true 30-day readmissions |
| **Specificity** | **73.84%** | Correctly clears **9,726 out of 13,171** stable encounters |
| **Precision** | **20.00%** | 1 in 5 flagged encounters is readmitted within 30 days |
| **F1-Score** | **0.2894** | Well-balanced operating point for post-discharge screening |
| **Brier Score** | **0.0929** | Excellent raw probability calibration |
| **Inference Latency** | **0.0028 ms** | Real-time response (<3 ms per 1,000 predictions) |

*Full analysis available in [`reports/MODEL_SELECTION.md`](reports/MODEL_SELECTION.md).*

---

## 9. Explainable AI (SHAP) Integration

Every prediction is explained via game-theoretic **TreeSHAP** in log-odds space:
- **Mathematical Consistency:** Base value ($\phi_0 = -2.057$) plus sum of SHAP attributions exactly equals model raw log-odds margin within numerical tolerance ($< 10^{-6}$).
- **Global Explanations:** Saved in `artifacts/plots/shap/` (feature importance bar chart, beeswarm plot, and dependence plots).
- **Local Explanations:** Real-time breakdown of top positive drivers (factors elevating risk) and top protective factors (factors attenuating risk), coupled with plain-language clinical interpretations.

---

## 10. Flask Web Interface

- Deep navy/teal clinical analytics aesthetic.
- Pages for the dashboard, risk prediction, SHAP explainability, population analytics, model performance, and project limitations.
- Pre-filled interactive Demo Personas (Low Risk, Moderate Risk, Complex High Utilizer).
- Real-time interactive Chart.js local waterfall charts.

---

## 11. Testing & Code Quality

The project includes an automated **pytest test suite** (29 tests at the time of writing):
- `tests/test_data_loader.py`: Ingestion, IDS mapping parsing, target derivation.
- `tests/test_features.py`: ICD-9 mapping, comorbidity indicators, polypharmacy, age midpoint.
- `tests/test_preprocessing.py`: Scalers, encoders, ColumnTransformer, patient-aware splitting.
- `tests/test_explainability.py`: TreeExplainer initialization, additive consistency, local explanations.
- `tests/test_prediction.py`: Framework-independent prediction service, validation, risk tiers.
- `tests/test_flask_app.py`: Flask route availability, JSON API endpoints.

Run the suite from the repository root:
```bash
python -m pytest tests/ -q
```

---

## 12. References

1. **Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014).** *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, 781670.
2. **Lundberg, S. M., et al. (2020).** *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56-67.
3. **Centers for Medicare & Medicaid Services (CMS).** *Hospital Readmissions Reduction Program (HRRP).* https://www.cms.gov/medicare/quality/initiatives/hospital-quality-initiative/hrrp
