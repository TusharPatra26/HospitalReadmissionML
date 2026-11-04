"""Streamlit Application for Hospital Readmission Risk Predictor.

Reuses the shared prediction service, validation, and SHAP explainability modules.
Designed for Streamlit Community Cloud and local execution.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "logs" / ".matplotlib_cache")

from src.prediction.prediction_service import PredictionService, predict_readmission
from src.utils.file_handler import load_json

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Healthcare Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .disclaimer-box { background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 12px 16px; border-radius: 6px; color: #92400E; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .metric-card { background: white; padding: 16px; border-radius: 10px; border: 1px solid #E2E8F0; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }
</style>
""", unsafe_allow_html=True)

# Cache prediction service
@st.cache_resource
def get_service():
    return PredictionService.get_instance()

service = get_service()

# Load metadata
@st.cache_data
def get_metadata():
    meta_path = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"
    eval_path = PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_metrics.json"
    comp_path = PROJECT_ROOT / "artifacts" / "evaluation" / "model_comparison.csv"
    meta = load_json(meta_path) if meta_path.is_file() else {}
    metrics = load_json(eval_path) if eval_path.is_file() else {}
    comp_df = pd.read_csv(comp_path) if comp_path.is_file() else pd.DataFrame()
    return meta, metrics, comp_df

metadata, test_metrics, comp_df = get_metadata()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
st.sidebar.title("Hospital Readmission")
st.sidebar.caption("Explainable Healthcare Machine Learning")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview Dashboard",
        "Patient Risk Predictor",
        "Explainable AI (SHAP)",
        "Population Analytics",
        "Model Performance",
        "About & Limitations",
    ],
    index=1,
)

# Prominent Research Disclaimer in Sidebar
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Academic Prototype:** Historical dataset (1999–2008). Not a certified medical device. For research use only."
)

# --- 1. OVERVIEW DASHBOARD ---
if page == "Overview Dashboard":
    st.markdown('<div class="main-header">Clinical Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive summary of the UCI Diabetes cohort and selected ML model performance.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Academic & Research Disclaimer:</strong> Developed using retrospective EHR claims from 130 US hospitals. 
        Predictions estimate pre-discharge statistical readmission risk for clinical workflow research.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cohort Encounters", "101,766", "130 Hospitals")
    with col2:
        st.metric("Unique Patients", "71,518", "Group-aware split")
    with col3:
        st.metric("30-Day Readmission Rate", "11.16%", "11,357 positive cases")
    with col4:
        st.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0.686):.3f}", f"PR-AUC: {test_metrics.get('pr_auc', 0.246):.3f}")

    st.markdown("### Model Benchmark Summary")
    st.write(
        f"**Selected Model:** `{metadata.get('model_name', 'XGBoost')}` evaluated at optimal decision threshold "
        f"**`{metadata.get('best_threshold', 0.130):.3f}`** on an untouched test partition of 14,815 encounters."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sensitivity (Recall)", f"{test_metrics.get('recall', 0.524)*100:.1f}%")
    k2.metric("Specificity", f"{test_metrics.get('specificity', 0.738)*100:.1f}%")
    k3.metric("F1-Score", f"{test_metrics.get('f1', 0.289):.3f}")
    k4.metric("Brier Score", f"{test_metrics.get('brier_score', 0.093):.4f}")


# --- 2. PATIENT RISK PREDICTOR ---
elif page == "Patient Risk Predictor":
    st.markdown('<div class="main-header">Patient 30-Day Readmission Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter patient discharge attributes to estimate readmission risk with local SHAP attribution.</div>', unsafe_allow_html=True)

    # Demo personas
    st.write("##### Quick-Load Illustrative Clinical Personas:")
    p_col1, p_col2, p_col3 = st.columns(3)
    persona = None
    if p_col1.button("🟢 Persona A: Routine Low Risk"):
        persona = "low"
    if p_col2.button("🟡 Persona B: Moderate Risk Inpatient"):
        persona = "mod"
    if p_col3.button("🔴 Persona C: Complex High Utilizer"):
        persona = "high"

    defaults = {
        "age": "[60-70)", "gender": "Female", "race": "Caucasian",
        "admission_type": "Emergency", "discharge_disposition": "Discharged to home",
        "admission_source": "Emergency Room", "medical_specialty": "InternalMedicine",
        "time_in_hospital": 4, "num_lab_procedures": 45, "num_procedures": 1,
        "num_medications": 16, "number_diagnoses": 7, "number_inpatient": 1,
        "number_emergency": 0, "number_outpatient": 0, "diag_1_group": "Circulatory",
        "diag_2_group": "Diabetes", "diag_3_group": "Respiratory",
        "max_glu_serum": "None", "A1Cresult": "None", "insulin": "Up",
        "change": "Ch", "diabetesMed": "Yes",
    }

    if persona == "low":
        defaults.update({"age": "[20-30)", "time_in_hospital": 2, "admission_type": "Elective", "number_inpatient": 0, "insulin": "No", "change": "No", "diabetesMed": "No", "num_medications": 5, "number_diagnoses": 3})
    elif persona == "high":
        defaults.update({"age": "[70-80)", "time_in_hospital": 8, "number_inpatient": 3, "number_emergency": 2, "insulin": "Up", "change": "Ch", "num_medications": 28, "number_diagnoses": 9, "A1Cresult": ">8", "discharge_disposition": "Discharged/transferred to SNF"})

    with st.form("prediction_form"):
        st.subheader("1. Patient Demographics & Encounter Details")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.selectbox("Age Bracket", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6 if persona != "low" else 2)
        gender = c2.selectbox("Gender", ["Female", "Male"])
        race = c3.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing/Unknown"])
        time_in_hospital = c4.number_input("Length of Stay (Days)", min_value=1, max_value=14, value=defaults["time_in_hospital"])

        c5, c6, c7, c8 = st.columns(4)
        admission_type = c5.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Trauma Center", "Unknown"], index=0 if defaults["admission_type"] == "Emergency" else 2)
        discharge_disposition = c6.selectbox("Discharge Disposition", ["Discharged to home", "Discharged/transferred to SNF", "Discharged/transferred to home with home health service", "Discharged/transferred to another short term hospital", "Left AMA", "Unknown"], index=0 if defaults["discharge_disposition"] == "Discharged to home" else 1)
        admission_source = c7.selectbox("Admission Source", ["Emergency Room", "Physician Referral", "Clinic Referral", "Transfer from a hospital", "Unknown"])
        medical_specialty = c8.selectbox("Medical Specialty", ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology", "Surgery", "Missing/Unknown"])

        st.subheader("2. Clinical Diagnoses & Laboratory Procedures")
        d1, d2, d3, d4 = st.columns(4)
        diag_1_group = d1.selectbox("Primary Diagnosis", ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Genitourinary", "Injury", "Musculoskeletal", "Neoplasms", "Other/Unknown"], index=0)
        diag_2_group = d2.selectbox("Secondary Diagnosis", ["Diabetes", "Circulatory", "Respiratory", "Genitourinary", "Other/Unknown"], index=0)
        diag_3_group = d3.selectbox("Tertiary Diagnosis", ["Respiratory", "Circulatory", "Diabetes", "Genitourinary", "Other/Unknown"], index=0)
        number_diagnoses = d4.number_input("Number of Diagnoses", min_value=1, max_value=16, value=defaults["number_diagnoses"])

        l1, l2, l3, l4 = st.columns(4)
        num_lab_procedures = l1.number_input("Lab Procedures", min_value=1, max_value=130, value=defaults["num_lab_procedures"])
        num_procedures = l2.number_input("Non-Lab Procedures", min_value=0, max_value=6, value=defaults["num_procedures"])
        A1Cresult = l3.selectbox("HbA1c Result", ["None", "Norm", ">7", ">8"], index=0 if defaults["A1Cresult"] == "None" else 3)
        max_glu_serum = l4.selectbox("Glucose Serum Result", ["None", "Norm", ">200", ">300"])

        st.subheader("3. Medication Regimen & Prior Healthcare Utilization")
        m1, m2, m3, m4 = st.columns(4)
        insulin = m1.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"], index=2 if defaults["insulin"] == "Up" else (1 if defaults["insulin"] == "Steady" else 0))
        change = m2.selectbox("Diabetes Med Dosage Changed?", ["Ch", "No"], index=0 if defaults["change"] == "Ch" else 1)
        diabetesMed = m3.selectbox("Prescribed Diabetes Medication?", ["Yes", "No"], index=0 if defaults["diabetesMed"] == "Yes" else 1)
        num_medications = m4.number_input("Total Medications Count", min_value=1, max_value=50, value=defaults["num_medications"])

        u1, u2, u3 = st.columns(3)
        number_inpatient = u1.number_input("Prior Inpatient Admissions (Last 12 Mo)", min_value=0, max_value=25, value=defaults["number_inpatient"], help="Primary predictive driver in cohort")
        number_emergency = u2.number_input("Prior Emergency Visits", min_value=0, max_value=25, value=defaults["number_emergency"])
        number_outpatient = u3.number_input("Prior Outpatient Visits", min_value=0, max_value=25, value=defaults["number_outpatient"])

        submitted = st.form_submit_button("Predict 30-Day Readmission Risk", use_container_width=True)

    if submitted:
        payload = {
            "race": race, "gender": gender, "age": age,
            "admission_type": admission_type, "discharge_disposition": discharge_disposition,
            "admission_source": admission_source, "medical_specialty": medical_specialty,
            "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures, "num_medications": num_medications,
            "number_diagnoses": number_diagnoses, "diag_1_group": diag_1_group,
            "diag_2_group": diag_2_group, "diag_3_group": diag_3_group,
            "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult,
            "insulin": insulin, "change": change, "diabetesMed": diabetesMed,
            "number_inpatient": number_inpatient, "number_emergency": number_emergency,
            "number_outpatient": number_outpatient,
        }

        with st.spinner("Analyzing patient encounter via shared inference pipeline..."):
            res = predict_readmission(payload)

        if not res["success"]:
            st.error("Validation error: " + "; ".join(res.get("errors", [])))
        else:
            st.markdown("---")
            st.subheader("Prediction & Personalized SHAP Explanation")
            
            prob_pct = res["probability_pct"]
            thresh_pct = res["threshold"] * 100
            interp = res["risk_interpretation"]
            shap_info = res["local_explanation"]

            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                st.metric("Estimated Readmission Probability", f"{prob_pct}%", f"Threshold: {thresh_pct:.1f}%")
            with r2:
                st.markdown(f"**Risk Classification:**")
                st.markdown(f"### :{interp['badge_class']}[{interp['risk_tier']}]")
            with r3:
                st.info(f"**Clinical Rationale:** {interp['rationale']}")

            # Recommendations
            st.write("##### Actionable Transitional Care Recommendations:")
            for rec in interp["actionable_recommendations"]:
                st.markdown(f"- ✅ {rec}")

            # Local SHAP Horizontal Bar Chart
            st.write("##### Local SHAP Feature Contributions (Log-Odds Impact):")
            pos = shap_info["top_positive_factors"][:4]
            neg = shap_info["top_negative_factors"][:4]
            combined = pos + neg
            combined.sort(key=lambda x: x["shap_value"])

            fig = go.Figure(go.Bar(
                x=[item["shap_value"] for item in combined],
                y=[item["feature_label"] for item in combined],
                orientation='h',
                marker_color=["#EF4444" if v["shap_value"] > 0 else "#0D9488" for v in combined]
            ))
            fig.update_layout(
                title="Top Risk Drivers (Red = Elevates Risk, Teal = Attenuates Risk)",
                xaxis_title="SHAP Impact (Log-Odds)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"**Plain-Language Summary:** {shap_info['summary_narrative']}")


# --- 3. EXPLAINABLE AI ---
elif page == "Explainable AI (SHAP)":
    st.markdown('<div class="main-header">Explainable AI & SHAP Global Explanations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Global feature importance, beeswarm summary, and non-linear feature dependence.</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.write("### Global Feature Importance")
        st.image("artifacts/plots/shap/shap_importance_bar.png", caption="Mean Absolute SHAP Value across cohort")
    with g2:
        st.write("### SHAP Beeswarm Summary")
        st.image("artifacts/plots/shap/shap_summary_beeswarm.png", caption="Directional impact of feature values on readmission log-odds")

    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        st.write("### Dependence: Prior Inpatient Admissions")
        st.image("artifacts/plots/shap/shap_dependence_inpatient.png", caption="Prior admissions sharply increase readmission log-odds")
    with d2:
        st.write("### Dependence: Length of Stay (Days)")
        st.image("artifacts/plots/shap/shap_dependence_hospital_time.png", caption="Prolonged initial hospitalization increases estimated risk")


# --- 4. POPULATION ANALYTICS ---
elif page == "Population Analytics":
    st.markdown('<div class="main-header">Population Analytics & Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploratory distributions and statistical tests across 101,766 encounters.</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.image("artifacts/plots/eda/01_target_distribution.png", caption="Target Distribution (Severe 11.2% Class Imbalance)")
        st.image("artifacts/plots/eda/04_prior_utilization.png", caption="Prior Utilization vs Readmission Rate")
    with a2:
        st.image("artifacts/plots/eda/02_demographics_readmission.png", caption="Demographic Stratification by Age, Gender, Race")
        st.image("artifacts/plots/eda/03_clinical_factors.png", caption="Inpatient Clinical Factors (LOS, Meds, Diagnoses)")


# --- 5. MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.markdown('<div class="main-header">Model Performance & Experimentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-algorithm benchmark across 11 experiments on the patient-aware validation split.</div>', unsafe_allow_html=True)

    if not comp_df.empty:
        st.dataframe(comp_df.style.highlight_max(subset=["roc_auc", "pr_auc", "f1"], color="#DEF7EC"), use_container_width=True)

    m1, m2 = st.columns(2)
    with m1:
        st.image("artifacts/plots/training/roc_curve.png", caption="Test Set ROC Curve (AUC = 0.686)")
        st.image("artifacts/plots/training/confusion_matrix.png", caption="Test Confusion Matrix (52.4% Recall, 73.8% Specificity)")
    with m2:
        st.image("artifacts/plots/training/pr_curve.png", caption="Test Set Precision-Recall Curve (PR-AUC = 0.246)")
        st.image("artifacts/plots/training/threshold_analysis.png", caption="Validation Threshold Sweep (Optimal Threshold = 0.130)")


# --- 6. ABOUT & LIMITATIONS ---
elif page == "About & Limitations":
    st.markdown('<div class="main-header">About the Project & Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Clinical & Research Context
    - **Dataset:** UCI Diabetes 130-US Hospitals for Years 1999–2008 (101,766 hospital encounters).
    - **Objective:** Retrospective estimation of 30-day readmission risk to support care coordination research.
    - **Architecture:** Shared, framework-independent prediction pipeline in `src/prediction/` shared by Flask and Streamlit.

    ### Non-Clinical Research Disclaimer
    This application is an educational and scientific prototype. Predictions are not clinically validated and must not be used for diagnosis, clinical treatment, discharge timing, or real-time patient triage.
    """)

