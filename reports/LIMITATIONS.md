# Clinical, Technical & Ethical Limitations
## Hospital Readmission Risk Predictor — UCI Diabetes 130-US Hospitals (1999–2008)

---

### 1. Clinical & Epidemiological Limitations

#### A. Historical Temporal Mismatch (1999–2008 Cohort)
- **Diagnostic Coding Shift:** The dataset uses ICD-9-CM codes. Modern healthcare facilities utilize ICD-10-CM / ICD-11, which provide substantially greater clinical granularity.
- **Therapeutic Evolution:** Between 1999 and 2008, standard diabetes management relied on older sulfonylureas, early thiazolidinediones (e.g. troglitazone, later withdrawn), metformin, and NPH/regular insulins. Landmark modern antidiabetic agents—such as Sodium-Glucose Cotransporter-2 (SGLT2) inhibitors and modern Glucagon-Like Peptide-1 (GLP-1) receptor agonists—are absent from this historical cohort.
- **Hospital Discharge Practices:** Clinical practice guidelines, discharge planning protocols, and transitional care coordination under the Affordable Care Act (enacted 2010) significantly altered readmission dynamics post-2010 compared to the retrospective study period.

#### B. Absence of Outpatient Continuity Data
- The dataset captures encounters at participating inpatient facilities. Outpatient visits, emergency room visits, or subsequent readmissions occurring at **non-participating hospitals** were not captured in the study registry, resulting in potential outcome undercounting.

---

### 2. Social Determinants of Health (SDOH) & Unobserved Confounders

Readmission risk is heavily determined by social and environmental determinants that do not appear in administrative billing records:
1. **Socioeconomic Status (SES):** Income, housing stability, and education are not recorded.
2. **Medication Affordability & Supply Chain:** Barriers to purchasing insulin or glucometer strips post-discharge are unobserved.
3. **Health Literacy & Social Support:** Living alone, cognitive impairment, and caregiver availability are uncaptured.
4. **Food Security & Dietary Access:** Nutritional access critical for glycemic stability is unrecorded.

Because these dominant factors are missing, the theoretical upper bound on discriminatory accuracy ($ROC\text{-}AUC$) for pure inpatient billing data in this cohort is naturally constrained to approximately $0.68 - 0.72$.

---

### 3. Data Quality & Non-Random Missingness

| Feature | Missing Rate | Limitation & Residual Bias |
| :--- | :---: | :--- |
| `weight` | **96.86%** | Body Mass Index (BMI) and obesity severity could not be modeled due to extreme absence across community hospital records. |
| `medical_specialty` | **49.08%** | Missingness was correlated with smaller non-teaching community facilities without departmental subdivision. |
| `payer_code` | **39.56%** | Commercial insurance vs. Medicare/Medicaid payer status was incomplete, limiting socioeconomic adjustment. |
| `A1Cresult` / `max_glu_serum` | **83.3% / 94.7%** | Inpatient blood glucose and HbA1c tests were ordered selectively by clinicians based on suspected decompensation (confounding by indication). |

---

### 4. Algorithmic & Explainability Constraints

#### A. Observational Association vs. Biological Causation
- **SHAP values are descriptive attributions, not causal interventions.**
- A positive SHAP value for `insulin = Up` does not mean that increasing insulin caused the patient to be readmitted. Rather, insulin dose increases serve as a surrogate marker for acute inpatient glycemic instability and disease complexity.
- Modifying a feature in real life does not guarantee an equivalent shift in patient outcome.

#### B. Biological Survivor Bias & Terminal Discharges
- Patients who expired during hospitalization or were discharged directly into hospice care cannot experience an acute 30-day readmission.
- While our pipeline systematically excluded known terminal discharge codes ($11, 13, 14, 19, 20, 21$), post-discharge out-of-hospital mortality within 30 days is unobserved, creating residual survivorship censoring.

---

### 5. Algorithmic Fairness & Subgroup Performance Audit

| Subgroup | Sample Size (Val Set) | True Positive Rate (Sensitivity) | False Positive Rate | Brier Score | Fairness Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Caucasian** | 11,042 | 52.1% | 25.8% | 0.093 | Primary majority population in cohort (74.8%). |
| **African American** | 2,810 | 53.6% | 26.5% | 0.096 | Slightly higher baseline utilization; comparable sensitivity. |
| **Hispanic** | 296 | 50.0% | 24.2% | 0.091 | Smaller subgroup sample size introduces statistical variance. |
| **Asian** | 94 | 44.4% | 22.4% | 0.088 | Smallest demographic cohort (<1% of encounters); wider error margins. |

> **Ethical Note:** Subgroup differences reflect disparities in healthcare access, insurance coverage, and historical systemic inequities rather than biological differences. Models must never be used to ration care or restrict services for vulnerable demographic groups.

---

### 6. Regulatory, Legal & Deployment Boundaries

1. **Non-Medical Device Status:** This software has not received FDA clearance, CE mark certification, or clinical trial validation under Software as a Medical Device (SaMD) standards.
2. **HIPAA & Privacy Boundary:** The application was built on historical, publicly de-identified datasets. It does not possess HIPAA BAA certification, encrypted clinical audit trails, or EHR integration bridges.
3. **Operational Scope:** Predictions must strictly be restricted to non-interventional educational exploration, operational capacity planning research, and supportive care coordination workflows.

