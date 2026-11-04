# Feature Engineering & Data Cleansing Report
## Hospital Readmission Risk Predictor — UCI Diabetes 130-US Hospitals (1999–2008)

---

### 1. Data Cleansing Protocols & Rationale

| Operation | Strategy & Implementation | Rationale & Evidence | Assumptions & Limitations |
| :--- | :--- | :--- | :--- |
| **Missing Markers** | Standardized `?`, empty strings, and `"None"` into explicit missing representations. | The raw UCI dataset uses `?` in categorical columns and string `"None"` in lab results. Uniform encoding prevents silent string matching errors. | Assumes `?` represents unrecorded data rather than a specific clinical code. |
| **High Missingness Column Exclusion** | `weight` was dropped completely (98,569 missing encounters; **96.86%**). | Weight was recorded in fewer than 3.2% of encounters. Imputing 97% of values introduces artificial bias and severe noise. | Weight / BMI is clinically important in diabetes management, but retrospective hospital capture was too sparse in 1999–2008. |
| **Informative Missingness Preservation** | `medical_specialty`, `payer_code`, and `race` preserve an explicit `'Missing/Unknown'` level. | In healthcare administration, unassigned specialty often signifies community hospital general admission rather than specialized tertiary care. | Do not treat missing specialty as equivalent to Internal Medicine. |
| **Laboratory Test Ordering Signals** | `A1Cresult` and `max_glu_serum` retain `'None'` as a first-class categorical level. | Tests were not ordered in 83.3% and 94.7% of encounters. In clinical practice, tests are ordered conditionally based on patient acuity and physician suspicion. | Testing absence is not equivalent to a normal test result; it represents clinical process information. |
| **Terminal Discharge Disposition Exclusion** | Excluded encounters with `discharge_disposition_id` in $\{11, 13, 14, 19, 20, 21\}$ (Expired / Hospice). | Patients who expired during hospitalization or were discharged directly into hospice care are biologically not at risk of acute 30-day readmission. | Including expired encounters creates biological survivor bias and post-event outcome leakage. |
| **Duplicate Row Audit** | Exact row duplicates audited and removed (0 found). Legitimate repeated encounters for the same `patient_nbr` (**30,248 repeat admissions**) were strictly preserved. | A patient can have multiple hospitalizations over 10 years. Dropping repeated patients would discard critical chronic disease trajectories. | Requires patient-aware group splitting (`GroupShuffleSplit`) to prevent cross-partition leakage. |
| **Outlier Capping on Utilization** | `number_inpatient` capped at 15; `number_emergency` capped at 20; `number_outpatient` capped at 20. | Extreme counts (e.g. 42 emergency visits) represent heavy-tailed outliers that distort linear gradients while representing the same clinical tier. | Preserves ordinal rank and bounds leverage. |

---

### 2. Feature Engineering Catalog

#### A. Age Features
- **Original Feature:** `age` (categorical 10-year brackets: `[0-10)`, `[10-20)`, ..., `[90-100)`).
- **Engineered Features:**
  1. `age_midpoint`: Numerical midpoint $\{5, 15, 25, 35, 45, 55, 65, 75, 85, 95\}$.
  2. `is_senior`: Binary indicator ($1$ if `age_midpoint` $\ge 65$, else $0$).
  3. `age_group`: Clinically interpretable life-stage category (`Young Adult (<30)`, `Adult (30-59)`, `Older Adult (60-74)`, `Elderly (75+)`).
- **Rationale:** Tree ensembles and linear models benefit from continuous monotonic representations rather than arbitrary unranked one-hot dummies.
- **Assumptions & Limitations:** The interval midpoint is an analytical surrogate, not the exact measured biological age.

#### B. Clinical Diagnosis Grouping (ICD-9 Mapping)
- **Original Features:** `diag_1`, `diag_2`, `diag_3` (raw alphanumeric ICD-9 codes with >700 distinct categories).
- **Engineered Features:**
  1. `diag_1_group`, `diag_2_group`, `diag_3_group`: Broad clinical categories based on Strack et al. (2014) and HCUP Clinical Classifications Software (CCS):
     - `Circulatory` (390–459, 785)
     - `Respiratory` (460–519, 786)
     - `Digestive` (520–579, 787)
     - `Diabetes` (250.xx)
     - `Injury` (800–999)
     - `Musculoskeletal` (710–739)
     - `Genitourinary` (580–629, 788)
     - `Neoplasms` (140–239)
     - `Other/Unknown` (all remaining numeric codes, V-codes, E-codes, or missing)
  2. `has_diabetes_primary`: $1$ if `diag_1_group == 'Diabetes'`, else $0$.
  3. `has_circulatory_primary`: $1$ if `diag_1_group == 'Circulatory'`, else $0$.
  4. `has_diabetes_any`: $1$ if diabetes appears in primary, secondary, or tertiary diagnosis.
  5. `num_distinct_diag_groups`: Cardinality of distinct non-unknown diagnostic groups ($1$ to $3$), measuring diagnostic heterogeneity.
- **Rationale:** Reduces diagnosis cardinality from $>700$ to $9$ meaningful pathophysiological groups, avoiding curse of dimensionality and severe sparsity.

#### C. Medication & Polypharmacy Features
- **Original Features:** 23 specific diabetes drug columns with values $\{No, Steady, Up, Down\}$, plus `change` and `diabetesMed`.
- **Engineered Features:**
  1. `num_diabetes_meds`: Integer count of active diabetes medications prescribed ($\sum \mathbb{I}[med \neq 'No']$).
  2. `num_med_changes`: Integer count of medications with active dose adjustments ($\sum \mathbb{I}[med \in \{'Up', 'Down'\}]$).
  3. `on_insulin`: Binary flag for any active insulin prescription ($1$ if `insulin != 'No'`).
  4. `insulin_titration`: Binary flag indicating active insulin dosage adjustment ($1$ if `insulin in ['Up', 'Down']`).
  5. `is_med_changed`: Binary flag ($1$ if `change == 'Ch'`).
  6. `is_diabetes_med`: Binary flag ($1$ if `diabetesMed == 'Yes'`).
- **Rationale:** Polypharmacy and active dosage titration reflect unstable glycemic control and acute medication management, both clinically known risk factors for readmission.

#### D. Prior Healthcare Utilization
- **Original Features:** `number_inpatient`, `number_emergency`, `number_outpatient`.
- **Engineered Features:**
  1. `total_prior_visits`: Total encounter frequency $\text{number\_inpatient} + \text{number\_emergency} + \text{number\_outpatient}$.
  2. `has_prior_inpatient`: $\mathbb{I}[\text{number\_inpatient} > 0]$.
  3. `has_prior_emergency`: $\mathbb{I}[\text{number\_emergency} > 0]$.
  4. `high_prior_utilizer`: $\mathbb{I}[(\text{number\_inpatient} \ge 2) \lor (\text{total\_prior\_visits} \ge 3)]$.
- **Rationale:** Historical hospital utilization is empirically the single strongest predictor of recurrent readmissions ($p < 10^{-300}$).

#### E. Medical Specialty Consolidation
- **Original Feature:** `medical_specialty` (73 distinct specialties; 49% missing).
- **Engineered Feature:** `medical_specialty_binned`: Consolidated into `InternalMedicine`, `Emergency/Trauma`, `Family/GeneralPractice`, `Cardiology`, `Surgery`, `Orthopedics`, `Nephrology`, `Other_Specialty`, and `Missing/Unknown`.

---

### 3. Temporal Availability & Leakage Review

| Feature / Attribute | Available at Discharge? | Leakage Risk Assessment & Action |
| :--- | :--- | :--- |
| `readmitted` (Raw Target) | **No** (Occurs 0–30+ days post-discharge) | **CRITICAL LEAKAGE:** Dropped immediately after creating binary target `readmitted_30_days`. Never used as feature. |
| `encounter_id`, `patient_nbr` | Administrative IDs | Excluded from feature inputs. `patient_nbr` retained solely for group splitting. |
| `discharge_disposition_id` | Yes (At discharge) | Terminal dispositions (death/hospice) excluded because acute readmission is impossible. Remaining disposition categories represent valid discharge destinations (e.g. Home, Skilled Nursing Facility). |
| `time_in_hospital`, `num_lab_procedures`, `num_medications` | Yes (Known at discharge) | Included as primary pre-discharge inpatient markers. |
| `A1Cresult`, `max_glu_serum` | Yes (Recorded during stay) | Included. Missingness retained as `'None'`. |
| Medication titrations (`insulin`, `change`) | Yes (Determined at discharge) | Included as discharge therapy indicators. |

---

### 4. Summary of Output Feature Matrices

- **Numerical Features (23):** `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, `age_midpoint`, `total_prior_visits`, `num_diabetes_meds`, `num_med_changes`, `num_distinct_diag_groups`, `is_senior`, `has_prior_inpatient`, `has_prior_emergency`, `high_prior_utilizer`, `has_diabetes_primary`, `has_circulatory_primary`, `on_insulin`, `insulin_titration`, `is_med_changed`, `is_diabetes_med`.
- **Categorical Features (15):** `race`, `gender`, `age_group`, `admission_type`, `discharge_disposition`, `admission_source`, `medical_specialty_binned`, `diag_1_group`, `diag_2_group`, `diag_3_group`, `max_glu_serum`, `A1Cresult`, `insulin`, `change`, `diabetesMed`.
- **Target Feature (1):** `readmitted_30_days` (Binary: $1 = <30$ days, $0 =$ otherwise).

