# Model Experiments & Class Imbalance Report
## Hospital Readmission Risk Predictor — UCI Diabetes 130-US Hospitals (1999–2008)

---

### 1. Experimental Setup & Methodological Safeguards

All model experiments were conducted strictly following gold-standard data science precautions:
1. **Patient-Aware Splitting:** 99,340 cleaned encounters were partitioned into **Train (70.2%, 69,702 encounters)**, **Validation (14.9%, 14,823 encounters)**, and **Test (14.9%, 14,815 encounters)** grouped strictly by `patient_nbr`. Zero patient overlap exists between any partitions.
2. **Leakage Safeguards:** Preprocessing imputers, one-hot encoders, and scalers were fitted **strictly on the Train partition**.
3. **Resampling Isolation:** SMOTE, RandomOverSampler, and RandomUnderSampler were applied **strictly inside the training partition**. Validation and Test sets remained completely untouched with natural class prevalence (~11.3%).
4. **Consistent Evaluation:** All candidate models were evaluated on the exact same untouched validation partition.

---

### 2. Comprehensive Model Comparison Table

Evaluated on the **Validation Partition (14,823 encounters, 11.32% positive prevalence)**:

| Model Name | Imbalance Strategy | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC | PR-AUC | Brier Score | Train Time (s) | Inf. Time (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Unweighted)** | **None (Natural)** | **0.8867** | 0.4615 | 0.0072 | **0.9989** | 0.0141 | **0.6724** | **0.2294** | **0.0953** | 6.17 | 0.0026 |
| **XGBoost (scale_pos_weight)** | Cost-Sensitive ($w=7.72$) | 0.6624 | 0.1834 | 0.5739 | 0.6737 | 0.2779 | 0.6713 | 0.2256 | 0.2146 | 2.06 | 0.0015 |
| **HistGradientBoosting** | Cost-Sensitive (`balanced`) | 0.6545 | 0.1803 | 0.5787 | 0.6642 | 0.2750 | 0.6703 | 0.2265 | 0.2191 | 4.00 | 0.0145 |
| **Random Forest (Balanced)** | Cost-Sensitive (`balanced`) | 0.7199 | 0.1959 | 0.4750 | 0.7512 | 0.2774 | 0.6647 | 0.2194 | 0.1995 | 3.89 | 0.0097 |
| **Extra Trees (Balanced)** | Cost-Sensitive (`balanced`) | 0.6731 | 0.1832 | 0.5459 | 0.6893 | 0.2743 | 0.6657 | 0.2251 | 0.2143 | 6.01 | 0.0052 |
| **Logistic Regression (Balanced)** | Cost-Sensitive (`balanced`) | 0.6630 | 0.1769 | 0.5411 | 0.6785 | 0.2666 | 0.6624 | 0.2147 | 0.2259 | 2.41 | 0.0004 |
| **Logistic Regression (Baseline)** | None (Unweighted) | 0.8869 | **0.5192** | 0.0161 | 0.9981 | 0.0312 | 0.6621 | 0.2154 | 0.0962 | 1.93 | 0.0004 |
| **Random Forest + ROS** | RandomOverSampler | 0.6963 | 0.1926 | 0.5274 | 0.7178 | 0.2822 | 0.6675 | 0.2204 | 0.2067 | 7.43 | 0.0078 |
| **Random Forest + RUS** | RandomUnderSampler | 0.6308 | 0.1719 | **0.5924** | 0.6358 | 0.2665 | 0.6634 | 0.2173 | 0.2279 | 0.86 | 0.0063 |
| **Decision Tree (Balanced)** | Cost-Sensitive (`balanced`) | 0.6157 | 0.1674 | 0.6025 | 0.6173 | 0.2620 | 0.6458 | 0.2081 | 0.2249 | 1.28 | 0.0016 |
| **Random Forest + SMOTE** | SMOTE Resampling | 0.8762 | 0.3179 | 0.0816 | 0.9776 | 0.1299 | 0.6466 | 0.1957 | 0.1299 | 8.50 | 0.0078 |

---

### 3. Key Findings & Scientific Observations

#### A. The Illusion of Raw Accuracy
- Unweighted models (`Logistic Regression Baseline` and `XGBoost Unweighted` at default threshold 0.5) achieved the highest accuracy (**88.69%** and **88.67%**), but their Recall was near zero (**1.6%** and **0.7%**).
- They achieved high accuracy purely by predicting the negative majority class for nearly all encounters. This provides rigorous empirical proof that **accuracy is entirely unsuitable as an optimization metric for imbalanced healthcare tasks**.

#### B. Cost-Sensitive Weighting vs Resampling
- **Cost-Sensitive Weighting (`scale_pos_weight` and `class_weight='balanced'`):** Directly penalizes false negatives during loss calculation. Boosted recall from ~1% to **54%–58%** without synthesizing artificial patient records.
- **SMOTE Oversampling Performance:** Generated the lowest PR-AUC (**0.1957**) and lower ROC-AUC (**0.6466**) among all tree models. In mixed tabular datasets with one-hot encoded categorical features, synthetic interpolation can fabricate clinically implausible feature combinations.
- **Random Under-Sampling (RUS):** Highly computationally efficient (0.86s training), but discarded 87% of majority encounters, leading to slightly depressed specificity (63.58%).

#### C. Model Family Superiority
- **Gradient Boosting (XGBoost & HistGradientBoosting)** demonstrated consistent superiority over linear models and individual decision trees, achieving the highest discriminatory power (**ROC-AUC 0.6724** and **PR-AUC 0.2294**).
- Probability calibration was highest in unweighted XGBoost (**Brier score 0.0953** vs >0.20 in resampled/heavily reweighted models).

