# Model Selection & Final Evaluation Report
## Hospital Readmission Risk Predictor — UCI Diabetes 130-US Hospitals (1999–2008)

---

### 1. Final Model Selection Decision

Based on empirical performance across 11 experiments on the patient-aware validation set (14,823 encounters), **XGBoost (with post-hoc validation threshold optimization)** was selected as the final production model.

#### Selection Rationale:
1. **Highest Area Under the Precision-Recall Curve (PR-AUC = 0.2294):** More than **2.05x higher** than the random no-skill baseline (0.1116).
2. **Superior Discrimination (ROC-AUC = 0.6724):** Consistently outperformed Random Forest (0.6647), Extra Trees (0.6657), and Logistic Regression (0.6624).
3. **Lowest Brier Score (0.0953):** Demonstrates superior raw probability calibration prior to artificial threshold shifting.
4. **Computational Efficiency:** 6.17 seconds training time and **0.0026 ms/sample inference latency**, making it ideal for real-time web deployment.

---

### 2. Validation Threshold Optimization

In clinical decision support for 30-day readmissions, using the arbitrary default threshold of $T = 0.50$ is counterproductive because positive class prevalence is only ~11.2%.

Using the validation partition exclusively, a fine-grained sweep from $T = 0.05$ to $0.90$ was conducted:
- **Optimal Threshold Selected:** **$T = 0.130$** (chosen via F1-maximization criterion).
- **Validation Metrics at $T = 0.130$:**
  - **Recall (Sensitivity):** **51.2%** (detects $>51\%$ of readmitted encounters).
  - **Precision (PPV):** **19.8%** (nearly double baseline prevalence).
  - **F1-Score:** **0.285** (vs 0.014 at $T = 0.50$).
  - **Specificity:** **74.1%** (avoids triggering excessive false alarms for stable patients).

![Threshold Analysis](../artifacts/plots/training/threshold_analysis.png)

---

### 3. Single Final Evaluation on Untouched Test Set

The frozen winning pipeline and optimal threshold ($T = 0.130$) were evaluated **exactly once** on the untouched Test partition (**14,815 encounters, 10,499 unique patients, 11.10% positive prevalence**).

| Final Metric | Test Set Score | Baseline Comparison | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.6859** | 0.5000 (No-skill) | Moderate-to-strong discrimination given pre-discharge electronic record limits. |
| **PR-AUC** | **0.2463** | 0.1110 (Prevalence) | Over **2.21x enhancement** over random baseline. |
| **Sensitivity (Recall)** | **0.5237 (52.4%)** | 0.0072 (at $T=0.5$) | Captures **861 out of 1,644** high-risk readmission encounters. |
| **Precision (PPV)** | **0.2000 (20.0%)** | 0.1110 (Prevalence) | 1 out of 5 flagged patients experiences 30-day readmission. |
| **Specificity** | **0.7384 (73.8%)** | 0.9989 (at $T=0.5$) | Correctly clears **9,726 out of 13,171** non-readmitted patients. |
| **F1-Score** | **0.2894** | 0.0141 (at $T=0.5$) | Well-balanced trade-off for screening allocation. |
| **Brier Score** | **0.0950** | 0.0987 | Excellent mean squared error calibration. |
| **Inference Latency** | **0.0028 ms** | Real-time | Under 3 milliseconds per 1,000 predictions. |

---

### 4. Test Set Confusion Matrix

```
                      PREDICTED NEGATIVE (0)    PREDICTED POSITIVE (1)
ACTUAL NEGATIVE (0):        9,726 (65.6%)              3,445 (23.3%)
ACTUAL POSITIVE (1):          783 ( 5.3%)                861 ( 5.8%)
----------------------------------------------------------------------
Total Test Encounters: 14,815
True Positives (Captured Readmissions):  861
False Negatives (Missed Readmissions):   783
True Negatives (Correct Non-Readmissions): 9,726
False Positives (Screening False Alarms): 3,445
```

![Confusion Matrix](../artifacts/plots/training/confusion_matrix.png)
![ROC Curve](../artifacts/plots/training/roc_curve.png)
![PR Curve](../artifacts/plots/training/pr_curve.png)
![Calibration Curve](../artifacts/plots/training/calibration_curve.png)

---

### 5. Probability Calibration & Clinical Bands

The Brier score of **0.0950** confirms strong raw probability calibration.

For UI presentation, model probabilities are translated into transparent, project-defined display bands (clearly labelled as heuristic research bands, not validated clinical tiers):
- **Lower Risk Tier:** Estimated probability $< 0.10$
- **Moderate Risk Tier:** Estimated probability $0.10 \le p < 0.25$ (Actionable screening zone around threshold $0.130$)
- **Elevated Risk Tier:** Estimated probability $\ge 0.25$ (High intensity post-discharge intervention candidate)

