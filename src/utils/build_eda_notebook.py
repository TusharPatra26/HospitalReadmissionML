"""
Build the combined, detailed EDA notebook for Hospital Readmission ML project.
Outputs: notebooks/1. EDA HOSPITAL READMISSION.ipynb
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NB_PATH = PROJECT_ROOT / "notebooks" / "1. EDA HOSPITAL READMISSION.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip(),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip(),
    }


cells = []

# ─────────────────────────────────────────────
# Title & Clinical Context
# ─────────────────────────────────────────────
cells.append(md("""# 🏥 Hospital Readmission Risk Predictor — Exploratory Data Analysis

**Dataset:** UCI Diabetes 130-US Hospitals for Years 1999–2008  
**Source:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
**Author:** Hospital Readmission ML Project  
**Purpose:** End-to-end exploratory analysis of ~101 K diabetic patient encounters to identify drivers of 30-day readmission.

---

## 🩺 Clinical Background

Hospital readmission within 30 days is a key **quality-of-care** indicator.  
The **Hospital Readmissions Reduction Program (HRRP)** penalises US hospitals with excess readmissions for conditions including heart failure, pneumonia, COPD, and uncontrolled diabetes.

**Our Target:**  
`readmitted_30_days = 1` if the encounter column `readmitted == "<30"`, else `0`.  
This is a **binary classification** problem. Encounters labelled `>30` (later readmission) are treated as negative.

> ⚠️ **Disclaimer:** This is an academic research prototype and **not a clinically validated medical device**."""))

# ─────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────
cells.append(md("## 0. Environment Setup & Imports"))

cells.append(code("""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── project root on path ──────────────────────
PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── standard ─────────────────────────────────
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_colwidth", 60)

# ── visualisation ─────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {"0 (Not Readmitted)": "#2196F3", "1 (≤30 days)": "#F44336"}

# ── stats ─────────────────────────────────────
from scipy import stats

# ── data loader ───────────────────────────────
from src.data_ingestion.data_loader import load_raw_data, parse_ids_mapping

print("All imports OK ✓")
print(f"NumPy {np.__version__} | Pandas {pd.__version__} | Seaborn {sns.__version__}")
"""))

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
cells.append(md("""---
## 1. Load Dataset

We load:
- `diabetic_data.csv` — 101 766 hospital encounter records, 50 columns
- `IDs_mapping.csv` — integer → human-readable label mappings for admission type, discharge disposition, and admission source"""))

cells.append(code("""
df, ids_map = load_raw_data()
print(f"diabetic_data : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"IDs mapping   : {len(ids_map)} entries")
"""))

cells.append(code("""
print("── First 5 rows ──")
df.head()
"""))

cells.append(code("""
print("── Last 5 rows ──")
df.tail()
"""))

cells.append(code("""
print("── Random sample of 5 rows ──")
df.sample(5, random_state=42)
"""))

# ─────────────────────────────────────────────
# 2. Data Overview
# ─────────────────────────────────────────────
cells.append(md("""---
## 2. Dataset Overview

### 2.1 Schema & Dtypes (`df.info()`)"""))

cells.append(code("""
df.info(verbose=True, show_counts=True)
"""))

cells.append(md("### 2.2 Column Names"))

cells.append(code("""
print("All 50 columns:")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:>2}. {c}")
"""))

cells.append(md("### 2.3 Unique Value Counts per Column"))

cells.append(code("""
uniq = df.nunique().reset_index()
uniq.columns = ["Column", "Unique Values"]
uniq["Dtype"] = [str(df[c].dtype) for c in uniq["Column"]]
print(uniq.to_string(index=False))
"""))

# ─────────────────────────────────────────────
# 3. Descriptive Statistics
# ─────────────────────────────────────────────
cells.append(md("""---
## 3. Descriptive Statistics

### 3.1 Numerical Features — `df.describe().T`"""))

cells.append(code("""
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns ({len(num_cols)}): {num_cols}")
df[num_cols].describe().T.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["mean","std","50%"])
"""))

cells.append(md("### 3.2 Categorical / Object Features — `df.describe(include='O').T`"))

cells.append(code("""
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
df[cat_cols].describe(include="O").T
"""))

cells.append(md("### 3.3 Value Counts for Key Categorical Columns"))

cells.append(code("""
key_cats = ["race", "gender", "age", "admission_type_id",
            "discharge_disposition_id", "admission_source_id",
            "readmitted", "diabetesMed", "change"]
for col in key_cats:
    if col in df.columns:
        print(f"\\n── {col} ──────────────────────")
        vc = df[col].value_counts(dropna=False)
        pct = (vc / len(df) * 100).round(2)
        summary = pd.DataFrame({"Count": vc, "Pct%": pct})
        print(summary.to_string())
"""))

# ─────────────────────────────────────────────
# 4. Missing Values
# ─────────────────────────────────────────────
cells.append(md("""---
## 4. Missing Value Audit

This dataset uses **`?`** as a placeholder for unknown values in addition to standard `NaN`."""))

cells.append(code("""
# Replace ? with NaN for analysis
df_audit = df.replace("?", np.nan)

nan_count = df_audit.isnull().sum()
nan_pct   = (nan_count / len(df_audit) * 100).round(2)
miss = pd.DataFrame({
    "Missing Count": nan_count,
    "Missing %": nan_pct,
    "Dtype": df_audit.dtypes
}).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

print(f"Columns with missing values: {len(miss)}")
print(miss.to_string())
"""))

cells.append(code("""
fig, ax = plt.subplots(figsize=(9, 4))
miss_nonzero = miss[miss["Missing %"] > 0]
bars = ax.barh(miss_nonzero.index[::-1], miss_nonzero["Missing %"][::-1],
               color=["#E53935" if p > 50 else "#FB8C00" if p > 20 else "#42A5F5"
                      for p in miss_nonzero["Missing %"][::-1]])
for bar, pct in zip(bars, miss_nonzero["Missing %"][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Value Percentage by Column", fontsize=13, fontweight="bold")
ax.axvline(50, color="red", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/04_missing_values.png", dpi=120, bbox_inches="tight")
plt.show()
print("\\n📌 'weight' (96.9% missing) → will be DROPPED.")
print("📌 'medical_specialty' (49%) and 'payer_code' (40%) → kept as 'Missing/Unknown' category.")
print("📌 Lab result 'None' values are informative, not truly missing.")
"""))

# ─────────────────────────────────────────────
# 5. Duplicate Check & Patient Cardinality
# ─────────────────────────────────────────────
cells.append(md("""---
## 5. Duplicate Check & Patient Cardinality"""))

cells.append(code("""
exact_dups = df.duplicated().sum()
patient_dups = df.duplicated(subset=["encounter_id"]).sum()
n_patients = df["patient_nbr"].nunique()
n_encounters = len(df)

print(f"Total encounters       : {n_encounters:,}")
print(f"Exact duplicate rows   : {exact_dups:,}")
print(f"Duplicate encounter_id : {patient_dups:,}")
print(f"Unique patients        : {n_patients:,}")
print(f"Mean encounters/patient: {n_encounters/n_patients:.2f}")
print(f"Patients with >1 visit : {(df.groupby('patient_nbr').size() > 1).sum():,}")
"""))

cells.append(code("""
visits_per_patient = df.groupby("patient_nbr").size()
visit_dist = visits_per_patient.value_counts().sort_index().head(10)

fig, ax = plt.subplots(figsize=(8, 4))
visit_dist.plot(kind="bar", ax=ax, color="#42A5F5", edgecolor="white")
ax.set_xlabel("Number of Encounters per Patient")
ax.set_ylabel("Number of Patients")
ax.set_title("Distribution of Encounters per Patient", fontsize=12, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/05_patient_encounters.png", dpi=120, bbox_inches="tight")
plt.show()
print("\\n⚠️  PATIENT LEAKAGE RISK: Same patient can appear in train AND test sets.")
print("    → MUST use GroupShuffleSplit on patient_nbr to prevent data leakage.")
"""))

# ─────────────────────────────────────────────
# 6. Target Variable
# ─────────────────────────────────────────────
cells.append(md("""---
## 6. Target Variable Analysis

### 6.1 Original 3-class `readmitted`"""))

cells.append(code("""
readmit_counts = df["readmitted"].value_counts()
readmit_pct    = (readmit_counts / len(df) * 100).round(2)
print("Original readmitted column:")
print(pd.DataFrame({"Count": readmit_counts, "Pct%": readmit_pct}))
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 3-class bar
colors3 = ["#E53935", "#FB8C00", "#43A047"]
axes[0].bar(readmit_counts.index, readmit_counts.values, color=colors3, edgecolor="white", width=0.5)
for i, (v, p) in enumerate(zip(readmit_counts.values, readmit_pct.values)):
    axes[0].text(i, v + 500, f"{v:,}\\n({p:.1f}%)", ha="center", fontsize=10, fontweight="bold")
axes[0].set_title("Original: 3-Class Readmission", fontweight="bold")
axes[0].set_ylabel("Encounter Count")
axes[0].set_xlabel("Readmission Status")

# Binary target
df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)
bin_counts = df["readmitted_30_days"].value_counts().sort_index()
bin_labels = ["0 — Not Readmitted\\n(or >30 days)", "1 — Readmitted\\nwithin 30 days"]
axes[1].bar(bin_labels, bin_counts.values,
            color=["#2196F3", "#F44336"], edgecolor="white", width=0.4)
for i, v in enumerate(bin_counts.values):
    pct = v / len(df) * 100
    axes[1].text(i, v + 500, f"{v:,}\\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
axes[1].set_title("Binary Target: 30-Day Readmission", fontweight="bold")
axes[1].set_ylabel("Encounter Count")

plt.suptitle("Target Variable Distribution", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/01_target_distribution.png", dpi=120, bbox_inches="tight")
plt.show()

imbalance_ratio = bin_counts[0] / bin_counts[1]
print(f"\\nClass imbalance ratio (neg:pos) = {imbalance_ratio:.1f}:1")
print(f"Positive class (readmitted <30d): {bin_counts[1]:,} ({bin_counts[1]/len(df)*100:.2f}%)")
print(f"⚠️  Severe imbalance → use SMOTE / class_weight='balanced' + PR-AUC as primary metric.")
"""))

# ─────────────────────────────────────────────
# 7. Numerical Feature Distributions
# ─────────────────────────────────────────────
cells.append(md("""---
## 7. Numerical Feature Distributions

Histograms with KDE overlay for all 8 raw numerical features."""))

cells.append(code("""
num_features = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for ax, col in zip(axes, num_features):
    data = df[col].dropna()
    ax.hist(data, bins=30, color="#42A5F5", edgecolor="white", alpha=0.8, density=True)
    try:
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color="#1565C0", linewidth=2)
    except Exception:
        pass
    ax.axvline(data.mean(), color="#E53935", linestyle="--", linewidth=1.5, label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color="#43A047", linestyle="-.", linewidth=1.5, label=f"Median={data.median():.1f}")
    ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

plt.suptitle("Distribution of Numerical Features (Hist + KDE)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/06_numerical_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 7.1 Summary Statistics (numerical)"))

cells.append(code("""
df[num_features].agg(["mean","median","std","min","max","skew"]).T.round(3)
"""))

# ─────────────────────────────────────────────
# 8. Outlier Analysis
# ─────────────────────────────────────────────
cells.append(md("""---
## 8. Outlier Analysis — Boxplots

Box-plots for all 8 numerical features, coloured by binary target."""))

cells.append(code("""
df["readmit_label"] = df["readmitted_30_days"].map({0: "0 (Not Readmitted)", 1: "1 (≤30 days)"})

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for ax, col in zip(axes, num_features):
    sns.boxplot(data=df, x="readmit_label", y=col, ax=ax,
                hue="readmit_label", palette=PALETTE, legend=False,
                width=0.5, flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

plt.suptitle("Boxplots of Numerical Features by Readmission Status", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/07_outlier_boxplots.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""
# IQR-based outlier count
print("Outlier summary (IQR method):")
print(f"{'Feature':<25} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Lower':>10} {'Upper':>10} {'Outliers':>10} {'Out%':>8}")
print("-"*90)
for col in num_features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<25} {q1:>8.1f} {q3:>8.1f} {iqr:>8.1f} {lo:>10.1f} {hi:>10.1f} {n_out:>10,} {n_out/len(df)*100:>7.2f}%")
"""))

# ─────────────────────────────────────────────
# 9. Demographics
# ─────────────────────────────────────────────
cells.append(md("""---
## 9. Demographic Analysis

### 9.1 Age Distribution"""))

cells.append(code("""
age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
age_readmit = (df.groupby("age")["readmitted_30_days"]
                 .agg(["mean","count"])
                 .reindex(age_order)
                 .reset_index())
age_readmit.columns = ["age","readmit_rate","count"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count bar
axes[0].bar(age_readmit["age"], age_readmit["count"],
            color="#42A5F5", edgecolor="white")
axes[0].set_title("Encounter Count by Age Group", fontweight="bold")
axes[0].set_xlabel("Age Group"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

# Readmission rate line
axes[1].plot(age_readmit["age"], age_readmit["readmit_rate"]*100,
             marker="o", color="#E53935", linewidth=2.5, markersize=8)
axes[1].fill_between(range(len(age_readmit)), age_readmit["readmit_rate"]*100,
                     alpha=0.15, color="#E53935")
axes[1].set_xticks(range(len(age_readmit)))
axes[1].set_xticklabels(age_readmit["age"], rotation=45, ha="right")
axes[1].set_title("30-Day Readmission Rate by Age Group", fontweight="bold")
axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Readmission Rate (%)")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())

plt.suptitle("Age Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/08_age_analysis.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 9.2 Gender & Race"))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gender
gender_data = (df[df["gender"].isin(["Male","Female"])]
               .groupby(["gender","readmit_label"])
               .size().reset_index(name="count"))
sns.barplot(data=gender_data, x="gender", y="count", hue="readmit_label",
            palette=PALETTE, ax=axes[0])
axes[0].set_title("Gender vs Readmission Status", fontweight="bold")
axes[0].set_xlabel("Gender"); axes[0].set_ylabel("Count")
axes[0].legend(title="Readmitted", fontsize=9)

# Race readmission rate
race_rate = (df.groupby("race")["readmitted_30_days"]
              .agg(["mean","count"])
              .reset_index()
              .sort_values("mean", ascending=False))
race_rate = race_rate[race_rate["race"] != "?"]
bars = axes[1].bar(race_rate["race"], race_rate["mean"]*100,
                   color="#7E57C2", edgecolor="white")
for bar, row in zip(bars, race_rate.itertuples()):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"{bar.get_height():.1f}%\\n(n={row.count:,})",
                 ha="center", fontsize=8)
axes[1].set_title("30-Day Readmission Rate by Race", fontweight="bold")
axes[1].set_xlabel("Race"); axes[1].set_ylabel("Readmission Rate (%)")
axes[1].tick_params(axis="x", rotation=30)

plt.suptitle("Demographic Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/09_demographics.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─────────────────────────────────────────────
# 10. Clinical Factors
# ─────────────────────────────────────────────
cells.append(md("""---
## 10. Clinical Factor Analysis

Boxplots split by readmission status for key clinical features."""))

cells.append(code("""
clinical_features = ["time_in_hospital", "num_lab_procedures",
                     "num_medications", "number_diagnoses"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, clinical_features):
    sns.violinplot(data=df, x="readmit_label", y=col, ax=ax,
                   hue="readmit_label", palette=PALETTE,
                   inner="box", legend=False, density_norm="width")
    ax.set_title(col.replace("_"," ").title(), fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

plt.suptitle("Clinical Features by Readmission Status (Violin + Box)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/10_clinical_violin.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─────────────────────────────────────────────
# 11. Prior Utilization
# ─────────────────────────────────────────────
cells.append(md("""---
## 11. Prior Utilization Analysis

Outpatient, emergency, and inpatient visit counts in the year before this encounter.  
We cap values at 5+ to reduce the effect of extreme outliers."""))

cells.append(code("""
utilization_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
df_util = df.copy()
for col in utilization_cols:
    df_util[col + "_capped"] = df_util[col].clip(upper=5)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
titles = ["Prior Outpatient Visits", "Prior Emergency Visits", "Prior Inpatient Visits"]
for ax, col, title in zip(axes, utilization_cols, titles):
    cap_col = col + "_capped"
    util_rate = (df_util.groupby(cap_col)["readmitted_30_days"]
                 .agg(["mean","count"]).reset_index())
    bars = ax.bar(util_rate[cap_col].astype(str).str.replace("5","5+"),
                  util_rate["mean"]*100,
                  color=plt.cm.Blues(np.linspace(0.3, 0.9, len(util_rate))),
                  edgecolor="white")
    for bar, (_, row) in zip(bars, util_rate.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8)
    ax.set_title(f"30-Day Readmit Rate by\\n{title}", fontweight="bold")
    ax.set_xlabel("# Prior Visits (capped at 5+)")
    ax.set_ylabel("Readmission Rate (%)")

plt.suptitle("Prior Utilization vs 30-Day Readmission Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/11_prior_utilization.png", dpi=120, bbox_inches="tight")
plt.show()
print("\\n📌 Higher prior inpatient visits → markedly higher 30-day readmission rate.")
"""))

# ─────────────────────────────────────────────
# 12. Glycemic Control
# ─────────────────────────────────────────────
cells.append(md("""---
## 12. Glycemic Control & Medication Analysis

### 12.1 HbA1c Test Result"""))

cells.append(code("""
hba1c_rate = (df.groupby("A1Cresult")["readmitted_30_days"]
               .agg(["mean","count"]).reset_index()
               .sort_values("mean", ascending=False))
print("HbA1c Result vs 30-Day Readmission:")
hba1c_rate["readmit_rate%"] = (hba1c_rate["mean"]*100).round(2)
print(hba1c_rate.to_string(index=False))
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# HbA1c
hba1c_rate2 = (df.groupby("A1Cresult")["readmitted_30_days"]
                .agg(["mean","count"]).reset_index())
axes[0].bar(hba1c_rate2["A1Cresult"], hba1c_rate2["mean"]*100,
            color="#26A69A", edgecolor="white")
for i, row in hba1c_rate2.iterrows():
    axes[0].text(i, row["mean"]*100+0.1, f"{row['mean']*100:.1f}%\\n(n={row['count']:,})",
                 ha="center", fontsize=8)
axes[0].set_title("30-Day Readmit Rate\\nby HbA1c Result", fontweight="bold")
axes[0].set_xlabel("A1C Result"); axes[0].set_ylabel("Readmission Rate (%)")

# Glucose serum
gluc_rate = (df.groupby("max_glu_serum")["readmitted_30_days"]
              .agg(["mean","count"]).reset_index())
axes[1].bar(gluc_rate["max_glu_serum"], gluc_rate["mean"]*100,
            color="#EC407A", edgecolor="white")
for i, row in gluc_rate.iterrows():
    axes[1].text(i, row["mean"]*100+0.1, f"{row['mean']*100:.1f}%\\n(n={row['count']:,})",
                 ha="center", fontsize=8)
axes[1].set_title("30-Day Readmit Rate\\nby Glucose Serum Result", fontweight="bold")
axes[1].set_xlabel("Max Glucose Serum"); axes[1].set_ylabel("Readmission Rate (%)")

# Medication change
change_rate = (df.groupby("change")["readmitted_30_days"]
                .agg(["mean","count"]).reset_index())
axes[2].bar(change_rate["change"], change_rate["mean"]*100,
            color=["#42A5F5","#EF5350"], edgecolor="white")
for i, row in change_rate.iterrows():
    axes[2].text(i, row["mean"]*100+0.1, f"{row['mean']*100:.1f}%\\n(n={row['count']:,})",
                 ha="center", fontsize=10)
axes[2].set_title("30-Day Readmit Rate\\nby Medication Change", fontweight="bold")
axes[2].set_xlabel("Medication Changed?"); axes[2].set_ylabel("Readmission Rate (%)")

plt.suptitle("Glycemic Control & Medication Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/12_glycemic_control.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("### 12.2 Insulin Usage"))

cells.append(code("""
insulin_rate = (df.groupby("insulin")["readmitted_30_days"]
                 .agg(["mean","count"]).reset_index()
                 .sort_values("mean", ascending=False))
insulin_rate["readmit_rate%"] = (insulin_rate["mean"]*100).round(2)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(insulin_rate["insulin"], insulin_rate["mean"]*100,
              color=plt.cm.Oranges(np.linspace(0.4, 0.9, len(insulin_rate))),
              edgecolor="white")
for bar, (_, row) in zip(bars, insulin_rate.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{bar.get_height():.1f}%\\n(n={row['count']:,})",
            ha="center", fontsize=9)
ax.set_title("30-Day Readmit Rate by Insulin Usage", fontweight="bold")
ax.set_xlabel("Insulin Dosage Change"); ax.set_ylabel("Readmission Rate (%)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/13_insulin_analysis.png", dpi=120, bbox_inches="tight")
plt.show()
print(insulin_rate.to_string(index=False))
"""))

# ─────────────────────────────────────────────
# 13. Admission & Discharge Patterns
# ─────────────────────────────────────────────
cells.append(md("""---
## 13. Admission & Discharge Patterns

Using `IDs_mapping.csv` to decode integer IDs to meaningful labels."""))

cells.append(code("""
# Build mapping dictionaries
def get_map(ids_map, category):
    sub = ids_map[ids_map["description"].str.contains(category, case=False, na=False)] if hasattr(ids_map,"description") else pd.DataFrame()
    return {}

# Try to parse structured mapping
try:
    adm_type_map = dict(zip(
        ids_map[ids_map.iloc[:,0]=="admission_type_id"].iloc[:,1].astype(int),
        ids_map[ids_map.iloc[:,0]=="admission_type_id"].iloc[:,2]
    ))
except Exception:
    adm_type_map = {}

try:
    disc_disp_map = dict(zip(
        ids_map[ids_map.iloc[:,0]=="discharge_disposition_id"].iloc[:,1].astype(int),
        ids_map[ids_map.iloc[:,0]=="discharge_disposition_id"].iloc[:,2]
    ))
except Exception:
    disc_disp_map = {}

print("Sample Admission Type mapping:")
for k, v in list(adm_type_map.items())[:5]:
    print(f"  {k}: {v}")
print("\\nSample Discharge Disposition mapping:")
for k, v in list(disc_disp_map.items())[:5]:
    print(f"  {k}: {v}")
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Admission type readmission rate
adm_rate = (df.groupby("admission_type_id")["readmitted_30_days"]
              .agg(["mean","count"]).reset_index()
              .sort_values("mean", ascending=False).head(8))
adm_rate["label"] = adm_rate["admission_type_id"].map(adm_type_map).fillna(adm_rate["admission_type_id"].astype(str))
adm_rate["label"] = adm_rate["label"].str[:25]

axes[0].barh(adm_rate["label"][::-1], adm_rate["mean"][::-1]*100,
             color="#5C6BC0", edgecolor="white")
for i, row in adm_rate[::-1].iterrows():
    axes[0].text(row["mean"]*100+0.1, len(adm_rate)-1-adm_rate.index.get_loc(i),
                 f"{row['mean']*100:.1f}%", va="center", fontsize=8)
axes[0].set_title("30-Day Readmit Rate by Admission Type", fontweight="bold")
axes[0].set_xlabel("Readmission Rate (%)")

# Discharge disposition readmission rate
disc_rate = (df.groupby("discharge_disposition_id")["readmitted_30_days"]
               .agg(["mean","count"]).reset_index()
               .query("count > 100")
               .sort_values("mean", ascending=False).head(8))
disc_rate["label"] = disc_rate["discharge_disposition_id"].map(disc_disp_map).fillna(disc_rate["discharge_disposition_id"].astype(str))
disc_rate["label"] = disc_rate["label"].str[:30]

axes[1].barh(disc_rate["label"][::-1], disc_rate["mean"][::-1]*100,
             color="#26A69A", edgecolor="white")
for i, row in disc_rate[::-1].iterrows():
    axes[1].text(row["mean"]*100+0.1, len(disc_rate)-1-disc_rate.index.get_loc(i),
                 f"{row['mean']*100:.1f}%", va="center", fontsize=8)
axes[1].set_title("30-Day Readmit Rate by Discharge Disposition\\n(min 100 encounters)", fontweight="bold")
axes[1].set_xlabel("Readmission Rate (%)")

plt.suptitle("Admission & Discharge Patterns", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/14_admission_discharge.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─────────────────────────────────────────────
# 14. Medication Analysis
# ─────────────────────────────────────────────
cells.append(md("""---
## 14. Medication Polypharmacy Analysis"""))

cells.append(code("""
med_cols = ["metformin","repaglinide","nateglinide","chlorpropamide",
            "glimepiride","acetohexamide","glipizide","glyburide",
            "tolbutamide","pioglitazone","rosiglitazone","acarbose",
            "miglitol","troglitazone","tolazamide","examide","citoglipton",
            "insulin","glyburide-metformin","glipizide-metformin",
            "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone"]
med_cols = [c for c in med_cols if c in df.columns]

# Count how many medications are "active" (not "No")
med_active = (df[med_cols] != "No").sum().sort_values(ascending=False)
print(f"Total medication columns found: {len(med_cols)}")
print("\\nTop 10 most-used medications (active encounters):")
print(med_active.head(10).to_string())
"""))

cells.append(code("""
top_meds = med_active.head(10).index.tolist()
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for ax, med in zip(axes, top_meds):
    med_rate = (df.groupby(med)["readmitted_30_days"]
                 .agg(["mean","count"]).reset_index()
                 .sort_values("mean", ascending=False))
    ax.bar(med_rate[med], med_rate["mean"]*100,
           color=plt.cm.Purples(np.linspace(0.4, 0.9, len(med_rate))),
           edgecolor="white")
    for i, row in med_rate.iterrows():
        ax.text(i, row["mean"]*100+0.3, f"{row['mean']*100:.0f}%", ha="center", fontsize=7)
    ax.set_title(med, fontsize=9, fontweight="bold")
    ax.set_xlabel("Dosage Status"); ax.set_ylabel("Readmit Rate %")
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("30-Day Readmission Rate by Top 10 Medication Dosage", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/15_medications.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─────────────────────────────────────────────
# 15. Diagnosis Analysis
# ─────────────────────────────────────────────
cells.append(md("""---
## 15. Primary Diagnosis Analysis (ICD-9 Codes)

The `diag_1` column contains ICD-9 codes. We map them to 9 clinical groups."""))

cells.append(code("""
def map_icd9(code):
    try:
        c = str(code).upper()
        if c.startswith("V") or c.startswith("E"):
            return "External/Supplemental"
        c_num = float(c.split(".")[0])
        if 390 <= c_num <= 459 or c_num == 785:
            return "Circulatory"
        elif 460 <= c_num <= 519 or c_num == 786:
            return "Respiratory"
        elif 520 <= c_num <= 579 or c_num == 787:
            return "Digestive"
        elif c_num == 250:
            return "Diabetes"
        elif 800 <= c_num <= 999:
            return "Injury"
        elif 710 <= c_num <= 739:
            return "Musculoskeletal"
        elif 580 <= c_num <= 629 or c_num == 788:
            return "Genitourinary"
        elif 140 <= c_num <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except Exception:
        return "Other"

df["diag1_group"] = df["diag_1"].apply(map_icd9)

diag_rate = (df.groupby("diag1_group")["readmitted_30_days"]
               .agg(["mean","count"]).reset_index()
               .sort_values("mean", ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(diag_rate["diag1_group"][::-1], diag_rate["count"][::-1],
             color="#42A5F5", edgecolor="white")
axes[0].set_title("Encounter Count by Primary Diagnosis Group", fontweight="bold")
axes[0].set_xlabel("Encounter Count")

axes[1].barh(diag_rate["diag1_group"][::-1], diag_rate["mean"][::-1]*100,
             color=["#E53935" if r > 12 else "#FB8C00" if r > 10 else "#43A047"
                    for r in diag_rate["mean"][::-1]*100],
             edgecolor="white")
for i, row in diag_rate[::-1].iterrows():
    axes[1].text(row["mean"]*100+0.1, len(diag_rate)-1-diag_rate.index.get_loc(i),
                 f"{row['mean']*100:.1f}%  (n={row['count']:,})", va="center", fontsize=8)
axes[1].set_title("30-Day Readmit Rate by Primary Diagnosis", fontweight="bold")
axes[1].set_xlabel("Readmission Rate (%)")

plt.suptitle("Primary Diagnosis (ICD-9 Groups)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/16_diagnosis_groups.png", dpi=120, bbox_inches="tight")
plt.show()
print(diag_rate.to_string(index=False))
"""))

# ─────────────────────────────────────────────
# 16. Correlation Heatmap
# ─────────────────────────────────────────────
cells.append(md("""---
## 16. Correlation Heatmap

Pearson correlation between all numerical features and the binary target."""))

cells.append(code("""
corr_cols = num_features + ["readmitted_30_days"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Pearson Correlation — Numerical Features + Target", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/17_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()

# Print correlations with target
target_corr = corr_matrix["readmitted_30_days"].drop("readmitted_30_days").sort_values(key=abs, ascending=False)
print("\\nCorrelation with readmitted_30_days (sorted by absolute magnitude):")
print(target_corr.round(4).to_string())
"""))

# ─────────────────────────────────────────────
# 17. Statistical Hypothesis Tests
# ─────────────────────────────────────────────
cells.append(md("""---
## 17. Statistical Hypothesis Tests

### 17.1 Mann-Whitney U Test (Numerical Features)

Non-parametric test comparing feature distributions between readmitted and not-readmitted groups.  
H₀: The two groups come from the same distribution."""))

cells.append(code("""
group0 = df[df["readmitted_30_days"] == 0]
group1 = df[df["readmitted_30_days"] == 1]

print(f"{'Feature':<25} {'U-statistic':>14} {'p-value':>12} {'Significant':>13}")
print("-" * 70)
mw_results = []
for col in num_features:
    u_stat, p_val = stats.mannwhitneyu(
        group0[col].dropna(), group1[col].dropna(), alternative="two-sided"
    )
    sig = "✓ YES" if p_val < 0.05 else "✗ NO"
    mw_results.append({"Feature": col, "U-stat": u_stat, "p-value": p_val, "Significant": sig})
    print(f"{col:<25} {u_stat:>14,.0f} {p_val:>12.4e} {sig:>13}")

print("\\nAll features are statistically different between groups (p < 0.05)?",
      all(r['p-value'] < 0.05 for r in mw_results))
"""))

cells.append(md("### 17.2 Chi-Square Test + Cramér's V (Categorical Features)"))

cells.append(code("""
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    r, c = ct.shape
    v = np.sqrt(chi2 / (n * (min(r, c) - 1)))
    return chi2, p, v

test_cats = ["race", "gender", "age", "admission_type_id",
             "discharge_disposition_id", "diabetesMed", "change",
             "A1Cresult", "max_glu_serum", "insulin"]
test_cats = [c for c in test_cats if c in df.columns]

print(f"{'Feature':<30} {'Chi2':>12} {'p-value':>12} {'Cramer V':>10} {'Effect':>10}")
print("-" * 80)
for col in test_cats:
    chi2, p, v = cramers_v(df[col].fillna("Unknown"), df["readmitted_30_days"])
    effect = "Large" if v > 0.3 else "Medium" if v > 0.1 else "Small"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{col:<30} {chi2:>12,.1f} {p:>12.4e} {v:>10.4f} {effect + sig:>10}")
"""))

# ─────────────────────────────────────────────
# 18. Medical Specialty
# ─────────────────────────────────────────────
cells.append(md("""---
## 18. Medical Specialty Analysis"""))

cells.append(code("""
df_spec = df[df["medical_specialty"] != "?"].copy()
spec_rate = (df_spec.groupby("medical_specialty")["readmitted_30_days"]
               .agg(["mean","count"]).reset_index()
               .query("count >= 200")
               .sort_values("mean", ascending=False).head(15))

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(spec_rate["medical_specialty"][::-1],
               spec_rate["mean"][::-1]*100,
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(spec_rate))),
               edgecolor="white")
for bar, (_, row) in zip(bars, spec_rate[::-1].iterrows()):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
            f"{bar.get_width():.1f}% (n={row['count']:,})",
            va="center", fontsize=8)
ax.set_title("Top 15 Medical Specialties by 30-Day Readmission Rate\\n(min 200 encounters)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("30-Day Readmission Rate (%)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts/plots/eda/18_medical_specialty.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─────────────────────────────────────────────
# 19. Feature Engineering Preview
# ─────────────────────────────────────────────
cells.append(md("""---
## 19. Feature Engineering Preview

Applying the production feature-engineering pipeline to preview engineered features."""))

cells.append(code("""
from src.cleansing.data_cleaner import clean_data
from src.feature_engineering.feature_builder import build_all_features, get_feature_columns

df_clean = clean_data(df.drop(columns=["readmit_label","diag1_group","readmitted_30_days"],
                               errors="ignore"))
df_eng   = build_all_features(df_clean)
feat_cols = get_feature_columns()
print(f"\\nAfter cleaning  : {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
print(f"After engineering: {df_eng.shape[0]:,} rows × {df_eng.shape[1]} cols")
print(f"\\nEngineered feature columns ({len(feat_cols['numerical'])+len(feat_cols['categorical'])}):")
print(f"  Numerical ({len(feat_cols['numerical'])}): {feat_cols['numerical']}")
print(f"  Categorical ({len(feat_cols['categorical'])}): {feat_cols['categorical']}")
"""))

cells.append(code("""
# Show engineered features
all_feat = feat_cols["numerical"] + feat_cols["categorical"]
available = [f for f in all_feat if f in df_eng.columns]
df_eng[available].head(5)
"""))

# ─────────────────────────────────────────────
# 20. Summary & Conclusions
# ─────────────────────────────────────────────
cells.append(md("""---
## 20. EDA Summary & Conclusions

| # | Finding | Action |
|---|---------|--------|
| 1 | **11.2% positive class** (severe imbalance) | Use SMOTE + class-weight; primary metric = PR-AUC |
| 2 | **Patient leakage**: 71K unique patients, 1.42 visits avg | GroupShuffleSplit on `patient_nbr` mandatory |
| 3 | **`weight` column** 96.9% missing | Drop entirely |
| 4 | **`medical_specialty`** 49% missing | Encode as 'Missing' category |
| 5 | **Terminal discharges** (codes 11,13,14,19,20,21) | Exclude — patient cannot be readmitted |
| 6 | **Prior inpatient visits** strongest utilization predictor | Feature: `number_inpatient_capped` |
| 7 | **Age** shows clear monotonic increase in readmission rate | Encode age midpoint + is_senior flag |
| 8 | **Circulatory diagnoses** highest readmission rate | ICD-9 group feature |
| 9 | **Medication changes** (Ch vs No) → higher readmit rate | Binary `change` feature |
| 10 | **HbA1c & glucose tests** often not taken (>80% 'None') | 'None' is informative — keep as category |

### Next Steps → `2. MODEL TRAINING.ipynb`

1. **Clean** data using `src/cleansing/data_cleaner.py`
2. **Engineer features** using `src/feature_engineering/feature_builder.py`
3. **Split** using `GroupShuffleSplit` on `patient_nbr`
4. **Preprocess** using `src/preprocessing/preprocessing_pipeline.py`
5. **Train 11 models** (Logistic, RF variants, XGBoost variants, SMOTE, threshold-tuned)
6. **Select** best model by ROC-AUC + PR-AUC on held-out test set
7. **Explain** with SHAP TreeExplainer
"""))

# ─────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Notebook written: {NB_PATH}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")

