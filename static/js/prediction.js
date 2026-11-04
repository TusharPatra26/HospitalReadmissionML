/**
 * Interactive Client-Side Logic for Patient Readmission Prediction & Local SHAP
 */

const DEMO_PERSONAS = {
  low: {
    race: "Caucasian",
    gender: "Male",
    age: "[20-30)",
    time_in_hospital: 2,
    admission_type: "Elective",
    discharge_disposition: "Discharged to home",
    admission_source: "Physician Referral",
    medical_specialty: "InternalMedicine",
    num_lab_procedures: 20,
    num_procedures: 1,
    num_medications: 6,
    number_outpatient: 0,
    number_emergency: 0,
    number_inpatient: 0,
    number_diagnoses: 3,
    diag_1_group: "Other/Unknown",
    diag_2_group: "Other/Unknown",
    diag_3_group: "Other/Unknown",
    max_glu_serum: "None",
    A1Cresult: "None",
    insulin: "No",
    change: "No",
    diabetesMed: "No"
  },
  moderate: {
    race: "Caucasian",
    gender: "Female",
    age: "[60-70)",
    time_in_hospital: 4,
    admission_type: "Emergency",
    discharge_disposition: "Discharged to home",
    admission_source: "Emergency Room",
    medical_specialty: "InternalMedicine",
    num_lab_procedures: 48,
    num_procedures: 0,
    num_medications: 16,
    number_outpatient: 1,
    number_emergency: 0,
    number_inpatient: 1,
    number_diagnoses: 7,
    diag_1_group: "Circulatory",
    diag_2_group: "Diabetes",
    diag_3_group: "Respiratory",
    max_glu_serum: "None",
    A1Cresult: "Norm",
    insulin: "Steady",
    change: "No",
    diabetesMed: "Yes"
  },
  high: {
    race: "AfricanAmerican",
    gender: "Female",
    age: "[70-80)",
    time_in_hospital: 8,
    admission_type: "Emergency",
    discharge_disposition: "Discharged/transferred to SNF",
    admission_source: "Emergency Room",
    medical_specialty: "Emergency/Trauma",
    num_lab_procedures: 78,
    num_procedures: 2,
    num_medications: 26,
    number_outpatient: 2,
    number_emergency: 1,
    number_inpatient: 3,
    number_diagnoses: 9,
    diag_1_group: "Circulatory",
    diag_2_group: "Diabetes",
    diag_3_group: "Genitourinary",
    max_glu_serum: ">300",
    A1Cresult: ">8",
    insulin: "Up",
    change: "Ch",
    diabetesMed: "Yes"
  }
};

let shapChart = null;

function loadPersona(type) {
  const data = DEMO_PERSONAS[type];
  if (!data) return;

  const form = document.getElementById("patientPredictionForm");
  for (const [key, value] of Object.entries(data)) {
    const input = form.elements[key];
    if (input) {
      input.value = value;
    }
  }

  // Highlight selected persona
  document.querySelectorAll(".persona-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`btn-persona-${type}`)?.classList.add("active");
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("patientPredictionForm");
  const resultCard = document.getElementById("predictionResultCard");
  const btnPredict = document.getElementById("btnPredict");

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btnPredict.disabled = true;
    btnPredict.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing Encounter...`;

    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = value;
    });

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const res = await response.json();
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<i class="fa-solid fa-calculator me-2"></i>Predict 30-Day Readmission Risk`;

      if (!res.success) {
        alert("Validation error: " + res.errors.join("\n"));
        return;
      }

      // Display results card
      resultCard.style.display = "block";
      resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

      const probPct = res.probability_pct;
      const interp = res.risk_interpretation;
      const shap = res.local_explanation;

      // Update UI elements
      document.getElementById("resProb").innerText = `${probPct}%`;
      document.getElementById("resThreshold").innerText = `Optimized Threshold: ${(res.threshold * 100).toFixed(1)}%`;
      
      const badge = document.getElementById("resRiskTierBadge");
      badge.innerText = interp.risk_tier;
      badge.className = `badge ${interp.badge_class === 'danger' ? 'badge-high' : (interp.badge_class === 'warning' ? 'badge-mod' : 'badge-low')}`;

      document.getElementById("resRationale").innerText = interp.rationale;
      document.getElementById("resNarrative").innerText = shap.summary_narrative;

      // Populate recommendations list
      const recList = document.getElementById("resRecommendations");
      recList.innerHTML = "";
      interp.actionable_recommendations.forEach(rec => {
        const li = document.createElement("li");
        li.className = "mb-1 text-secondary";
        li.innerHTML = `<i class="fa-solid fa-check text-success me-2"></i>${rec}`;
        recList.appendChild(li);
      });

      // Render local SHAP waterfall chart
      renderLocalShapChart(shap.top_positive_factors, shap.top_negative_factors);

    } catch (err) {
      console.error(err);
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<i class="fa-solid fa-calculator me-2"></i>Predict 30-Day Readmission Risk`;
      alert("Error occurred while generating prediction.");
    }
  });
});

function renderLocalShapChart(posFactors, negFactors) {
  const canvas = document.getElementById("localShapCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  // Combine top positive and negative
  const factors = [...negFactors.slice(0, 4), ...posFactors.slice(0, 4)];
  factors.sort((a, b) => a.shap_value - b.shap_value);

  const labels = factors.map(f => f.feature_label);
  const values = factors.map(f => f.shap_value);
  const bgColors = values.map(v => v > 0 ? "rgba(239, 68, 68, 0.85)" : "rgba(13, 148, 136, 0.85)");

  if (shapChart) {
    shapChart.destroy();
  }

  shapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "SHAP Impact (Log-Odds)",
        data: values,
        backgroundColor: bgColors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw;
              return val > 0 
                ? `Elevates Risk (+${val.toFixed(3)} log-odds)` 
                : `Attenuates Risk (${val.toFixed(3)} log-odds)`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Attribution Impact on Model Log-Odds (SHAP Value)"
          },
          grid: { color: "#E2E8F0" }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}

