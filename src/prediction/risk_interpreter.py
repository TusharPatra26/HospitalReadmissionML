"""Clinical risk banding, recommendations, and plain-language interpretation."""

from typing import Any, Dict


def interpret_readmission_risk(
    probability: float,
    threshold: float = 0.130,
) -> Dict[str, Any]:
    """Interprets model readmission probability into project-defined clinical screening tiers.

    Note: Risk tiers are heuristic research categories, not clinically validated diagnostic criteria.
    """
    prob_pct = round(probability * 100, 1)

    if probability < 0.10:
        tier = "Lower Risk Tier"
        color = "#10B981"  # Emerald Green
        badge_class = "success"
        recommendations = [
            "Standard routine discharge instructions and self-management education.",
            "Schedule standard outpatient follow-up within 14–21 days.",
            "Verify patient understanding of diabetes medications and supply access.",
        ]
        rationale = "The estimated probability falls below both the cohort average (~11.3%) and the optimized decision threshold."
    elif probability < 0.25:
        tier = "Moderate Risk Tier"
        color = "#F59E0B"  # Amber
        badge_class = "warning"
        recommendations = [
            "Conduct structured post-discharge telephone follow-up within 48–72 hours.",
            "Comprehensive medication reconciliation to assess insulin titration and adherence.",
            "Expedite outpatient primary care or endocrinology visit within 7–10 days.",
            "Review prior emergency department or inpatient visit triggers.",
        ]
        rationale = (
            f"The estimated probability ({prob_pct}%) exceeds the screening decision threshold ({threshold*100:.1f}%), "
            "warranting proactive transitional care monitoring."
        )
    else:
        tier = "Elevated Risk Tier"
        color = "#EF4444"  # Coral Red
        badge_class = "danger"
        recommendations = [
            "High-intensity transitional care management (TCM) enrollment prior to discharge.",
            "Multi-disciplinary bedside medication review and diabetes educator consultation.",
            "Direct appointment booking with PCP within 5 business days.",
            "Identify home health nursing or community paramedicine support needs.",
        ]
        rationale = (
            f"The estimated probability ({prob_pct}%) is more than double the cohort average prevalence. "
            "Patient demonstrates compounding utilization and clinical comorbidity signals."
        )

    disclaimer = (
        "Academic & Research Prototype Notice: This prediction is an algorithmic estimate generated from "
        "historical de-identified hospital encounter records (1999–2008). It is not a clinical diagnosis and "
        "must not be used as the sole basis for discharge timing, treatment selection, or patient management."
    )

    return {
        "probability_percentage": prob_pct,
        "raw_probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "is_readmission_predicted": probability >= threshold,
        "risk_tier": tier,
        "badge_class": badge_class,
        "color": color,
        "rationale": rationale,
        "actionable_recommendations": recommendations,
        "disclaimer": disclaimer,
    }

