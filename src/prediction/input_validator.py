"""Input validation and sanitization for patient inference payloads."""

from typing import Any, Dict, List, Tuple


# Valid ranges and categories based on UCI Diabetes dataset
ALLOWED_CATEGORIES: Dict[str, List[str]] = {
    "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing/Unknown"],
    "gender": ["Female", "Male"],
    "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
    "admission_type": ["Emergency", "Urgent", "Elective", "Newborn", "Not Available", "NULL", "Trauma Center", "Unknown"],
    "discharge_disposition": ["Discharged to home", "Discharged/transferred to SNF", "Discharged/transferred to home with home health service", "Discharged/transferred to another short term hospital", "Left AMA", "Unknown"],
    "admission_source": ["Emergency Room", "Physician Referral", "Clinic Referral", "Transfer from a hospital", "HMO Referral", "Unknown"],
    "medical_specialty": ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology", "Surgery", "Orthopedics", "Nephrology", "Other_Specialty", "Missing/Unknown"],
    "diag_1_group": ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other/Unknown"],
    "diag_2_group": ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other/Unknown"],
    "diag_3_group": ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other/Unknown"],
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
    "A1Cresult": ["None", "Norm", ">7", ">8"],
    "insulin": ["No", "Steady", "Up", "Down"],
    "change": ["No", "Ch"],
    "diabetesMed": ["No", "Yes"],
}

NUMERICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 150),
    "num_procedures": (0, 10),
    "num_medications": (1, 100),
    "number_outpatient": (0, 50),
    "number_emergency": (0, 50),
    "number_inpatient": (0, 50),
    "number_diagnoses": (1, 20),
}


def validate_patient_payload(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validates and sanitizes single patient encounter payload.

    Args:
        data: Dict of raw incoming user input.

    Returns:
        (is_valid, sanitized_dict, errors_list)
    """
    errors: List[str] = []
    sanitized: Dict[str, Any] = {}

    # 1. Validate numerical features
    for col, (min_val, max_val) in NUMERICAL_BOUNDS.items():
        val = data.get(col)
        if val is None or val == "":
            errors.append(f"Missing required numeric field: '{col}'")
            continue
        try:
            num_val = float(val)
            if num_val < min_val or num_val > max_val:
                errors.append(f"Field '{col}' must be between {min_val} and {max_val} (received {num_val}).")
            else:
                sanitized[col] = int(num_val) if num_val.is_integer() else num_val
        except (ValueError, TypeError):
            errors.append(f"Field '{col}' must be a valid number (received '{val}').")

    # 2. Validate categorical features
    for col, allowed in ALLOWED_CATEGORIES.items():
        val = data.get(col)
        if val is None or val == "":
            # Assign neutral/unknown default
            if col in ["max_glu_serum", "A1Cresult"]:
                sanitized[col] = "None"
            elif col == "medical_specialty":
                sanitized[col] = "Missing/Unknown"
            elif col in ["diag_1_group", "diag_2_group", "diag_3_group"]:
                sanitized[col] = "Other/Unknown"
            elif col in ["race", "admission_type", "discharge_disposition", "admission_source"]:
                sanitized[col] = "Unknown"
            else:
                errors.append(f"Missing required categorical field: '{col}'")
        else:
            str_val = str(val).strip()
            if str_val not in allowed:
                # If close match or unlisted, map to Unknown/Other if permitted, else report error
                if "Unknown" in allowed or "Other/Unknown" in allowed or "Missing/Unknown" in allowed:
                    sanitized[col] = "Unknown" if "Unknown" in allowed else ("Other/Unknown" if "Other/Unknown" in allowed else "Missing/Unknown")
                else:
                    errors.append(f"Invalid option for '{col}': '{str_val}'. Allowed: {allowed}")
            else:
                sanitized[col] = str_val

    is_valid = len(errors) == 0
    return is_valid, sanitized, errors

