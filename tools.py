"""Tools the assistant can call on its own.

The LLM decides which tool to use and fills in the arguments.
These functions do the real work, so every number the assistant
reports comes from the trained model, never from the LLM itself.
"""
from api import EVALUATION, FEATURES, LABELS, META, SCHEMA, caveats, default_for, score

import json
from pathlib import Path as _Path

# Global feature-importance ranking, precomputed in models/shap_comparison.json.
try:
    _SHAP = json.loads((_Path(__file__).parent / "models" / "shap_comparison.json").read_text())
    _TOP_DRIVERS = _SHAP.get("shap_top5", {})
except (FileNotFoundError, json.JSONDecodeError):
    _TOP_DRIVERS = {}


def check_fields(fields: dict) -> dict:
    """Reject unknown fields and invalid values instead of guessing."""
    clean = {}
    for name, value in fields.items():
        if name in SCHEMA["numeric"]:
            info = SCHEMA["numeric"][name]
            number = float(value)
            if not info["min"] <= number <= info["max"]:
                raise ValueError(f"{name} must be between {info['min']} and {info['max']}, got {value}")
            clean[name] = number
        elif name in SCHEMA["categorical"]:
            options = SCHEMA["categorical"][name]["options"]
            if str(value) not in options:
                raise ValueError(f"{name} must be one of {options}, got {value!r}")
            clean[name] = str(value)
        else:
            raise ValueError(f"unknown field {name!r}. Valid fields: {FEATURES}")
    return clean


def get_field_options() -> dict:
    """Every field the model uses, its valid values, and what the ID codes mean."""
    return {"numeric": SCHEMA["numeric"], "categorical": SCHEMA["categorical"], "code_labels": LABELS}


def predict_patient(fields: dict) -> dict:
    """Risk for a patient described by only some fields. Missing fields use typical values."""
    given = check_fields(fields)
    patient = {f: given.get(f, default_for(f)) for f in FEATURES}
    result = score(patient)
    result["fields_given"] = given
    result["fields_assumed"] = {f: patient[f] for f in FEATURES if f not in given}
    return result


def what_if(fields: dict, changes: dict) -> dict:
    """Same patient, before and after changing some fields."""
    before = predict_patient(fields)
    after = predict_patient({**fields, **changes})
    return {
        "before": {"probability": before["probability"], "band": before["band"]},
        "after": {"probability": after["probability"], "band": after["band"]},
        "change_in_probability": round(after["probability"] - before["probability"], 4),
        "changes": check_fields(changes),
        "caveats": list(dict.fromkeys(before["caveats"] + after["caveats"])),
    }


def get_evaluation() -> dict:
    """How predicted risk compared with real readmissions for held-out patients."""
    if EVALUATION is None:
        return {"error": "no held-out sample shipped with this model"}
    return EVALUATION


def get_model_info() -> dict:
    """What the model is, how it was trained, its scores, and where it is known to fail."""
    known_limitations = (
        caveats({"age": "[90-100)", "race": "Caucasian"})
        + caveats({"race": "AfricanAmerican"})
        + caveats({"race": ""})
    )
    return {
        "model": META["model"],
        "cv_pr_auc": round(META["cv_pr_auc"], 4),
        "cv_pr_auc_std": round(META["cv_pr_auc_std"], 4),
        "baseline_readmission_rate": round(META["baseline_positive_rate"], 4),
        "all_cv_scores": {name: round(s, 4) for name, s in META["all_scores"].items()},
        "n_encounters": META["n_rows"],
        "trained_on": META["trained_on"],
        "known_limitations": known_limitations,
        "top_drivers_overall": _TOP_DRIVERS,
    }
