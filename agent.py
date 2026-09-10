"""Ask questions about the readmission model in plain English.

Runs 100% locally with Ollama: no API key, no internet, no cost.
The model decides which tool to call; the tools in tools.py do the real work.

    python3 agent.py
"""
import json
import os

import ollama

import tools
from api import SCHEMA

MODEL = os.getenv("ASSISTANT_MODEL", "llama3.1:8b")
MAX_TOOL_ROUNDS = 4  # stop a confused model from looping forever

# What each field means, so the model can map plain English to field names.
FIELD_MEANINGS = {
    "time_in_hospital": "days in hospital this stay",
    "num_lab_procedures": "lab tests this stay",
    "num_procedures": "non-lab procedures this stay",
    "num_medications": "distinct medications this stay",
    "number_outpatient": "outpatient visits in the year before",
    "number_emergency": "emergency visits in the year before",
    "number_inpatient": "inpatient hospital stays in the year before",
    "number_diagnoses": "diagnoses recorded",
    "race": "race",
    "gender": "gender",
    "age": "age as a 10-year bracket; a 75-year-old is [70-80)",
    "admission_type_id": "admission type code (1 = Emergency; call get_field_options for others)",
    "discharge_disposition_id": "where the patient went after discharge (1 = home; call get_field_options for others)",
    "admission_source_id": "where the patient came from (7 = emergency room; call get_field_options for others)",
    "A1Cresult": "HbA1c test result",
    "max_glu_serum": "glucose serum test result",
    "insulin": "insulin dose change",
    "change": "diabetes medication changed (Ch) or not (No)",
    "diabetesMed": "on diabetes medication",
    "diag_1_group": "primary diagnosis group",
    "diag_2_group": "secondary diagnosis group",
    "diag_3_group": "additional diagnosis group",
}


def field_guide() -> str:
    lines = []
    for name, info in SCHEMA["numeric"].items():
        lines.append(f"- {name}: {FIELD_MEANINGS[name]} (number {info['min']:g}-{info['max']:g})")
    for name, info in SCHEMA["categorical"].items():
        options = info["options"]
        shown = ", ".join(options) if len(options) <= 10 else "codes"
        lines.append(f"- {name}: {FIELD_MEANINGS[name]} ({shown})")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You answer questions about a machine learning model that predicts 30-day hospital
readmission for diabetic patients (UCI Diabetes 130-US Hospitals data).

Rules:
1. Every number you state must come from a tool result. Never estimate or calculate risk yourself.
2. For a patient's risk, call predict_patient with only the details the user gave.
3. For "what if" questions about a patient already discussed, call what_if with that
   patient's fields and the changes.
4. If a tool returns an error, fix the field names or values and call it again.
5. When you report a prediction, say which details were given and that every other field
   was set to a typical value.
6. Always repeat any caveats the tool returns.
7. Use the *_pct values for percentages.
8. You do not give medical advice. If asked whether to discharge, treat, or change a patient's
   care, say this is a research demo and must not be used for clinical decisions.
9. Answer in 2-4 plain sentences. No bullet points, no headings, no lists.
10. Explain scores using the how_to_read note in the tool result. Never call PR-AUC or ROC-AUC
    "accuracy" or "precision".

Fields the model uses:
{field_guide()}"""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "predict_patient",
            "description": "Predict 30-day readmission risk for a patient. Pass only the fields the user mentioned; the rest use typical values.",
            "parameters": {
                "type": "object",
                "properties": {"fields": {"type": "object", "description": "Field name -> value"}},
                "required": ["fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "what_if",
            "description": "Compare the same patient's risk before and after changing some fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {"type": "object", "description": "The patient's original fields"},
                    "changes": {"type": "object", "description": "Only the fields that change, with new values"},
                },
                "required": ["fields", "changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evaluation",
            "description": "How accurate the model is: predicted risk vs real readmissions on held-out patients, by risk band.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "What the model is, its cross-validated scores, and the groups where it is known to perform poorly.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_field_options",
            "description": "Every field's valid values and what the admission/discharge ID codes mean.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_FUNCTIONS = {
    "predict_patient": tools.predict_patient,
    "what_if": tools.what_if,
    "get_evaluation": tools.get_evaluation,
    "get_model_info": tools.get_model_info,
    "get_field_options": tools.get_field_options,
}

RATE_KEYS = {"probability", "baseline_rate", "baseline_readmission_rate", "predicted_mean",
             "observed_rate", "change_in_probability"}


def add_percents(value):
    """Add ready-made percentages so the model never has to do the conversion itself."""
    if isinstance(value, dict):
        out = {k: add_percents(v) for k, v in value.items()}
        for k, v in value.items():
            if k in RATE_KEYS and isinstance(v, (int, float)):
                out[f"{k}_pct"] = f"{v * 100:.1f}%"
        return out
    if isinstance(value, list):
        return [add_percents(v) for v in value]
    return value


HOW_TO_READ = {
    "get_evaluation": (
        "n held-out patients the model never saw in training. baseline_rate_pct: share actually readmitted. "
        "roc_auc: how well the model ranks patients by risk, where 0.5 is random guessing and 1.0 is perfect. "
        "It is NOT accuracy. pr_auc: ranking quality focused on readmitted patients; compare it to "
        "baseline_rate, which is what random guessing would score. It is NOT precision. "
        "bands: patients grouped by predicted risk. predicted_mean_pct is the average risk the model predicted "
        "for that group; observed_rate_pct is the share of that group actually readmitted; observed_ci95 is the "
        "95% range for that share, which is wide when the group is small."
    ),
    "get_model_info": (
        "cv_pr_auc: ranking quality from 5-fold cross-validation, compared with baseline_readmission_rate, "
        "which is what random guessing would score. It is NOT accuracy or precision. all_cv_scores compares "
        "gradient boosting (boost) with logistic regression (logreg). known_limitations: groups where the "
        "model performs poorly; mention them when asked about accuracy or fairness."
    ),
}


def run_tool(name: str, arguments) -> dict:
    """Run one tool call. Errors go back to the model so it can correct itself."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"unknown tool {name!r}. Available: {list(TOOL_FUNCTIONS)}"}
    if isinstance(arguments, str):  # small models sometimes send JSON as a string
        arguments = json.loads(arguments or "{}")
    for key in ("fields", "changes"):  # ...or a nested object as a string
        if isinstance(arguments.get(key), str):
            arguments[key] = json.loads(arguments[key])
    try:
        result = add_percents(TOOL_FUNCTIONS[name](**arguments))
        if name in HOW_TO_READ:
            result["how_to_read"] = HOW_TO_READ[name]
        return result
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return {"error": str(e)}


def ask(messages: list, chat=ollama.chat) -> tuple[str, list]:
    """Send the conversation to the model, run any tools it picks, return its answer."""
    tool_log = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOL_SPECS,
            options={"temperature": 0, "num_ctx": 8192},
        )
        message = response.message
        messages.append(message)
        if not message.tool_calls:
            return message.content, tool_log
        for call in message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            tool_log.append({"tool": call.function.name, "arguments": call.function.arguments, "result": result})
            messages.append({"role": "tool", "tool_name": call.function.name, "content": json.dumps(result)})
    return "Sorry, I couldn't work that out. Try rephrasing the question.", tool_log


if __name__ == "__main__":
    print(f"Ask about the readmission model (local {MODEL}). Type 'quit' to stop.\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        history.append({"role": "user", "content": question})
        answer, tool_log = ask(history)
        for entry in tool_log:
            print(f"  [tool] {entry['tool']}({entry['arguments']})")
        print(f"Assistant: {answer}\n")
