"""Ask questions about the readmission model in plain English.

The model decides which tool to call; the tools in tools.py do the real work.

Two backends, picked with ASSISTANT_BACKEND:
  ollama (default)  local, no API key, no internet, no cost
  groq              hosted, needs GROQ_API_KEY in the environment

    python3 agent.py
    ASSISTANT_BACKEND=groq python3 agent.py
"""
import json
import os

import tools
from api import SCHEMA

BACKEND = os.getenv("ASSISTANT_BACKEND", "ollama").strip().lower()
DEFAULT_MODELS = {"ollama": "llama3.1:8b", "groq": "openai/gpt-oss-20b"}
if BACKEND not in DEFAULT_MODELS:
    raise SystemExit(f"ASSISTANT_BACKEND must be one of {sorted(DEFAULT_MODELS)}, got {BACKEND!r}")
MODEL = os.getenv("ASSISTANT_MODEL") or DEFAULT_MODELS[BACKEND]
MAX_TOOL_ROUNDS = 6  # stop a confused model from looping forever

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
2. Call predict_patient whenever the user gives ANY patient detail or asks for a risk number, even just an age like "a 92-year-old" - predict_patient fills every unspecified field with a typical value, so a partial description is enough. Never ask the user for more details before predicting. Only for greetings, small talk, or questions not about a patient or the model do you reply briefly with NO tool call.
3. For "what if" questions about a patient already discussed, call what_if with that
   patient's fields and the changes.
4. If a tool returns an error, fix the field names or values and call it again.
5. When you report a prediction, say which details were given and that every other field
   was set to a typical value.
6. Always repeat any caveats the tool returns.
7. Use the *_pct values for percentages.
8. You do not give medical advice. If the question asks what should be done for a patient --
   whether to discharge, admit, treat, medicate, or otherwise change their care -- do NOT call
   any tool and do NOT state any risk number, percentage, or band. Reply with the refusal only:
   this is a research demo and must not be used for clinical decisions, and the decision belongs
   to the treating clinician. This rule overrides rule 2: a request for a care decision is not a
   request for a prediction, even if the user describes the patient in detail. Only if the user
   then asks separately for the risk number may you call predict_patient.
9. Answer in 2-4 plain sentences. No bullet points, no headings, no lists.
10. Explain scores using the how_to_read note in the tool result. Never call PR-AUC or ROC-AUC
    "accuracy" or "precision".
11. Questions about fairness or performance across race, age, or other groups are legitimate and
    expected. Call get_model_info and report the model's own measured numbers and known_limitations
    plainly. This is transparency about the model's measured behavior, not a statement about any group.
12. For "how accurate is THIS model" or its overall performance numbers, call get_evaluation
    (held-out results) and report its ROC-AUC. But if the user only asks what a metric MEANS
    (e.g. "what does PR-AUC mean?"), just explain the concept in plain words with NO tool call.
    Use get_model_info only for what the model is or for group/fairness questions.
13. For "what drives the model" or "most important features", call get_model_info and report
    top_drivers_overall in order (the first key is the strongest driver).

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
            "description": "How accurate the model is on held-out patients it never saw in training: ROC-AUC, PR-AUC, and predicted vs real readmission rates by risk band. Use this for any question about accuracy or performance.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "What the model IS: its type, training data, and the groups where it is known to perform poorly (fairness/limitations). Use for what-it-is and fairness questions, NOT for a general accuracy number - use get_evaluation for that.",
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
    try:
        if isinstance(arguments, str):  # Groq always sends JSON as a string; Ollama sometimes does
            arguments = json.loads(arguments or "{}")
        if not isinstance(arguments, dict):
            return {"error": f"arguments must be a JSON object, got {type(arguments).__name__}"}
        # Some models write {"": ""} rather than {} for a tool that takes no arguments.
        # Unpacking that raised "unexpected keyword argument ''", and the model answered
        # the error by sending the very same call again, burning a round each time.
        arguments = {k: v for k, v in arguments.items() if k != ""}
        for key in ("fields", "changes"):  # ...or a nested object as a string
            if isinstance(arguments.get(key), str):
                arguments[key] = json.loads(arguments[key])
        result = add_percents(TOOL_FUNCTIONS[name](**arguments))
        if name in HOW_TO_READ:
            result["how_to_read"] = HOW_TO_READ[name]
        return result
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return {"error": str(e)}


# --- Backends ---------------------------------------------------------------
# Both providers take the same TOOL_SPECS and speak the same tool-calling loop.
# They differ in where the assistant turn sits in the response, how it has to be
# echoed back, and how a tool result is addressed. Each send/result pair below
# hides those differences so ask() stays one loop.
#
# send(messages) -> (message to append, answer text, [(tool name, raw arguments, call id)])

def _ollama_send(messages: list):
    import ollama

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=TOOL_SPECS,
        options={"temperature": 0, "num_ctx": 8192},
    )
    message = response.message  # Ollama puts it at the top level
    # Ollama hands back arguments already parsed into a dict.
    calls = [(c.function.name, c.function.arguments, None) for c in (message.tool_calls or [])]
    return message, message.content, calls  # appending the object itself is what Ollama expects


def _ollama_tool_result(name: str, call_id, result: dict) -> dict:
    return {"role": "tool", "tool_name": name, "content": json.dumps(result)}


_groq_client = None


def _groq_send(messages: list):
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq()  # reads GROQ_API_KEY from the environment

    response = _groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=TOOL_SPECS,
        temperature=0,
    )
    message = response.choices[0].message  # Groq wraps it in choices
    tool_calls = message.tool_calls or []
    # Groq gives arguments as a JSON string; run_tool parses it.
    calls = [(c.function.name, c.function.arguments, c.id) for c in tool_calls]
    # Groq will not accept its own response object back, so rebuild the assistant
    # turn as plain OpenAI-shaped JSON.
    echo = {"role": "assistant", "content": message.content or ""}
    if tool_calls:
        echo["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in tool_calls
        ]
    return echo, message.content, calls


def _groq_tool_result(name: str, call_id, result: dict) -> dict:
    # OpenAI shape: a tool result must name the call it answers.
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}


BACKENDS = {
    "ollama": (_ollama_send, _ollama_tool_result),
    "groq": (_groq_send, _groq_tool_result),
}


def ask(messages: list, backend: str = None) -> tuple[str, list]:
    """Send the conversation to the model, run any tools it picks, return its answer."""
    send, tool_result = BACKENDS[backend or BACKEND]
    tool_log = []
    for _ in range(MAX_TOOL_ROUNDS):
        message, content, calls = send(messages)
        messages.append(message)
        if not calls:
            return content or "", tool_log
        for name, arguments, call_id in calls:
            result = run_tool(name, arguments)
            tool_log.append({"tool": name, "arguments": arguments, "result": result})
            messages.append(tool_result(name, call_id, result))
    return "Sorry, I couldn't work that out. Try rephrasing the question.", tool_log


if __name__ == "__main__":
    where = "local" if BACKEND == "ollama" else BACKEND
    print(f"Ask about the readmission model ({where} {MODEL}). Type 'quit' to stop.\n")
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
