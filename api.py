import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PIPELINE = joblib.load("models/pipeline.joblib")
META = json.loads(Path("models/metadata.json").read_text())
SCHEMA = json.loads(Path("models/schema.json").read_text())
LABELS_PATH = Path("models/labels.json")   # code -> description for the three ID fields (optional)
LABELS = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
FEATURES = META["numeric"] + META["categorical"]

app = FastAPI(title="Readmission Risk", version="1.0")

# uvicorn owns the handlers in production; piggyback so these reach Render logs.
log = logging.getLogger("uvicorn.error")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- pages ----------------------------------------------------------------
# One static file per page; each carries the same header/nav and marks its own
# link with aria-current. Kept out of the OpenAPI schema, which documents the API.

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/home.html")


@app.get("/about", include_in_schema=False)
def about():
    return FileResponse("static/about.html")


@app.get("/how-it-works", include_in_schema=False)
def how_it_works():
    return FileResponse("static/how.html")


@app.get("/try", include_in_schema=False)
def try_it():
    return FileResponse("static/index.html")


# GET /ask is the chat page; POST /ask (below) is the assistant endpoint it calls.
# FastAPI routes on path *and* method, so the two coexist on the same URL.
@app.get("/ask", include_in_schema=False)
def ask_page():
    return FileResponse("static/ask.html")


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Feature name -> value")


BANDS = [("low", 0.0, 0.15), ("elevated", 0.15, 0.25), ("high", 0.25, 1.01)]


def band_for(prob: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= prob < hi:
            return name
    return "high"


def caveats(feats: dict) -> list[str]:
    """Subgroup limitations documented in the error analysis."""
    out = []
    age = str(feats.get("age", ""))
    race = str(feats.get("race", "")).strip()
    if "90-100" in age:
        out.append(
            "Ages 90-100: model performs near chance for this group "
            "(ROC-AUC 0.540). Not suitable for use here."
        )
    if race in ("AfricanAmerican", "African American"):
        out.append(
            "Measured performance gap: PR-AUC 0.190 for African American "
            "patients vs 0.238 for Caucasian patients at similar base rates."
        )
    if race in ("", "nan", "None", "?", "Unknown"):
        out.append(
            "Race unrecorded: weakest subgroup measured (PR-AUC 0.168), "
            "though on a small sample."
        )
    return out


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Feature dicts -> DataFrame in the pipeline's column order, numerics coerced."""
    df = pd.DataFrame([{f: r.get(f) for f in FEATURES} for r in rows])
    for col in META["numeric"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def default_for(feature: str):
    if feature in SCHEMA["numeric"]:
        return SCHEMA["numeric"][feature]["median"]
    return SCHEMA["categorical"][feature]["default"]


def explain(feats: dict, prob: float, top_n: int = 5) -> list[dict]:
    """One-at-a-time counterfactuals.

    For each feature not already at its default, re-predict with that single
    feature set to its default (median or modal value). delta = actual risk
    minus counterfactual risk, so a positive delta means this feature's value
    is pushing the risk *up*. Cheap and easy to explain; it does not capture
    interactions between features (SHAP would).
    """
    names, rows = [], []
    for f in FEATURES:
        default = default_for(f)
        if feats.get(f) == default:
            continue
        names.append(f)
        rows.append({**feats, f: default})
    if not rows:
        return []
    cf_probs = PIPELINE.predict_proba(to_frame(rows))[:, 1]
    drivers = [
        {
            "feature": f,
            "value": feats.get(f),
            "default": default_for(f),
            "delta": round(prob - float(cf), 4),
        }
        for f, cf in zip(names, cf_probs)
    ]
    drivers.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return drivers[:top_n]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% confidence interval for a proportion k/n (Wilson score interval)."""
    if n == 0:
        return [None, None]
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def evaluate_holdout() -> dict | None:
    """Score the shipped held-out sample once, at startup.

    These patients were excluded from training by patient id, so the observed
    readmission rate per band is a fair check of calibration.
    """
    path = Path("models/holdout_sample.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={c: str for c in META["categorical"]})
    y = df.pop("readmitted_30").astype(int).to_numpy()
    probs = PIPELINE.predict_proba(to_frame(df.to_dict("records")))[:, 1]
    bands = []
    for name, lo, hi in BANDS:
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        bands.append({
            "band": name,
            "range": [lo, min(hi, 1.0)],
            "n": n,
            "predicted_mean": round(float(probs[mask].mean()), 4) if n else None,
            "observed_rate": round(float(y[mask].mean()), 4) if n else None,
            "observed_ci95": wilson(int(y[mask].sum()), n),
        })
    return {
        "n": int(len(y)),
        "baseline_rate": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, probs)), 4),
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "bands": bands,
    }


EVALUATION = evaluate_holdout()


@app.get("/evaluation")
def evaluation():
    if EVALUATION is None:
        raise HTTPException(404, "no held-out sample shipped with this model")
    return EVALUATION


@app.get("/metadata")
def metadata():
    return META


@app.get("/schema")
def schema():
    return {**SCHEMA, "labels": LABELS}


def score(features: dict) -> dict:
    """Everything /predict returns, as a plain function the assistant's tools can reuse."""
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"missing features: {missing}")

    prob = float(PIPELINE.predict_proba(to_frame([features]))[0, 1])
    base = META["baseline_positive_rate"]

    return {
        "probability": round(prob, 4),
        "baseline_rate": round(base, 4),
        "lift_vs_baseline": round(prob / base, 2),
        "band": band_for(prob),
        "caveats": caveats(features),
        "drivers": explain(features, prob),
        "model": META["model"],
        "cv_pr_auc": round(META["cv_pr_auc"], 4),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return score(req.features)
    except ValueError as e:
        raise HTTPException(422, str(e))


# --- Ask AI endpoint -------------------------------------------------------
import os
import time
from collections import defaultdict

from pydantic import BaseModel as _BaseModel


class AskRequest(_BaseModel):
    messages: list[dict]


_MAX_TOTAL_CHARS = 4000
_MAX_MESSAGES = 20
_RATE_LIMIT = 10          # requests per minute per client
_RATE_WINDOW = 60
_recent_calls: dict = defaultdict(list)


def _rate_ok(client: str) -> bool:
    now = time.time()
    hits = [t for t in _recent_calls[client] if now - t < _RATE_WINDOW]
    _recent_calls[client] = hits
    if len(hits) >= _RATE_LIMIT:
        return False
    hits.append(now)
    return True


@app.get("/ask/status")
def ask_status():
    """Let the UI know whether the assistant is usable, and which backend."""
    backend = os.getenv("ASSISTANT_BACKEND", "ollama").lower()
    if backend == "groq":
        available = bool(os.getenv("GROQ_API_KEY"))
    else:
        available = True  # assume local Ollama is reachable; the call will error friendly if not
    return {"available": available, "backend": backend}


@app.post("/ask")
def ask_ai(req: AskRequest, request: Request):
    if len(req.messages) > _MAX_MESSAGES:
        raise HTTPException(413, "Too many messages in one request.")
    total = sum(len(str(m.get("content", ""))) for m in req.messages)
    if total > _MAX_TOTAL_CHARS:
        raise HTTPException(413, "Message too long.")

    client = request.client.host if request.client else "unknown"
    if not _rate_ok(client):
        raise HTTPException(429, "Too many requests. Please wait a moment.")

    try:
        # Lazy import so /predict etc. keep working even if the assistant's
        # modules or backend are missing; a failure here becomes the 503 below.
        import agent

        answer, tool_log = agent.ask(req.messages)
    except Exception as exc:
        # Log the real cause; the client still gets a generic message so we do
        # not leak internals, but the traceback lands in the service logs.
        log.exception("POST /ask failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(503, "The assistant is unavailable right now. Try again shortly.") from exc

    tools_used = [entry["tool"] for entry in tool_log]
    return {"answer": answer, "tools_used": tools_used}
