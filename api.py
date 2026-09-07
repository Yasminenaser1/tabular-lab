import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PIPELINE = joblib.load("models/pipeline.joblib")
META = json.loads(Path("models/metadata.json").read_text())
SCHEMA = json.loads(Path("models/schema.json").read_text())
FEATURES = META["numeric"] + META["categorical"]

app = FastAPI(title="Readmission Risk", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Feature name -> value")


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


@app.get("/metadata")
def metadata():
    return META


@app.get("/schema")
def schema():
    return SCHEMA


@app.post("/predict")
def predict(req: PredictRequest):
    missing = [f for f in FEATURES if f not in req.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")

    row = pd.DataFrame([{f: req.features[f] for f in FEATURES}])
    for col in META["numeric"]:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    prob = float(PIPELINE.predict_proba(row)[0, 1])
    base = META["baseline_positive_rate"]

    return {
        "probability": round(prob, 4),
        "baseline_rate": round(base, 4),
        "lift_vs_baseline": round(prob / base, 2),
        "band": "high" if prob >= 0.25 else "elevated" if prob >= 0.15 else "low",
        "caveats": caveats(req.features),
        "model": META["model"],
        "cv_pr_auc": round(META["cv_pr_auc"], 4),
    }
