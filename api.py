import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


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
