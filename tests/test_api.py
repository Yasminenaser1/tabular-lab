import pytest
from fastapi.testclient import TestClient

from api import app, META, SCHEMA

client = TestClient(app)


def default_features():
    """A complete request built from the schema: medians and modal categories."""
    feats = {name: info["median"] for name, info in SCHEMA["numeric"].items()}
    feats.update({name: info["default"] for name, info in SCHEMA["categorical"].items()})
    return feats


def test_home_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "30-day readmission risk" in r.text


def test_schema_matches_model_features():
    numeric = set(SCHEMA["numeric"])
    categorical = set(SCHEMA["categorical"])
    assert numeric == set(META["numeric"])
    assert categorical == set(META["categorical"])


def test_predict_defaults():
    r = client.post("/predict", json={"features": default_features()})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["band"] in {"low", "elevated", "high"}
    assert body["caveats"] == []


def test_missing_feature_is_rejected():
    feats = default_features()
    del feats["number_inpatient"]
    r = client.post("/predict", json={"features": feats})
    assert r.status_code == 422
    assert "number_inpatient" in r.json()["detail"]


def test_age_90_plus_adds_caveat():
    feats = default_features() | {"age": "[90-100)"}
    body = client.post("/predict", json={"features": feats}).json()
    assert any("90-100" in c for c in body["caveats"])


def test_unrecorded_race_is_accepted_with_caveat():
    feats = default_features() | {"race": None}
    r = client.post("/predict", json={"features": feats})
    assert r.status_code == 200
    assert any("unrecorded" in c.lower() for c in r.json()["caveats"])


def test_more_prior_admissions_raises_risk():
    low = default_features() | {"number_inpatient": 0}
    high = default_features() | {"number_inpatient": 5}
    p_low = client.post("/predict", json={"features": low}).json()["probability"]
    p_high = client.post("/predict", json={"features": high}).json()["probability"]
    assert p_high > p_low
