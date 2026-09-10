import pytest
from fastapi.testclient import TestClient

from api import FEATURES, app
from tools import get_evaluation, get_model_info, predict_patient, what_if

client = TestClient(app)


def test_predict_patient_fills_in_missing_fields():
    r = predict_patient({"age": "[70-80)", "number_inpatient": 3})
    assert set(r["fields_given"]) == {"age", "number_inpatient"}
    assert set(r["fields_given"]) | set(r["fields_assumed"]) == set(FEATURES)
    assert 0.0 <= r["probability"] <= 1.0


def test_tool_gives_same_answer_as_predict_endpoint():
    """The assistant must never report a different number than the app does."""
    r = predict_patient({"age": "[70-80)", "number_inpatient": 3})
    full_patient = r["fields_given"] | r["fields_assumed"]
    body = client.post("/predict", json={"features": full_patient}).json()
    assert r["probability"] == body["probability"]
    assert r["band"] == body["band"]


@pytest.mark.parametrize("bad_fields", [
    {"blood_type": "A"},           # field doesn't exist
    {"age": "75"},                 # wrong format for a category
    {"number_inpatient": 99},      # outside the range seen in the data
])
def test_invalid_fields_are_rejected(bad_fields):
    with pytest.raises(ValueError):
        predict_patient(bad_fields)


def test_what_if_fewer_prior_stays_lowers_risk():
    patient = {"age": "[70-80)", "number_inpatient": 3}
    r = what_if(patient, {"number_inpatient": 0})
    assert r["change_in_probability"] < 0
    assert r["after"]["probability"] == predict_patient({"age": "[70-80)", "number_inpatient": 0})["probability"]


def test_what_if_keeps_caveats():
    r = what_if({"age": "[90-100)"}, {"number_inpatient": 2})
    assert any("90-100" in c for c in r["caveats"])


def test_model_info_lists_each_limitation_once():
    limits = get_model_info()["known_limitations"]
    assert len(limits) == 3
    assert len(set(limits)) == 3


def test_evaluation_tool_matches_endpoint():
    assert get_evaluation() == client.get("/evaluation").json()
