# Hospital Readmission Prediction

**Live demo:** https://readmission-risk.onrender.com — a five-page site:
what the model is and where it fails, how it works, a form to try it on a
patient, and an AI assistant that answers with tool calls to the live API.


[![CI](https://github.com/Yasminenaser1/tabular-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasminenaser1/tabular-lab/actions/workflows/ci.yml)

**Model card:** [MODEL_CARD.md](MODEL_CARD.md)


Predicting 30-day readmission for diabetic patients using the UCI Diabetes
130-US Hospitals dataset — 101,766 encounters across 71,518 unique patients (99,343 after exclusions, below),
de-identified real clinical records from 130 US hospitals (1999–2008),
CC BY 4.0.

![results](figures/results.png)

## Headline

The model separates risk well: patients in the **top predicted-risk decile
readmit at 27.8%, versus 4.0% in the bottom decile — a 6.9x lift.**
Predicted probabilities are well calibrated (see centre panel), so the scores
can be read as actual risk, not just a ranking.

## Why this problem is harder than it looks

**Class imbalance.** Only 11.4% of encounters are 30-day readmissions.
Accuracy is meaningless here — predicting "no" for everyone scores 88.6%.
All results are reported as PR-AUC.

**Patient leakage.** Patients appear multiple times (~30,000 repeat
encounters). A random row split places the same patient in both train and
test, silently inflating every score. All splits are grouped by `patient_nbr`
and asserted non-overlapping.

**Informative missingness.** `A1Cresult` and `max_glu_serum` are 85–95%
missing, but a test *not* being ordered is itself a signal. These are encoded
as their own category rather than dropped. `weight` (97% missing) is dropped.

**Structural negatives.** 2,423 encounters (2.4%) end in death (discharge
codes 11, 19, 20, 21) or hospice transfer (13, 14). These patients cannot be
readmitted — their readmission rate is 1.8% vs 11.4% for everyone else. They
are excluded, because a model that learns "discharge code 11 means no
readmission" is correct and clinically useless.

**High-cardinality diagnoses.** `diag_1/2/3` hold ~700 distinct ICD-9 codes
each. These are grouped into nine clinical categories (circulatory,
respiratory, diabetes, etc.) rather than one-hot encoded raw.

## Results

5-fold `GroupKFold` cross-validation, PR-AUC:

| Model | PR-AUC | vs. base rate |
|-------|--------|---------------|
| Dummy (predicts base rate) | 0.1139 ± 0.0019 | — |
| Logistic regression | 0.2153 ± 0.0048 | +89% |
| Gradient boosting | 0.2306 ± 0.0040 | +102% |

Boosting beats logistic regression in **5 of 5 folds** (mean gain 0.0154,
worst fold +0.0087), so the improvement survives resampling rather than
being an artifact of one split.

ROC-AUC for the boosting model on the held-out test set is 0.669 — much higher than PR-AUC because
the large negative class flatters the false-positive rate. PR-AUC is the
honest metric at this base rate.

## What drives the model

Permutation importance (drop in PR-AUC when a feature is shuffled):

| Feature | Drop |
|---------|------|
| `number_inpatient` | 0.0795 |
| `discharge_disposition_id` | 0.0567 |
| `number_emergency` | 0.0099 |
| `diag_3_group` | 0.0058 |

Prior hospitalizations dominate — past admissions are the strongest predictor
of future ones. Discharge destination is second, which is clinically
sensible: where a patient goes after release shapes whether they come back.

## Where the model fails

Subgroup PR-AUC on the held-out test set:

- **Patients aged 90–100 (n=547): PR-AUC 0.194, ROC-AUC 0.540** — close to
  chance, against ~0.65 ROC-AUC for other age bands. This does not appear to be a
  base-rate artifact: that group has the *highest* base rate of any age band (12.8%).
  The model is weakest at ranking the patients most likely to be readmitted.
  Plausibly the features here don't capture what drives readmission in the
  very elderly — frailty, social support, care setting.
- **African American patients (n=3,707): PR-AUC 0.190 vs 0.238 for Caucasian
  patients (n=14,886)** at similar base rates (10.7% vs 11.5%). Both samples
  are large enough for this gap to be a stable estimate.
- **Patients with race unrecorded (n=418): PR-AUC 0.168** — the weakest
  group overall, though see the caveat below.

Subgroups below roughly 500 encounters (~40 positives) produce unstable
PR-AUC estimates. That includes the unrecorded-race group above and puts the
90–100 band close to the line, so both should be read as indicative rather
than settled. Small subgroups that score *well* (e.g. ages 20–30, PR-AUC
0.537 on n=388) are not reported as strengths for the same reason.

**This model should not be deployed for the 90+ population, and the
performance gap across race groups would need to be addressed before any
clinical use.** Reporting subgroup performance matters on real patient data
where race, gender, and age are present.

## API

A FastAPI service wraps the trained pipeline:

- `GET /schema` — field names and valid values, so a client can build its form from the model
- `GET /metadata` — model version and CV performance
- `POST /predict` — returns a probability, the baseline rate, lift, and a risk band

**Known limitations are returned with every prediction.** If a request falls
into a subgroup where the model underperforms, the response includes a caveat:

```json
{
  "probability": 0.349,
  "lift_vs_baseline": 3.06,
  "band": "high",
  "caveats": ["Ages 90-100: model performs near chance for this group (ROC-AUC 0.540). Not suitable for use here."]
}
```

A consumer is warned at the point of use rather than being expected to have
read the documentation.

The web UI is a five-page site — a landing page, `/about` (what the model is,
how it scores, where it fails), `/how-it-works` (the grouped splits, PR-AUC
and missingness decisions, each checked against the running service), `/try`,
and `/ask`.

`/try` is a three-step flow generated from `/schema` — fields, valid values and
defaults come from the model, not a hand-written form — and renders those
caveats in red above the score. Three example patients let a visitor see the
model react without filling in 22 fields. Light, dark and system themes.

`/ask` is an assistant with tool access to this service: it looks up measured
performance, explains what drives a score, and runs real predictions for a
patient described in plain English. The numbers come from the same endpoints
the rest of the site uses, not from the language model's memory.

**Every prediction is explained.** The response also carries `drivers`: the
fields moving this patient's risk the most, computed by one-at-a-time
counterfactuals. Each field is set to its typical value (median or mode), the
model is re-run, and the change in risk is reported; all 22 what-ifs go
through the model in one batch, so it costs one extra prediction. For a
patient with four prior inpatient stays:

```json
"drivers": [
  {"feature": "number_inpatient", "value": 4, "default": 0.0, "delta": 0.100},
  {"feature": "time_in_hospital", "value": 7, "default": 4.0, "delta": -0.051},
  {"feature": "number_emergency", "value": 2, "default": 0.0, "delta": 0.020}
]
```

The method is deliberately simple and its limitation is stated in the UI:
fields are varied one at a time, so interactions between them are not
captured. The ordering agrees with the permutation-importance table above,
and a test asserts that prior inpatient stays stay the top driver.

**Validated against SHAP.** `shap_compare.py` computes TreeExplainer SHAP values
for 2,000 held-out patients (one-hot columns folded back into their source
feature) and the counterfactual drivers for the same patients via the API's
own `explain()`. Globally the two agree — Spearman rank correlation 0.70
across all 22 features, and the same top three (`number_inpatient`,
`discharge_disposition_id`, `diag_1_group`). Per patient they name the same
lead driver 44% of the time (chance: 4.5%), which is the cost of ignoring
interactions, quantified. SHAP stays out of the served image on purpose:
it would add ~100 MB to a 512 MB container for a method the UI can't
explain in one sentence.

![SHAP vs counterfactual](figures/shap_vs_counterfactual.png)

**Accuracy is shown, not asserted.** The served model is fit on the patient-grouped
80% split, and the untouched 20% — all 19,802 encounters — ships with the app.
At startup `GET /evaluation` scores it and groups patients by the same low /
elevated / high bands the UI uses; the result page shows, per band, how many
patients, what the model predicted, and how many were actually readmitted,
with 95% Wilson intervals. Observed readmission rises band by band
(8.5% → 18.0% → 35.4% against 8.8% / 18.6% / 33.7% predicted), and a test
asserts that ordering so a retrain that breaks it fails CI.

The app reports PR-AUC 0.229 and ROC-AUC 0.669 on that split — the same
numbers as the table above, because it is the same held-out data. An earlier
build shipped a random 1,000-row sample instead, which was enough for the
draw itself to move PR-AUC by about ±0.03 and land on 0.299; the whole split
ships now so the live numbers can't depend on a seed.

Run it with:

```bash
uvicorn api:app --reload --port 8000
```

Built with scikit-learn 1.9.0; the pickled pipeline may not load under a
different version.

## Ask AI

A tool-calling assistant answers questions about the model in plain English, at `/ask`. It never invents numbers: every figure it quotes comes from the model itself through tools (`predict_patient`, `what_if`, `get_evaluation`, `get_model_info`, `check_fields`, `get_field_options`). The language model only decides which tool to call and phrases the reply; the numbers come from the same endpoints the rest of the site uses.

It runs on a local Ollama backend (`llama3.1:8b`) or hosted Groq (`openai/gpt-oss-20b`), switchable with the `ASSISTANT_BACKEND` environment variable, with no change to `agent.py`.

Guardrails: it refuses clinical-advice questions like "should I discharge this patient" without calling any tool; the `/ask` endpoint caps message length and rate-limits requests; and the API key is whitespace-stripped so a stray newline can't break authentication.

An eval harness in `evals/` scores the assistant's answers on which tool it chose and what it said. This surfaced a concrete lesson: the local 8b model followed the guardrails inconsistently (about 5 of 8 cases), while hosted `gpt-oss` was reliable (8 of 8) — a reminder that model choice, not just prompting, drives how dependable an agent is.

## Tests & deployment

`tests/test_api.py` exercises the service without the dataset: schema/model
agreement, the 422 on missing features, the subgroup caveats, and a behavioral
check that more prior inpatient stays raise predicted risk — a retrained model
that got that backwards would fail CI. `tests/test_split.py` needs the data
and skips when it is absent. GitHub Actions runs the suite on every push.

The `Dockerfile` builds a slim image from `requirements-api.txt` (no plotting
or download dependencies) and reads the port from `PORT`; Render deploys it
from `main` automatically.

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# download dataset id 296 from UCI into data/

python explore.py          # split integrity + data audit
python baseline.py         # dummy vs. logistic on numeric features
python features.py         # + categorical encoding and ICD-9 grouping
python boost.py            # gradient boosting
python cv.py               # 5-fold GroupKFold
python error_analysis.py   # calibration, subgroups, permutation importance
python plots.py            # figures/results.png
python save_model.py       # fit on all data -> models/pipeline.joblib
python build_schema.py     # field metadata -> models/schema.json
```

## Data

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US
Hospitals for Years 1999-2008*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5230J

