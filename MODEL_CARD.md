# Model card — 30-day readmission risk (diabetic inpatients)

## Model details

- **Model:** scikit-learn `HistGradientBoostingClassifier` (300 iterations, learning rate 0.05,
  31 max leaf nodes, L2 = 1.0, early stopping) inside a `Pipeline` with median imputation +
  scaling for numerics and constant-`MISSING` imputation + one-hot encoding (min frequency 20)
  for categoricals.
- **Inputs:** 8 numeric and 14 categorical fields describing one hospital encounter
  (`models/schema.json` lists them with valid values).
- **Output:** probability of readmission within 30 days of discharge, plus a risk band
  (low < 15% ≤ elevated < 25% ≤ high), subgroup caveats, and per-prediction drivers.
- **Version:** pipeline trained 2026-09-05 on all 99,343 rows (`models/pipeline.joblib`,
  scikit-learn 1.9.0). Author: Yasmine Naser. Personal portfolio project.

## Intended use

- **Primary use:** demonstrate an end-to-end tabular ML workflow — leakage-safe validation,
  calibration, error analysis, subgroup reporting, serving and explanation — on real
  de-identified clinical data.
- **Intended users:** people evaluating the engineering and analysis, via the live demo, the
  API or the repository.
- **Out of scope:** any clinical, operational or insurance decision about a real patient.
  The data are from 1999–2008 US hospitals; care patterns, coding and populations have changed.
  The model has not been externally validated, and it underperforms in the groups listed below.

## Data

UCI *Diabetes 130-US Hospitals for Years 1999–2008* (Clore et al., 2014; CC BY 4.0):
101,766 encounters from 71,518 patients. Encounters ending in death or hospice transfer
(discharge codes 11, 13, 14, 19, 20, 21; 2,423 rows) are excluded as structural negatives,
leaving 99,343. `weight` (97% missing) is dropped; `A1Cresult` and `max_glu_serum` keep
"not tested" as a category. ICD-9 diagnosis codes are grouped into nine clinical categories.
Positive rate: 11.4%.

## Evaluation

All splits are grouped by `patient_nbr` so no patient appears in both train and test.

| Metric (5-fold GroupKFold) | Value |
|---|---|
| PR-AUC, gradient boosting | 0.2306 ± 0.0040 |
| PR-AUC, logistic regression | 0.2153 ± 0.0048 |
| PR-AUC, base-rate dummy | 0.1139 ± 0.0019 |
| ROC-AUC, held-out test | 0.669 |
| Top vs bottom predicted-risk decile (readmission rate) | 27.8% vs 4.0% (6.9× lift) |

Probabilities are calibrated (see `figures/results.png`), so a score can be read as a risk,
not only a ranking. PR-AUC is reported because accuracy and ROC-AUC flatter any model at an
11% base rate.

## Where it fails

Held-out subgroup performance (PR-AUC / ROC-AUC):

| Group | n | PR-AUC | Note |
|---|---|---|---|
| Age 90–100 | 547 | 0.194 (ROC 0.540) | near chance; highest base rate of any age band |
| African American | 3,707 | 0.190 | vs 0.238 for Caucasian (n = 14,886) at similar base rates |
| Race unrecorded | 418 | 0.168 | weakest group, but small sample |

Subgroups under ~500 encounters give unstable estimates; small groups that score *well*
are not reported as strengths for the same reason. The API and UI return these caveats with
every prediction that falls into an affected group.

## Explanations

Each prediction ships with `drivers`: one-at-a-time counterfactuals (each field set to its
typical value, model re-run, change in risk reported). Cheap and legible; does not capture
interactions between fields. Ordering agrees with permutation importance
(`number_inpatient`, `discharge_disposition_id`, `number_emergency` dominate).

## Ethical considerations and recommendations

- Race, gender and age are model inputs. They are kept because the analysis reports
  performance *by* group and the service warns *at the point of use*; removing them would
  hide the gap, not close it. A deployable model would need that gap addressed first.
- Do not use for the 90+ population.
- Prior hospital utilisation dominates the signal; a model like this can reinforce existing
  patterns of care if used to allocate resources without oversight.
- Retraining on current data and external validation would be required before any real use.

## Reproduce

See `README.md` → Reproducing. Tests: `pytest -q`. Serving: `uvicorn api:app`.
