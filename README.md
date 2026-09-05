# Hospital Readmission Prediction

Predicting 30-day readmission for diabetic patients using the UCI Diabetes
130-US Hospitals dataset — 101,766 encounters across 71,518 unique patients,
de-identified real clinical records from 130 US hospitals (1999–2008),
CC BY 4.0.

![results](figures/results.png)

## Headline

The model separates risk well: patients in the **top predicted-risk decile
readmit at 26.0%, versus 3.3% in the bottom decile — a 7.8x lift.**
Predicted probabilities are well calibrated (see centre panel), so the scores
can be read as actual risk, not just a ranking.

## Why this problem is harder than it looks

**Class imbalance.** Only 11.2% of encounters are 30-day readmissions.
Accuracy is meaningless here — predicting "no" for everyone scores 88.8%.
All results are reported as PR-AUC.

**Patient leakage.** Patients appear multiple times (~30,000 repeat
encounters). A random row split places the same patient in both train and
test, silently inflating every score. All splits are grouped by `patient_nbr`
and asserted non-overlapping.

**Informative missingness.** `A1Cresult` and `max_glu_serum` are 85–95%
missing, but a test *not* being ordered is itself a signal. These are encoded
as their own category rather than dropped. `weight` (97% missing) is dropped.

**High-cardinality diagnoses.** `diag_1/2/3` hold ~700 distinct ICD-9 codes
each. These are grouped into nine clinical categories (circulatory,
respiratory, diabetes, etc.) rather than one-hot encoded raw.

## Results

5-fold `GroupKFold` cross-validation, PR-AUC:

| Model | PR-AUC | vs. base rate |
|-------|--------|---------------|
| Dummy (predicts base rate) | 0.1116 ± 0.0014 | — |
| Logistic regression | 0.2144 ± 0.0054 | +92% |
| Gradient boosting | 0.2288 ± 0.0054 | +105% |

Boosting beats logistic regression in **5 of 5 folds** (mean gain 0.0144,
worst fold +0.0097), so the improvement survives resampling rather than
being an artifact of one split.

ROC-AUC for the boosting model is 0.674 — much higher than PR-AUC because
the large negative class flatters the false-positive rate. PR-AUC is the
honest metric on an 11% problem.

## What drives the model

Permutation importance (drop in PR-AUC when a feature is shuffled):

| Feature | Drop |
|---------|------|
| `number_inpatient` | 0.0589 |
| `discharge_disposition_id` | 0.0448 |
| `diag_1_group` | 0.0116 |
| `number_emergency` | 0.0053 |

Prior hospitalizations dominate — past admissions are the strongest predictor
of future ones. Discharge destination is second, which is clinically
sensible: where a patient goes after release shapes whether they come back.

## Where the model fails

Subgroup PR-AUC on the held-out test set:

- **Patients aged 90–100 (n=528): PR-AUC 0.128, ROC-AUC 0.578** — barely
  above chance, against ~0.67 ROC-AUC for every other age band. This is not a
  base-rate artifact; that group has the *lowest* base rate (8.5%). The
  features here don't capture what actually drives readmission in the very
  elderly (frailty, social support, care setting).
- **Patients with unrecorded race (n=410): PR-AUC 0.085** — worse than any
  recorded group.

**This model should not be deployed for those populations without further
work.** Reporting subgroup performance matters on real patient data where
race, gender, and age are present.

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas scikit-learn matplotlib
# download dataset id 296 from UCI into data/

python explore.py          # split integrity + data audit
python baseline.py         # dummy vs. logistic on numeric features
python features.py         # + categorical encoding and ICD-9 grouping
python boost.py            # gradient boosting
python cv.py               # 5-fold GroupKFold
python error_analysis.py   # calibration, subgroups, permutation importance
python plots.py            # figures/results.png
```

## Data

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US
Hospitals for Years 1999-2008*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5230J
