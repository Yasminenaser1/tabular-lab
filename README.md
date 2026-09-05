# Hospital Readmission Prediction

Predicting 30-day readmission for diabetic patients using the UCI Diabetes
130-US Hospitals dataset (101,766 encounters, 71,518 unique patients).

## Why this is harder than it looks

**Class imbalance.** Only 11.2% of encounters are readmissions. Accuracy is
useless here — predicting "no" for everyone scores 88.8%. All results are
reported as PR-AUC.

**Patient leakage.** Patients appear multiple times (30k repeat encounters).
A random row split puts the same patient in train and test, silently
inflating every score. Splits are grouped by `patient_nbr` and asserted
non-overlapping.

**Informative missingness.** `A1Cresult` and `max_glu_serum` are ~85-95%
missing, but a test not being ordered is itself a signal. These are encoded
as a category rather than dropped.

## Results

| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| Dummy (predicts base rate) | 0.1067 | 0.5000 |
| Logistic regression, 8 numeric features | 0.1801 | 0.6292 |

ROC-AUC looks much better than PR-AUC because the large negative class
flatters the false-positive rate. PR-AUC is the honest metric on this problem.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas scikit-learn matplotlib
# download the dataset into data/ from UCI (id 296)
python explore.py    # split integrity + data audit
python baseline.py   # baseline models
```

## Next

Categorical encoding, ICD-9 diagnosis grouping, gradient boosting,
cross-validation, error analysis.
