import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

df = pd.read_csv("data/diabetic_data.csv", na_values="?")
y = (df["readmitted"] == "<30").astype(int)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups=df["patient_nbr"]))

X = df[NUMERIC]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

models = {
    "dummy": DummyClassifier(strategy="prior"),
    "logreg": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
}

print(f"{'model':<10} {'PR-AUC':>8} {'ROC-AUC':>9}")
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    pr = average_precision_score(y_test, probs)
    roc = roc_auc_score(y_test, probs)
    print(f"{name:<10} {pr:>8.4f} {roc:>9.4f}")
