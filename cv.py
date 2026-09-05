import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from features import load, build_preprocessor, NUMERIC, CATEGORICAL

df, y = load()
X = df[NUMERIC + CATEGORICAL]
groups = df["patient_nbr"]

def make(kind):
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    clf = (LogisticRegression(max_iter=2000) if kind == "logreg"
           else HistGradientBoostingClassifier(
               max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
               l2_regularization=1.0, early_stopping=True,
               validation_fraction=0.15, random_state=42))
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])

cv = GroupKFold(n_splits=5)
results = {}

for kind in ["dummy", "logreg", "boost"]:
    scores = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        assert not (set(groups.iloc[tr]) & set(groups.iloc[te]))
        model = make(kind)
        Xtr = X.iloc[tr] if kind != "dummy" else X.iloc[tr][NUMERIC]
        Xte = X.iloc[te] if kind != "dummy" else X.iloc[te][NUMERIC]
        model.fit(Xtr, y.iloc[tr])
        probs = model.predict_proba(Xte)[:, 1]
        s = average_precision_score(y.iloc[te], probs)
        scores.append(s)
        print(f"{kind:<7} fold {fold}: {s:.4f}")
    results[kind] = np.array(scores)
    print()

print("=== 5-fold GroupKFold, PR-AUC ===")
for kind, s in results.items():
    print(f"{kind:<7} {s.mean():.4f} +/- {s.std():.4f}   [{s.min():.4f}, {s.max():.4f}]")

diff = results["boost"] - results["logreg"]
print(f"\nboost - logreg per fold: {np.round(diff, 4)}")
print(f"mean gain {diff.mean():.4f}, wins {(diff > 0).sum()}/5 folds")
