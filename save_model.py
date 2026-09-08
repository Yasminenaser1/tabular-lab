import json, time
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from features import load, split, build_preprocessor, NUMERIC, CATEGORICAL

df, y = load()
X = df[NUMERIC + CATEGORICAL]
groups = df["patient_nbr"]

def make(kind):
    clf = (LogisticRegression(max_iter=2000) if kind == "logreg"
           else HistGradientBoostingClassifier(
               max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
               l2_regularization=1.0, early_stopping=True,
               validation_fraction=0.15, random_state=42))
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])

cv = GroupKFold(n_splits=5)
scores = {}

for kind in ["logreg", "boost"]:
    fold_scores = []
    for tr, te in cv.split(X, y, groups):
        m = make(kind).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        fold_scores.append(average_precision_score(y.iloc[te], p))
    scores[kind] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))
    print(f"{kind:8s} PR-AUC {scores[kind][0]:.4f} +/- {scores[kind][1]:.4f}")

best = max(scores, key=lambda k: scores[k][0])
print(f"winner: {best}")

# Serve a model fit on the patient-grouped 80% split, and keep the 20% truly
# held out so the app can report calibration on patients it has never seen.
tr, te = split(df, y)
final = make(best).fit(X.iloc[tr], y.iloc[tr])
Path("models").mkdir(exist_ok=True)
joblib.dump(final, "models/pipeline.joblib")

holdout = X.iloc[te].sample(1000, random_state=7).copy()
holdout["readmitted_30"] = y.iloc[holdout.index].values
holdout.to_csv("models/holdout_sample.csv", index=False)
print(f"held-out sample: {len(holdout)} rows, {holdout['readmitted_30'].mean():.3f} positive")

meta = {
    "model": best,
    "cv_pr_auc": scores[best][0],
    "cv_pr_auc_std": scores[best][1],
    "baseline_positive_rate": float(y.mean()),
    "all_scores": {k: v[0] for k, v in scores.items()},
    "numeric": NUMERIC,
    "categorical": CATEGORICAL,
    "n_rows": int(len(X)),
    "n_train": int(len(tr)),
    "n_test": int(len(te)),
    "trained_on": "80% patient-grouped split; 20% held out (models/holdout_sample.csv)",
    "trained_at": time.strftime("%Y-%m-%d %H:%M"),
}
Path("models/metadata.json").write_text(json.dumps(meta, indent=2))
print("saved models/pipeline.joblib + models/metadata.json")
