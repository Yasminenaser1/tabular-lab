import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
from features import load, split, build_preprocessor, NUMERIC, CATEGORICAL

df, y = load()
tr, te = split(df, y)

X = df[NUMERIC + CATEGORICAL]
X_train, X_test = X.iloc[tr], X.iloc[te]
y_train, y_test = y.iloc[tr], y.iloc[te]
test_rows = df.iloc[te]

model = Pipeline([
    ("prep", build_preprocessor()),
    ("clf", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=42,
    )),
])
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

print(f"overall PR-AUC: {average_precision_score(y_test, probs):.4f}\n")

# --- 1. Calibration: are predicted probabilities honest? ---
print("=== calibration by predicted-risk decile ===")
deciles = pd.qcut(probs, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"decile": deciles, "pred": probs, "actual": y_test.values})
summary = cal.groupby("decile").agg(
    mean_predicted=("pred", "mean"),
    actual_rate=("actual", "mean"),
    n=("actual", "size"),
)
print(summary.round(4).to_string())

top = summary.iloc[-1]
bottom = summary.iloc[0]
print(f"\ntop decile actual readmission rate:    {top['actual_rate']:.1%}")
print(f"bottom decile actual readmission rate: {bottom['actual_rate']:.1%}")
print(f"lift: {top['actual_rate'] / bottom['actual_rate']:.1f}x\n")

# --- 2. Subgroup performance: does it fail unevenly? ---
def subgroup_report(col, min_n=300):
    print(f"=== PR-AUC by {col} ===")
    rows = []
    for value, idx in test_rows.groupby(col, dropna=False).groups.items():
        mask = test_rows.index.isin(idx)
        yt, pt = y_test[mask], probs[mask]
        if len(yt) < min_n or yt.sum() < 10:
            continue
        rows.append({
            "group": value, "n": len(yt),
            "base_rate": yt.mean(),
            "pr_auc": average_precision_score(yt, pt),
            "roc_auc": roc_auc_score(yt, pt),
        })
    out = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    print(out.round(4).to_string(index=False), "\n")
    return out

subgroup_report("race")
subgroup_report("gender")
subgroup_report("age")

# --- 3. Which features actually carry the model? ---
print("=== permutation importance (top 12) ===")
sample = np.random.RandomState(42).choice(len(X_test), 5000, replace=False)
imp = permutation_importance(
    model, X_test.iloc[sample], y_test.iloc[sample],
    n_repeats=5, random_state=42, scoring="average_precision", n_jobs=-1,
)
ranked = pd.DataFrame({
    "feature": X_test.columns,
    "drop_in_pr_auc": imp.importances_mean,
    "std": imp.importances_std,
}).sort_values("drop_in_pr_auc", ascending=False).head(12)
print(ranked.round(4).to_string(index=False))
