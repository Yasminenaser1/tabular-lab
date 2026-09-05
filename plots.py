import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score
from features import load, split, build_preprocessor, NUMERIC, CATEGORICAL

df, y = load()
tr, te = split(df, y)
X = df[NUMERIC + CATEGORICAL]
y_test = y.iloc[te]

model = Pipeline([
    ("prep", build_preprocessor()),
    ("clf", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=42)),
])
model.fit(X.iloc[tr], y.iloc[tr])
probs = model.predict_proba(X.iloc[te])[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. CV comparison with error bars
names = ["Dummy", "Logistic", "Boosting"]
means = [0.1139, 0.2153, 0.2306]
stds = [0.0019, 0.0048, 0.0040]
axes[0].bar(names, means, yerr=stds, capsize=6,
            color=["#bbb", "#7aa6c2", "#2b6a8f"])
axes[0].axhline(0.1139, ls="--", c="k", lw=1, alpha=0.5)
axes[0].set_ylabel("PR-AUC")
axes[0].set_title("5-fold GroupKFold CV\n(dashed = base rate)")

# 2. Calibration
deciles = pd.qcut(probs, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"d": deciles, "p": probs, "a": y_test.values}).groupby("d").mean()
axes[1].plot([0, 0.3], [0, 0.3], ls="--", c="k", lw=1, alpha=0.5, label="perfect")
axes[1].plot(cal["p"], cal["a"], "o-", c="#2b6a8f", label="model")
axes[1].set_xlabel("mean predicted probability")
axes[1].set_ylabel("actual readmission rate")
axes[1].set_title("Calibration by risk decile")
axes[1].legend()

# 3. Precision-recall curve
prec, rec, _ = precision_recall_curve(y_test, probs)
axes[2].plot(rec, prec, c="#2b6a8f")
axes[2].axhline(y_test.mean(), ls="--", c="k", lw=1, alpha=0.5)
axes[2].set_xlabel("recall")
axes[2].set_ylabel("precision")
axes[2].set_title(f"PR curve (AP = {average_precision_score(y_test, probs):.3f})")

plt.tight_layout()
plt.savefig("figures/results.png", dpi=140)
print("wrote figures/results.png")
