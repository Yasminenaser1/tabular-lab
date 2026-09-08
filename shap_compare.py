"""Validate the API's counterfactual drivers against SHAP on held-out patients.

SHAP values are in log-odds; counterfactual deltas are in probability. So this
compares *rankings*, not magnitudes: global feature order, and per-patient
top-driver agreement.
"""
import json
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import load, split, NUMERIC, CATEGORICAL
from api import explain, FEATURES

N_SAMPLE = 2000
SEED = 42

pipe = joblib.load("models/pipeline.joblib")
prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]

df, y = load()
_, te = split(df, y)
X_test = df.iloc[te][NUMERIC + CATEGORICAL]
X = X_test.sample(N_SAMPLE, random_state=SEED)
print(f"{len(X)} held-out patients")

# ---- SHAP on the transformed matrix, folded back to the 22 original features ----
Xt = prep.transform(X)
col_names = prep.get_feature_names_out()

def origin(col):
    """'cat__diag_1_group_circulatory' -> 'diag_1_group'; 'num__number_inpatient' -> 'number_inpatient'."""
    name = col.split("__", 1)[1]
    for f in sorted(FEATURES, key=len, reverse=True):   # longest match first
        if name == f or name.startswith(f + "_"):
            return f
    raise ValueError(col)

groups = defaultdict(list)
for j, c in enumerate(col_names):
    groups[origin(c)].append(j)

sv = shap.TreeExplainer(clf).shap_values(Xt)
if isinstance(sv, list):          # older shap returns [class0, class1]
    sv = sv[1]
shap_by_feature = pd.DataFrame(
    {f: sv[:, idx].sum(axis=1) for f, idx in groups.items()}, index=X.index
)[FEATURES]

# ---- counterfactual drivers for the same patients, via the API's own function ----
probs = pipe.predict_proba(X)[:, 1]
cf_rows = []
for (_, row), p in zip(X.iterrows(), probs):
    feats = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    drivers = explain(feats, float(p), top_n=len(FEATURES))
    cf_rows.append({d["feature"]: d["delta"] for d in drivers})
cf_by_feature = pd.DataFrame(cf_rows, index=X.index).reindex(columns=FEATURES).fillna(0.0)

# ---- compare ----
shap_global = shap_by_feature.abs().mean().sort_values(ascending=False)
cf_global = cf_by_feature.abs().mean().sort_values(ascending=False)
rho, _ = spearmanr(shap_global.reindex(FEATURES), cf_global.reindex(FEATURES))

top_shap = shap_by_feature.abs().idxmax(axis=1)
top_cf = cf_by_feature.abs().idxmax(axis=1)
top1_agree = float((top_shap == top_cf).mean())

def top3(df_):
    return df_.abs().apply(lambda r: set(r.nlargest(3).index), axis=1)
top3_overlap = float(np.mean([len(a & b) / 3 for a, b in zip(top3(shap_by_feature), top3(cf_by_feature))]))

summary = {
    "n_patients": int(len(X)),
    "spearman_global_rank": round(float(rho), 3),
    "top1_driver_agreement": round(top1_agree, 3),
    "top3_driver_overlap": round(top3_overlap, 3),
    "shap_top5": shap_global.head(5).round(4).to_dict(),
    "counterfactual_top5": cf_global.head(5).round(4).to_dict(),
}
print(json.dumps(summary, indent=2))
json.dump(summary, open("models/shap_comparison.json", "w"), indent=2)

# ---- figure: top-10 by each method, side by side ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
for ax, series, title, color in [
    (axes[0], shap_global.head(10), "SHAP — mean |value| (log-odds)", "#4f8cff"),
    (axes[1], cf_global.head(10), "Counterfactual — mean |Δ risk|", "#22c55e"),
]:
    s = series[::-1]
    ax.barh(s.index, s.values, color=color, height=0.55)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    f"Feature importance, two ways — {len(X)} held-out patients · "
    f"rank ρ = {rho:.2f} · same top driver {top1_agree:.0%}",
    fontsize=11, x=0.01, ha="left",
)
fig.tight_layout()
fig.savefig("figures/shap_vs_counterfactual.png", dpi=160)
print("saved figures/shap_vs_counterfactual.png")
