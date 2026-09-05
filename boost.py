from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from features import load, split, build_preprocessor, NUMERIC, CATEGORICAL

df, y = load()
tr, te = split(df, y)

X = df[NUMERIC + CATEGORICAL]
X_train, X_test = X.iloc[tr], X.iloc[te]
y_train, y_test = y.iloc[tr], y.iloc[te]

model = Pipeline([
    ("prep", build_preprocessor()),
    ("clf", HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )),
])
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

print(f"iterations used: {model.named_steps['clf'].n_iter_}")
print(f"PR-AUC:  {average_precision_score(y_test, probs):.4f}  (logreg 0.2026)")
print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}  (logreg 0.6645)")
