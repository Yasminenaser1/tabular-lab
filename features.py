import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from diag import add_diag_groups

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

CATEGORICAL = [
    "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "A1Cresult", "max_glu_serum",
    "insulin", "change", "diabetesMed",
    "diag_1_group", "diag_2_group", "diag_3_group",
]


def load():
    df = pd.read_csv("data/diabetic_data.csv", na_values="?", low_memory=False)
    y = (df["readmitted"] == "<30").astype(int)
    # ID columns are codes, not quantities — treat as categories
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)
    df = add_diag_groups(df)
    # Expired (11,19,20,21) and hospice (13,14) discharges cannot be
    # readmitted — structural negatives that inflate any score.
    keep = ~df["discharge_disposition_id"].astype(int).isin([11, 13, 14, 19, 20, 21])
    df = df[keep].reset_index(drop=True)
    y = y[keep.values].reset_index(drop=True)
    return df, y


def split(df, y, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(df, y, groups=df["patient_nbr"]))
    assert not (set(df.iloc[tr]["patient_nbr"]) & set(df.iloc[te]["patient_nbr"]))
    return tr, te


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]), CATEGORICAL),
    ])


if __name__ == "__main__":
    df, y = load()
    tr, te = split(df, y)

    X = df[NUMERIC + CATEGORICAL]
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]

    model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    n_features = model.named_steps["prep"].transform(X_train[:1]).shape[1]
    print(f"features after encoding: {n_features}")
    print(f"PR-AUC:  {average_precision_score(y_test, probs):.4f}  (was 0.1801)")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}  (was 0.6292)")
