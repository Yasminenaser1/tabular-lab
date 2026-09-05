import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("data/diabetic_data.csv", na_values="?")
print("shape:", df.shape)

print("\nmissing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nreadmitted classes:")
print(df["readmitted"].value_counts())

y = (df["readmitted"] == "<30").astype(int)
print("\npositive rate:", round(y.mean(), 4))
print("rows:", len(df), "unique patients:", df["patient_nbr"].nunique())

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups=df["patient_nbr"]))

train, test = df.iloc[train_idx], df.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

overlap = set(train["patient_nbr"]) & set(test["patient_nbr"])
assert len(overlap) == 0, f"leak: {len(overlap)} patients in both splits"

print("\ntrain:", len(train), "test:", len(test))
print("train pos rate:", round(y_train.mean(), 4))
print("test pos rate:", round(y_test.mean(), 4))
