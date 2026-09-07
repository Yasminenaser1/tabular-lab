from pathlib import Path

import pytest
from features import load, split, build_preprocessor, NUMERIC, CATEGORICAL

pytestmark = pytest.mark.skipif(
    not Path("data/diabetic_data.csv").exists(),
    reason="dataset not downloaded",
)

EXPIRED_HOSPICE = [11, 13, 14, 19, 20, 21]


def test_no_patient_overlap():
    df, y = load()
    tr, te = split(df, y)
    overlap = set(df.iloc[tr]["patient_nbr"]) & set(df.iloc[te]["patient_nbr"])
    assert len(overlap) == 0


def test_random_split_would_leak():
    df, y = load()
    n = len(df)
    tr, te = range(0, int(n * 0.8)), range(int(n * 0.8), n)
    overlap = set(df.iloc[tr]["patient_nbr"]) & set(df.iloc[te]["patient_nbr"])
    assert len(overlap) > 0


def test_expired_and_hospice_excluded():
    df, _ = load()
    assert df["discharge_disposition_id"].astype(int).isin(EXPIRED_HOSPICE).sum() == 0
    assert len(df) == 99343


def test_preprocessor_rejects_missing_column():
    df, y = load()
    prep = build_preprocessor()
    prep.fit(df[NUMERIC + CATEGORICAL])
    broken = df[NUMERIC + CATEGORICAL].drop(columns=["number_inpatient"])
    with pytest.raises(Exception):
        prep.transform(broken)
