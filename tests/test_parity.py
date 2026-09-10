"""The gate: the exported model must reproduce sklearn on every held-out row.

Scores models/holdout_sample.csv twice - once through the real
pipeline.joblib, once through web_reference.py reading only web/model.json -
and requires the probabilities to agree to 1e-6 everywhere. If this fails the
browser demo would quietly report different numbers than the model card, so
the tolerance is not negotiable.

    python3 tests/test_parity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import META, PIPELINE, to_frame          # noqa: E402
from web_reference import MODEL_PATH, WebModel    # noqa: E402

TOLERANCE = 1e-6
HOLDOUT = Path("models/holdout_sample.csv")


def load_rows() -> list[dict]:
    """Read the held-out sample exactly as api.evaluate_holdout does."""
    df = pd.read_csv(HOLDOUT, dtype={c: str for c in META["categorical"]})
    df.pop("readmitted_30")
    return df.to_dict("records")


def compare() -> tuple[int, float, int]:
    """Returns (rows checked, largest absolute difference, index of that row)."""
    rows = load_rows()
    expected = PIPELINE.predict_proba(to_frame(rows))[:, 1]
    model = WebModel.load()
    actual = np.array([model.predict_proba(r) for r in rows])
    diff = np.abs(expected - actual)
    worst = int(diff.argmax())
    return len(rows), float(diff[worst]), worst


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run export_web.py first")
def test_exported_model_matches_sklearn():
    rows = load_rows()
    expected = PIPELINE.predict_proba(to_frame(rows))[:, 1]
    model = WebModel.load()

    failures = []
    for i, (row, want) in enumerate(zip(rows, expected)):
        got = model.predict_proba(row)
        if abs(got - want) > TOLERANCE:
            failures.append((i, want, got))

    assert not failures, (
        f"{len(failures)} of {len(rows)} rows differ by more than {TOLERANCE:g}; "
        f"first: row {failures[0][0]} sklearn={failures[0][1]!r} exported={failures[0][2]!r}"
    )


if __name__ == "__main__":
    n, worst, index = compare()
    print(f"rows checked:       {n}")
    print(f"max abs difference: {worst:.3e}  (row {index})")
    print(f"tolerance:          {TOLERANCE:.0e}")
    print("PASS" if worst <= TOLERANCE else "FAIL")
    sys.exit(0 if worst <= TOLERANCE else 1)
