"""Reference scorer that reads web/model.json and nothing else.

This is the specification for predict.js: preprocess -> tree traversal ->
sigmoid, using only what the exported file contains. Keeping it in Python
first means tests/test_parity.py can hold it against the real sklearn
pipeline row by row before any of it is translated to JavaScript.
"""
import json
import math
from pathlib import Path

MODEL_PATH = Path("web/model.json")


def to_number(value) -> float:
    """Mirror pandas' pd.to_numeric(errors='coerce'): unparseable -> NaN."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


class WebModel:
    def __init__(self, model: dict):
        self.n_features = model["n_features"]
        self.baseline = model["baseline"]
        self.numeric = model["numeric"]
        self.categorical = model["categorical"]
        self.trees = model["trees"]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "WebModel":
        return cls(json.loads(Path(path).read_text()))

    def preprocess(self, features: dict) -> list[float]:
        """Feature dict -> the 107-long dense row the ensemble expects."""
        row = [0.0] * self.n_features

        num = self.numeric
        for i, name in enumerate(num["features"]):
            value = to_number(features.get(name))
            if math.isnan(value):
                value = num["medians"][i]
            row[i] = (value - num["means"][i]) / num["scales"][i]

        cat = self.categorical
        for name in cat["features"]:
            value = features.get(name)
            # NaN is the only value not equal to itself; it means "missing" here.
            if value is None or (isinstance(value, float) and value != value):
                key = cat["missing_fill"]
            else:
                key = str(value)
            column = cat["mapping"][name].get(key)
            if column is not None:  # unknown categories stay all-zero
                row[column] = 1.0

        return row

    def raw(self, row: list[float]) -> float:
        total = self.baseline
        for tree in self.trees:
            node = 0
            while not tree["leaf"][node]:
                value = row[tree["feature"][node]]
                if value != value:  # NaN
                    node = tree["left"][node] if tree["missing_left"][node] else tree["right"][node]
                elif value <= tree["threshold"][node]:
                    node = tree["left"][node]
                else:
                    node = tree["right"][node]
            total += tree["value"][node]
        return total

    def predict_proba(self, features: dict) -> float:
        """Probability of 30-day readmission for one feature dict."""
        return 1.0 / (1.0 + math.exp(-self.raw(self.preprocess(features))))
