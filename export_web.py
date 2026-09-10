"""Pack the trained pipeline into static JSON the browser can score with.

Writes web/model.json (preprocessing constants + one-hot mapping + the tree
ensemble) plus the three payloads the frontend currently fetches from the API:
schema.json, metadata.json and evaluation.json.

Nothing here is approximated: thresholds and leaf values are written at full
float64 precision, because tests/test_parity.py re-scores every held-out row
from this file alone and requires agreement with sklearn to 1e-6.

    python3 export_web.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# api.py owns the held-out evaluation and the /schema payload shape; import it
# rather than re-deriving, so the static files match what the API serves today.
from api import EVALUATION, LABELS, META, SCHEMA

OUT = Path("web")
SENTINEL = "__UNSEEN_CATEGORY__"  # a value guaranteed absent from every fitted category

NUMERIC = META["numeric"]
CATEGORICAL = META["categorical"]


def numeric_block(prep) -> dict:
    """Median impute then standard scale: value = (x or median - mean) / scale."""
    num = prep.named_transformers_["num"]
    impute, scale = num.named_steps["impute"], num.named_steps["scale"]
    assert list(num.named_steps) == ["impute", "scale"]
    assert impute.strategy == "median"
    return {
        "features": NUMERIC,
        "medians": [float(v) for v in impute.statistics_],
        "means": [float(v) for v in scale.mean_],
        "scales": [float(v) for v in scale.scale_],
    }


def categorical_block(prep, offset: int) -> dict:
    """Map every fitted category to the global column it switches on.

    Derived by probing the fitted transformer rather than reading its private
    attributes: each probe row sets one feature to one category and every other
    feature to SENTINEL, which handle_unknown='ignore' encodes as all zeros. So
    whichever column lights up belongs to the feature under test, and the
    min_frequency infrequent bucket falls out as a column that several
    categories share.
    """
    cat = prep.named_transformers_["cat"]
    impute, ohe = cat.named_steps["impute"], cat.named_steps["onehot"]
    assert impute.strategy == "constant"
    assert ohe.handle_unknown == "ignore"
    fill = str(impute.fill_value)

    probes, index = [], []
    for i, feature in enumerate(CATEGORICAL):
        for category in ohe.categories_[i]:
            row = dict.fromkeys(CATEGORICAL, SENTINEL)
            row[feature] = category
            probes.append(row)
            index.append((feature, str(category)))
    probes.append(dict.fromkeys(CATEGORICAL, SENTINEL))  # the all-unknown control

    encoded = cat.transform(pd.DataFrame(probes, columns=CATEGORICAL))
    assert not encoded[-1].any(), "unknown categories must encode as all zeros"

    mapping: dict[str, dict[str, int]] = {f: {} for f in CATEGORICAL}
    for (feature, category), row in zip(index, encoded[:-1]):
        cols = np.flatnonzero(row)
        assert len(cols) == 1, f"{feature}={category!r} lit {len(cols)} columns"
        mapping[feature][category] = offset + int(cols[0])

    # A column shared by more than one category is that feature's infrequent bucket.
    infrequent: dict[str, int | None] = {}
    for i, feature in enumerate(CATEGORICAL):
        rare = ohe.infrequent_categories_[i]
        infrequent[feature] = None if rare is None else mapping[feature][str(rare[0])]
        if rare is not None:
            shared = {mapping[feature][str(c)] for c in rare}
            assert shared == {infrequent[feature]}, f"{feature}: infrequent bucket not shared"

    return {
        "features": CATEGORICAL,
        "missing_fill": fill,
        "mapping": mapping,
        "infrequent_column": infrequent,
        "width": int(encoded.shape[1]),
    }


def trees_block(clf) -> list[dict]:
    """Flatten the ensemble to struct-of-arrays, one entry per boosting iteration."""
    trees = []
    for stage in clf._predictors:
        assert len(stage) == 1, "binary classification expects one tree per iteration"
        nodes = stage[0].nodes
        assert not nodes["is_categorical"].any(), "categorical splits are not supported"
        trees.append({
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": [float(v) for v in nodes["num_threshold"]],
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "value": [float(v) for v in nodes["value"]],
            "leaf": nodes["is_leaf"].astype(int).tolist(),
            "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
        })
    return trees


def main() -> None:
    pipeline = joblib.load("models/pipeline.joblib")
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    assert [t[0] for t in prep.transformers_] == ["num", "cat"], "column order assumed [num, cat]"

    numeric = numeric_block(prep)
    categorical = categorical_block(prep, offset=len(NUMERIC))
    n_features = len(NUMERIC) + categorical["width"]
    assert n_features == clf.n_features_in_, f"{n_features} != {clf.n_features_in_}"

    model = {
        "n_features": n_features,
        "baseline": float(np.ravel(clf._baseline_prediction)[0]),
        "numeric": numeric,
        "categorical": categorical,
        "trees": trees_block(clf),
    }

    OUT.mkdir(exist_ok=True)
    written = {
        "model.json": model,
        "schema.json": {**SCHEMA, "labels": LABELS},
        "metadata.json": META,
        "evaluation.json": EVALUATION,
    }
    for name, payload in written.items():
        (OUT / name).write_text(json.dumps(payload, separators=(",", ":")))

    nodes = sum(len(t["value"]) for t in model["trees"])
    print(f"{len(model['trees'])} trees, {nodes} nodes, {n_features} features")
    for name in written:
        print(f"  web/{name:16} {(OUT / name).stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
