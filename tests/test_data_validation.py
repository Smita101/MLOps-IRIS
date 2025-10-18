
import os
import pathlib
import pandas as pd
import pytest


EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "data/iris.csv")

REQUIRED_COLUMNS = [
    "sepal_length", "sepal_width", "petal_length", "petal_width"
]

def _find_existing_label_column(df):
    for col in ["species", "target"]:
        if col in df.columns:
            return col
    return None

@pytest.mark.skipif(not pathlib.Path(EVAL_DATA_PATH).exists(), reason="Eval data not found yet (set EVAL_DATA_PATH in CI after dvc pull)")
def test_schema_and_nulls():
    df = pd.read_csv(EVAL_DATA_PATH)
    # columns present
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert col in df.columns, f"Missing column: {col}"
    label = _find_existing_label_column(df)
    assert label is not None, "Missing label column: expected 'species' or 'target'"

    # no nulls in features/label
    cols_to_check = ["sepal_length", "sepal_width", "petal_length", "petal_width", label]
    assert df[cols_to_check].isnull().sum().sum() == 0, "Nulls found in required columns"

@pytest.mark.skipif(not pathlib.Path(EVAL_DATA_PATH).exists(), reason="Eval data not found yet (set EVAL_DATA_PATH in CI after dvc pull)")
def test_basic_ranges():
    df = pd.read_csv(EVAL_DATA_PATH)
    # sanity ranges for iris features (broad bounds to avoid flakiness)
    assert (df["sepal_length"].between(0.0, 10.0)).all()
    assert (df["sepal_width"].between(0.0, 10.0)).all()
    assert (df["petal_length"].between(0.0, 10.0)).all()
    assert (df["petal_width"].between(0.0, 10.0)).all()
    # at least some rows
    assert len(df) > 0
