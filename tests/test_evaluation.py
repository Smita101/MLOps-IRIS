import os
import pathlib
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
import joblib

# CI will set these env vars to DVC-pulled artifacts
EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "data/iris.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

MIN_EXPECTED_ACCURACY = float(os.getenv("MIN_EXPECTED_ACCURACY", "0.90"))

data_exists = pathlib.Path(EVAL_DATA_PATH).exists()
model_exists = pathlib.Path(MODEL_PATH).exists()

@pytest.mark.skipif(not data_exists, reason="Eval data not found yet (set EVAL_DATA_PATH in CI after dvc pull)")
@pytest.mark.skipif(not model_exists, reason="Model not found yet (set MODEL_PATH in CI after dvc pull)")
def test_model_meets_accuracy_threshold():
    df = pd.read_csv(EVAL_DATA_PATH)

    # figure out label column
    label_col = "species" if "species" in df.columns else ("target" if "target" in df.columns else None)
    assert label_col is not None, "Label column not found (expected 'species' or 'target')"

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
    y = df[label_col].to_numpy()

    # if labels are strings ('setosa' etc), encode to integers for metric
    if y.dtype.kind in {"U", "S", "O"}:
        classes, y = np.unique(y, return_inverse=True)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    assert acc >= MIN_EXPECTED_ACCURACY, f"Accuracy {acc:.3f} < expected {MIN_EXPECTED_ACCURACY:.3f}"
