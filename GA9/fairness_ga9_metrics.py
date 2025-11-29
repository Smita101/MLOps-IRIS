import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    artifacts_dir = base_dir / "artifacts"

    test_path = data_dir / "iris_test_with_location.csv"
    model_path = artifacts_dir / "model_ga9_random_forest.pkl"

    print(f"Loading test data from: {test_path}")
    df = pd.read_csv(test_path)

    # Adjust target column name if yours is different
    target_col = "species"

    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found in test data.")

    if "location" not in df.columns:
        raise ValueError("Expected sensitive attribute column 'location' not found in test data.")

    # Separate features, target, and sensitive attribute
    X_test = df.drop(columns=[target_col, "location"])
    y_test = df[target_col]
    sensitive = df["location"]

    print("Test shape:", X_test.shape)
    print("Target value counts:")
    print(y_test.value_counts())

    print(f"Loading model from: {model_path}")
    clf = joblib.load(model_path)

    # Predictions
    y_pred = clf.predict(X_test)

    # ----- Overall accuracy -----
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Overall accuracy: {overall_acc:.3f}")

    # ----- Fairness metrics by 'location' -----
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    print("\n📊 Metrics by location group (0 vs 1):")
    print(mf.by_group)

    # Optional: Also show just accuracy by group
    print("\nAccuracy by location:")
    print(mf.by_group["accuracy"])

    print("\nSelection rate by location (fraction predicted positive for each class label overall):")
    print(mf.by_group["selection_rate"])


if __name__ == "__main__":
    main()
