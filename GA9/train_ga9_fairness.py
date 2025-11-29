import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    # Base directories
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "iris.csv"
    loc_path = data_dir / "iris_with_location.csv"

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Add random location column (0 or 1) if not already present
    if "location" not in df.columns:
        rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
        df["location"] = rng.integers(0, 2, size=len(df))
        print("✅ Added 'location' column.")
    else:
        print("ℹ️ 'location' column already present in data.")

    # Save dataset with location for later use
    df.to_csv(loc_path, index=False)
    print(f"✅ Saved data with 'location' to: {loc_path}")

    # ----- Prepare features and target -----
    # Update this if your target column has a different name
    target_col = "species"  # CHANGE HERE if your target column name is different

    if target_col not in df.columns:
        raise ValueError(
            f"Expected target column '{target_col}' not found. "
            f"Please update target_col in train_ga9_fairness.py."
        )

    # Features = all columns except target and sensitive attribute
    feature_cols = [c for c in df.columns if c not in [target_col, "location"]]
    X = df[feature_cols]
    y = df[target_col]
    sensitive = df["location"]

    print("Feature columns:", feature_cols)
    print("Target column:", target_col)

    # ----- Train / test split -----
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----- Train model -----
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Test accuracy: {acc:.3f}")

    # ----- Save model -----
    model_path = artifacts_dir / "model_ga9_random_forest.pkl"
    joblib.dump(clf, model_path)
    print(f"✅ Saved trained model to: {model_path}")

    # ----- Save test data with sensitive attribute for fairness analysis -----
    test_data = X_test.copy()
    test_data[target_col] = y_test
    test_data["location"] = s_test
    test_path = data_dir / "iris_test_with_location.csv"
    test_data.to_csv(test_path, index=False)
    print(f"✅ Saved test subset with sensitive attribute to: {test_path}")


if __name__ == "__main__":
    main()

