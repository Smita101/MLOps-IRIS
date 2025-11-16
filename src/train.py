import json, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import argparse
import numpy as np
import yaml
import mlflow
import mlflow.sklearn


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--poison_fraction",
        type=float,
        default=0.0,
        help="Fraction of training samples to poison (0.0, 0.05, 0.10, 0.50, etc.)"
    )
    parser.add_argument(
        "--poison_mode",
        type=str,
        default="none",
        choices=["none", "features", "labels", "both"],
        help="Type of data poisoning to apply."
    )
    return parser.parse_args()


def apply_poisoning(X_train, y_train, poison_fraction, poison_mode, random_state=42):
    """Apply simple data poisoning to a subset of the training data."""
    if poison_fraction <= 0 or poison_mode == "none":
        return X_train, y_train

    rng = np.random.RandomState(random_state)
    n_samples = X_train.shape[0]
    n_poison = int(poison_fraction * n_samples)

    if n_poison == 0:
        return X_train, y_train

    poisoned_idx = rng.choice(n_samples, size=n_poison, replace=False)

    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    # For features: replace rows with random values (within min/max of each feature)
    if poison_mode in ["features", "both"]:
        feature_min = X_train.min(axis=0)
        feature_max = X_train.max(axis=0)
        random_features = rng.uniform(
            low=feature_min,
            high=feature_max,
            size=(n_poison, X_train.shape[1])
        )
        X_poisoned.iloc[poisoned_idx] = random_features  # use iloc to be explicit

    # For labels: flip to a random *different* class
    if poison_mode in ["labels", "both"]:
        unique_labels = np.unique(y_train)
        for idx in poisoned_idx:
            original = y_poisoned.iloc[idx]
            other_labels = unique_labels[unique_labels != original]
            y_poisoned.iloc[idx] = rng.choice(other_labels)

    return X_poisoned, y_poisoned


def main():
    params = load_params()
    data_path = params["data_path"]
    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))
    n_estimators = int(params.get("n_estimators", 200))
    max_depth = params.get("max_depth", None)

    args = parse_args()

    df = pd.read_csv(data_path)
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # GA8: apply optional data poisoning (OFF by default)
    X_train, y_train = apply_poisoning(
        X_train,
        y_train,
        poison_fraction=args.poison_fraction,
        poison_mode=args.poison_mode,
        random_state=random_state,
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    # Optional: set a dedicated experiment name
    mlflow.set_experiment("GA8_Data_Poisoning_IRIS")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Log poisoning setup
        mlflow.log_param("poison_fraction", args.poison_fraction)
        mlflow.log_param("poison_mode", args.poison_mode)

        # Train
        clf.fit(X_train, y_train)

        # Evaluate
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        metrics = {"accuracy": acc, "f1_macro": f1}

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # Log model to MLflow
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # Also keep your existing artifacts for the GA structure
        os.makedirs("artifacts", exist_ok=True)
        dump(clf, "artifacts/model.joblib")
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("Saved artifacts/model.joblib and artifacts/metrics.json")
        print("Metrics:", metrics)


if __name__ == "__main__":
    main()
