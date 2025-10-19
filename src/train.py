import json, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

import yaml

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    data_path = params["data_path"]
    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))
    n_estimators = int(params.get("n_estimators", 200))
    max_depth = params.get("max_depth", None)

    
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

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    os.makedirs("artifacts", exist_ok=True)
    dump(clf, "artifacts/model.joblib")

    metrics = {"accuracy": acc, "f1_macro": f1}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved artifacts/model.joblib and artifacts/metrics.json")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
