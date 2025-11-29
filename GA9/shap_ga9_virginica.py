import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    data_path = data_dir / "iris_with_location.csv"
    model_path = artifacts_dir / "model_ga9_random_forest.pkl"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    target_col = "species"

    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")

    if "location" not in df.columns:
        raise ValueError("Expected 'location' column not found.")

    # Features = all columns except target and sensitive attribute
    feature_cols = [c for c in df.columns if c not in [target_col, "location"]]
    X = df[feature_cols]

    print("Feature columns used for SHAP:", feature_cols)

    print(f"Loading model from: {model_path}")
    clf = joblib.load(model_path)

    # Create SHAP explainer for tree-based model
    explainer = shap.TreeExplainer(clf)

    # For multi-class classifiers, shap_values can be:
    #  - a list of arrays (one per class), OR
    #  - a single 3D array (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
    print("Computing SHAP values for full dataset...")
    shap_values = explainer.shap_values(X)

    # Inspect type / shape for debugging
    print(f"type(shap_values): {type(shap_values)}")
    if isinstance(shap_values, np.ndarray):
        print(f"shap_values.shape: {shap_values.shape}")
    else:
        print(f"len(shap_values): {len(shap_values)}")

    # Find index of class 'virginica'
    class_labels = clf.classes_
    print("Model classes:", class_labels)

    try:
        virginica_index = list(class_labels).index("virginica")
    except ValueError:
        raise ValueError(
            "Class 'virginica' not found in model classes. "
            f"Available classes: {class_labels}"
        )

    print(f"Index for class 'virginica': {virginica_index}")

    # ---- Handle different SHAP output formats ----
    if isinstance(shap_values, list):
        # Standard old behaviour: list of arrays, one per class
        shap_virginica = shap_values[virginica_index]
    else:
        # Newer behaviour: single ndarray, likely 3D
        sv = shap_values
        if sv.ndim == 3:
            # Try (n_samples, n_features, n_classes)
            if sv.shape[0] == X.shape[0] and sv.shape[1] == X.shape[1]:
                shap_virginica = sv[:, :, virginica_index]
            # Try (n_classes, n_samples, n_features)
            elif sv.shape[0] == len(class_labels) and sv.shape[2] == X.shape[1]:
                shap_virginica = sv[virginica_index, :, :]
            else:
                raise ValueError(
                    f"Unexpected shap_values shape {sv.shape} for SHAP. "
                    "Cannot align with feature matrix X."
                )
        else:
            raise ValueError(
                f"Unsupported shap_values ndim={sv.ndim}. "
                "Expected list or 3D numpy array."
            )

    print("shap_virginica.shape:", getattr(shap_virginica, "shape", None))
    print("X.shape:", X.shape)

    # Final sanity check
    if shap_virginica.shape[0] != X.shape[0] or shap_virginica.shape[1] != X.shape[1]:
        raise ValueError(
            f"Mismatch between SHAP values shape {shap_virginica.shape} "
            f"and X shape {X.shape}."
        )

    # Create summary plot for class virginica
    plt.figure()
    shap.summary_plot(
        shap_virginica,
        X,
        show=False,        # do not open an interactive window
        plot_type="dot"    # standard dot summary plot
    )

    output_path = artifacts_dir / "shap_summary_virginica.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ SHAP summary plot for class 'virginica' saved to: {output_path}")


if __name__ == "__main__":
    main()

