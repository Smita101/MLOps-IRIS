import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt


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

    # For multi-class classifiers, shap_values is a list: one array per class
    print("Computing SHAP values for full dataset...")
    shap_values = explainer.shap_values(X)

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

    # SHAP values for class 'virginica'
    shap_virginica = shap_values[virginica_index]

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
