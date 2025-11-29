import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Locate the GA9/data/iris.csv file relative to this script
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "iris.csv"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Add random location column (0 or 1)
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    df["location"] = rng.integers(0, 2, size=len(df))

    # Save new file with location attribute
    output_path = base_dir / "data" / "iris_with_location.csv"
    df.to_csv(output_path, index=False)

    print("✅ Added 'location' column and saved to:", output_path)
    print("Columns now:", list(df.columns))
    print("Location value counts:")
    print(df["location"].value_counts())


if __name__ == "__main__":
    main()
