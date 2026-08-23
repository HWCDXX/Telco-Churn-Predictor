# src/load_data.py
from pathlib import Path
import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """Loads the raw Telco Churn dataset into a pandas DataFrame.

    Args:
        file_path (str | Path): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at specified path: {path}")

    df = pd.read_csv(path)
    print(f"Data successfully loaded. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Local test execution
    DEFAULT_PATH = (
        r"C:\Users\waghm\MY PROJECTS\Telco Churn Predictor\data\raw\telco_churn.csv"
    )
    df = load_raw_data(DEFAULT_PATH)
    print(df.head())