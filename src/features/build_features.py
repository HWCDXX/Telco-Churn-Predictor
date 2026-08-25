# src/build_features.py
import pandas as pd
from src.data.load_data import load_raw_data
from src.data.preprocess_data import run_preprocessing_pipeline


def cast_booleans_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all boolean columns (True/False) into integers (1/0)."""
    df = df.copy()
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def consolidate_redundant_service_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidates redundant multi-column 'No internet service' and 'No phone service'

    dummy variables into singular indicator columns.
    """
    df = df.copy()

    # Consolidate 'No internet service' columns if present
    internet_no_service_cols = [
        col for col in df.columns if "No internet service" in col
    ]

    if internet_no_service_cols:
        # Check across any of the related service dummy columns
        df["No_internet_service"] = df[internet_no_service_cols].any(axis=1).astype(int)
        df = df.drop(columns=internet_no_service_cols)

    # Consolidate 'No phone service' column if present
    if "MultipleLines_No phone service" in df.columns:
        df["No_phone_service"] = df["MultipleLines_No phone service"].astype(int)
        df = df.drop(columns=["MultipleLines_No phone service"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline function running feature engineering steps."""
    df = cast_booleans_to_int(df)
    df = consolidate_redundant_service_dummies(df)
    return df


if __name__ == "__main__":
    # Example local testing setup
    from load_data import load_raw_data
    from preprocess import run_preprocessing_pipeline

    raw_path = (
        r"C:\Users\waghm\MY PROJECTS\Telco Churn Predictor\data\raw\telco_churn.csv"
    )
    raw_df = load_raw_data(raw_path)
    prep_df = run_preprocessing_pipeline(raw_df)
    feat_df = engineer_features(prep_df)

    print(f"Features Engineered Shape: {feat_df.shape}")
