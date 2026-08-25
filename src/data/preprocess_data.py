# src/data/preprocess_data.py
import pandas as pd
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

DEFAULT_BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

DEFAULT_MULTI_CAT_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def clean_raw_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Phase 1: Cleans column names, removes ID columns, fixes data types, and handles NaNs."""
    df = df.copy()

    df.columns = df.columns.str.strip()

    id_cols = ["customerID", "CustomerID", "customer_id"]
    df = df.drop(columns=[col for col in id_cols if col in df.columns], errors="ignore")

    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def encode_features(
    df: pd.DataFrame,
    binary_cols: list[str] | None = None,
    multi_cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Phase 2: Encodes binary and multi-class categorical features into numeric formats."""
    df = df.copy()
    b_cols = binary_cols if binary_cols is not None else DEFAULT_BINARY_COLS
    m_cols = multi_cat_cols if multi_cat_cols is not None else DEFAULT_MULTI_CAT_COLS

    valid_b_cols = [col for col in b_cols if col in df.columns]
    mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    # Fix: Added .infer_objects(copy=False) to suppress Pandas downcasting warning
    df[valid_b_cols] = (
        df[valid_b_cols].replace(mapping).infer_objects(copy=False).astype("int64")
    )

    valid_m_cols = [col for col in m_cols if col in df.columns]
    df = pd.get_dummies(df, columns=valid_m_cols, drop_first=True)

    return df


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executes full preprocessing: Data Cleaning -> Categorical Encoding."""
    df_clean = clean_raw_data(df)
    df_encoded = encode_features(df_clean)
    return df_encoded
