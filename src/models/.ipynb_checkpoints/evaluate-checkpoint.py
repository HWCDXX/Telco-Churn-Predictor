# src/evaluate.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_target_correlation(
    df: pd.DataFrame, 
    target_col: str = "Churn", 
    figsize: tuple[int, int] = (4, 12)
) -> None:
    """Computes numeric correlations with the target column and plots a heatmap."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    corr_matrix = df.corr(numeric_only=True)
    churn_corr = corr_matrix[[target_col]].sort_values(by=target_col, ascending=False)

    plt.figure(figsize=figsize)
    sns.heatmap(churn_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation of features with {target_col}")
    plt.tight_layout()
    plt.show()


def calculate_vif(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Calculates Variance Inflation Factor (VIF) for feature columns 

    to detect multicollinearity.
    """
    # Exclude target and drop non-finite values
    X = df.drop(columns=[target_col], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)


def check_class_imbalance(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Returns frequency counts and percentage distribution of the target variable."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    counts = df[target_col].value_counts().to_frame(name="count")
    counts["percentage"] = df[target_col].value_counts(normalize=True) * 100

    return counts


if __name__ == "__main__":
    from build_features import engineer_features
    from load_data import load_raw_data
    from preprocess import run_preprocessing_pipeline

    raw_path = r"C:\Users\waghm\MY PROJECTS\Telco Churn Predictor\data\raw\telco_churn.csv"
    raw_df = load_raw_data(raw_path)
    df_prep = run_preprocessing_pipeline(raw_df)
    df_feat = engineer_features(df_prep)

    # Run evaluations
    print("--- Class Imbalance ---")
    print(check_class_imbalance(df_feat))

    print("\n--- Variance Inflation Factor (Top 5) ---")
    print(calculate_vif(df_feat).head(5))

    # Plot correlation
    plot_target_correlation(df_feat)