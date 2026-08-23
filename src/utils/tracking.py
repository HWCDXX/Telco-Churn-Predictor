# src/utils/tracking.py
import os
import tempfile
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.xgboost
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Dynamically locate project root (src/utils/tracking.py -> src/utils -> src -> Root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLRUNS_PATH = PROJECT_ROOT / "mlruns"


def setup_mlflow_experiment(experiment_name: str = "Telco Churn - XGBoost") -> str:
    """Configures active MLflow tracking URI and experiment name."""
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment(experiment_name)
    return experiment_name


def log_experiment_run(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    threshold: float,
    train_time: float,
    pred_time: float,
    probas,
    preds,
):
    """Logs parameters, metrics, feature schema, and XGBoost model artifacts to MLflow."""
    # Ensure tracking URI is set before run initialization
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("prediction_threshold", threshold)

        mlflow.log_metric("precision", precision_score(y_test, preds, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, preds, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, preds, zero_division=0))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probas))
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("pred_time_sec", pred_time)

        feature_cols = X_test.columns.tolist()
        mlflow.xgboost.log_model(xgb_model=model, artifact_path="model")

        with tempfile.TemporaryDirectory() as tmp_dir:
            feat_file_path = os.path.join(tmp_dir, "feature_columns.txt")
            with open(feat_file_path, "w") as f:
                for col in feature_cols:
                    f.write(f"{col}\n")

            mlflow.log_artifact(feat_file_path, artifact_path="model")


def print_markdown_report(y_true, y_pred, threshold: float = 0.3):
    """Prints formatted evaluation report table to console."""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    report_df.loc["accuracy", ["precision", "recall"]] = report_df.loc["accuracy", "f1-score"]
    report_df = report_df.reset_index().rename(columns={"index": ""})

    print("=" * 60)
    print(f"📊 XGBoost Trial Output (Threshold = {threshold})")
    print("=" * 60)
    print(tabulate(report_df, headers="keys", tablefmt="github", showindex=False))