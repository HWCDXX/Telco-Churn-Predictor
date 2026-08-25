# src/models/train.py
import time
from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.load_data import load_raw_data
from src.data.preprocess_data import run_preprocessing_pipeline
from src.features.build_features import engineer_features
from src.models.metrics import calculate_scale_pos_weight
from src.models.tune import optimize_xgboost
from src.utils.tracking import (
    log_experiment_run,
    print_markdown_report,
    setup_mlflow_experiment,
)


def train_and_track_xgboost(
    df: pd.DataFrame,
    params: Dict[str, Any] | None = None,
    threshold: float = 0.3,
    experiment_name: str = "Telco Churn - XGBoost",
) -> None:
    """Trains an XGBoost model and logs the full experiment run to MLflow."""
    setup_mlflow_experiment(experiment_name=experiment_name)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    model_params = params.copy() if params else {}
    model_params.update(
        {
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }
    )

    start_train = time.time()
    xgb = XGBClassifier(**model_params)
    xgb.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    probas = xgb.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)
    pred_time = time.time() - start_pred

    log_experiment_run(
        model=xgb,
        X_test=X_test,
        y_test=y_test,
        params=model_params,
        threshold=threshold,
        train_time=train_time,
        pred_time=pred_time,
        probas=probas,
        preds=preds,
    )

    print_markdown_report(y_test, preds, threshold)


if __name__ == "__main__":
    raw_path = (
        r"C:\Users\waghm\MY PROJECTS\Telco Churn Predictor\data\raw\telco_churn.csv"
    )

    df_raw = load_raw_data(raw_path)
    df_prep = run_preprocessing_pipeline(df_raw)
    df_feat = engineer_features(df_prep)

    sample_best_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
    }

    train_and_track_xgboost(df=df_feat, params=sample_best_params, threshold=0.3)
