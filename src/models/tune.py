# src/tune.py
from typing import Any, Dict
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier
from src.models.metrics import calculate_scale_pos_weight

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.3,
    n_trials: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs an Optuna study to optimize XGBoost hyperparameters for target recall."""
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": random_state,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_val)[:, 1]
        preds = (probas >= threshold).astype(int)

        # Optimize for Recall on positive class (Churn = 1)
        return float(optuna.metrics.recall_score(y_val, preds, pos_label=1))

    print(f"Starting Optuna hyperparameter search ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best Trial Recall: {study.best_value:.4f}")
    print("Best Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from build_features import engineer_features
    from load_data import load_raw_data
    from preprocess import run_preprocessing_pipeline

    raw_path = r"C:\Users\waghm\MY PROJECTS\Telco Churn Predictor\data\raw\telco_churn.csv"
    df = engineer_features(run_preprocessing_pipeline(load_raw_data(raw_path)))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_params = optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=10)