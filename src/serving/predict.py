# Top section of src/serving/predict.py
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow.pyfunc
import pandas as pd

from src.data.preprocess_data import run_preprocessing_pipeline
from src.features.build_features import engineer_features

# Dynamically locate project root (src/serving/predict.py -> src/serving -> src -> Root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "mlruns"


class ChurnPredictor:
    """Production Inference Service for Telco Churn Predictions."""

    def __init__(self, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model, self.feature_cols = self._load_artifacts()

    def _resolve_model_path(self) -> Path:
        """Resolves active MLflow model folder path."""
        if self.model_dir.exists() and (self.model_dir / "MLmodel").exists():
            return self.model_dir.resolve()

        search_pattern = str(PROJECT_ROOT / "mlruns" / "**" / "MLmodel")
        all_mlmodels = glob.glob(search_pattern, recursive=True)

        if all_mlmodels:
            latest_mlmodel = max(all_mlmodels, key=os.path.getmtime)
            return Path(latest_mlmodel).parent.resolve()

        raise FileNotFoundError(
            f"Could not locate active MLmodel file at {self.model_dir} or inside {PROJECT_ROOT / 'mlruns'}"
        )
        

    def _load_artifacts(self) -> Tuple[Any, List[str]]:
        """Loads MLflow pyfunc model and extracts exact feature column requirements."""
        resolved_path = self._resolve_model_path()
        model_uri = resolved_path.as_uri()
        model = mlflow.pyfunc.load_model(model_uri)

        feature_cols: List[str] = []

        # Strategy 1: Check feature_columns.txt artifact
        feature_file = resolved_path / "feature_columns.txt"
        if feature_file.exists():
            with open(feature_file, "r") as f:
                feature_cols = [line.strip() for line in f if line.strip()]

        # Strategy 2: Extract directly from underlying XGBoost Booster
        if not feature_cols:
            try:
                booster = model._model_impl.xgb_model.get_booster()
                if booster.feature_names:
                    feature_cols = list(booster.feature_names)
            except Exception:
                pass

        # Strategy 3: Check MLflow model input schema metadata
        if not feature_cols:
            try:
                schema = model.metadata.get_input_schema()
                if schema:
                    feature_cols = list(schema.input_names())
            except Exception:
                pass

        print(f"✅ Loaded model & {len(feature_cols)} feature columns from {resolved_path}")
        return model, feature_cols

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies training preprocessing and feature engineering to input payload."""
        # 1. Clean & Encode
        df_prep = run_preprocessing_pipeline(df)

        # 2. Engineer features
        df_feat = engineer_features(df_prep)

        # 3. Drop target column if present
        if "Churn" in df_feat.columns:
            df_feat = df_feat.drop(columns=["Churn"])

        # 4. Reindex to align exactly with model input schema
        if self.feature_cols:
            return df_feat.reindex(columns=self.feature_cols, fill_value=0)

        return df_feat

    def predict(
        self, input_data: Union[Dict[str, Any], pd.DataFrame], threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Runs end-to-end inference for a single customer payload or DataFrame."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        transformed_df = self.transform_features(df)
        raw_preds = self.model.predict(transformed_df)

        if hasattr(raw_preds, "tolist"):
            raw_preds = raw_preds.tolist()

        pred_val = raw_preds[0] if isinstance(raw_preds, (list, tuple)) else raw_preds
        is_churn = int(pred_val >= threshold) if isinstance(pred_val, float) else int(pred_val)

        return {
            "prediction": "Likely to churn" if is_churn == 1 else "Not likely to churn",
            "churn_class": is_churn,
            "raw_output": pred_val,
        }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    service = ChurnPredictor(model_dir="./mlruns")
    result = service.predict(sample_customer, threshold=0.3)
    print("\nInference Output:")
    print(result)