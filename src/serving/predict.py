# src/serving/predict.py
import os
import logging
import pandas as pd
import mlflow.pyfunc

logger = logging.getLogger("telco-churn-api")

class ChurnPredictor:
    def __init__(self, model_uri: str = None):
        if not model_uri:
            # Fallback to MLflow Model Registry URI or environment variable
            model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/TelcoChurnXGBoost/Production")
        
        logger.info(f"Loading MLflow model from URI: {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)
        logger.info("MLflow model loaded successfully into memory.")

    def predict(self, input_dict: dict, threshold: float = 0.3) -> dict:
        df = pd.DataFrame([input_dict])
        
        # MLflow pyfunc predict
        raw_probs = self.model.predict(df)
        probability = float(raw_probs[0]) if hasattr(raw_probs, "__len__") else float(raw_probs)
        
        churn_class = 1 if probability >= threshold else 0
        label = "Likely to churn" if churn_class == 1 else "Not likely to churn"

        return {
            "prediction": label,
            "churn_class": churn_class,
            "raw_output": round(probability, 4)
        }
