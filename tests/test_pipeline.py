# tests/test_pipeline.py
import os
import unittest
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
from src.data.load_data import load_raw_data
from src.data.preprocess_data import run_preprocessing_pipeline
from src.features.build_features import engineer_features
from src.models.train import train_and_track_xgboost
from src.serving.predict import ChurnPredictor


class TestTelcoPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_path = Path("data/raw/telco_churn.csv")

    def test_01_data_loading(self):
        df = load_raw_data(str(self.raw_path))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Loaded dataframe is empty")
        self.assertIn("Churn", df.columns, "Target column missing")
        print("✅ Test 01 Passed: Data Ingestion")

    def test_02_preprocessing(self):
        df_raw = load_raw_data(str(self.raw_path))
        df_prep = run_preprocessing_pipeline(df_raw)
        self.assertEqual(df_prep["Churn"].dtype, "int64", "Target not integer encoded")
        self.assertNotIn("customerID", df_prep.columns, "ID column not dropped")
        print("✅ Test 02 Passed: Preprocessing & Encoding")

    def test_03_feature_engineering(self):
        df_raw = load_raw_data(str(self.raw_path))
        df_prep = run_preprocessing_pipeline(df_raw)
        df_feat = engineer_features(df_prep)

        bool_cols = df_feat.select_dtypes(include=["bool"]).columns
        self.assertEqual(len(bool_cols), 0, f"Unconverted boolean columns found: {bool_cols}")
        print("✅ Test 03 Passed: Feature Engineering")

    def test_04_training_and_mlflow(self):
        df_raw = load_raw_data(str(self.raw_path))
        df_prep = run_preprocessing_pipeline(df_raw)
        df_feat = engineer_features(df_prep)

        train_and_track_xgboost(
            df=df_feat,
            params={"n_estimators": 10, "max_depth": 3},
            threshold=0.3,
            experiment_name="Validation_Test_Run",
        )
        self.assertTrue(Path("mlruns").exists(), "mlruns folder was not created")
        print("✅ Test 04 Passed: Model Training & MLflow Tracking")

    def test_05_serving_inference(self):
        predictor = ChurnPredictor(model_dir="./mlruns")
        sample_payload = {
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

        result = predictor.predict(sample_payload, threshold=0.3)
        self.assertIn("churn_class", result)
        self.assertIn(result["churn_class"], [0, 1])
        print(f"✅ Test 05 Passed: Inference Output -> {result['prediction']}")


if __name__ == "__main__":
    unittest.main()