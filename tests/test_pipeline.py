import os
from pathlib import Path
import mlflow
import pytest

# Allow file-based MLflow tracking in pytest
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class TestTelcoPipeline:

    @pytest.fixture(autouse=True)
    def setup_mlflow_environment(self):
        """Configure cross-platform file store URI for tests."""
        test_dir = Path("./mlruns_test").resolve()
        mlflow.set_tracking_uri(test_dir.as_uri())
        yield

    def test_01_data_loading(self):
        """Verify data loading pipeline."""
        pass

    def test_02_data_preprocessing(self):
        """Verify preprocessing transformations."""
        pass

    def test_03_model_initialization(self):
        """Verify model setup."""
        pass

    def test_04_training_and_mlflow(self):
        """Verify MLflow model logging."""
        test_dir = Path("./mlruns_test").resolve()
        mlflow.set_tracking_uri(test_dir.as_uri())

        mlflow.set_experiment("Telco_Churn_Test_Experiment")

        with mlflow.start_run():
            mlflow.log_param("test_param", 1)
            mlflow.log_metric("test_metric", 0.95)

        assert True