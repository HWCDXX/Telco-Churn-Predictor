# MLOps Pipeline Post-Mortem & Incident Log
**Project:** Telco Customer Churn Predictor  
**Author:** Engineering Team  
**Date:** August 19, 2026  
**Status:** Resolved (All 5 Pipeline Tests Passing)  

---

## 1. Executive Summary

During automated integration testing (`python -m unittest tests/test_pipeline.py`), the MLOps pipeline failed across two critical operational boundaries: tracking backend initialization (`test_04`) and inference schema alignment (`test_05`). 

Through systematic root-cause analysis, we resolved a deprecated MLflow store file lock, eliminated Train-Serve feature skew, and implemented a resilient schema recovery fallback strategy.

| Incident ID | Affected Module | Error Type | Root Cause | Engineering Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **INC-01** | `tracking.py` | `MlflowException` | MLflow file store local tracking disabled in modern releases | Injected `MLFLOW_ALLOW_FILE_STORE=true` globally across execution contexts |
| **INC-02** | `predict.py` | `ValueError` | Raw prediction payload lacked full 25 engineered training features | Enforced identical preprocessing pipeline & `.reindex()` during inference |
| **INC-03** | `predict.py` | `KeyError` / Loss | Model wrapper abstracted native XGBoost booster column schema | Designed 3-tiered artifact schema resolution fallback mechanism |

---

## 2. Detailed Incident Analysis & Fixes

### INC-01: MLflow FileStore Maintenance Mode Exception
* **Symptom:** `test_04_training_and_mlflow` raised `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode`.
* **Root Cause:** Modern MLflow versions restrict raw local directory tracking (`./mlruns`) to prevent file lock corruption across concurrent processes, requiring explicit opt-in for local execution.
* **Resolution:**
  * Added `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` at the top of `src/utils/tracking.py`, `src/serving/predict.py`, and `tests/test_pipeline.py`.

---

### INC-02: Train-Serve Feature Skew & Schema Mismatch
* **Symptom:** `test_05_serving_inference` failed with `ValueError: feature_names mismatch`. XGBoost expected 25 feature columns (including one-hot encoded categories), but received only 9 raw payload features.
* **Root Cause:** The `ChurnPredictor.predict()` method passed raw JSON inputs directly into the model without routing through feature engineering (`engineer_features()`), leading to missing dummy variables (e.g., `Contract_One year`, `No_internet_service`).
* **Resolution:**
  * Updated `transform_features()` to execute both `run_preprocessing_pipeline()` and `engineer_features()`.
  * Applied explicit schema alignment using Pandas `.reindex(columns=self.feature_cols, fill_value=0)` to guarantee zero-filling for unobserved categories at inference time.

---

### INC-03: PyFunc Artifact Schema Deserialization Failure
* **Symptom:** In initial runs, `ChurnPredictor` reported `Loaded model & 0 feature columns`, leading to feature schema validation failure.
* **Root Cause:** Loading models via generic `mlflow.pyfunc.load_model()` wraps XGBoost in a generic interface that omits framework-native metadata like `xgb_model.get_booster().feature_names`.
* **Resolution:**
  * Modified `log_experiment_run()` to write an explicit `feature_columns.txt` artifact directly into the MLflow model folder during training.
  * Built a 3-tier fallback loader in `_load_artifacts()`:
    1. Read explicit `feature_columns.txt` artifact file.
    2. Fallback to extracting `get_booster().feature_names` from underlying model instance.
    3. Fallback to MLflow `metadata.get_input_schema()`.

---

## 3. Pipeline Validation Results

Executing the full suite after fixes yielded 100% test passage across all five pipeline stages:

```bash
(telco_env) C:\Users\xyz\MY PROJECTS\Telco Churn Predictor>python -m unittest tests/test_pipeline.py
Data successfully loaded. Shape: (7043, 21)
✅ Test 01 Passed: Data Ingestion
.Data successfully loaded. Shape: (7043, 21)
✅ Test 02 Passed: Preprocessing & Encoding
.Data successfully loaded. Shape: (7043, 21)
✅ Test 03 Passed: Feature Engineering
.Data successfully loaded. Shape: (7043, 21)
✅ Test 04 Passed: Model Training & MLflow Tracking
.✅ Loaded model & 25 feature columns from mlruns/...
✅ Test 05 Passed: Inference Output -> Likely to churn
.
----------------------------------------------------------------------
Ran 5 tests in 31.085s - OK