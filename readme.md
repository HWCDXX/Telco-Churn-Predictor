# 🔮 The Architecture of Retention: An MLOps Chronicle
> *An End-to-End Production Guide to Predicting Customer Churn with XGBoost, FastAPI, Streamlit, Docker, GitHub Actions, and Render.*

![CI/CD Pipeline](https://img.shields.io/github/actions/workflow/status/HWCDXX/Telco-Churn-Predictor/ci.yml?branch=main&style=for-the-badge&logo=github)
![Docker Pulls](https://img.shields.io/docker/pulls/hwcdxx/telco-fastapi?style=for-the-badge&logo=docker)
![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi)
![Frontend](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit)

---

## 📚 Telco Customer Churn Predictor: Complete Engineering Masterbook & Architectural Blueprint
📚 Master Table of Contents

Executive Summary & Business Genesis

Full System Architecture & Technical Specifications

Data Sourcing, Exploratory Intelligence & Schema Breakdown

Feature Engineering, Data Preprocessing & Scikit-Learn Pipelines

Machine Learning Engine: XGBoost & Optuna Optimization

MLOps & Lifecycle Management: MLflow Experiment Tracking

REST API Microservice Architecture: FastAPI Infrastructure

User Interface Design: Interactive Streamlit Dashboard

Containerization Strategy: Docker & Multi-Container Orchestration

Continuous Integration & Delivery: GitHub Actions Automation

Cloud Infrastructure & Production Provisioning on Render

The Crucible of Errors: Comprehensive Post-Mortems & Bug Resolutions

Production Observability & Zero-Downtime Keep-Alive Strategy

Future Engineering Roadmap & MLOps Maturity Model

---

## Prologue: System Architecture & Design Philosophy

This project bridges the gap between statistical modeling and production software engineering. Rather than stopping at a Jupyter Notebook evaluation, the **Telco Churn Predictor** is implemented as a decoupled, microservice-driven MLOps system.

            
              ┌─────────────────────────────────────────────────────────┐
              │                    USER INTERFACE                       │
              │             Streamlit Web Application                   │
              │            (Port 8501 / Render Frontend)                │
              └────────────────────────────┬────────────────────────────┘
                                           │
                                 HTTP POST │ /predict
                                (JSON Payload)
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │                 INFERENCE REST API                      │
              │             FastAPI + Uvicorn Web Server                │
              │            (Port 8000 / Render Backend)                 │
              └────────────────────────────┬────────────────────────────┘
                                           │
                                 Loads Trained Artifacts
                                 from MLflow FileStore
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │               PREDICTIVE ENGINE & MODEL                 │
              │           XGBoost Classifier (mlruns/)                  │
              └─────────────────────────────────────────────────────────┘

---

## Chapter I: Data Sourcing & Exploratory Intelligence

* **Dataset Origin:** IBM Telco Customer Churn dataset (7,043 rows, 21 feature columns).
* **Target Variable:** `Churn` (`Yes` / `No` mapped to binary `1` / `0`).
* **Exploratory Insights:**
  1. **Tenure & Contract Type:** Customers on Month-to-Month contracts with short tenure ($\le 12$ months) exhibit the highest probability of churn.
  2. **Service Subscriptions:** Fiber Optic internet subscribers churn at significantly higher rates compared to DSL, often correlated with high `MonthlyCharges` without supplemental `TechSupport` or `OnlineSecurity`.
  3. **Financial Factors:** Electronic Check is the primary payment method associated with high risk.

---

## Chapter II: Feature Engineering & Model Development

* **Data Preprocessing:** Categorical features (`InternetService`, `Contract`, `PaymentMethod`, etc.) are transformed via One-Hot Encoding. Missing values in numerical attributes (e.g., `TotalCharges`) are imputed using median values.
* **Hyperparameter Optimization:** Utilized **Optuna** to run automated Bayesian optimization trials over XGBoost hyperparameters:
  * `max_depth`, `learning_rate`, `n_estimators`, `subsample`, and `colsample_bytree`.
* **Evaluation Metrics:** Optimized specifically for **ROC-AUC** and **Recall** on Class 1 (Churners) to minimize false negatives in proactive retention campaigns.

---

## Chapter III: Experiment Tracking & Artifact Lifecycle (MLflow)

Model metrics, parameters, and serialization states are managed through **MLflow**.

* **Run Logs:** Metrics logged across trials include Precision, Recall, F1-score, ROC-AUC, and log-loss.
* **Artifact Logging:** Models are saved natively as MLflow artifacts under `./mlruns/` containing:
  * `MLmodel` specification file.
  * `model.xgb` serialized weights.
  * `conda.yaml` and `requirements.txt` environment locks.

---

## Chapter IV: Containerization & Microservice Decoupling

The platform is split into two lightweight, isolated Docker containers orchestrated locally via `docker-compose.yml`.

### 1. FastAPI Inference Backend (`Dockerfile.fastapi`)
* Exposes `/health` for liveness checks and `/predict` for ML inference.
* Accepts structured data validated via **Pydantic** (`CustomerPayload`).

### 2. Streamlit UI Frontend (`Dockerfile.streamlit`)
* Provides an intuitive interface for field agents to input customer profiles.
* Renders real-time gauge visualizations (via Plotly) and strategic risk mitigation recommendations.

---

## Chapter V: Automated Quality Gates & CI/CD Pipeline

Every code push to the `main` branch triggers an automated GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Code Formatting & Linting:** Enforces PEP8 compliance using `black` and `flake8`.
2. **Automated Unit Testing:** Executes `pytest` across pipeline and preprocessing modules.
3. **Container Build & Registry Push:** Builds Docker images for both services and pushes them directly to Docker Hub:
   * `hwcdxx/telco-fastapi:latest`
   * `hwcdxx/telco-streamlit:latest`

---

## Chapter VI: Cloud Provisioning & Production Deployment (Render)

Both services are deployed on **Render** using standalone Docker Web Service instances.

### Environment Variable Matrix

| Service | Key | Value | Purpose |
| :--- | :--- | :--- | :--- |
| **FastAPI Backend** | `PORT` | `8000` | Exposes Uvicorn server to Render routing |
| | `MLFLOW_ALLOW_FILE_STORE` | `true` | Permits file-based MLflow artifact loading |
| **Streamlit Frontend** | `PORT` | `8501` | Exposes Streamlit server to Render routing |
| | `BACKEND_URL` | `https://<fastapi-service>.onrender.com` | Directs REST API traffic to live backend |

---

## Chapter VII: The Crucible of Errors (Failures & Technical Fixes)

During development and deployment, several non-trivial engineering issues emerged. Below is the complete diagnostic log and resolution matrix:

### Issue 1: Render Deployment Cancellation (Missing Model Files)
* **Symptom:** Render deployments failed repeatedly with `503 Service Unavailable` and logs stating:
  `⚠️ Warning: Could not initialize ChurnPredictor on startup: Could not locate active MLmodel file at mlruns`
* **Root Cause:** The `mlruns/` directory was listed inside `.gitignore`/`.dockerignore`, preventing Git from pushing model binaries to GitHub. The GitHub Actions builder generated a clean container without model artifacts.
* **Resolution:** Removed `mlruns` from ignore files, executed `git add -f mlruns/`, and committed model binaries directly so `COPY . /app` included trained weights inside the image.

### Issue 2: Cross-Platform Path Failure in CI Runner (`PermissionError: '/C:'`)
* **Symptom:** `pytest` failed inside the Linux GitHub Actions runner (`ubuntu-latest`) with:
  `PermissionError: [Errno 13] Permission denied: '/C:'`
* **Root Cause:** Hardcoded Windows drive paths (`C:\...`) inside test MLflow tracking URIs were parsed as non-existent root directories on Linux filesystem containers.
* **Resolution:** Replaced raw string paths in `tests/test_pipeline.py` with `pathlib.Path.resolve().as_uri()` to generate cross-platform URIs (`file:./mlruns_test`).

### Issue 3: MLflow Local FileStore Exception in Pytest
* **Symptom:** `pytest` threw `MlflowException`:
  `The filesystem tracking backend (e.g., './mlruns') is in maintenance mode... set MLFLOW_ALLOW_FILE_STORE=true`
* **Root Cause:** Modern MLflow versions enforce security blocks against local directory tracking unless explicitly overridden by an environment variable.
* **Resolution:** Injected `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` at the top of `tests/test_pipeline.py` and set the variable in Render runtime configs.

### Issue 4: Streamlit Frontend False "Backend Disconnected" Errors
* **Symptom:** Streamlit rendered a red `🔴 Backend Disconnected` banner upon opening the application.
* **Root Cause:** Render's free tier spins down inactive containers. Streamlit's `check_backend_health()` had a strict `timeout=3` seconds, failing before FastAPI finished cold-start bootup (which takes 30–40s).
* **Resolution:** Sanitized `BACKEND_URL.rstrip("/")`, added schema fallback keys (`raw_output`/`churn_probability`), and increased health check request timeouts to `timeout=15`.

---

## Epilogue: Live Endpoints & Observability Keep-Alive

### Live Production Services
* **Interactive Frontend:** [https://telco-streamlit-frontend.onrender.com](https://telco-streamlit-frontend.onrender.com)
* **FastAPI Docs (Swagger):** [https://telco-churn-fastapi.onrender.com/docs](https://telco-churn-fastapi.onrender.com/docs)
* **Backend Liveness Check:** [https://telco-churn-fastapi.onrender.com/health](https://telco-churn-fastapi.onrender.com/health)

### Eliminating Cold Starts
To prevent Render free instances from entering sleep mode after 15 minutes of inactivity, an external 5-minute HTTP ping is configured on the backend liveness endpoint:
* **Target URL:** `https://telco-churn-fastapi.onrender.com/health`
* **Schedule:** `*/5 * * * *` (Every 5 minutes via UptimeRobot / cron-job.org)
* **Expected Result:** `200 OK` status, ensuring zero-latency response times for end users.
