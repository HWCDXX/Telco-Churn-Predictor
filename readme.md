# 🔮 The Architecture of Retention: An MLOps Chronicle
> *An End-to-End Production Guide to Predicting Customer Churn with XGBoost, FastAPI, Streamlit, Docker, GitHub Actions, and Render.*

![CI/CD Pipeline](https://img.shields.io/github/actions/workflow/status/HWCDXX/Telco-Churn-Predictor/ci.yml?branch=main&style=for-the-badge&logo=github)
![Docker Pulls](https://img.shields.io/docker/pulls/hwcdxx/telco-fastapi?style=for-the-badge&logo=docker)
![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi)
![Frontend](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit)

---

📑 Master Index & Navigation
Click on any section or chapter heading below to navigate directly to that phase of the project audit.

Executive Summary

System Architecture Blueprint

Chapter 1: Strategic Inception & Business Rationale

Chapter 2: Data Sourcing & Exploratory Intelligence

Chapter 3: Feature Engineering & Preprocessing Pipeline

Chapter 4: Model Exploration, Optimization & Tuning

Chapter 5: Experiment Tracking & Artifact Lifecycle Management

Chapter 6: Microservice Architecture & API Engineering

Chapter 7: Containerization & Environment Isolation

Chapter 8: Automated Quality Gates & CI/CD Pipeline

Chapter 9: Cloud Infrastructure & Production Deployment

Chapter 10: The Engineering Crucible (Detailed Failure Ledger & Root Cause Analyses)

Chapter 11: Production Observability, Monitoring & Keep-Alive Systems

Chapter 12: Visual Audit & Project Screenshot Repository

---

Executive Summary
The Telco Customer Churn Predictor represents a complete, production-grade Machine Learning Operations (MLOps) ecosystem designed to quantify, predict, and mitigate subscriber attrition for enterprise telecommunications providers. Customer churn is one of the most critical financial metrics in subscription-based business models; acquiring a new customer costs between 5 to 7x more than retaining an existing one.

This technical report documents the end-to-end lifecycle of the solution—transitioning from exploratory data science to a decoupled, containerized, microservice-driven cloud architecture. Operating on an automated Continuous Integration and Continuous Deployment (CI/CD) pipeline, the platform ingests raw data, processes tabular features through robust transformation gates, evaluates risk via a optimized gradient-boosted decision tree (XGBoost), and delivers real-time probability assessments through a interactive web interface and REST API.

Every phase of this initiative has been framed around four fundamental questions designed for executive and engineering review:

What? Defines the technical boundary, component, or objective under scrutiny.

How? Details the operational implementation, algorithmic mechanics, and software engineering protocols.

Why? Explains the strategic business rationale, trade-offs, and design choices.

Where? Pinpoints the precise location of the component within the physical and logical system architecture.

System Architecture Blueprint
The platform follows a decoupled, microservice-based design pattern that isolates machine learning inference from user interaction, ensuring independent scalability, maintainability, and security.

User Interaction Layer: A Streamlit-based web portal that captures user inputs, constructs validated JSON payloads, renders interactive gauge visualization charts, and displays churn mitigation strategies.

Inference API Layer: A FastAPI application hosted on Uvicorn that exposes structured endpoints for health checks and model inference, enforcing strict data schemas via Pydantic.

Predictive Core: An XGBoost classification model managed and versioned through MLflow artifact directories, loaded dynamically into memory upon API bootup.

Automated CI/CD Engine: A GitHub Actions runner that triggers code quality linting, unit testing with Pytest, Docker image construction, and image publishing to Docker Hub upon every code commit.

Cloud Infrastructure: Multi-container Web Services hosted on Render cloud infrastructure, protected by external automated keep-alive polling.

Chapter 1: Strategic Inception & Business Rationale
What?
This phase establishes the business scope and mathematical formalization of customer churn prediction. Churn is defined as the event where a subscriber terminates their service contract within a specified timeframe. The business goal is to transform reactive customer service into a proactive retention strategy.

How?
The problem is modeled as a supervised binary classification task. The predictive target, Churn, is assigned a binary value of 1 for accounts that closed within the observation window and 0 for accounts that remained active. The output of the machine learning model is not merely a hard binary decision, but a continuous probability score ranging from 0.00 to 1.00. This score reflects the likelihood of churn and allows account managers to segment customers into actionable risk tiers: Low Risk (0.00 to 0.35), Moderate Risk (0.35 to 0.65), and High Risk (0.65 to 1.00).

Why?
In telecommunications, margins are tightly coupled with Customer Lifetime Value (CLTV). Uncontrolled subscriber churn directly reduces Annual Recurring Revenue (ARR) and inflates Customer Acquisition Cost (CAC). By predicting churn risk before a cancellation notice is submitted, marketing and customer success teams can deploy targeted retention offers—such as discounted contract renewals or service upgrades to high-risk subscribers. Applying financial incentives selectively to high-probability churners maximizes retention yield while preventing unnecessary discount spending on loyal, low-risk accounts.

Where?
This strategic framework forms the foundation of the system. It governs the design of loss functions in the model, API request/response structures, and UI layout.


Chapter 2: Data Sourcing & Exploratory Intelligence
What?
This phase focuses on the ingestion, inspection, and exploratory audit of the IBM Telco Customer Churn dataset. The dataset comprises 7,043 unique customer records across 21 distinct feature variables, encompassing demographic data, account tenure, subscribed services, contract structures, and financial metrics.

How?
Exploratory Data Analysis (EDA) was performed across all 21 features to evaluate distributions, detect anomalies, evaluate target class imbalance, and measure statistical correlations with churn.

Feature Category,Features Included,Key Business Insight Identified
Account Metadata,"CustomerID, Tenure, Contract",Short tenure (under 12 months) combined with Month-to-Month contracts produces the highest churn density.
Subscribed Services,"PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies",Fiber Optic subscribers exhibit significantly higher churn than DSL users due to pricing friction and lack of bundled security.
Financial Indicators,"PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges",Electronic Check users churn at nearly twice the rate of automated credit card or bank transfer users.
Demographics,"Gender, SeniorCitizen, Partner, Dependents",Senior citizens without dependents display higher baseline risk metrics.

Data auditing identified a key structural anomaly in the TotalCharges feature: 11 records contained blank string whitespace characters where numerical float values were expected. These records corresponded to newly onboarded customers with a Tenure of 0 months who had not yet received their first billing statement.

Why?
Raw data cannot be passed directly into gradient-boosted decision trees without rigorous validation. Identifying blank values in TotalCharges prevented runtime type-casting crashes during model ingestion. Furthermore, identifying target class imbalance—approximately 26.5% churned (1,869 accounts) versus 73.5% retained (5,174 accounts)—dictated that standard accuracy metrics could not be used to judge model performance. Evaluating model performance strictly on raw accuracy would yield a naive baseline accuracy of 73.5% simply by predicting "No Churn" for every customer, while missing 100% of at-risk accounts.

Where?
Data auditing occurs at the boundary between raw storage and the data processing pipeline, executed inside interactive exploratory notebooks before model pipeline serialization.

Chapter 3: Feature Engineering & Preprocessing Pipeline
What?
The feature engineering phase transforms raw, heterogeneous categorical strings and numerical attributes into standardized numeric arrays optimized for matrix multiplication and tree-based node splitting.

How?
A deterministic preprocessing pipeline was constructed using scikit-learn transformers:

Numerical Imputation & Conversion: The 11 blank entries in TotalCharges were coerced to float values and imputed using the feature median. Numerical features (Tenure, MonthlyCharges, TotalCharges) were audited for scaling requirements.

Categorical Encoding: Categorical features with low cardinality (such as Gender, Partner, Dependents, PaperlessBilling) were converted to binary indicators. Multi-class categorical variables (such as InternetService, Contract, PaymentMethod) were encoded using One-Hot Encoding to avoid implying an artificial ordinal ranking among un-ordered categories.

Categorical Consolidation: Redundant sub-categories such as "No phone service" and "No internet service" across security and support features were grouped cleanly to streamline feature space dimensions.

Why?
Ensuring that data transformations are encapsulated inside a single, reusable pipeline object eliminates Data Leakage. In data science workflows, calculating statistics (such as mean, median, or encoding mappings) across the entire dataset before splitting introduces future test-set knowledge into the training phase. Enforcing fit-transform steps strictly on the training partition ensures valid model evaluation. Furthermore, serializing the fitted preprocessor guarantees that incoming production REST API requests undergo the exact same mathematical transformation as historical training data.

Where?
The preprocessing logic resides within the core Python module library (src/preprocessing.py) and is executed both during offline model training and dynamically inside the FastAPI inference route upon payload ingestion.


Chapter 4: Model Exploration, Optimization & Tuning
What?
This phase covers algorithm selection, hyperparameter tuning using OPTUNA, and metric optimization to construct a model with strong predictive accuracy and generalization performance.

How?
Multiple classification algorithms were benchmarked using 5-fold cross-validation: Logistic Regression, Random Forest, and XGBoost (Extreme Gradient Boosting). XGBoost consistently outperformed simpler models due to its handling of non-linear feature interactions, missing values, and gradient-boosted decision trees.

To optimize the XGBoost classifier, Optuna was integrated to execute Bayesian hyperparameter optimization across the search space:

Number of Estimators: Evaluated from 100 to 500 trees.

Max Depth: Evaluated from 3 to 10 to balance model capacity and prevent overfitting.

Learning Rate (Eta): Tuned between 0.01 and 0.20 to control step sizes along the loss gradient.

Subsample & Colsample By Tree: Tuned between 0.6 and 1.0 to introduce stochastic regularization.

Scale Pos Weight: Adjusted to account for the 3:1 target class imbalance.

       Hyperparameter Optimization Search Space & Final Selection
       ┌──────────────────────────────────────────────────────────┐
       │ Parameter: max_depth         --> Final Selected: 4       │
       │ Parameter: learning_rate     --> Final Selected: 0.03    │
       │ Parameter: n_estimators      --> Final Selected: 250     │
       │ Parameter: subsample         --> Final Selected: 0.8     │
       │ Parameter: scale_pos_weight  --> Final Selected: 2.76    │
       └──────────────────────────────────────────────────────────┘
Why?
In customer churn modeling, the costs of classification errors are asymmetrical:

False Negative (Unidentified Churner): The system predicts a customer will stay, but they cancel their account. The company loses 100% of the customer's future lifetime value.

False Positive (False Alarm): The system predicts a customer will leave, but they intended to stay. The company incurs a minor expense by sending a promotional retention discount.

Because a False Negative is significantly more costly than a False Positive, hyperparameter optimization focused on maximizing Recall and Area Under the Receiver Operating Characteristic Curve (ROC-AUC) rather than overall accuracy. Setting scale_pos_weight to 2.76 adjusted the loss function gradient to penalize missed churners more heavily.

Where?
Model optimization runs within the offline execution scripts (src/train.py), outputting optimized hyperparameter logs and trained binary model weight artifacts.


Chapter 5: Experiment Tracking & Artifact Lifecycle Management
What?
This phase establishes formal model tracking and governance using MLflow. MLflow tracks hyperparameter search runs, logs evaluation metrics, records environment configurations, and versions generated model artifacts.

How?
The training workflow initializes an MLflow experiment context. During every execution run, the script logs parameters, metric evaluations, and serialized weights to a structured directory structure (mlruns/):

Parameters Logged: Learning rate, tree depth, sample split ratios, algorithm type, random seeds.

Metrics Logged: Training ROC-AUC, Validation ROC-AUC, Precision, Recall, F1-score, Log-Loss.

Artifacts Stored: Serialized XGBoost model binary (model.xgb), MLflow environment spec (MLmodel), Conda configuration dependencies, and preprocessing pickle objects.

Why?
Without centralized experiment tracking, machine learning models become un-auditable "black boxes." If a production model's performance degrades over time, engineers must be able to inspect the exact training code, dataset split, and hyperparameters used to generate those weights. MLflow provides complete reproducibility, enabling team members to inspect historic runs, compare performance metrics across model iterations, and safely roll back to previous model versions if required.

Where?
MLflow writes metrics and binary objects locally to the ./mlruns/ tracking directory. During containerization, these verified artifact folders are packaged into the Docker container image to serve active API inference calls.


Chapter 6: Microservice Architecture & API Engineering
What?
The microservice layer decouples model inference logic from user interface presentation. It provides a RESTful web service engineered using FastAPI and Uvicorn, alongside an interactive user dashboard developed using Streamlit.

How?
The platform uses two dedicated microservice applications:

1. FastAPI Inference Backend
Data Validation: Uses Pydantic models (CustomerPayload) to validate incoming JSON POST payloads. If field values fall outside accepted formats or types, the API returns an informative 422 Unprocessable Entity response before execution reaches the model.

Health Monitoring Route: Exposes a /health endpoint returning a 200 OK status and health JSON body, allowing load balancers and orchestrators to perform liveness probes.

Inference Route: Exposes a /predict endpoint that transforms validated data payloads, feeds features into the loaded XGBoost model, calculates churn probabilities, and returns a structured response containing prediction labels and confidence scores.

2. Streamlit Web Dashboard
Input Interface: Renders interactive form inputs (sliders, drop-downs, numeric inputs) mapped to model feature variables.

API Integration: Formats user inputs into a JSON payload, issues an asynchronous HTTP POST request to the FastAPI backend /predict endpoint, and receives prediction results.

Visual Presentation: Uses Plotly to render a gauge chart showing risk level, accompanied by dynamic retention recommendations based on the calculated risk tier.

REST API Input/Response Data Schema
  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
  │ INCOMING REQUEST PAYLOAD        │   │ OUTGOING INFERENCE RESPONSE     │
  ├─────────────────────────────────┤   ├─────────────────────────────────┤
  │ {                               │   │ {                               │
  │   "tenure": 2,                  │   │   "status": "success",          │
  │   "Contract": "Month-to-month", │   │   "churn_probability": 0.784,   │
  │   "MonthlyCharges": 85.70,      │ ──┼─> "churn_prediction": 1,        │
  │   "InternetService": "Fiber",   │   │   "risk_level": "High Risk",    │
  │   "PaymentMethod": "Electronic" │   │   "recommended_action": "Offer" │
  │ }                               │   │ }                               │
  └─────────────────────────────────┘   └─────────────────────────────────┘

  Why?
Combining user interface rendering and model inference inside a single monolith application creates tight coupling: changing a dashboard UI element risks breaking inference code, and high frontend traffic can starve model execution resources. Decoupling the system into distinct microservices allows each component to scale independently based on demand. Furthermore, an isolated REST API allows third-party tools—such as mobile apps, external enterprise CRMs, or automated marketing workflows—to consume churn predictions directly without interacting with the Streamlit interface.

Where?
FastAPI runs on host port 8000, while Streamlit operates on host port 8501. In production, these services run in isolated container runtimes communicating securely over HTTPS routes.


Chapter 7: Containerization & Environment Isolation
What?
Containerization packages application code, runtime dependencies, system libraries, configuration settings, and model binary artifacts into isolated, reproducible execution units using Docker.

How?
The project uses two independent Dockerfile configurations alongside a master docker-compose.yml file for multi-container orchestration.

Base Layer Selection: Built using python:3.12-slim to reduce image size and minimize security vulnerability exposure.

Layer Caching Optimization: requirements.txt dependencies are copied and installed in an early build layer, ensuring Docker reuses cached package layers unless dependency manifests change.

Environment Provisioning: Configures python environment variables (PYTHONUNBUFFERED=1, PYTHONDONTWRITEBYTECODE=1) to keep container logs clear and lightweight.

Port Mapping: Exposes port 8000 for the FastAPI container and port 8501 for the Streamlit container.

                        Multi-Container Docker Network
  ┌───────────────────────────────────────────────────────────────────────┐
  │ DOCKER BRIDGE NETWORK: telco-network                                  │
  │                                                                       │
  │   ┌───────────────────────────┐         ┌───────────────────────────┐ │
  │   │ Container 1: Streamlit    │         │ Container 2: FastAPI      │ │
  │   │ Image: telco-streamlit    │ ──────> │ Image: telco-fastapi      │ │
  │   │ Internal Port: 8501       │  HTTP   │ Internal Port: 8000       │ │
  │   └─────────────┬─────────────┘         └─────────────┬─────────────┘ │
  └─────────────────┼─────────────────────────────────────┼───────────────┘
                    │ Port Forwarding                     │ Port Forwarding
                    ▼                                     ▼
             Host Port: 8501                       Host Port: 8000
Why?
Containerization eliminates environmental inconsistencies between local developer workstations and cloud production servers. Packaging dependencies inside Docker containers ensures that Python library updates or host OS differences do not cause unexpected failures in production environments.

Where?
Docker definitions reside in Dockerfile.fastapi, Dockerfile.streamlit, and docker-compose.yml within the repository root. Docker images are published to Docker Hub (hwcdxx/telco-fastapi and hwcdxx/telco-streamlit) for cloud deployment.


Chapter 8: Automated Quality Gates & CI/CD Pipeline
What?
Continuous Integration and Continuous Deployment (CI/CD) automates code quality validation, unit testing, and container deployment upon every repository update using GitHub Actions.

How?
The CI/CD pipeline workflow (.github/workflows/ci.yml) triggers automatically whenever code is pushed to the main branch. The pipeline executes across sequential jobs:
                      GitHub Actions Automated Pipeline Flow
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
 │  GIT PUSH TO   │ ──> │ LINTING GATE   │ ──> │ PYTEST SUITE   │ ──> │ DOCKER BUILD & │
 │  MAIN BRANCH   │     │ Black / Flake8 │     │ Unit Tests     │     │ PUSH TO HUB    │
 └────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘

Environment Setup: Provisions an ubuntu-latest virtual runner, checks out repository source code, and sets up Python 3.12.

Static Analysis Gate: Runs black --check and flake8 to enforce PEP8 formatting standards across source modules.

Automated Unit Testing Gate: Executes pytest across test modules (tests/test_pipeline.py) to verify that data loading, transformer pipeline execution, and model predictions match expected outputs.

Container Build & Registry Publishing: Authenticates with Docker Hub using encrypted repository secrets, builds multi-stage Docker images for both services, tags them with the commit hash and latest, and pushes them to the public image registry.

Why?
Automated CI/CD pipelines prevent broken code, unformatted modules, or failing tests from entering production environments. Automating image builds ensures that container registries always stay synchronized with verified source code, removing manual deployment steps and reducing human error.

Where?
Pipeline configurations are managed in .github/workflows/ci.yml and executed on GitHub's cloud-hosted Linux runner environments.


Chapter 9: Cloud Infrastructure & Production Deployment
What?
Production deployment hosts the containerized applications on Render cloud infrastructure, making the API and web app publicly accessible over HTTPS.

How?
Both services are configured as Render Web Services connected to Docker Hub image repositories:
            Cloud Infrastructure Topology
  ┌─────────────────────────────────────────────────────────────────────┐
  │ RENDER MANAGED CLOUD INFRASTRUCTURE                                 │
  │                                                                     │
  │   ┌──────────────────────────────┐   ┌──────────────────────────┐   │
  │   │ Web Service 1: Frontend      │   │ Web Service 2: Backend   │   │
  │   │ Streamlit Application        │   │ FastAPI Inference Engine │   │
  │   │ Public HTTPS URL             │   │ Public HTTPS URL         │   │
  │   └──────────────┬───────────────┘   └─────────────▲────────────┘   │
  └──────────────────┼─────────────────────────────────┼────────────────┘
                     │                                 │
                     └────────────── REST API ─────────┘
                              (JSON over HTTPS)

Web Service Configuration Parameters
FastAPI Backend Service:

Image Endpoint: hwcdxx/telco-fastapi:latest

Service Runtime: Docker Container

Environment Variables: PORT=8000, MLFLOW_ALLOW_FILE_STORE=true

Health Check Path: /health

Streamlit Frontend Service:

Image Endpoint: hwcdxx/telco-streamlit:latest

Service Runtime: Docker Container

Environment Variables: PORT=8501, BACKEND_URL=[https://telco-churn-fastapi.onrender.com](https://telco-churn-fastapi.onrender.com)

Health Check Path: /

Why?
Deploying containerized services to managed cloud infrastructure provides automated TLS/SSL certificate generation, built-in DDoS protection, and public domain routing. Using environment variables allows service connections to be configured dynamically without editing container source code.

Where?
The application infrastructure runs on Render cloud web servers, exposing public HTTPS entry points for enterprise integration.


Chapter 10: The Engineering Crucible (Detailed Failure Ledger & Root Cause Analyses)
During system development and deployment, four major technical roadblocks emerged. This section details the diagnostic process, root cause analysis, and resolution for each issue.

Failure Incident 1: Deployment Failure Due to Missing Model Files
What Happened?
During early Render deployment trials, the FastAPI backend container repeatedly failed health checks and crashed upon bootup with a 503 Service Unavailable error. Inspecting application logs revealed the following error trace:
                  ⚠️ Warning: Could not initialize ChurnPredictor on startup: Could not locate active MLmodel file at mlruns/0/...
FAILURE DIAGNOSTIC FLOW 1
  ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
  │ Local Git Add Command  │ ──> │ Gitignore Rules        │ ──> │ Container Missing      │
  │ Git Add . executed     │     │ Silently Blocks mlruns │     │ Model Binary Weights   │
  └────────────────────────┘     └────────────────────────┘     └────────────────────────┘

Root Cause Analysis
The project .gitignore file contained default rules excluding mlruns/ directories to prevent temporary experiment data from entering source control. Consequently, when executing git add ., model artifact weights (model.xgb and MLmodel) were silently ignored and left uncommitted. When GitHub Actions built the Docker image from repository source code, the COPY . /app directive produced an image without trained model files. When the container booted on Render, FastAPI attempted to load missing weights and raised an unhandled startup exception.

Technical Resolution
Removed mlruns/ rules from .gitignore and .dockerignore.

Executed a force-add command in PowerShell to stage model files: git add -f mlruns/

Verified inclusion using git log -1 --stat, confirming 329 model tracking artifacts were committed.

Pushed changes to GitHub, triggering a fresh CI build containing the required model binary weights.

Failure Incident 2: Linux CI Runner Crash (PermissionError: '/C:')
What Happened?
After updating tracking paths, local test suites passed on Windows, but the GitHub Actions runner crashed during Pytest execution on Linux with the following error:
          FAILED tests/test_pipeline.py::TestTelcoPipeline::test_04_training_and_mlflow - PermissionError: [Errno 13] Permission denied: '/C:'

          FAILURE DIAGNOSTIC FLOW 2
  ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
  │ Windows Absolute Path  │ ──> │ Linux File System      │ ──> │ Permission Denied      │
  │ file:///C:/Users/...   │     │ Evaluates Path as /C:  │     │ Non-root Access Blocks │
  └────────────────────────┘     └────────────────────────┘     └────────────────────────┘

Root Cause Analysis
In tests/test_pipeline.py, the MLflow tracking URI was configured using absolute Windows file paths (file:///C:/Users/waghm/MY PROJECTS/...). While Windows handles drive letters natively, Linux environments interpret C: as a top-level directory off the root folder (/C:). Non-root processes running inside Linux GitHub Actions runners lack write permissions to create folders in root (/), causing OS-level permission denied errors when MLflow attempted to write tracking directories.

Technical Resolution
Updated tests/test_pipeline.py to use Python's cross-platform pathlib.Path library. This dynamically generates valid file URIs on both Windows and POSIX-compliant Linux filesystems:

Imported pathlib.Path inside test suites.

Transformed local directory references into normalized URIs: Path("./mlruns_test").resolve().as_uri()

Validated that URIs evaluate correctly across both local Windows workstations and Linux CI runners.

Failure Incident 3: MLflow Local FileStore Security Block in Pytest
What Happened?
After resolving path issues, local Pytest execution failed when updating MLflow versions, raising a blocking exception:
                   
                    MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not receive further updates... If the filesystem backend is required, set MLFLOW_ALLOW_FILE_STORE=true to opt out of this exception.

     FAILURE DIAGNOSTIC FLOW 3
  ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
  │ Updated MLflow Library │ ──> │ FileStore Tracking     │ ──> │ Local Test Execution   │
  │ Package Enforcement    │     │ Flagged Maintenance    │     │ Raises Exception       │
  └────────────────────────┘     └────────────────────────┘     └────────────────────────┘

  Root Cause Analysis
Recent versions of MLflow enforce security restrictions to deprecate local folder-based tracking (./mlruns) in favor of relational database backends (such as SQLite or PostgreSQL). To prevent accidental file-backed tracking in enterprise environments, MLflow raises a runtime exception unless an explicit override flag is passed in the system environment.

Technical Resolution
Added environment variable overrides at the top of test modules: os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

Updated Render runtime environment variables to include MLFLOW_ALLOW_FILE_STORE=true.

Re-ran local test suites, confirming tests executed without throwing maintenance mode exceptions.

Failure Incident 4: Streamlit False "Backend Disconnected" Errors
What Happened?
After deploying both microservices to Render, opening the Streamlit web app displayed a red error banner:
               🔴 Backend Disconnected ([https://telco-churn-fastapi.onrender.com](https://telco-churn-fastapi.onrender.com)). Ensure FastAPI container is running.
Directly visiting the backend health route (/health) in a browser succeeded, but Streamlit continued reporting the backend as offline.

FAILURE DIAGNOSTIC FLOW 4
  ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
  │ Render Free Instances  │ ──> │ Cold Start Spin-Up     │ ──> │ Strict 3s UI Timeout   │
  │ Enter Sleep Mode       │     │ Requires 30-40 Seconds │     │ Raises False Error     │
  └────────────────────────┘     └────────────────────────┘     └────────────────────────┘

Root Cause Analysis
This issue resulted from two compounding factors:

Cloud Cold Starts: On Render's free tier, idle containers automatically spin down after 15 minutes of inactivity. Waking up a sleeping container takes between 30 and 40 seconds.

Aggressive UI Timeout Settings: Streamlit's health check function (check_backend_health()) had a strict 3-second HTTP timeout (timeout=3). When accessing the app while the backend container was sleeping, the health check timed out after 3 seconds and displayed a failure banner—even though the backend was actively booting up in the background.

Technical Resolution
Increased the health check request timeout in app.py from 3 seconds to 15 seconds, giving sleeping containers time to wake up.

Sanitized the configuration URL (BACKEND_URL.rstrip("/")) to prevent double-slash formatting issues in API paths.

Implemented automated keep-alive polling to prevent the backend container from entering sleep mode.


Chapter 11: Production Observability, Monitoring & Keep-Alive Systems
What?
Production observability maintains continuous health monitoring and availability for deployed cloud services, preventing performance degradation caused by cloud platform spin-downs.

How?
Because Render's free tier spins down inactive web services after 15 minutes, an external monitoring job was configured using cron-job.org to ping the API health endpoint around the clock:
                           
                    Automated Keep-Alive Architecture
 ┌──────────────────────┐    HTTP GET /health     ┌──────────────────────┐
 │ External Monitor     │ ──────────────────────> │ FastAPI Cloud Server │
 │ (Cron-Job.org Engine)│ <────────────────────── │ (Render Infrastructure)│
 └──────────────────────┘  Returns 200 OK Status  └──────────────────────┘
                          (Executes Every 5 Mins)

Keep-Alive Configuration Parameters
Target Ping Endpoint: [https://telco-churn-fastapi.onrender.com/health](https://telco-churn-fastapi.onrender.com/health)

HTTP Protocol Method: GET

Cron Execution Schedule: */5 * * * * (Pings every 5 minutes)

Success Criteria: HTTP Response Code 200 OK

Response Payload Validated: {"status": "healthy"}

Why?
Pinging the lightweight /health route every 5 minutes keeps the backend container active in memory. This eliminates 40-second cold start delays for end users, ensuring sub-second API response times whenever a user submits a churn evaluation request through the Streamlit frontend.

Where?
Monitoring executes via external edge servers targeting the public HTTPS endpoints of the Render cloud platform.


Chapter 12: Visual Audit & Project Screenshot Repository
This chapter provides visual documentation placeholders for auditing the project's user interface, API endpoints, CI/CD pipeline, and monitoring systems.

Audit View 1: Streamlit Interactive User Dashboard
The primary user interface allows field staff to configure customer feature attributes and view instant churn risk assessments.

Audit View 2: FastAPI Interactive OpenAPI (Swagger) Documentation
FastAPI automatically generates interactive API documentation, allowing developers to test endpoints directly from the browser.

Audit View 3: GitHub Actions Automated CI/CD Pipeline
The GitHub Actions workflow dashboard tracks code quality linting, unit test execution, and Docker Hub image publishing.

Audit View 4: Cloud Infrastructure Deployment Management
The Render management dashboard displays instance liveness, environment variable configurations, and deployment logs for both microservices.
