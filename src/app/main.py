# src/app/main.py
import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import mlflow
from src.serving.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telco-churn-api")

# Configure MLflow Tracking Server URL from Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/TelcoChurnXGBoost/Production")
        logger.info(f"Connecting to MLflow server at {MLFLOW_TRACKING_URI}...")
        predictor = ChurnPredictor(model_uri=model_uri)
    except Exception as e:
        logger.error(f"Failed to load model from MLflow tracking server: {e}")
        predictor = None
    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production REST API loading XGBoost model dynamically from MLflow.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerPayload(BaseModel):
    gender: str = Field(..., description="Gender ('Male', 'Female')")
    SeniorCitizen: int = Field(..., description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., description="'Yes' or 'No'")
    Dependents: str = Field(..., description="'Yes' or 'No'")
    tenure: int = Field(..., ge=0, description="Months customer has stayed with company")
    PhoneService: str = Field(..., description="'Yes' or 'No'")
    MultipleLines: str = Field(..., description="'Yes', 'No', 'No phone service'")
    InternetService: str = Field(..., description="'DSL', 'Fiber optic', 'No'")
    OnlineSecurity: str = Field(..., description="'Yes', 'No', 'No internet service'")
    OnlineBackup: str = Field(..., description="'Yes', 'No', 'No internet service'")
    DeviceProtection: str = Field(..., description="'Yes', 'No', 'No internet service'")
    TechSupport: str = Field(..., description="'Yes', 'No', 'No internet service'")
    StreamingTV: str = Field(..., description="'Yes', 'No', 'No internet service'")
    StreamingMovies: str = Field(..., description="'Yes', 'No', 'No internet service'")
    Contract: str = Field(..., description="'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(..., description="'Yes' or 'No'")
    PaymentMethod: str = Field(..., description="Payment method string")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0.0, description="Total charge amount")


class ChurnResponse(BaseModel):
    prediction: str
    churn_class: int
    raw_output: float


@app.get("/health", tags=["Health"])
def health_check():
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service not connected to MLflow.",
        )
    return {"status": "healthy", "mlflow_tracking_uri": MLFLOW_TRACKING_URI}


@app.post("/predict", response_model=ChurnResponse, tags=["Inference"])
def predict_churn(
    payload: CustomerPayload,
    threshold: float = Query(0.3, ge=0.0, le=1.0),
):
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    try:
        return predictor.predict(payload.model_dump(), threshold=threshold)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
