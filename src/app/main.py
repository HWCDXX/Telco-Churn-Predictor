# src/app/main.py
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.serving.predict import ChurnPredictor

# Global predictor instance
predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model into memory on API startup."""
    global predictor
    try:
        predictor = ChurnPredictor(model_dir="./mlruns")
        print("🚀 Model successfully loaded into FastAPI app memory.")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize ChurnPredictor on startup: {e}")
        predictor = None
    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production REST API providing real-time XGBoost churn predictions.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------
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
    PaymentMethod: str = Field(
        ...,
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'",
    )
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0.0, description="Total charge amount")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ChurnResponse(BaseModel):
    prediction: str = Field(..., description="'Likely to churn' or 'Not likely to churn'")
    churn_class: int = Field(..., description="1 for churn, 0 for non-churn")
    raw_output: float = Field(..., description="Raw probability score from XGBoost model")


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root_check():
    """Root endpoint welcoming users and confirming API availability."""
    return {
        "status": "online",
        "service": "Telco Churn Prediction API",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming model availability."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not loaded.",
        )
    return {"status": "healthy", "features_loaded": len(predictor.feature_cols)}


@app.post("/predict", response_model=ChurnResponse, tags=["Inference"])
def predict_churn(
    payload: CustomerPayload,
    threshold: float = Query(
        0.3, ge=0.0, le=1.0, description="Decision threshold for classification"
    ),
):
    """Runs real-time inference on a customer payload."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable.",
        )

    try:
        input_dict: Dict[str, Any] = payload.model_dump()
        result = predictor.predict(input_dict, threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )