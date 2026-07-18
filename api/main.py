"""
main.py
-------
FastAPI application exposing the churn prediction model as a REST API.

Design principle: the API layer contains NO business logic itself — it
only validates input (via Pydantic), delegates to src/models/predict.py,
and formats the response. This keeps the prediction logic testable and
reusable outside the API (e.g. in batch scoring scripts) without
duplicating it here.
"""

from fastapi import FastAPI, HTTPException
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.predict import score_customer
from api.schemas import CustomerInput, ChurnPrediction

app = FastAPI(
    title="SaaS Subscription Churn Prediction API",
    description="Predicts churn probability for a streaming subscription customer.",
    version="1.0.0"
)

# Load artifacts once at startup, not per-request — loading a model from
# disk on every call would add significant latency and is unnecessary
# since these files don't change while the API is running.
MODEL = joblib.load("models/model_final.pkl")
SCALER = joblib.load("models/feature_scaler.pkl")
FEATURE_STATS = joblib.load("models/feature_engineering_stats.pkl")


@app.get("/health")
def health_check():
    """Basic liveness check — used by Docker/orchestrators to confirm the API is up."""
    return {"status": "ok", "model": "xgboost_tuned"}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(customer: CustomerInput):
    """
    Score a single customer and return their churn probability and
    risk segment. Raises a 500 with a clear message if the underlying
    pipeline fails (e.g. an unexpected category), rather than leaking a
    raw stack trace to the API consumer.
    """
    try:
        result = score_customer(customer.dict(), MODEL, SCALER, FEATURE_STATS)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    