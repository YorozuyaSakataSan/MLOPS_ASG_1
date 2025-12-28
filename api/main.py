from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")


# ---------------- Model load ----------------
MODEL_PATH = Path("models/best_model.joblib")
model = None


def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Generate it using: python src/heart/train.py"
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH.resolve())


# ---------------- FastAPI ----------------
app = FastAPI(title="Heart Disease Risk API", version="1.0.0")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=50, le=250)
    chol: int = Field(..., ge=50, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("Request: %s", req.model_dump())

    X = pd.DataFrame([req.model_dump()])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    logger.info("Response: pred=%s proba=%s", pred, proba)
    return PredictResponse(prediction=pred, probability=proba)
