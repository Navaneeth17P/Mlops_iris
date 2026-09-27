from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import predict, load_model, load_scaler

app = FastAPI(
    title="Iris Classifier API",
    description="MLOps demo — predicts Iris species from flower measurements",
    version="1.0.0",
)

model = None
scaler = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.on_event("startup")
async def startup_event():
    global model, scaler
    try:
        model = load_model(os.path.join(BASE_DIR, "models", "model.joblib"))
        scaler = load_scaler(os.path.join(BASE_DIR, "models", "scaler.joblib"))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Model not found: {e}. Run `make train` first.")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., example=0.2, description="Petal width in cm")


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Dict[str, float]


@app.get("/")
def root():
    return {"message": "Iris Classifier API is running 🌸"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_iris(features: IrisFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")

    result = predict(
        [features.sepal_length, features.sepal_width,
         features.petal_length, features.petal_width],
        model=model,
        scaler=scaler,
    )
    return result