"""Prediction API for the NYC green taxi ride duration model."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

MODEL: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once, when the API starts."""
    global MODEL
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}. Run `uv run dvc repro` first.")
    yield


app = FastAPI(title="Green Taxi Ride Duration API", version="1.0.0", lifespan=lifespan)


class TripRequest(BaseModel):
    """One completed taxi trip. trip_distance is the taximeter distance, known after the trip."""

    PULocationID: int = Field(..., ge=1, le=265, examples=[74])
    DOLocationID: int = Field(..., ge=1, le=265, examples=[75])
    trip_distance: float = Field(..., gt=0, le=100, allow_inf_nan=False, examples=[1.5])


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check that also reports whether the model is loaded."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest) -> PredictionResponse:
    """Predict the ride duration in minutes for one trip."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `uv run dvc repro` first.")

    # Same format as in training: location IDs as text, one dict per trip.
    record = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }
    prediction = float(MODEL.predict([record])[0])
    return PredictionResponse(predicted_duration_minutes=round(prediction, 2))
