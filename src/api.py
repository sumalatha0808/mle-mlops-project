"""Prediction API for the NYC green taxi ride duration model, with monitoring."""

import csv
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"

MODEL: Any = None
REQUEST_LOG_LOCK = threading.Lock()

# --- Prometheus metrics -------------------------------------------------------
PREDICTION_REQUESTS = Counter(
    "api_prediction_requests_total", "Total number of prediction requests."
)
PREDICTION_ERRORS = Counter(
    "api_prediction_errors_total", "Prediction requests that failed."
)
PREDICTION_LATENCY = Histogram(
    "api_prediction_latency_seconds", "Time spent computing one prediction."
)
PREDICTED_DURATION = Histogram(
    "api_predicted_duration_minutes",
    "Distribution of predicted ride durations (model behaviour).",
    buckets=(2, 5, 10, 15, 20, 30, 45, 60, 90),
)


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


app = FastAPI(title="Green Taxi Ride Duration API", version="1.1.0", lifespan=lifespan)


class TripRequest(BaseModel):
    """One completed taxi trip. trip_distance is the taximeter distance, known after the trip."""

    PULocationID: int = Field(..., ge=1, le=265, examples=[74])
    DOLocationID: int = Field(..., ge=1, le=265, examples=[75])
    trip_distance: float = Field(..., gt=0, le=100, allow_inf_nan=False, examples=[1.5])


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float


def _log_request(trip: TripRequest, prediction: float) -> None:
    """Append inputs and prediction to a CSV file used by the Evidently report."""
    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {**trip.model_dump(), "prediction": prediction}
    with REQUEST_LOG_LOCK:
        new_file = not REQUEST_LOG_PATH.exists()
        with REQUEST_LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check that also reports whether the model is loaded."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest) -> PredictionResponse:
    """Predict the ride duration in minutes for one trip."""
    PREDICTION_REQUESTS.inc()
    if MODEL is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Run `uv run dvc repro` first.")

    start = perf_counter()
    record = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }
    try:
        prediction = round(float(MODEL.predict([record])[0]), 2)
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    PREDICTION_LATENCY.observe(perf_counter() - start)
    PREDICTED_DURATION.observe(prediction)
    _log_request(trip, prediction)
    return PredictionResponse(predicted_duration_minutes=prediction)
