"""Smoke test for the taxi API, without starting a server or touching the real request log."""

import asyncio
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from . import api


async def check_api() -> None:
    payload = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 1.5}

    with TemporaryDirectory() as tmp:
        api.REQUEST_LOG_PATH = Path(tmp) / "requests.csv"
        # ASGITransport does not run the lifespan, so load the model here.
        async with api.lifespan(api.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://test"
            ) as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json() == {"status": "ok", "model_loaded": True}

                prediction = await client.post("/predict", json=payload)
                assert prediction.status_code == 200
                assert prediction.json()["predicted_duration_minutes"] > 0

                invalid = await client.post("/predict", json={**payload, "trip_distance": -1})
                assert invalid.status_code == 422

                metrics = await client.get("/metrics")
                assert metrics.status_code == 200
                assert "api_prediction_requests_total" in metrics.text

        with api.REQUEST_LOG_PATH.open(newline="") as f:
            assert len(list(csv.DictReader(f))) == 1

    print("API checks passed: health, prediction, HTTP 422, /metrics and request logging.")
    print("Example prediction:", prediction.json())


if __name__ == "__main__":
    asyncio.run(check_api())
