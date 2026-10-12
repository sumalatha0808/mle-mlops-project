"""Smoke test for the taxi API, without starting a server."""

import asyncio

import httpx

from . import api


async def check_api() -> None:
    payload = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 1.5}

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

    print("API checks passed: health, prediction, and input validation (HTTP 422).")
    print("Example prediction:", prediction.json())


if __name__ == "__main__":
    asyncio.run(check_api())
