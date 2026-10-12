"""Send a batch of real taxi trips to the API to fill the monitoring data.

Normal batch:   uv run python src/send_sample_requests.py
Shifted batch:  uv run python src/send_sample_requests.py --shifted
(The shifted batch multiplies trip_distance by 3 to simulate data drift.)
"""

import argparse

import pandas as pd
import requests

from prepare_data import DATA_PATH
from preprocess import FEATURES, preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("-n", type=int, default=200, help="number of requests")
    parser.add_argument("--shifted", action="store_true", help="simulate drift")
    args = parser.parse_args()

    trips = preprocess(pd.read_parquet(DATA_PATH))[FEATURES].sample(n=args.n, random_state=7)
    if args.shifted:
        trips["trip_distance"] = (trips["trip_distance"] * 3).clip(upper=100)

    ok = 0
    for trip in trips.to_dict(orient="records"):
        payload = {
            "PULocationID": int(trip["PULocationID"]),
            "DOLocationID": int(trip["DOLocationID"]),
            "trip_distance": float(trip["trip_distance"]),
        }
        response = requests.post(f"{args.url}/predict", json=payload, timeout=10)
        ok += response.status_code == 200

    kind = "shifted" if args.shifted else "normal"
    print(f"Sent {len(trips)} {kind} requests, {ok} succeeded.")


if __name__ == "__main__":
    main()
