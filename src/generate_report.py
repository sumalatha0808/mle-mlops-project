"""Compare logged API requests with the training data using an Evidently drift report."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from prepare_data import DATA_PATH
from preprocess import FEATURES, preprocess, to_records

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"
REPORTS_DIR = ROOT_DIR / "monitoring" / "reports"
COLUMNS = FEATURES + ["prediction"]


def load_reference(n: int = 2000) -> pd.DataFrame:
    """A sample of the training data, with the model's predictions added."""
    df = preprocess(pd.read_parquet(DATA_PATH))[FEATURES].sample(n=n, random_state=42)
    df["prediction"] = joblib.load(MODEL_PATH).predict(to_records(df))
    return df


def load_current() -> pd.DataFrame:
    """All requests the API has received so far."""
    if not REQUEST_LOG_PATH.exists():
        raise FileNotFoundError("No request log yet. Run src/send_sample_requests.py first.")
    df = pd.read_csv(REQUEST_LOG_PATH)
    if len(df) < 5:
        raise ValueError("Send at least 5 requests before generating a report.")
    return df


def generate_report(name: str) -> Path:
    reference, current = load_reference(), load_current()
    # Location IDs are categories, not numbers: use text in both datasets.
    for df in (reference, current):
        df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)

    snapshot = Report(metrics=[DataDriftPreset()]).run(
        reference_data=reference[COLUMNS], current_data=current[COLUMNS]
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.html"
    snapshot.save_html(str(path))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="data_drift_report", help="report file name")
    print(f"Wrote Evidently report to {generate_report(parser.parse_args().name)}")
