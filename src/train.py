"""Train the taxi duration model, save it locally and log it to MLflow."""

import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

from prepare_data import COLOR, DATA_PATH, MONTH, YEAR
from preprocess import FEATURES, TARGET, preprocess, split_data, to_records

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "train_metrics.json"

EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"


def train(df: pd.DataFrame):
    """Fit the pipeline and return it with its test RMSE."""
    X_train, X_test, y_train, y_test = split_data(preprocess(df))
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(to_records(X_train), y_train)
    rmse = root_mean_squared_error(y_test, pipeline.predict(to_records(X_test)))
    return pipeline, float(rmse), len(X_train), len(X_test)


def save_local(pipeline, rmse: float, n_train: int, n_test: int) -> None:
    """Save the model and metrics to predictable paths (used later by the API and DVC)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps({"rmse": rmse, "n_train": n_train, "n_test": n_test}, indent=2)
    )


def log_to_mlflow(pipeline, rmse: float) -> None:
    """Log params, metric and model, then register it with the 'production' alias."""
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "model": "linear regression",
            "developer": os.getenv("DEVELOPER", "Sumalatha"),
            "dataset": f"{COLOR}-taxi",
        })
        mlflow.log_params({"year": YEAR, "month": MONTH, "features": FEATURES, "target": TARGET})
        mlflow.log_metric("rmse", rmse)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )

        client = MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME, description="NYC green taxi ride duration")
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        version = client.create_model_version(
            name=MODEL_NAME, source=model_info.model_uri, run_id=run.info.run_id
        )
        client.set_registered_model_alias(MODEL_NAME, "production", version.version)
        print(f"Registered {MODEL_NAME} version {version.version} as 'production'")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    pipeline, rmse, n_train, n_test = train(df)
    print(f"RMSE: {rmse:.2f} minutes (train={n_train}, test={n_test})")
    save_local(pipeline, rmse, n_train, n_test)
    log_to_mlflow(pipeline, rmse)


if __name__ == "__main__":
    main()
