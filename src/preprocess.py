"""Turn raw taxi trips into model features and target."""

import pandas as pd
from sklearn.model_selection import train_test_split

# trip_distance is the taximeter distance, known only AFTER the trip ends.
# The model therefore replays completed trips; it is not a pickup-time estimate.
FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duration in minutes, remove outliers, and cast location IDs to text."""
    df = df.copy()
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)].copy()
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)
    return df[FEATURES + [TARGET]]


def to_records(X: pd.DataFrame) -> list[dict]:
    """DictVectorizer expects a list of dicts, one per trip."""
    return X.to_dict(orient="records")


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split the processed data into train and test sets."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
