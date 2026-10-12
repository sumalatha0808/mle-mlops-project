"""Download the raw green taxi data used for training."""

from pathlib import Path
from urllib.request import urlretrieve

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"

COLOR = "green"
YEAR = 2024
MONTH = 1

DATA_PATH = RAW_DIR / f"{COLOR}_tripdata_{YEAR}-{MONTH:02d}.parquet"
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    f"{COLOR}_tripdata_{YEAR}-{MONTH:02d}.parquet"
)


def download_data() -> Path:
    """Download the parquet file once and reuse it afterwards."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading {DATA_URL}")
        urlretrieve(DATA_URL, DATA_PATH)
    else:
        print(f"Using existing file {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    download_data()
