"""Data ingestion module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT, load_data_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _make_json_safe(value: Any) -> Any:
    """Convert values to JSON-safe Python types."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def ingest_data() -> pd.DataFrame:
    """Load the raw dataset and save a short ingestion report."""
    config = load_data_config()

    raw_data_path = PROJECT_ROOT / config["data"]["raw_data_path"]
    report_path = PROJECT_ROOT / config["artifacts"]["ingestion_report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    df = pd.read_csv(raw_data_path)

    LOGGER.info("Loaded dataset from %s", raw_data_path)
    LOGGER.info("Dataset shape: %s", df.shape)
    LOGGER.info("Column dtypes: %s", df.dtypes.astype(str).to_dict())
    LOGGER.info("Null counts: %s", df.isna().sum().to_dict())

    report = {
        "file_path": str(raw_data_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": {column: int(count) for column, count in df.isna().sum().items()},
        "sample_record": {
            key: _make_json_safe(value) for key, value in df.iloc[0].to_dict().items()
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    LOGGER.info("Saved ingestion report to %s", report_path)
    return df


if __name__ == "__main__":
    dataframe = ingest_data()
    print(dataframe.head())
