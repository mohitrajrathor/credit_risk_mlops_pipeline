"""Dataset validation module."""

from __future__ import annotations

import json

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT, load_data_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def validate_data(df: pd.DataFrame) -> dict[str, object]:
    """Validate schema and null percentages for the input dataframe."""
    config = load_data_config()
    expected_columns = config["schema"]["expected_columns"]
    target_column = config["schema"]["target_column"]
    null_threshold = float(config["preprocessing"]["null_threshold"])
    report_path = PROJECT_ROOT / config["artifacts"]["validation_report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    null_percentages = (df.isna().mean() * 100).round(2)
    high_null_columns = [
        column
        for column, percent in null_percentages.items()
        if percent > null_threshold * 100
    ]

    if high_null_columns:
        LOGGER.warning(
            "Columns above null threshold %.2f%%: %s",
            null_threshold * 100,
            high_null_columns,
        )

    report = {
        "is_valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "target_column": target_column,
        "null_threshold_percent": null_threshold * 100,
        "null_percentages": {
            column: float(percent) for column, percent in null_percentages.items()
        },
        "high_null_columns": high_null_columns,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    LOGGER.info("Saved validation report to %s", report_path)
    return report


if __name__ == "__main__":
    from src.data_processing.ingestion import ingest_data

    validation_report = validate_data(ingest_data())
    print(validation_report)
