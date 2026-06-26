"""Feature engineering module."""

from __future__ import annotations

import json

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT, load_data_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple derived features."""
    engineered_df = df.copy()

    engineered_df["loan_to_income"] = engineered_df["loan_amnt"] / engineered_df["person_income"].replace(0, 1)
    engineered_df["emp_length_to_age"] = engineered_df["person_emp_length"] / engineered_df["person_age"].replace(0, 1)

    return engineered_df


def _drop_high_correlation_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop highly correlated features based on the training set only."""
    feature_df = train_df.drop(columns=[target_column])
    correlation_matrix = feature_df.corr().abs()
    upper_triangle = correlation_matrix.where(
        pd.DataFrame(
            [[column_index > row_index for column_index in range(correlation_matrix.shape[1])]
             for row_index in range(correlation_matrix.shape[0])],
            index=correlation_matrix.index,
            columns=correlation_matrix.columns,
        )
    )

    columns_to_drop = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)
    ]

    return (
        train_df.drop(columns=columns_to_drop, errors="ignore"),
        test_df.drop(columns=columns_to_drop, errors="ignore"),
        columns_to_drop,
    )


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create derived features and drop highly correlated ones."""
    config = load_data_config()
    target_column = config["schema"]["target_column"]
    correlation_threshold = float(config["preprocessing"]["correlation_threshold"])
    feature_list_path = PROJECT_ROOT / config["artifacts"]["feature_list_path"]
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)

    engineered_train_df = _add_derived_features(train_df)
    engineered_test_df = _add_derived_features(test_df)

    engineered_train_df, engineered_test_df, dropped_columns = _drop_high_correlation_features(
        engineered_train_df,
        engineered_test_df,
        target_column,
        correlation_threshold,
    )

    feature_list = [
        column for column in engineered_train_df.columns if column != target_column
    ]

    with feature_list_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_list": feature_list,
                "dropped_due_to_correlation": dropped_columns,
            },
            file,
            indent=2,
        )

    LOGGER.info("Saved feature list to %s", feature_list_path)
    return engineered_train_df, engineered_test_df


if __name__ == "__main__":
    from src.data_processing.ingestion import ingest_data
    from src.data_processing.transformation import transform_data

    train_dataframe, test_dataframe = transform_data(ingest_data())
    final_train_df, final_test_df = engineer_features(train_dataframe, test_dataframe)
    print(final_train_df.shape, final_test_df.shape)
