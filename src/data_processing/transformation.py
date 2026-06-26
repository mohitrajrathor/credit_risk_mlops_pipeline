"""Data transformation module."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config_loader import PROJECT_ROOT, load_data_config, load_training_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, encode, split, and scale the dataset."""
    data_config = load_data_config()
    training_config = load_training_config()

    target_column = data_config["schema"]["target_column"]
    numerical_features = list(data_config["schema"]["numerical_features"])
    categorical_features = list(data_config["schema"]["categorical_features"])
    drop_columns = list(data_config["schema"]["drop_columns"])

    encoders_path = PROJECT_ROOT / data_config["artifacts"]["encoders_path"]
    scaler_path = PROJECT_ROOT / data_config["artifacts"]["scaler_path"]
    train_path = PROJECT_ROOT / data_config["data"]["train_path"]
    test_path = PROJECT_ROOT / data_config["data"]["test_path"]

    for path in [encoders_path, scaler_path, train_path, test_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    transformed_df = df.copy()

    if drop_columns:
        transformed_df = transformed_df.drop(columns=drop_columns, errors="ignore")

    for column in numerical_features:
        transformed_df[column] = transformed_df[column].fillna(transformed_df[column].median())

    for column in categorical_features:
        mode_value = transformed_df[column].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        transformed_df[column] = transformed_df[column].fillna(fill_value)

    transformed_df[numerical_features] = transformed_df[numerical_features].astype(float)
    transformed_df[target_column] = transformed_df[target_column].astype(int)

    label_encoders: dict[str, LabelEncoder] = {}
    for column in categorical_features:
        encoder = LabelEncoder()
        transformed_df[column] = encoder.fit_transform(transformed_df[column].astype(str))
        label_encoders[column] = encoder

    X = transformed_df.drop(columns=[target_column])
    y = transformed_df[target_column]

    train_settings = training_config["training"]
    stratify_values = y if bool(train_settings.get("stratify", True)) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(train_settings["test_size"]),
        random_state=int(train_settings["random_state"]),
        stratify=stratify_values,
    )

    scaler = StandardScaler()
    X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

    train_df = X_train.copy()
    train_df[target_column] = y_train.values

    test_df = X_test.copy()
    test_df[target_column] = y_test.values

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(label_encoders, encoders_path)
    joblib.dump(scaler, scaler_path)

    LOGGER.info("Saved transformed train data to %s", train_path)
    LOGGER.info("Saved transformed test data to %s", test_path)
    LOGGER.info("Saved encoders to %s", encoders_path)
    LOGGER.info("Saved scaler to %s", scaler_path)

    return train_df, test_df


if __name__ == "__main__":
    from src.data_processing.ingestion import ingest_data

    train_dataframe, test_dataframe = transform_data(ingest_data())
    print(train_dataframe.shape, test_dataframe.shape)
