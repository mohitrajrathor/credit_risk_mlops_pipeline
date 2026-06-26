"""Model training module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.utils.config_loader import (
    PROJECT_ROOT,
    load_data_config,
    load_model_config,
    load_training_config,
)
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def build_models() -> dict[str, Any]:
    """Create model objects from the config file."""
    model_config = load_model_config()["models"]

    models: dict[str, Any] = {}

    if model_config["logistic_regression"].get("enabled", True):
        models["logistic_regression"] = LogisticRegression(
            **model_config["logistic_regression"]["params"]
        )

    if model_config["xgboost"].get("enabled", True):
        models["xgboost"] = XGBClassifier(**model_config["xgboost"]["params"])

    if model_config["lightgbm"].get("enabled", True):
        models["lightgbm"] = LGBMClassifier(**model_config["lightgbm"]["params"])

    return models


def train_models(train_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Train all configured models and save them to disk."""
    data_config = load_data_config()
    training_config = load_training_config()
    target_column = data_config["schema"]["target_column"]

    models_dir = PROJECT_ROOT / training_config["artifacts"]["models_dir"]
    training_results_path = PROJECT_ROOT / training_config["artifacts"]["training_results_path"]
    models_dir.mkdir(parents=True, exist_ok=True)
    training_results_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    cv_folds = int(training_config["training"]["cv_folds"])
    primary_metric = str(training_config["training"]["primary_metric"])

    results: dict[str, dict[str, Any]] = {}

    for model_name, model in build_models().items():
        LOGGER.info("Training model: %s", model_name)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=primary_metric,
        )

        model.fit(X_train, y_train)

        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        results[model_name] = {
            "cv_scores": [float(score) for score in cv_scores],
            "mean_cv_score": float(cv_scores.mean()),
            "model_path": str(model_path),
        }

        LOGGER.info("%s mean CV score: %.4f", model_name, cv_scores.mean())
        LOGGER.info("Saved model to %s", model_path)

    with training_results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    LOGGER.info("Saved training results to %s", training_results_path)
    return results


if __name__ == "__main__":
    from src.data_processing.ingestion import ingest_data
    from src.data_processing.transformation import transform_data
    from src.features.feature_engineering import engineer_features

    raw_df = ingest_data()
    train_df, test_df = transform_data(raw_df)
    featured_train_df, _ = engineer_features(train_df, test_df)
    print(train_models(featured_train_df))
