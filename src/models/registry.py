"""Model registry module for MLflow model registration and stage transition."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config_loader import PROJECT_ROOT
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def register_best_model() -> None:
    """Read best_model_info.json, register the model in MLflow, and move to Staging."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    best_model_info_path = PROJECT_ROOT / "artifacts" / "best_model_info.json"
    if not best_model_info_path.exists():
        LOGGER.error("Best model info file not found at %s", best_model_info_path)
        return

    with best_model_info_path.open("r", encoding="utf-8") as file:
        best_info = json.load(file)

    model_name = best_info.get("model_name")
    run_id = best_info.get("run_id")

    if not model_name or not run_id:
        LOGGER.error("Missing model_name or run_id in best_model_info.json")
        return

    model_uri = f"runs:/{run_id}/model"
    LOGGER.info("Registering model '%s' from URI: %s", model_name, model_uri)

    # Register the model in MLflow Model Registry
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered_model.version)
    LOGGER.info("Registered model '%s' version %s", model_name, version)

    # Use MlflowClient to transition stage to Staging
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
    )
    LOGGER.info("Transitioned model '%s' version %s to Staging stage", model_name, version)


if __name__ == "__main__":
    register_best_model()
