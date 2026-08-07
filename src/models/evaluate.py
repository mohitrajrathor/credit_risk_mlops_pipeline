"""Model evaluation module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config_loader import PROJECT_ROOT, load_data_config, load_training_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _get_model_probability(model: Any, X_test: pd.DataFrame) -> list[float]:
    """Return positive-class probabilities when supported."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        return probabilities[:, 1].tolist()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return pd.Series(scores).tolist()

    raise ValueError("Model does not support probability or decision scores.")


def _save_confusion_matrix_plot(
    matrix: Any,
    model_name: str,
    plots_dir: Path,
) -> Path:
    """Save a confusion matrix image and return its path."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plots_dir / f"{model_name}_confusion_matrix.png"

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def evaluate_models(test_df: pd.DataFrame) -> str:
    """Evaluate all saved models, log metrics/artifacts to MLflow, and return best model name."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-experiment")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data_config = load_data_config()
    training_config = load_training_config()

    target_column = data_config["schema"]["target_column"]
    models_dir = PROJECT_ROOT / training_config["artifacts"]["models_dir"]
    plots_dir = PROJECT_ROOT / training_config["artifacts"]["plots_dir"]
    metrics_path = PROJECT_ROOT / training_config["artifacts"]["metrics_path"]
    run_ids_path = PROJECT_ROOT / "artifacts" / "run_ids.json"
    best_model_info_path = PROJECT_ROOT / "artifacts" / "best_model_info.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    run_ids: dict[str, str] = {}
    if run_ids_path.exists():
        with run_ids_path.open("r", encoding="utf-8") as file:
            run_ids = json.load(file)

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    all_metrics: dict[str, dict[str, float]] = {}

    for model_path in sorted(models_dir.glob("*.pkl")):
        model_name = model_path.stem
        model = joblib.load(model_path)

        predictions = model.predict(X_test)
        scores = _get_model_probability(model, X_test)
        matrix = confusion_matrix(y_test, predictions)
        figure_path = _save_confusion_matrix_plot(matrix, model_name, plots_dir)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
            "precision_weighted": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, scores)),
        }

        all_metrics[model_name] = metrics
        LOGGER.info("Metrics for %s: %s", model_name, metrics)

        # Log evaluation metrics and confusion matrix artifact to MLflow run
        run_id = run_ids.get(model_name)
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(figure_path))
        else:
            with mlflow.start_run(run_name=f"evaluate_{model_name}"):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(figure_path))

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    metrics_df = pd.DataFrame(all_metrics).T.sort_values("f1_weighted", ascending=False)
    print("\nModel Comparison:")
    print(metrics_df.round(4).to_string())

    best_model_name = str(metrics_df.index[0])
    LOGGER.info("Best model by F1 weighted: %s", best_model_name)

    # Save best model info JSON (with version placeholder)
    best_model_info = {
        "model_name": best_model_name,
        "version": "N/A",
        "run_id": run_ids.get(best_model_name, ""),
        "metrics": all_metrics[best_model_name],
    }
    with best_model_info_path.open("w", encoding="utf-8") as file:
        json.dump(best_model_info, file, indent=2)

    LOGGER.info("Saved best model info to %s", best_model_info_path)
    return best_model_name


if __name__ == "__main__":
    from src.data_processing.ingestion import ingest_data
    from src.data_processing.transformation import transform_data
    from src.features.feature_engineering import engineer_features
    from src.models.train import train_models

    raw_df = ingest_data()
    train_df, test_df = transform_data(raw_df)
    featured_train_df, featured_test_df = engineer_features(train_df, test_df)
    train_models(featured_train_df)
    print(evaluate_models(featured_test_df))
