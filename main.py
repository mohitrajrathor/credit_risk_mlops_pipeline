"""Pipeline entry point for the credit risk project."""

from __future__ import annotations

import json

from src.data_processing.ingestion import ingest_data
from src.data_processing.transformation import transform_data
from src.data_processing.validation import validate_data
from src.explainability.shap_explainer import generate_shap_explanations
from src.features.feature_engineering import engineer_features
from src.models.evaluate import evaluate_models
from src.models.train import train_models
from src.utils.config_loader import PROJECT_ROOT, load_training_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """Run the full local machine learning pipeline."""
    LOGGER.info("Starting data ingestion")
    raw_df = ingest_data()
    LOGGER.info("Finished data ingestion")

    LOGGER.info("Starting data validation")
    validate_data(raw_df)
    LOGGER.info("Finished data validation")

    LOGGER.info("Starting data transformation")
    train_df, test_df = transform_data(raw_df)
    LOGGER.info("Finished data transformation")

    LOGGER.info("Starting feature engineering")
    featured_train_df, featured_test_df = engineer_features(train_df, test_df)
    LOGGER.info("Finished feature engineering")

    LOGGER.info("Starting model training")
    train_models(featured_train_df)
    LOGGER.info("Finished model training")

    LOGGER.info("Starting model evaluation")
    best_model_name = evaluate_models(featured_test_df)
    LOGGER.info("Finished model evaluation")

    LOGGER.info("Starting SHAP explainability")
    generate_shap_explanations(best_model_name, featured_train_df, featured_test_df)
    LOGGER.info("Finished SHAP explainability")

    metrics_path = PROJECT_ROOT / load_training_config()["artifacts"]["metrics_path"]
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    print("\nFinal Metrics Summary:")
    print(json.dumps(metrics, indent=2))
    print(f"\nBest model: {best_model_name}")


if __name__ == "__main__":
    main()
