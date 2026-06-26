"""SHAP explainability module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.config_loader import PROJECT_ROOT, load_data_config, load_training_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


class ShapExplainer:
    """Small wrapper around a SHAP explainer and trained model."""

    def __init__(self, model: Any, model_name: str, X_background: pd.DataFrame) -> None:
        """Create a SHAP explainer for the given model."""
        self.model = model
        self.model_name = model_name
        self.X_background = X_background
        self.explainer = self._build_explainer()

    def _build_explainer(self) -> Any:
        """Choose a SHAP explainer based on the model type."""
        if self.model_name in {"xgboost", "lightgbm"}:
            return shap.TreeExplainer(self.model)

        return shap.LinearExplainer(self.model, self.X_background)

    def explain_single(self, input_df: pd.DataFrame) -> dict[str, Any]:
        """Return SHAP contributions for one row."""
        shap_values = self.explainer(input_df)
        row_values = shap_values.values[0]
        base_value = shap_values.base_values[0]

        return {
            "base_value": float(base_value),
            "feature_contributions": {
                column: float(value)
                for column, value in zip(input_df.columns, row_values, strict=False)
            },
        }


def generate_shap_explanations(
    best_model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> ShapExplainer:
    """Generate and save SHAP plots and explainer object."""
    training_config = load_training_config()
    models_dir = PROJECT_ROOT / training_config["artifacts"]["models_dir"]
    plots_dir = PROJECT_ROOT / training_config["artifacts"]["plots_dir"]
    shap_explainer_path = PROJECT_ROOT / training_config["artifacts"]["shap_explainer_path"]
    target_column = load_data_config()["schema"]["target_column"]

    plots_dir.mkdir(parents=True, exist_ok=True)
    shap_explainer_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = train_df.drop(columns=[target_column])
    X_test = test_df.drop(columns=[target_column])

    model_path = models_dir / f"{best_model_name}.pkl"
    model = joblib.load(model_path)
    shap_wrapper = ShapExplainer(model, best_model_name, X_train)

    sample_test = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = shap_wrapper.explainer(sample_test)

    plt.figure()
    shap.summary_plot(shap_values, sample_test, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_importance.png", bbox_inches="tight")
    plt.close()

    joblib.dump(shap_wrapper, shap_explainer_path)
    LOGGER.info("Saved SHAP explainer to %s", shap_explainer_path)

    return shap_wrapper


if __name__ == "__main__":
    from src.data_processing.ingestion import ingest_data
    from src.data_processing.transformation import transform_data
    from src.features.feature_engineering import engineer_features
    from src.models.evaluate import evaluate_models
    from src.models.train import train_models

    raw_df = ingest_data()
    train_df, test_df = transform_data(raw_df)
    featured_train_df, featured_test_df = engineer_features(train_df, test_df)
    train_models(featured_train_df)
    best_model = evaluate_models(featured_test_df)
    explainer = generate_shap_explanations(best_model, featured_train_df, featured_test_df)
    print(explainer.explain_single(featured_test_df.drop(columns=["loan_status"]).head(1)))
