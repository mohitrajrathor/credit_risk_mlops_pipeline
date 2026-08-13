"""Tests for the credit risk machine learning pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Import src modules for testing
from src.utils.config_loader import load_data_config, load_model_config, load_training_config
from src.utils.logger import get_logger
from src.data_processing.ingestion import ingest_data
from src.data_processing.validation import validate_data
from src.data_processing.transformation import transform_data
from src.features.feature_engineering import engineer_features, _add_derived_features
from src.models.train import train_models, build_models
from src.models.evaluate import evaluate_models
from src.models.registry import register_best_model
from src.explainability.shap_explainer import ShapExplainer, generate_shap_explanations
import main


@pytest.fixture(autouse=True)
def mock_paths(tmp_path, monkeypatch):
    """Redirect PROJECT_ROOT to a temporary path across all loaded modules."""
    import src.utils.config_loader
    import src.data_processing.ingestion
    import src.data_processing.transformation
    import src.data_processing.validation
    import src.features.feature_engineering
    import src.models.train
    import src.models.evaluate
    import src.models.registry
    import src.explainability.shap_explainer
    import main

    # Find the real configs directory in the project root (one level up from tests/)
    real_root = Path(__file__).resolve().parents[1]
    
    # 1. Patch config_loader module variables
    monkeypatch.setattr(src.utils.config_loader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.utils.config_loader, "CONFIG_DIR", real_root / "configs")
    
    # 2. Patch PROJECT_ROOT in all modules where it was imported at module level
    monkeypatch.setattr(src.data_processing.ingestion, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.data_processing.transformation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.data_processing.validation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.features.feature_engineering, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.models.train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.models.evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.models.registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src.explainability.shap_explainer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)


@pytest.fixture(autouse=True)
def mock_mlflow_apis(monkeypatch):
    """Patch mlflow globally at the module level across src and main.py."""
    # Create a mock run object
    run_mock = MagicMock()
    run_mock.info.run_id = "mock_run_123"
    
    # Create start_run context manager mock
    mock_context = MagicMock()
    mock_context.__enter__.return_value = run_mock
    
    # Mock mlflow core methods
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value = mock_context
    mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="mock_exp_123")
    mock_mlflow.register_model.return_value = MagicMock(version="1")
    
    # Mock tracking client
    mock_client = MagicMock()
    mock_client.search_runs.return_value = []
    mock_client.search_model_versions.return_value = [MagicMock(version="1")]
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    
    # Apply monkeypatching to each imported module in the pipeline using string paths
    monkeypatch.setattr("src.models.train.mlflow", mock_mlflow)
    monkeypatch.setattr("src.models.evaluate.mlflow", mock_mlflow)
    monkeypatch.setattr("src.models.registry.mlflow", mock_mlflow)
    monkeypatch.setattr("src.models.registry.MlflowClient", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr("main.mlflow", mock_mlflow)


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    """Provide a realistic 20-row dataframe to satisfy CV and stratified split requirements."""
    np.random.seed(42)
    n_samples = 20
    
    data = {
        "person_age": np.random.randint(20, 60, size=n_samples),
        "person_income": np.random.randint(20000, 120000, size=n_samples),
        "person_home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], size=n_samples),
        "person_emp_length": np.random.uniform(0, 15, size=n_samples).round(1),
        "loan_intent": np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"], size=n_samples),
        "loan_grade": np.random.choice(["A", "B", "C", "D"], size=n_samples),
        "loan_amnt": np.random.randint(1000, 30000, size=n_samples),
        "loan_int_rate": np.random.uniform(5.0, 18.0, size=n_samples).round(2),
        "loan_status": [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], # 12 zeros, 8 ones
        "loan_percent_income": np.random.uniform(0.05, 0.40, size=n_samples).round(2),
        "cb_person_default_on_file": np.random.choice(["Y", "N"], size=n_samples),
        "cb_person_cred_hist_length": np.random.randint(1, 15, size=n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def create_raw_data_csv(tmp_path, dummy_data) -> Path:
    """Write dummy dataset to the temporary directory's raw path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_data_path = raw_dir / "credit_risk_dataset.csv"
    dummy_data.to_csv(raw_data_path, index=False)
    return raw_data_path


# --- UTILS TESTS ---

def test_config_loader():
    """Test that configurations load and have the expected structure."""
    data_config = load_data_config()
    model_config = load_model_config()
    training_config = load_training_config()

    assert isinstance(data_config, dict)
    assert "schema" in data_config
    assert isinstance(model_config, dict)
    assert "models" in model_config
    assert isinstance(training_config, dict)
    assert "training" in training_config


def test_logger():
    """Test that logger helper returns a logger instance."""
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"


# --- DATA PROCESSING TESTS ---

def test_ingest_data(tmp_path, create_raw_data_csv):
    """Test that ingestion loads CSV and writes report correctly."""
    df = ingest_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 20
    assert df.shape[1] == 12
    
    report_path = tmp_path / "artifacts" / "ingestion_report.json"
    assert report_path.exists()
    with report_path.open("r") as f:
        report = json.load(f)
    assert report["shape"]["rows"] == 20


def test_validate_data(tmp_path, dummy_data):
    """Test validation detects schema completeness and column violations."""
    report = validate_data(dummy_data)
    assert report["is_valid"] is True
    assert (tmp_path / "artifacts" / "validation_report.json").exists()

    # Drop target column to trigger validation failure
    df_invalid = dummy_data.drop(columns=["loan_status"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_data(df_invalid)

    # Drop another expected column
    df_missing_col = dummy_data.drop(columns=["person_age"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_data(df_missing_col)


def test_transform_data(tmp_path, dummy_data):
    """Test that transform handles missing values, encodes categories, and scales values."""
    # Introduce some NaNs to test imputation
    df_with_nan = dummy_data.copy()
    df_with_nan.loc[0, "person_age"] = np.nan
    df_with_nan.loc[1, "person_home_ownership"] = np.nan

    train_df, test_df = transform_data(df_with_nan)

    assert not train_df.isna().any().any()
    assert not test_df.isna().any().any()
    assert (tmp_path / "data" / "processed" / "train.csv").exists()
    assert (tmp_path / "data" / "processed" / "test.csv").exists()
    assert (tmp_path / "artifacts" / "scaler.pkl").exists()
    assert (tmp_path / "artifacts" / "encoders.pkl").exists()


# --- FEATURE ENGINEERING TESTS ---

def test_feature_engineering(tmp_path, dummy_data):
    """Test that derived features are correctly added and correlation is checked."""
    # Transformed data is numeric, so we copy and transform dummy_data features
    train_df, test_df = transform_data(dummy_data)
    train_feat, test_feat = engineer_features(train_df, test_df)

    assert "loan_to_income" in train_feat.columns
    assert "emp_length_to_age" in train_feat.columns
    assert (tmp_path / "artifacts" / "feature_list.json").exists()


# --- MODEL TESTS ---

def test_build_models():
    """Test that correct classification models are constructed."""
    models = build_models()
    assert "logistic_regression" in models
    assert "xgboost" in models
    assert "lightgbm" in models


def test_train_models(tmp_path, dummy_data):
    """Test that training fits models, dumps pickles, and writes metrics."""
    train_df, test_df = transform_data(dummy_data)
    feat_train, feat_test = engineer_features(train_df, test_df)

    results = train_models(feat_train)
    assert "logistic_regression" in results
    assert "xgboost" in results
    assert "lightgbm" in results
    
    assert (tmp_path / "artifacts" / "models" / "logistic_regression.pkl").exists()
    assert (tmp_path / "artifacts" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "training_results.json").exists()


def test_evaluate_models(tmp_path, dummy_data):
    """Test model evaluation predicts outputs, logs metrics, and outputs best_model_info."""
    train_df, test_df = transform_data(dummy_data)
    feat_train, feat_test = engineer_features(train_df, test_df)

    # Save a simple model trained on the actual engineered features
    target_column = "loan_status"
    X_train = feat_train.drop(columns=[target_column])
    y_train = feat_train[target_column]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "logistic_regression.pkl")

    best_model = evaluate_models(feat_test)
    assert best_model == "logistic_regression"
    assert (tmp_path / "artifacts" / "best_model_info.json").exists()
    assert (tmp_path / "artifacts" / "metrics.json").exists()


def test_register_best_model(tmp_path):
    """Test model registration transitions and records versions."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    best_info = {
        "model_name": "logistic_regression",
        "version": "N/A",
        "run_id": "mock_run_123",
        "metrics": {"f1_weighted": 0.8},
    }
    with open(artifacts_dir / "best_model_info.json", "w") as f:
        json.dump(best_info, f)

    register_best_model()

    with open(artifacts_dir / "best_model_info.json", "r") as f:
        updated_info = json.load(f)
        
    assert updated_info["version"] == "1"


# --- EXPLAINABILITY TESTS ---

def test_shap_explainer(tmp_path, dummy_data):
    """Test ShapExplainer returns contributions and summary plots are created."""
    train_df, test_df = transform_data(dummy_data)
    feat_train, feat_test = engineer_features(train_df, test_df)

    target_column = "loan_status"
    X_train = feat_train.drop(columns=[target_column])
    y_train = feat_train[target_column]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "logistic_regression.pkl")

    explainer = generate_shap_explanations("logistic_regression", feat_train, feat_test)
    assert explainer is not None
    assert (tmp_path / "artifacts" / "shap_explainer.pkl").exists()
    assert (tmp_path / "artifacts" / "plots" / "shap_summary.png").exists()
    assert (tmp_path / "artifacts" / "plots" / "shap_importance.png").exists()

    single_row = feat_test.drop(columns=[target_column]).head(1)
    explanation = explainer.explain_single(single_row)
    assert "base_value" in explanation
    assert "feature_contributions" in explanation


# --- MAIN PIPELINE INTEGRATION TEST ---

def test_main_pipeline_e2e(tmp_path, create_raw_data_csv):
    """Test running the full pipeline from ingestion to explainability."""
    # Execute full pipeline
    main.main()

    # Assert that all stage output files and reports were successfully generated in tmp_path
    assert (tmp_path / "artifacts" / "ingestion_report.json").exists()
    assert (tmp_path / "artifacts" / "validation_report.json").exists()
    assert (tmp_path / "data" / "processed" / "train.csv").exists()
    assert (tmp_path / "data" / "processed" / "test.csv").exists()
    assert (tmp_path / "artifacts" / "feature_list.json").exists()
    assert (tmp_path / "artifacts" / "models" / "logistic_regression.pkl").exists()
    assert (tmp_path / "artifacts" / "best_model_info.json").exists()
    assert (tmp_path / "artifacts" / "plots" / "shap_summary.png").exists()
