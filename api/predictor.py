"""Artifact loading and prediction helpers for the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from api.schemas import ApplicantInput, PredictionResponse
from src.utils.config_loader import PROJECT_ROOT, load_data_config, load_training_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

MODEL: Any | None = None
SCALER: Any | None = None
ENCODERS: dict[str, Any] | None = None
FEATURE_LIST: list[str] = []
LABEL_MAPPING: dict[str, Any] = {}
BEST_MODEL_INFO: dict[str, Any] = {}
DATA_CONFIG: dict[str, Any] | None = None
TRAINING_CONFIG: dict[str, Any] | None = None


def _create_label_mapping_file(label_mapping_path: Path, encoders: dict[str, Any]) -> dict[str, Any]:
    """Create a readable label mapping file from saved encoders."""
    label_mapping = {
        column: {label: int(index) for index, label in enumerate(encoder.classes_)}
        for column, encoder in encoders.items()
    }

    with label_mapping_path.open("w", encoding="utf-8") as file:
        json.dump(label_mapping, file, indent=2)

    return label_mapping


def _create_best_model_info_file(best_model_info_path: Path) -> dict[str, Any]:
    """Create best model metadata from the metrics artifact."""
    metrics_path = PROJECT_ROOT / load_training_config()["artifacts"]["metrics_path"]

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    best_model_name, best_metrics = max(
        metrics.items(),
        key=lambda item: item[1]["f1_weighted"],
    )

    best_model_info = {
        "model_name": best_model_name,
        "version": "1.0.0",
        "metrics": best_metrics,
    }

    with best_model_info_path.open("w", encoding="utf-8") as file:
        json.dump(best_model_info, file, indent=2)

    return best_model_info


def load_artifacts() -> None:
    """Load all required prediction artifacts once at startup."""
    global MODEL, SCALER, ENCODERS, FEATURE_LIST, LABEL_MAPPING, BEST_MODEL_INFO, DATA_CONFIG, TRAINING_CONFIG

    DATA_CONFIG = load_data_config()
    TRAINING_CONFIG = load_training_config()

    data_artifacts = DATA_CONFIG["artifacts"]
    training_artifacts = TRAINING_CONFIG["artifacts"]

    encoders_path = PROJECT_ROOT / data_artifacts["encoders_path"]
    scaler_path = PROJECT_ROOT / data_artifacts["scaler_path"]
    feature_list_path = PROJECT_ROOT / data_artifacts["feature_list_path"]
    label_mapping_path = PROJECT_ROOT / data_artifacts["label_mapping_path"]
    best_model_info_path = PROJECT_ROOT / data_artifacts["best_model_info_path"]
    models_dir = PROJECT_ROOT / training_artifacts["models_dir"]

    ENCODERS = joblib.load(encoders_path)
    SCALER = joblib.load(scaler_path)

    with feature_list_path.open("r", encoding="utf-8") as file:
        feature_data = json.load(file)
    FEATURE_LIST = list(feature_data["feature_list"])

    if label_mapping_path.exists():
        with label_mapping_path.open("r", encoding="utf-8") as file:
            LABEL_MAPPING = json.load(file)
    else:
        LABEL_MAPPING = _create_label_mapping_file(label_mapping_path, ENCODERS)

    if best_model_info_path.exists():
        with best_model_info_path.open("r", encoding="utf-8") as file:
            BEST_MODEL_INFO = json.load(file)
    else:
        BEST_MODEL_INFO = _create_best_model_info_file(best_model_info_path)

    model_name = BEST_MODEL_INFO["model_name"]
    model_path = models_dir / f"{model_name}.pkl"
    MODEL = joblib.load(model_path)

    LOGGER.info("Loaded prediction artifacts for model: %s", model_name)


def _ensure_artifacts_loaded() -> None:
    """Make sure startup loading happened before using artifacts."""
    if MODEL is None or SCALER is None or ENCODERS is None or not FEATURE_LIST:
        raise RuntimeError("Artifacts are not loaded. Call load_artifacts() first.")


def preprocess(input_data: ApplicantInput) -> pd.DataFrame:
    """Convert one applicant input into a model-ready dataframe."""
    _ensure_artifacts_loaded()

    assert DATA_CONFIG is not None
    assert ENCODERS is not None
    assert SCALER is not None

    schema_config = DATA_CONFIG["schema"]
    numerical_features = list(schema_config["numerical_features"])
    categorical_features = list(schema_config["categorical_features"])

    input_dict = input_data.model_dump()
    df = pd.DataFrame([input_dict])
    df[numerical_features] = df[numerical_features].astype(float)

    for column in categorical_features:
        encoder = ENCODERS[column]
        value = str(df.at[0, column])

        if value not in encoder.classes_:
            raise ValueError(f"Unknown category '{value}' for column '{column}'.")

        df[column] = encoder.transform(df[column].astype(str))

    df.loc[:, numerical_features] = SCALER.transform(df[numerical_features])
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"].replace(0, 1)
    df["emp_length_to_age"] = df["person_emp_length"] / df["person_age"].replace(0, 1)

    processed_df = df.reindex(columns=FEATURE_LIST)
    return processed_df


def predict(input_data: ApplicantInput) -> PredictionResponse:
    """Run prediction for one applicant."""
    _ensure_artifacts_loaded()
    assert MODEL is not None

    processed_df = preprocess(input_data)
    prediction = int(MODEL.predict(processed_df)[0])
    probability = float(MODEL.predict_proba(processed_df)[0][1])
    risk_label = "High Risk" if prediction == 1 else "Low Risk"

    return PredictionResponse(
        loan_status=prediction,
        default_probability=round(probability, 4),
        risk_label=risk_label,
    )


def get_model_info() -> dict[str, Any]:
    """Return loaded best model metadata."""
    _ensure_artifacts_loaded()
    return BEST_MODEL_INFO


if __name__ == "__main__":
    load_artifacts()
    sample = ApplicantInput(
        person_age=28,
        person_income=65000,
        person_home_ownership="RENT",
        person_emp_length=4.0,
        loan_intent="PERSONAL",
        loan_grade="C",
        loan_amnt=12000,
        loan_int_rate=11.5,
        loan_percent_income=0.18,
        cb_person_default_on_file="N",
        cb_person_cred_hist_length=5,
    )
    print(preprocess(sample))
    print(predict(sample).model_dump())
