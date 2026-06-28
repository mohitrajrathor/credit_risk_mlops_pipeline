"""SHAP explanation helpers for the API."""

from __future__ import annotations

import sys

import joblib

from api.predictor import FEATURE_LIST, load_artifacts, predict, preprocess
from api.schemas import ApplicantInput, ExplainResponse
from src.utils.config_loader import PROJECT_ROOT, load_training_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

SHAP_EXPLAINER = None


def _register_shap_explainer_compatibility() -> None:
    """Allow legacy pickles created from the training script to load in the API process."""
    try:
        from src.explainability.shap_explainer import ShapExplainer
    except Exception as error:  # pragma: no cover - defensive import guard
        LOGGER.debug("Unable to import SHAP explainer class for compatibility: %s", error)
        return

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    if getattr(main_module, "ShapExplainer", None) is not ShapExplainer:
        setattr(main_module, "ShapExplainer", ShapExplainer)


def load_shap_explainer() -> None:
    """Load the saved SHAP explainer once at startup."""
    global SHAP_EXPLAINER

    shap_explainer_path = PROJECT_ROOT / load_training_config()["artifacts"]["shap_explainer_path"]
    _register_shap_explainer_compatibility()

    try:
        SHAP_EXPLAINER = joblib.load(shap_explainer_path)
        LOGGER.info("Loaded SHAP explainer from %s", shap_explainer_path)
    except Exception as error:  # pragma: no cover - defensive runtime guard
        SHAP_EXPLAINER = None
        LOGGER.warning("Unable to load SHAP explainer from %s: %s", shap_explainer_path, error)


def explain(input_data: ApplicantInput) -> ExplainResponse:
    """Return prediction plus SHAP-based feature contributions."""
    global SHAP_EXPLAINER

    if SHAP_EXPLAINER is None:
        load_artifacts()
        load_shap_explainer()

    if SHAP_EXPLAINER is None:
        prediction = predict(input_data)
        return ExplainResponse(
            prediction=prediction,
            base_value=0.0,
            feature_contributions={},
            top_risk_factors=[],
        )

    processed_df = preprocess(input_data)
    explanation = SHAP_EXPLAINER.explain_single(processed_df)
    prediction = predict(input_data)

    feature_contributions = {
        feature: float(value)
        for feature, value in explanation["feature_contributions"].items()
    }

    top_risk_factors = [
        {"feature": feature, "contribution": feature_contributions[feature]}
        for feature in sorted(
            feature_contributions,
            key=lambda name: abs(feature_contributions[name]),
            reverse=True,
        )[:5]
    ]

    return ExplainResponse(
        prediction=prediction,
        base_value=float(explanation["base_value"]),
        feature_contributions=feature_contributions,
        top_risk_factors=top_risk_factors,
    )


if __name__ == "__main__":
    from api.schemas import ApplicantInput

    load_artifacts()
    load_shap_explainer()
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
    print(explain(sample).model_dump())
