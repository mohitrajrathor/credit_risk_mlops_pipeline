"""Pydantic schemas for the credit risk API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


HOME_OWNERSHIP_VALUES = {"RENT", "OWN", "MORTGAGE", "OTHER"}
LOAN_INTENT_VALUES = {
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
}
LOAN_GRADE_VALUES = {"A", "B", "C", "D", "E", "F", "G"}
DEFAULT_FILE_VALUES = {"Y", "N"}


class ApplicantInput(BaseModel):
    """Input schema for one loan applicant."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "person_age": 28,
                "person_income": 65000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "C",
                "loan_amnt": 12000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.18,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 5,
            }
        }
    )

    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(..., ge=0, le=80)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0, le=100)
    loan_percent_income: float = Field(..., ge=0, le=1.5)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0, le=100)

    @field_validator(
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        """Convert string fields to uppercase and trim spaces."""
        if not isinstance(value, str):
            raise ValueError("Value must be a string.")
        return value.strip().upper()

    @field_validator("person_home_ownership")
    @classmethod
    def validate_home_ownership(cls, value: str) -> str:
        """Validate home ownership categories."""
        if value not in HOME_OWNERSHIP_VALUES:
            raise ValueError(f"person_home_ownership must be one of {sorted(HOME_OWNERSHIP_VALUES)}")
        return value

    @field_validator("loan_intent")
    @classmethod
    def validate_loan_intent(cls, value: str) -> str:
        """Validate loan intent categories."""
        if value not in LOAN_INTENT_VALUES:
            raise ValueError(f"loan_intent must be one of {sorted(LOAN_INTENT_VALUES)}")
        return value

    @field_validator("loan_grade")
    @classmethod
    def validate_loan_grade(cls, value: str) -> str:
        """Validate loan grade categories."""
        if value not in LOAN_GRADE_VALUES:
            raise ValueError(f"loan_grade must be one of {sorted(LOAN_GRADE_VALUES)}")
        return value

    @field_validator("cb_person_default_on_file")
    @classmethod
    def validate_default_flag(cls, value: str) -> str:
        """Validate previous default flag."""
        if value not in DEFAULT_FILE_VALUES:
            raise ValueError(f"cb_person_default_on_file must be one of {sorted(DEFAULT_FILE_VALUES)}")
        return value


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    model_config = ConfigDict(protected_namespaces=())
    loan_status: Literal[0, 1]
    default_probability: float = Field(..., ge=0, le=1)
    risk_label: Literal["Low Risk", "High Risk"]


class ExplainResponse(BaseModel):
    """Response schema for prediction explanation."""

    model_config = ConfigDict(protected_namespaces=())
    prediction: PredictionResponse
    base_value: float
    feature_contributions: dict[str, float]
    top_risk_factors: list[dict[str, float | str]]


class HealthResponse(BaseModel):
    """Simple health check response."""

    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    explainer_loaded: bool


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded model."""

    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    version: str
    metrics: dict[str, float]


if __name__ == "__main__":
    example = ApplicantInput(
        person_age=28,
        person_income=65000,
        person_home_ownership="rent",
        person_emp_length=4.0,
        loan_intent="personal",
        loan_grade="c",
        loan_amnt=12000,
        loan_int_rate=11.5,
        loan_percent_income=0.18,
        cb_person_default_on_file="n",
        cb_person_cred_hist_length=5,
    )
    print(example.model_dump())
