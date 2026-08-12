"""FastAPI app for credit risk prediction and explanation."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import explainer as explainer_module
from api import predictor as predictor_module
from api.explainer import explain, load_shap_explainer
from api.predictor import get_model_info, load_artifacts, predict
from api.schemas import (
    ApplicantInput,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

app = FastAPI(title="Credit Risk Prediction API", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    """Return a clean 422 response for request validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Load all required artifacts when the API starts."""
    load_artifacts()
    load_shap_explainer()
    LOGGER.info(
        "API started with model '%s' version '%s'",
        predictor_module.BEST_MODEL_INFO.get("model_name", "unknown"),
        predictor_module.BEST_MODEL_INFO.get("version", "unknown"),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return a simple health status."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor_module.MODEL is not None,
        explainer_loaded=explainer_module.SHAP_EXPLAINER is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded best model."""
    try:
        return ModelInfoResponse(**get_model_info())
    except Exception as error:
        LOGGER.exception("Failed to fetch model info.")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/predict", response_model=PredictionResponse)
async def predict_route(input_data: ApplicantInput) -> PredictionResponse:
    """Return credit risk prediction for one applicant."""
    try:
        return predict(input_data)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Internal server error.") from error


@app.post("/explain", response_model=ExplainResponse)
async def explain_route(input_data: ApplicantInput) -> ExplainResponse:
    """Return prediction plus SHAP explanation."""
    try:
        return explain(input_data)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Explanation failed.")
        raise HTTPException(status_code=500, detail="Internal server error.") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8100, reload=True)
