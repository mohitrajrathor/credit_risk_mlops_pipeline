# Credit Risk MLOps Pipeline 🏦

An end-to-end, production-ready Machine Learning Operations (MLOps) pipeline for predicting credit risk and loan default probability. This project integrates data versioning, automated training pipelines, experiment tracking, model explainability, and a RESTful API service.

---

## 📌 Project Overview

Assessing credit risk accurately is crucial for financial institutions to minimize loan default losses while maximizing lending efficiency. This project implements a complete MLOps workflow to:

- **Ingest & Validate Data**: Validate schema consistency, handle missing data, and transform numerical/categorical financial attributes.
- **Engineer Features**: Generate key financial indicators such as `loan_to_income` and `emp_length_to_age`.
- **Train & Compare Models**: Benchmark multiple classifiers (**Logistic Regression**, **XGBoost**, **LightGBM**) using 5-fold cross-validation.
- **Track Experiments**: Log model parameters, evaluation metrics (`F1-score`, `ROC-AUC`, `Precision`, `Recall`), and artifacts using **MLflow**.
- **Explain Model Predictions**: Provide feature importance and local instance explanations using **SHAP**.
- **Serve via REST API**: Expose real-time prediction and explanation endpoints powered by **FastAPI** and **Pydantic**.
- **Orchestrate & Containerize**: Reproduce pipeline stages with **DVC** and deploy using **Docker** & **Docker Compose**.

---

## 🛠️ Tech Stack

- **Language**: Python 3.12
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, SHAP
- **MLOps & Pipeline**: DVC (Data Version Control), MLflow (Experiment Tracking & Model Registry)
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Containerization**: Docker, Docker Compose
- **Environment & Tools**: `uv` / `venv`, PyYAML

---

## 📁 Directory Structure

```text
.
├── api/                     # FastAPI application endpoints & schema definitions
│   ├── main.py              # Application entry point & route handlers
│   ├── predictor.py         # Model loading & inference logic
│   ├── explainer.py         # SHAP explanation service
│   └── schemas.py           # Pydantic request & response validation schemas
├── configs/                 # YAML configuration files
│   ├── data_config.yaml     # Dataset schema, features & preprocessing settings
│   ├── model_config.yaml    # Algorithm hyperparameters
│   └── traning_config.yaml  # Train/test split, cross-validation & evaluation settings
├── data/                    # Managed dataset directory (DVC tracked)
│   ├── raw/                 # Raw input datasets
│   └── processed/           # Processed & engineered datasets
├── src/                     # Core pipeline source code
│   ├── data_processing/     # Data ingestion, schema validation & transformation
│   ├── features/            # Feature engineering modules
│   ├── models/              # Model training, benchmarking & registry
│   ├── explainability/      # SHAP explainer utilities
│   └── utils/               # Logging and configuration loading helpers
├── notebooks/               # EDA and experiment notebooks
├── main.py                  # CLI runner for the end-to-end ML pipeline
├── dvc.yaml                 # DVC pipeline stage definitions
├── Dockerfile               # Container build configuration for FastAPI service
├── docker-compose.yml       # Docker Compose definition (FastAPI + MLflow server)
└── requirements.txt         # Project Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose *(optional for containerized deployment)*

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/mohitrajrathor/credit_risk_mlops_pipeline.git
cd credit_risk_mlops_pipeline

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Complete ML Pipeline
You can run the entire machine learning pipeline locally via `main.py` or through `dvc repro`:

```bash
# Execute via python script
python main.py

# Or run stages via DVC
dvc repro
```

This will ingest data, run schema validation, preprocess features, train candidate models, evaluate performance, register the top model, and pre-compute SHAP explainers into `artifacts/`.

---

## 🌐 Serving the API

### Option A: Local Run with Uvicorn
Start the FastAPI server locally:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Option B: Docker Compose
Spin up the FastAPI server and MLflow tracking server together:

```bash
docker-compose up --build
```
- **FastAPI Service**: `http://localhost:8000`
- **MLflow Tracking UI**: `http://localhost:5000`

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Service status and model/explainer load state |
| `/model-info` | `GET` | Metadata and performance metrics of active model |
| `/predict` | `POST` | Single applicant credit risk classification (`0` = Low Risk, `1` = High Risk) |
| `/explain` | `POST` | Prediction outcome accompanied by SHAP feature contribution scores |

### Sample Prediction Request (`POST /predict`)

```json
{
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
  "cb_person_cred_hist_length": 5
}
```

### Sample Response

```json
{
  "loan_status": 0,
  "default_probability": 0.142,
  "risk_label": "Low Risk"
}
```

---

## 📊 Model Evaluation & Metrics

Models are evaluated on an 80/20 train-test split using weighted F1-score, ROC-AUC, Precision, and Recall. Metrics are saved to `artifacts/metrics.json` and logged to MLflow during training.

---

## 📄 License

This project is licensed under the MIT License.
