# Credit Risk MLOps

> An end-to-end Machine Learning pipeline for Credit Risk Analysis and Loan Default Prediction - built with production-grade MLOps practices.

**By Mohit Raj Rathor**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4)](https://cloud.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is this project?

Most ML projects end at a Jupyter notebook. This one doesn't.

This project takes a real-world Indian bank and CIBIL dataset and builds a **complete production pipeline** — from raw data ingestion all the way to a live REST API deployed on Google Cloud. The goal was to go beyond just training a model and actually understand what it takes to ship ML to production.

The system predicts whether a loan applicant falls into **Low**, **Medium**, or **High** credit risk category, with full model explainability via SHAP — so predictions aren't just black boxes.

---

## Architecture

```
Raw Data (Kaggle)
      │
      ▼
┌─────────────────────────────────────────────────┐
│              DVC Pipeline (Reproducible)         │
│                                                   │
│  Data Ingestion → Validation → Transformation    │
│       → Feature Engineering → Training           │
│       → Evaluation → Best Model to Registry      │
└─────────────────────────────────────────────────┘
      │                        │
      ▼                        ▼
  GCS Bucket              MLflow Server
 (Data + Models)          (GCP VM)
                               │
                               ▼
                    ┌─────────────────────┐
                    │  FastAPI (Docker)    │
                    │  /predict            │
                    │  /explain (SHAP)     │
                    │  /health             │
                    │  /model-info         │
                    └─────────────────────┘
                               │
                               ▼
                      GCP Cloud Run
                      (us-central1)
                               ▲
                               │
                    GitHub Actions CI/CD
                    (push to main → deploy)
```

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.12 |
| Data Versioning | DVC + GCS |
| Experiment Tracking | MLflow (GCP VM) |
| ML Models | Logistic Regression, XGBoost, LightGBM |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| Cloud | GCP (Cloud Run, GCS, Compute Engine, Artifact Registry) |
| CI/CD | GitHub Actions |
| Config Management | YAML (per component) |

---

## Dataset

**Leading Indian Bank & CIBIL Real World Dataset** from Kaggle.

- Problem: Multi-class Classification (Loan Default Risk)
- Target: Risk Category → `Low` / `Medium` / `High`
- Features: Credit score, income, loan amount, EMI history, and more

Data is version-controlled with DVC. The actual CSV files are not in this repo — they live in GCS and are pulled via `dvc pull`.

---

## Project Structure

```
credit-risk-MLOps/
├── .github/workflows/          # CI/CD pipelines
│   ├── dev_pipeline.yml        # Manual trigger on dev branch
│   └── main_deploy.yml         # Auto deploy on push to main
├── config/                     # YAML configs per component
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── gcp_config.yaml
├── data/                       # DVC tracked (not in git)
│   ├── raw/
│   └── processed/
├── src/                        # Core ML pipeline
│   ├── data/                   # Ingestion, validation, transformation
│   ├── features/               # Feature engineering
│   ├── models/                 # Training, evaluation, registry
│   ├── explainability/         # SHAP explainer
│   └── utils/                  # Logger, config loader
├── api/                        # FastAPI app
│   ├── main.py
│   ├── schemas.py
│   ├── predictor.py
│   └── explainer.py
├── notebooks/                  # EDA + experiment analysis
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # DVC tracked params
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.12
- Docker
- GCP account with a service account key
- DVC installed (`pip install dvc[gs]`)

### 1. Clone the repo
```bash
git clone https://github.com/mohitrajrathor/credit-risk-MLOps.git
cd credit-risk-MLOps
```

### 2. Setup environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure GCP credentials
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
export MLFLOW_TRACKING_URI="http://<gcp-vm-ip>:5000"
```

### 4. Pull data with DVC
```bash
dvc pull
```

### 5. Run the full pipeline
```bash
dvc repro
```
This runs all stages: ingestion → validation → transformation → feature engineering → training → evaluation → model registration.

### 6. Run the API locally
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger docs available at: `http://localhost:8000/docs`

---

## API Endpoints

### `POST /predict`
Predict credit risk category for a loan applicant.

**Request:**
```json
{
  "cibil_score": 720,
  "annual_income": 850000,
  "loan_amount": 500000,
  "loan_tenure": 36,
  "existing_emis": 2,
  ...
}
```

**Response:**
```json
{
  "risk_category": "Low",
  "probabilities": {
    "Low": 0.72,
    "Medium": 0.21,
    "High": 0.07
  },
  "model_version": "3",
  "model_name": "XGBoostClassifier"
}
```

### `POST /explain`
Get SHAP explanation for a prediction.

**Response:**
```json
{
  "base_value": 0.33,
  "feature_contributions": {
    "cibil_score": 0.18,
    "annual_income": 0.12,
    "existing_emis": -0.08,
    ...
  }
}
```

### `GET /health`
```json
{ "status": "ok", "model": "XGBoostClassifier", "version": "3" }
```

### `GET /model-info`
Returns active model details, training metrics, and registration timestamp.

---

## MLflow Experiment Tracking

All 3 models are tracked in MLflow with:
- Hyperparameters
- Cross-validation scores
- Test metrics (Accuracy, F1, ROC-AUC, Precision, Recall)
- Confusion matrix plots
- Model artifacts

Best model is automatically promoted to **MLflow Model Registry** and served via the API.

---

## CI/CD

| Branch | Trigger | What happens |
|---|---|---|
| `dev` | Manual (`workflow_dispatch`) | Pulls data → runs DVC pipeline → logs to MLflow |
| `main` | Push | Full pipeline → build Docker → push to Artifact Registry → deploy to Cloud Run |

---

## Model Comparison (Example)

| Model | Accuracy | F1 (Weighted) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | - | - | - |
| XGBoost | - | - | - |
| LightGBM | - | - | - |

*Results will be updated after training runs.*

---

## Reproducibility

The entire pipeline is defined in `dvc.yaml`. A fresh clone can reproduce all results:

```bash
git clone <repo>
dvc pull          # get data from GCS
dvc repro         # run full pipeline
```

DVC tracks data, models, and metrics. Changing any param in `params.yaml` invalidates downstream stages automatically.

---

## Future Scope

- [ ] LIME explainability (local interpretable model-agnostic explanations)
- [ ] Batch prediction endpoint (`/predict/batch`)
- [ ] Model drift monitoring with Evidently AI
- [ ] API authentication with JWT
- [ ] Streamlit dashboard for non-technical stakeholders

---

## Why I built this

I wanted to understand what happens *after* the model is trained. Most tutorials stop at `model.fit()`. This project is my attempt to build something I'd actually want to maintain in production — proper versioning, experiment tracking, explainability, and automated deployment.

The dataset is real-world Indian banking data which made the problem domain meaningful and the feature engineering genuinely interesting.

---

## Author

**Mohit Raj Rathor**  
[GitHub](https://github.com/mohitrajrathor) · [LinkedIn](https://linkedin.com/in/mohitrajrathor)

---

## License

MIT License — feel free to fork, adapt, and build on this.