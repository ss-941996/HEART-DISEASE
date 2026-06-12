❤️ Heart Disease Prediction API
A fully containerized, reproducible, end‑to‑end MLOps project for predicting heart disease using clinical features.
Built with FastAPI, scikit‑learn, DVC, MLflow, Docker, and CI/CD.

This project demonstrates:

real‑world ML preprocessing

reproducible training pipelines

experiment tracking

data versioning

automated testing

containerized inference

production‑ready API deployment

🚀 Quickstart (Run with Docker — Recommended)
The easiest way to run the API is using the pre‑built image hosted on GitHub Container Registry (GHCR).

1. Pull the latest image
Code
docker pull ghcr.io/ss-941996/heart-api:latest
2. Run the container
Code
docker run -p 8000:8000 ghcr.io/ss-941996/heart-api:latest
3. Open the interactive API docs
👉 http://localhost:8000/docs

You can now send predictions directly from the Swagger UI.

🧠 Project Overview
This repository implements a complete ML workflow for predicting heart disease:

Data ingestion & cleaning

Feature engineering

Model training & evaluation

Experiment tracking (MLflow)

Data versioning (DVC)

Model packaging

FastAPI inference service

Dockerized deployment

CI/CD pipeline (GitHub Actions)

The API exposes a /predict endpoint that accepts clinical features and returns:

predicted class (0 = no disease, 1 = disease)

probability

confidence percentage

model metadata

🏗️ Project Structure
Code
heart-disease/
│
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI app + inference pipeline
│   ├── data/
│   │   ├── ingest.py               # Raw data ingestion
│   │   ├── preprocess.py           # Offline preprocessing + preprocessor.pkl
│   │   └── eda.py                  # Exploratory analysis
│   ├── features/
│   │   └── build_features.py       # Feature engineering
│   ├── models/
│   │   ├── train.py                # Training pipeline
│   │   ├── evaluate.py             # Evaluation + metrics
│   │   └── predict.py              # Local inference
│   └── monitoring/
│       └── drift.py                # Data drift detection
│
├── models/
│   ├── champion.pkl                # Best model
│   ├── preprocessor.pkl            # Full preprocessing pipeline
│   └── metadata.json               # Model metadata
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── data.dvc                    # DVC tracking
│
├── mlruns/                         # MLflow experiment store
├── dvcstore/                       # DVC remote cache
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_models.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-inference.txt
├── README.md
└── config.yaml
🔬 Model & Preprocessing Pipeline
The model is a Logistic Regression classifier trained on the UCI Heart Disease dataset.

Preprocessing includes:
dropping high‑missingness columns

imputing numeric medians

adding missingness indicator flags

mapping binary categories (sex, fbs, exang)

one‑hot encoding cp + restecg

robust scaling of numeric features

saving the exact feature order

Inference uses the same pipeline
The API loads:

preprocessor.pkl

champion.pkl

metadata.json

and applies identical transformations to incoming JSON before prediction.

This guarantees training ↔ inference consistency.

🧪 Testing
The project includes:

API tests

Feature engineering tests

Model tests

Run all tests:

Code
pytest -q
📦 DVC (Data Version Control)
DVC tracks:

raw data

processed data

intermediate artifacts

Useful commands:

Code
dvc pull     # get latest data
dvc repro    # reproduce full pipeline
dvc push     # push updated artifacts
Even if users don’t need DVC, it shows the project is fully reproducible.

📊 MLflow Experiment Tracking
MLflow logs:

metrics

parameters

artifacts

model versions

Start the UI:

Code
mlflow ui --backend-store-uri mlflow.db
Open:

👉 http://localhost:5000

🔄 CI/CD Pipeline
This repo includes a full GitHub Actions pipeline:

linting

testing

building Docker image

pushing to GHCR

tagging releases

updating metadata

Every push to main triggers:

Run tests

Build Docker image

Push to GHCR

Update model metadata

Deploy new API image

This ensures the API is always up‑to‑date and reproducible.

🧠 Example Prediction Request
Code
POST /predict
Content-Type: application/json

{
  "age": 45,
  "sex": "Male",
  "cp": "asymptomatic",
  "trestbps": 130,
  "chol": 230,
  "fbs": "False",
  "restecg": "normal",
  "thalch": 150,
  "exang": "False",
  "oldpeak": 1.2
}