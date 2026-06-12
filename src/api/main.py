import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ───────────────────────────────────────────────────────────────
# 1. LOAD MODEL + PREPROCESSOR
# ───────────────────────────────────────────────────────────────

with open("models/champion.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

# ───────────────────────────────────────────────────────────────
# 2. FASTAPI APP
# ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease from clinical features.",
    version="1.0.0",
)

# ───────────────────────────────────────────────────────────────
# 3. INPUT SCHEMA
# ───────────────────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    age:      int             = Field(..., description="Age in years", ge=1, le=120)
    sex:      str             = Field(..., description="'Male' or 'Female'")
    cp:       str             = Field(..., description="Chest pain type")
    trestbps: Optional[float] = Field(None)
    chol:     Optional[float] = Field(None)
    fbs:      Optional[str]   = Field(None)
    restecg:  Optional[str]   = Field(None)
    thalch:   Optional[float] = Field(None)
    exang:    Optional[str]   = Field(None)
    oldpeak:  Optional[float] = Field(None)

# ───────────────────────────────────────────────────────────────
# 4. FIXED PREPROCESSING FUNCTION (MATCHES TRAINING EXACTLY)
# ───────────────────────────────────────────────────────────────

def preprocess_input(patient: PatientFeatures) -> pd.DataFrame:

    # Convert to DataFrame
    df = pd.DataFrame([{
        "age":      patient.age,
        "sex":      patient.sex or "Unknown",
        "cp":       patient.cp or "Unknown",
        "trestbps": patient.trestbps,
        "chol":     patient.chol,
        "fbs":      patient.fbs or "Unknown",
        "restecg":  patient.restecg or "Unknown",
        "thalch":   patient.thalch,
        "exang":    patient.exang or "Unknown",
        "oldpeak":  patient.oldpeak,
    }])

    # ───────────────────────────────────────────────
    # 1. Missingness flags (before imputation)
    # ───────────────────────────────────────────────
    for col in preprocessor["cols_with_missingness_flag"]:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ───────────────────────────────────────────────
    # 2. Numeric median imputation
    # ───────────────────────────────────────────────
    for col, median_val in preprocessor["numeric_medians"].items():
        df[col] = df[col].fillna(median_val)

    # ───────────────────────────────────────────────
    # 3. Categorical imputation
    # ───────────────────────────────────────────────
    for col in preprocessor["binary_cols"] + preprocessor["onehot_cols"]:
        df[col] = df[col].fillna("Unknown")

    # ───────────────────────────────────────────────
    # 4. Binary encoding (exact mapping)
    # ───────────────────────────────────────────────
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0, "Unknown": -1})
    df["fbs"] = df["fbs"].map({"True": 1, True: 1, "False": 0, False: 0, "Unknown": -1})
    df["exang"] = df["exang"].map({"True": 1, True: 1, "False": 0, False: 0, "Unknown": -1})

    # ───────────────────────────────────────────────
    # 5. One-hot encoding (cp + restecg)
    # ───────────────────────────────────────────────
    df = pd.get_dummies(df, columns=preprocessor["onehot_cols"], drop_first=False)

    # ───────────────────────────────────────────────
    # 6. Sanitize column names BEFORE alignment
    # ───────────────────────────────────────────────
    df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

    # ───────────────────────────────────────────────
    # 7. Add missing one-hot columns
    # ───────────────────────────────────────────────
    expected_cols = [
        c.replace(" ", "_").replace("-", "_")
        for c in preprocessor["feature_columns"]
    ]

    df = df.reindex(columns=expected_cols, fill_value=0)

    # ───────────────────────────────────────────────
    # 8. Scale numeric columns (AFTER alignment)
    # ───────────────────────────────────────────────
    numeric_cols = preprocessor["numeric_cols"]
    df[numeric_cols] = preprocessor["scaler"].transform(df[numeric_cols])

    return df

# ───────────────────────────────────────────────────────────────
# 5. ENDPOINTS
# ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata.get("champion_model", "unknown"),
        "f1": metadata.get("test_f1", "unknown"),
    }

@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }

@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        X = preprocess_input(patient)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        return {
            "prediction": prediction,
            "diagnosis": "Disease" if prediction == 1 else "No Disease",
            "probability": round(probability, 4),
            "confidence": f"{round(probability * 100, 1)}%",
            "model_used": metadata.get("champion_model", "unknown"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
