import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── 1. LOAD MODELS AT STARTUP ────────────────────────────────────────────────
# We load the model and preprocessor ONCE when the app starts, not on every
# request. Loading a pickle file is slow (relative to inference). If we loaded
# on every request, a busy endpoint would be extremely slow.
# By loading at module level, they stay in memory for the lifetime of the app.

with open("models/champion.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

# ── 2. FASTAPI APP ───────────────────────────────────────────────────────────
# FastAPI() creates the application instance.
# title, description, version appear in the auto-generated docs at /docs
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence of heart disease from clinical features.",
    version="1.0.0",
)

# ── 3. INPUT SCHEMA ───────────────────────────────────────────────────────────
# Pydantic's BaseModel defines what a valid request body looks like.
# FastAPI uses this to:
#   a) automatically validate incoming JSON — if a required field is missing
#      or the wrong type, FastAPI returns a 422 error before your code runs
#   b) generate the interactive docs at /docs automatically
#
# Optional[...] = None means the field can be absent (mirrors our missing value
# handling in preprocessing — we'll impute it just like we did in training).
# Field(..., description=) adds documentation to each field in /docs.

class PatientFeatures(BaseModel):
    age:      int            = Field(..., description="Age in years", ge=1, le=120)
    sex:      str            = Field(..., description="'Male' or 'Female'")
    cp:       str            = Field(..., description="Chest pain type: 'asymptomatic', 'non-anginal', 'atypical angina', 'typical angina'")
    trestbps: Optional[float] = Field(None, description="Resting blood pressure (mm Hg)")
    chol:     Optional[float] = Field(None, description="Serum cholesterol (mg/dl)")
    fbs:      Optional[str]   = Field(None, description="Fasting blood sugar > 120 mg/dl: 'True' or 'False'")
    restecg:  Optional[str]   = Field(None, description="Resting ECG: 'normal', 'lv hypertrophy', 'st-t abnormality'")
    thalch:   Optional[float] = Field(None, description="Maximum heart rate achieved")
    exang:    Optional[str]   = Field(None, description="Exercise induced angina: 'True' or 'False'")
    oldpeak:  Optional[float] = Field(None, description="ST depression induced by exercise")

# ── 4. PREPROCESSING FUNCTION ────────────────────────────────────────────────
# This function applies the exact same transformations as preprocess.py.
# The order must be identical. The values used (medians, scaler) come from
# the preprocessor.pkl we trained on — never recomputed from new data.

def preprocess_input(patient: PatientFeatures) -> pd.DataFrame:
    # Convert the Pydantic object to a dictionary, then to a single-row DataFrame.
    # A model always expects a 2D array (rows x columns), never a 1D array.
    # Even for one patient, we wrap it in a list to get shape (1, n_features).
    data = {
        "age":      [patient.age],
        "sex":      [patient.sex      if patient.sex      else "Unknown"],
        "cp":       [patient.cp       if patient.cp       else "Unknown"],
        "trestbps": [patient.trestbps],
        "chol":     [patient.chol],
        "fbs":      [patient.fbs      if patient.fbs      else "Unknown"],
        "restecg":  [patient.restecg  if patient.restecg  else "Unknown"],
        "thalch":   [patient.thalch],
        "exang":    [patient.exang    if patient.exang    else "Unknown"],
        "oldpeak":  [patient.oldpeak],
    }
    df = pd.DataFrame(data)

    # ── Missingness indicators ───────────────────────────────────────────────
    # Same columns we flagged in preprocess.py — before imputation.
    for col in preprocessor["cols_with_missingness_flag"]:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ── Numeric imputation ───────────────────────────────────────────────────
    # Fill missing numeric values with the medians learned from training data.
    # Never recompute medians from the incoming request.
    for col, median_val in preprocessor["numeric_medians"].items():
        df[col] = df[col].fillna(median_val)

    # ── Binary encoding ──────────────────────────────────────────────────────
    df["sex"]   = df["sex"].map({"Male": 1, "Female": 0, "Unknown": -1})
    df["fbs"]   = df["fbs"].map({"True": 1, True: 1, "False": 0, False: 0, "Unknown": -1})
    df["exang"] = df["exang"].map({"True": 1, True: 1, "False": 0, False: 0, "Unknown": -1})

    # ── One-hot encoding ─────────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["cp", "restecg"], drop_first=False)

    # ── Align columns ────────────────────────────────────────────────────────
    # This is critical. After one-hot encoding a single row, some category
    # columns may be missing entirely (e.g. if cp='asymptomatic', there's no
    # cp_typical_angina column in this row's DataFrame).
    # We reindex to the exact columns the model was trained on, filling any
    # missing columns with 0 (meaning that category was absent).
    # Then sanitise column names exactly as in train.py.
    expected_cols = [
        c.replace(" ", "_").replace("-", "_")
        for c in preprocessor["feature_columns"]
    ]
    df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")
    df = df.reindex(columns=expected_cols, fill_value=0)

    # ── Scaling ──────────────────────────────────────────────────────────────
    # Use the fitted scaler from training — .transform() only, never .fit()
    numeric_cols = preprocessor["numeric_cols"]
    df[numeric_cols] = preprocessor["scaler"].transform(df[numeric_cols])

    return df

# ── 5. ENDPOINTS ─────────────────────────────────────────────────────────────

# Health check — a simple GET endpoint that returns OK.
# Used by Docker, AWS, and monitoring tools to verify the service is alive.
# If this returns 200, the service is up. If it doesn't respond, it's down.
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  metadata.get("champion_model", "unknown"),
        "f1":     metadata.get("test_f1", "unknown"),
    }

# Root endpoint — basic info
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict",
    }

# Predict endpoint — the main endpoint.
# @app.post means it accepts POST requests (sending data to the server).
# We use POST not GET because we're sending a request body (patient data).
# GET requests have no body — they're for fetching data, not sending it.
@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        # Preprocess the incoming patient data
        X = preprocess_input(patient)

        # Get hard prediction (0 or 1)
        prediction = int(model.predict(X)[0])

        # Get probability of disease (class 1)
        # [0] = first (only) row, [1] = probability of class 1
        probability = float(model.predict_proba(X)[0][1])

        return {
            "prediction":        prediction,
            "diagnosis":         "Disease" if prediction == 1 else "No Disease",
            "probability":       round(probability, 4),
            "confidence":        f"{round(probability * 100, 1)}%",
            "model_used":        metadata.get("champion_model", "unknown"),
        }

    except Exception as e:
        # If anything goes wrong during preprocessing or inference,
        # return a 500 error with the message instead of crashing silently.
        raise HTTPException(status_code=500, detail=str(e))