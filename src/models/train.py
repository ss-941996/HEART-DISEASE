import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import yaml

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
PROCESSED_PATH   = "data/processed/heart_disease_processed.csv"
CHAMPION_PATH    = "models/champion.pkl"
METADATA_PATH    = "models/metadata.json"
CONFIG_PATH      = "config.yaml"
TARGET_COL       = "target"
TEST_SIZE        = 0.2   # 80% train, 20% test
RANDOM_STATE     = 42    # fixed seed → reproducible splits every run
MLFLOW_EXP_NAME  = "heart-disease-classification"

os.makedirs("models", exist_ok=True)

# ── 2. LOAD CONFIG ────────────────────────────────────────────────────────────
# config.yaml holds model hyperparameters outside the code.
# Why: if you hardcode hyperparameters in the script, changing them means
# editing source code. Keeping them in config means you can tune without
# touching the pipeline logic. MLflow will log these automatically.
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ── 3. LOAD PROCESSED DATA ───────────────────────────────────────────────────
df = pd.read_csv(PROCESSED_PATH)

# Sanitise column names — replace spaces with underscores.
# "cp_atypical angina" → "cp_atypical_angina"
# XGBoost specifically rejects column names with spaces or special characters.
# We do this here AND must do the same in FastAPI at inference time.
df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 4. SPLIT FEATURES AND TARGET ─────────────────────────────────────────────
# X = everything except the target column (the features)
# y = just the target column (what we're predicting)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"Features: {X.shape[1]} columns")
print(f"Target distribution:\n{y.value_counts().to_string()}")

# ── 5. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
# stratify=y ensures the class ratio (1.24:1) is preserved in both
# the training set and the test set.
# Without stratify, a random split might put most disease cases in train
# and leave test with almost none — your test metrics would be meaningless.
# random_state=42 means this exact split is reproducible every time you run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} rows")
print(f"Test set:  {X_test.shape[0]} rows")

# ── 6. DEFINE MODELS ─────────────────────────────────────────────────────────
# We train three models and let MLflow track all of them.
# Why three:
#   LogisticRegression → simple, interpretable, good baseline
#   RandomForest       → ensemble, handles non-linearity, robust
#   XGBoost            → gradient boosting, usually best on tabular data
# The best model on F1 score becomes the champion.
#
# class_weight='balanced' tells sklearn to internally upweight the minority
# class during training. Even though our imbalance is mild (1.24:1), it's
# good practice and costs nothing.
#
# Hyperparameters come from config.yaml — not hardcoded here.

lr_params  = config["models"]["logistic_regression"]
rf_params  = config["models"]["random_forest"]
xgb_params = config["models"]["xgboost"]

models = {
    "logistic_regression": LogisticRegression(
        **lr_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,  # default 100 often doesn't converge — 1000 is safe
    ),
    "random_forest": RandomForestClassifier(
        **rf_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "xgboost": XGBClassifier(
        **xgb_params,
        random_state=RANDOM_STATE,
        eval_metric="logloss",  # suppress XGBoost's default warning
        verbosity=0,            # suppress XGBoost console spam
    ),
}

# ── 7. MLFLOW SETUP ───────────────────────────────────────────────────────────
# An "experiment" in MLflow is a named container for related runs.
# All three models will live under the same experiment so you can compare them.
# set_experiment() creates it if it doesn't exist yet.
mlflow.set_experiment(MLFLOW_EXP_NAME)

# ── 8. TRAIN AND TRACK EACH MODEL ────────────────────────────────────────────
results = {}  # we'll store metrics here to pick the champion at the end

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")

    # mlflow.start_run() opens a new run — everything logged inside this
    # block gets attached to this run. When the block exits, the run closes.
    # run_name= gives it a human-readable label in the MLflow UI.
    with mlflow.start_run(run_name=model_name):

        # ── Log hyperparameters ──────────────────────────────────────────
        # mlflow.log_param() stores a single key-value pair as a parameter.
        # Parameters are things you SET before training — hyperparameters.
        # Metrics are things you MEASURE after training — accuracy, F1 etc.
        # MLflow keeps them separate so you can filter/sort by either.
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log all hyperparameters from config
        if model_name == "logistic_regression":
            for k, v in lr_params.items():
                mlflow.log_param(k, v)
        elif model_name == "random_forest":
            for k, v in rf_params.items():
                mlflow.log_param(k, v)
        elif model_name == "xgboost":
            for k, v in xgb_params.items():
                mlflow.log_param(k, v)

        # ── Cross-validation ─────────────────────────────────────────────
        # Before fitting on the full training set, we do 5-fold CV.
        # Why: a single train/test split might be lucky or unlucky.
        # CV trains and evaluates 5 times on different subsets of the data
        # and gives us a more reliable estimate of true model performance.
        # stratified = each fold preserves the class ratio.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()

        mlflow.log_metric("cv_f1_mean", round(cv_mean, 4))
        mlflow.log_metric("cv_f1_std",  round(cv_std, 4))
        print(f"  CV F1: {cv_mean:.4f} ± {cv_std:.4f}")

        # ── Train on full training set ───────────────────────────────────
        # After CV tells us the model is sensible, we fit on ALL training data.
        # This gives the model the maximum data to learn from before final eval.
        model.fit(X_train, y_train)

        # ── Evaluate on held-out test set ────────────────────────────────
        # The test set was never seen during training or CV.
        # These metrics are your honest estimate of real-world performance.
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # ── Log metrics ──────────────────────────────────────────────────
        # mlflow.log_metric() stores a single measured result.
        # All five metrics give a complete picture:
        #   accuracy  → overall correctness (misleading if imbalanced)
        #   f1        → harmonic mean of precision and recall (our champion metric)
        #   roc_auc   → how well the model separates classes across all thresholds
        #   precision → of patients predicted positive, how many actually are
        #   recall    → of all actual positive patients, how many did we catch
        # For clinical use, recall matters most — missing a sick patient is worse
        # than a false alarm. We use F1 as the champion metric as a balance.
        mlflow.log_metric("test_accuracy",  round(accuracy, 4))
        mlflow.log_metric("test_f1",        round(f1, 4))
        mlflow.log_metric("test_roc_auc",   round(roc_auc, 4))
        mlflow.log_metric("test_precision", round(precision, 4))
        mlflow.log_metric("test_recall",    round(recall, 4))

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

        # ── Log the model itself ─────────────────────────────────────────
        # mlflow.sklearn.log_model() saves the actual trained model object
        # inside the MLflow run. You can load any past run's model directly
        # from MLflow without needing the .pkl file.
        # artifact_path= is just the folder name inside the run's storage.
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Store results for champion selection
        results[model_name] = {
            "model":    model,
            "f1":       f1,
            "roc_auc":  roc_auc,
            "accuracy": accuracy,
            "recall":   recall,
        }

# ── 9. SELECT CHAMPION ───────────────────────────────────────────────────────
# The champion is the model with the highest F1 score on the test set.
# F1 balances precision and recall — appropriate for a medical classification
# task where both false positives and false negatives carry real cost.
champion_name   = max(results, key=lambda name: results[name]["f1"])
champion_model  = results[champion_name]["model"]
champion_metrics = results[champion_name]

print(f"\n{'='*60}")
print(f"CHAMPION: {champion_name}")
print(f"  F1:       {champion_metrics['f1']:.4f}")
print(f"  ROC-AUC:  {champion_metrics['roc_auc']:.4f}")
print(f"  Accuracy: {champion_metrics['accuracy']:.4f}")
print(f"  Recall:   {champion_metrics['recall']:.4f}")

# ── 10. SAVE CHAMPION ────────────────────────────────────────────────────────
# We save the champion model as a .pkl file separate from MLflow.
# MLflow stores models inside its own tracking directory (mlruns/).
# The champion.pkl in models/ is what FastAPI will load at startup —
# it's the "live" model in production. Clean, simple, one file.
with open(CHAMPION_PATH, "wb") as f:
    pickle.dump(champion_model, f)

print(f"\nChampion model saved → {CHAMPION_PATH}")

# ── 11. SAVE METADATA ────────────────────────────────────────────────────────
# metadata.json records which model is in production and what metrics it has.
# This is human-readable — you can open it anytime and see exactly what's
# deployed without opening MLflow or loading the pickle.
import json
metadata = {
    "champion_model":  champion_name,
    "test_f1":         round(champion_metrics["f1"], 4),
    "test_roc_auc":    round(champion_metrics["roc_auc"], 4),
    "test_accuracy":   round(champion_metrics["accuracy"], 4),
    "test_recall":     round(champion_metrics["recall"], 4),
    "trained_on_rows": len(X_train),
    "feature_columns": list(X.columns),
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved → {METADATA_PATH}")
print(f"\nAll runs tracked in MLflow. To view:")
print(f"  mlflow ui")
print(f"  Then open http://localhost:5000")