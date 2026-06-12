import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed/heart_disease_processed.csv"
CHAMPION_PATH  = "models/champion.pkl"
METADATA_PATH  = "models/metadata.json"
REPORTS_DIR    = "reports/evaluation"
TARGET_COL     = "target"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────
# We must apply the same column sanitisation as train.py.
# If we don't, the feature names won't match what the model was trained on.
from sklearn.model_selection import train_test_split

df = pd.read_csv(PROCESSED_PATH)
df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Recreate the exact same test split as train.py.
# Same test_size, same random_state, same stratify → identical test rows.
# This is why we fix random_state — so evaluate.py always sees the same
# 184 rows that train.py never trained on.
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")

# ── 3. LOAD CHAMPION MODEL ───────────────────────────────────────────────────
with open(CHAMPION_PATH, "rb") as f:
    model = pickle.load(f)

print(f"Loaded champion model: {type(model).__name__}")

# ── 4. GENERATE PREDICTIONS ──────────────────────────────────────────────────
# predict()      → hard class label (0 or 1)
# predict_proba  → probability scores; [:, 1] = probability of disease
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# ── 5. PRINT METRICS ─────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

print("\n" + "=" * 60)
print("CHAMPION MODEL EVALUATION")
print("=" * 60)
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ── 6. CONFUSION MATRIX ───────────────────────────────────────────────────────
# A confusion matrix shows the full breakdown of predictions:
#   True Negative  (TN): correctly predicted No Disease
#   False Positive (FP): predicted Disease, actually No Disease (false alarm)
#   False Negative (FN): predicted No Disease, actually Disease (missed case)
#   True Positive  (TP): correctly predicted Disease
#
# In clinical terms, FN is the most dangerous — a sick patient sent home.
# The confusion matrix makes this visible in a way a single F1 score doesn't.
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,          # show numbers inside cells
    fmt="d",             # format as integers not scientific notation
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
    ax=ax,
    linewidths=0.5,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — {type(model).__name__}", fontsize=13)

# Add TN/FP/FN/TP labels so the plot is self-explanatory in your README
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.15,
    f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
    ha="center", transform=ax.transAxes, fontsize=10, color="grey"
)

plt.tight_layout()
confusion_path = f"{REPORTS_DIR}/confusion_matrix.png"
plt.savefig(confusion_path, dpi=150)
plt.close()
print(f"\nConfusion matrix saved → {confusion_path}")

# ── 7. ROC CURVE ─────────────────────────────────────────────────────────────
# The ROC curve plots True Positive Rate vs False Positive Rate at every
# possible classification threshold (not just the default 0.5).
# A perfect model hugs the top-left corner. A random model is the diagonal.
# AUC (area under curve) summarises this in one number — higher is better.
# We use this instead of just accuracy because it's threshold-independent.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4C72B0", lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1,
        linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(f"ROC Curve — {type(model).__name__}", fontsize=13)
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
roc_path = f"{REPORTS_DIR}/roc_curve.png"
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"ROC curve saved → {roc_path}")

# ── 8. UPDATE METADATA ───────────────────────────────────────────────────────
# Load existing metadata and append evaluation results.
# This keeps a single source of truth about what's in production.
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

metadata["evaluation"] = {
    "accuracy":  round(accuracy, 4),
    "f1":        round(f1, 4),
    "roc_auc":   round(roc_auc, 4),
    "precision": round(precision, 4),
    "recall":    round(recall, 4),
    "tn": int(tn), "fp": int(fp),
    "fn": int(fn), "tp": int(tp),
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata updated → {METADATA_PATH}")
print("\nEvaluation complete.")