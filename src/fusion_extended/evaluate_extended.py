
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    classification_report
)

# -------------------------------------------------
# Configuration
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fusion.model import AdaptiveFusionModel

DATA_DIR = ROOT / "data" / "fusion_extended"
MODEL_DIR = ROOT / "models" / "fusion_extended"
OUT_DIR = ROOT / "results" / "xai_multimodal"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# -------------------------------------------------
# Load saved configuration
# -------------------------------------------------

with open(OUT_DIR / "training_summary.json") as f:
    config = json.load(f)

FEATURES = config["feature_columns"]
SEQ_LEN = config["sequence_length"]

# -------------------------------------------------
# Load datasets and check separation
# -------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

train_ids = set(train_df["session_id"])
val_ids = set(val_df["session_id"])
test_ids = set(test_df["session_id"])

assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)
assert train_ids.isdisjoint(val_ids)

print("Session overlap: None")

# -------------------------------------------------
# Load training-only scaler
# -------------------------------------------------

scaler = joblib.load(
    MODEL_DIR / "feature_scaler.pkl"
)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------

model = AdaptiveFusionModel(
    num_features=config["num_features"],
    hidden_size=config["hidden_size"],
    num_classes=config["num_classes"],
    dropout=config["dropout"]
).to(DEVICE)

checkpoint = torch.load(
    MODEL_DIR / "fusion_extended_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Extended model loaded successfully.")

# -------------------------------------------------
# Build test windows
# -------------------------------------------------

samples = []
labels = []
session_ids = []
window_starts = []

for session_id, group in test_df.groupby(
    "session_id", sort=False
):

    assert group["label"].nunique() == 1

    x = scaler.transform(
        group[FEATURES]
    ).astype(np.float32)

    label = int(group["label"].iloc[0])

    if len(x) < SEQ_LEN:
        raise ValueError(
            f"Session {session_id} is too short"
        )

    for start in range(len(x) - SEQ_LEN + 1):

        samples.append(
            x[start:start + SEQ_LEN]
        )

        labels.append(label)
        session_ids.append(session_id)
        window_starts.append(start)

X = torch.tensor(
    np.stack(samples),
    dtype=torch.float32
)

Y = torch.tensor(
    labels,
    dtype=torch.long
)

loader = DataLoader(
    TensorDataset(X, Y),
    batch_size=32,
    shuffle=False
)

print("Test sessions:", len(test_ids))
print("Test windows:", len(X))
print("Input shape:", X.shape)

# -------------------------------------------------
# Generate predictions
# -------------------------------------------------

probabilities = []

with torch.no_grad():

    for x, y in loader:

        x = x.to(DEVICE)

        logits, attention = model(x)

        prob = torch.softmax(
            logits, dim=1
        )[:, 1]

        probabilities.extend(
            prob.cpu().numpy()
        )

window_df = pd.DataFrame({
    "session_id": session_ids,
    "window_start": window_starts,
    "true_label": labels,
    "class_1_probability": probabilities
})

window_df["predicted_label"] = (
    window_df["class_1_probability"] >= 0.5
).astype(int)

window_df.to_csv(
    OUT_DIR / "test_window_predictions.csv",
    index=False
)

# -------------------------------------------------
# Session-level prediction
# -------------------------------------------------

session_df = window_df.groupby(
    "session_id", as_index=False
).agg(
    true_label=("true_label", "first"),
    mean_probability=("class_1_probability", "mean"),
    probability_std=("class_1_probability", "std"),
    windows=("true_label", "size")
)

session_df["predicted_label"] = (
    session_df["mean_probability"] >= 0.5
).astype(int)

session_df["correct"] = (
    session_df["true_label"]
    == session_df["predicted_label"]
)

session_df.to_csv(
    OUT_DIR / "test_session_predictions.csv",
    index=False
)

# -------------------------------------------------
# Evaluation metrics
# -------------------------------------------------

def calculate_metrics(y_true, y_pred, y_prob):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision_macro": float(
            precision_score(
                y_true, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),

        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        )
    }


window_metrics = calculate_metrics(
    window_df["true_label"],
    window_df["predicted_label"],
    window_df["class_1_probability"]
)

session_metrics = calculate_metrics(
    session_df["true_label"],
    session_df["predicted_label"],
    session_df["mean_probability"]
)

# -------------------------------------------------
# Confusion matrices
# -------------------------------------------------

def save_confusion(y_true, y_pred, name, title):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Synthetic class 0",
            "Synthetic class 1"
        ]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / name,
        dpi=300
    )

    plt.close()

    return cm.tolist()


window_cm = save_confusion(
    window_df["true_label"],
    window_df["predicted_label"],
    "test_window_confusion_matrix.png",
    "CogniView: Window-Level Confusion Matrix"
)

session_cm = save_confusion(
    session_df["true_label"],
    session_df["predicted_label"],
    "test_session_confusion_matrix.png",
    "CogniView: Session-Level Confusion Matrix"
)

# -------------------------------------------------
# ROC curves
# -------------------------------------------------

plt.figure(figsize=(8, 6))

for name, y, prob, metrics in [

    (
        "Window-level",
        window_df["true_label"],
        window_df["class_1_probability"],
        window_metrics
    ),

    (
        "Session-level",
        session_df["true_label"],
        session_df["mean_probability"],
        session_metrics
    )

]:

    if metrics["roc_auc"] is not None:

        fpr, tpr, thresholds = roc_curve(y, prob)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{name} "
                f"(AUC = {metrics['roc_auc']:.4f})"
            )
        )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CogniView: Synthetic Test ROC Curves")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUT_DIR / "test_roc_curve.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Session-wise probability chart
# -------------------------------------------------

colors = [
    "#2E8B57" if correct else "#C0392B"
    for correct in session_df["correct"]
]

plt.figure(figsize=(9, 5))

plt.bar(
    session_df["session_id"],
    session_df["mean_probability"],
    color=colors,
    edgecolor="black"
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Decision threshold"
)

plt.ylim(0, 1.05)
plt.xlabel("Synthetic test session")
plt.ylabel("Mean class 1 probability")
plt.title("CogniView: Session-Level Predictions")

from matplotlib.patches import Patch

plt.legend(handles=[
    Patch(color="#2E8B57", label="Correct"),
    Patch(color="#C0392B", label="Incorrect"),
    plt.Line2D(
        [0], [0],
        linestyle="--",
        color="gray",
        label="Threshold = 0.5"
    )
])

plt.tight_layout()

plt.savefig(
    OUT_DIR / "test_session_probabilities.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Save evaluation summary
# -------------------------------------------------

summary = {
    "model": "AdaptiveFusionModel",
    "num_features": len(FEATURES),
    "sequence_length": SEQ_LEN,
    "best_epoch": config["best_epoch"],
    "test_sessions": len(test_ids),
    "test_windows": len(window_df),
    "window_metrics": window_metrics,
    "session_metrics": session_metrics,
    "window_confusion_matrix": window_cm,
    "session_confusion_matrix": session_cm,
    "decision_threshold": 0.5,
    "session_aggregation": "mean_probability",
    "data_source": "synthetic",
    "clinical_validation": False,
    "real_plr_used_for_training": False,
    "retinal_modality_included": False,
    "limitations": [
        "Only four independent synthetic test sessions",
        "Overlapping windows are not independent samples",
        "Synthetic PLR features depend on generated labels",
        "Test sessions were part of earlier project data",
        "No clinical Alzheimer's disease validation"
    ]
}

with open(
    OUT_DIR / "evaluation_metrics.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)

# -------------------------------------------------
# Print final results
# -------------------------------------------------

print("\n========== WINDOW-LEVEL EVALUATION ==========")

for key, value in window_metrics.items():
    print(
        f"{key}: {value:.4f}"
        if value is not None
        else f"{key}: unavailable"
    )

print("\n========== SESSION-LEVEL EVALUATION ==========")

for key, value in session_metrics.items():
    print(
        f"{key}: {value:.4f}"
        if value is not None
        else f"{key}: unavailable"
    )

print("\n========== SESSION PREDICTIONS ==========")

print(session_df.to_string(index=False))

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    session_df["true_label"],
    session_df["predicted_label"],
    labels=[0, 1],
    target_names=[
        "Synthetic class 0",
        "Synthetic class 1"
    ],
    zero_division=0
))

print("\n========== EVALUATION COMPLETE ==========")

print("Test sessions:", len(test_ids))
print("Test windows:", len(window_df))
print("Data source: SYNTHETIC")
print("Clinical validation: False")
print("Outputs saved to:", OUT_DIR)
