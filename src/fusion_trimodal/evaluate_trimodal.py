
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =================================================
# CONFIGURATION
# =================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from src.fusion.model import AdaptiveFusionModel

DATA_DIR = ROOT / "data" / "fusion_trimodal"

MODEL_DIR = ROOT / "models" / "fusion_trimodal"

RESULTS_DIR = ROOT / "results" / "xai_trimodal"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
THRESHOLD = 0.5

print("Device:", DEVICE)

# =================================================
# LOAD TRAINING CONFIGURATION
# =================================================

with open(
    RESULTS_DIR / "training_summary.json",
    "r"
) as f:

    config = json.load(f)

FEATURES = config["feature_columns"]

SEQ_LEN = config["sequence_length"]

NUM_FEATURES = config["num_features"]

assert NUM_FEATURES == 25

assert len(FEATURES) == NUM_FEATURES

# =================================================
# LOAD DATASETS
# =================================================

train_df = pd.read_csv(
    DATA_DIR / "train.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "val.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "test.csv"
)

train_sessions = set(
    train_df["session_id"]
)

val_sessions = set(
    val_df["session_id"]
)

test_sessions = set(
    test_df["session_id"]
)

assert train_sessions.isdisjoint(val_sessions)

assert train_sessions.isdisjoint(test_sessions)

assert val_sessions.isdisjoint(test_sessions)

print("Session overlap: None")

# =================================================
# LOAD TRAINED SCALER
# =================================================

scaler = joblib.load(

    MODEL_DIR / "feature_scaler.pkl"
)

assert list(
    scaler.feature_names_in_
) == FEATURES

# =================================================
# LOAD TRAINED MODEL
# =================================================

model = AdaptiveFusionModel(

    num_features=NUM_FEATURES,

    hidden_size=config["hidden_size"],

    num_classes=config["num_classes"],

    dropout=config["dropout"]

).to(DEVICE)

model.load_state_dict(

    torch.load(

        MODEL_DIR /
        "fusion_trimodal_model.pth",

        map_location=DEVICE,

        weights_only=True
    )
)

model.eval()

print(
    "Three-modality model loaded successfully."
)

# =================================================
# CREATE SESSION-AWARE TEST WINDOWS
# =================================================

windows = []
labels = []
session_ids = []
window_starts = []

for session_id, group in test_df.groupby(

    "session_id",

    sort=False
):

    assert group["label"].nunique() == 1

    values = scaler.transform(

        group[FEATURES]

    ).astype(np.float32)

    label = int(
        group["label"].iloc[0]
    )

    for start in range(

        len(values) - SEQ_LEN + 1
    ):

        windows.append(

            values[start:start + SEQ_LEN]
        )

        labels.append(label)

        session_ids.append(session_id)

        window_starts.append(start)

X_test = torch.tensor(

    np.asarray(windows),

    dtype=torch.float32
)

y_test = np.asarray(
    labels,
    dtype=np.int64
)

assert torch.isfinite(X_test).all()

print("Test sessions:", len(test_sessions))

print("Test windows:", len(X_test))

print("Input shape:", X_test.shape)

# =================================================
# MODEL INFERENCE
# =================================================

probabilities = []

logit_margins = []

with torch.inference_mode():

    for start in range(

        0,

        len(X_test),

        BATCH_SIZE
    ):

        batch = X_test[

            start:start + BATCH_SIZE

        ].to(DEVICE)

        logits, attention = model(batch)

        probs = torch.softmax(

            logits,

            dim=1
        )[:, 1]

        margins = logits[:, 1] - logits[:, 0]

        probabilities.extend(

            probs.cpu().numpy()
        )

        logit_margins.extend(

            margins.cpu().numpy()
        )

probabilities = np.asarray(probabilities)

predictions = (

    probabilities >= THRESHOLD

).astype(int)

# =================================================
# SAVE WINDOW-LEVEL PREDICTIONS
# =================================================

window_df = pd.DataFrame({

    "session_id": session_ids,

    "window_start": window_starts,

    "true_label": y_test,

    "class_1_probability": probabilities,

    "logit_margin": logit_margins,

    "predicted_label": predictions
})

window_df.to_csv(

    RESULTS_DIR /
    "test_window_predictions.csv",

    index=False
)

# =================================================
# SESSION-LEVEL AGGREGATION
# =================================================

session_df = window_df.groupby(

    "session_id",

    as_index=False

).agg(

    true_label=("true_label", "first"),

    mean_probability=(
        "class_1_probability", "mean"
    ),

    probability_std=(
        "class_1_probability", "std"
    ),

    mean_logit_margin=(
        "logit_margin", "mean"
    ),

    windows=("window_start", "count")
)

session_df["predicted_label"] = (

    session_df["mean_probability"] >= THRESHOLD

).astype(int)

session_df["correct"] = (

    session_df["true_label"]

    == session_df["predicted_label"]
)

session_df.to_csv(

    RESULTS_DIR /
    "test_session_predictions.csv",

    index=False
)

# =================================================
# METRIC CALCULATION
# =================================================

def calculate_metrics(y_true, y_pred, y_prob):

    return {

        "accuracy": float(

            accuracy_score(y_true, y_pred)
        ),

        "precision_macro": float(

            precision_score(

                y_true,

                y_pred,

                average="macro",

                zero_division=0
            )
        ),

        "recall_macro": float(

            recall_score(

                y_true,

                y_pred,

                average="macro",

                zero_division=0
            )
        ),

        "f1_macro": float(

            f1_score(

                y_true,

                y_pred,

                average="macro",

                zero_division=0
            )
        ),

        "roc_auc": float(

            roc_auc_score(y_true, y_prob)
        )
    }


window_metrics = calculate_metrics(

    y_test,

    predictions,

    probabilities
)

session_metrics = calculate_metrics(

    session_df["true_label"],

    session_df["predicted_label"],

    session_df["mean_probability"]
)

# =================================================
# CONFUSION MATRICES
# =================================================

def save_confusion_matrix(

    y_true,

    y_pred,

    title,

    filename
):

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]
    )

    fig, ax = plt.subplots(

        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix,

        display_labels=[

            "Synthetic class 0",

            "Synthetic class 1"
        ]
    )

    display.plot(

        ax=ax,

        cmap="Blues",

        colorbar=False,

        values_format="d"
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(

        RESULTS_DIR / filename,

        dpi=300
    )

    plt.close()

    return matrix.tolist()


window_cm = save_confusion_matrix(

    y_test,

    predictions,

    "Three-Modality Window-Level Confusion Matrix",

    "test_window_confusion_matrix.png"
)

session_cm = save_confusion_matrix(

    session_df["true_label"],

    session_df["predicted_label"],

    "Three-Modality Session-Level Confusion Matrix",

    "test_session_confusion_matrix.png"
)

# =================================================
# ROC CURVE
# =================================================

window_fpr, window_tpr, _ = roc_curve(

    y_test,

    probabilities
)

session_fpr, session_tpr, _ = roc_curve(

    session_df["true_label"],

    session_df["mean_probability"]
)

plt.figure(figsize=(7, 6))

plt.plot(

    window_fpr,

    window_tpr,

    linewidth=2,

    label=(
        f"Window AUC: "
        f"{window_metrics['roc_auc']:.4f}"
    )
)

plt.plot(

    session_fpr,

    session_tpr,

    linewidth=2,

    linestyle="--",

    label=(
        f"Session AUC: "
        f"{session_metrics['roc_auc']:.4f}"
    )
)

plt.plot(

    [0, 1],

    [0, 1],

    linestyle=":",

    color="gray"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("CogniView: Three-Modality ROC Curve")

plt.legend(loc="lower right")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "test_roc_curve.png",

    dpi=300
)

plt.close()

# =================================================
# SESSION PROBABILITY VISUALIZATION
# =================================================

colors = [

    "#27966B" if correct else "#D97706"

    for correct in session_df["correct"]
]

plt.figure(figsize=(9, 5))

bars = plt.bar(

    session_df["session_id"],

    session_df["mean_probability"],

    color=colors
)

plt.axhline(

    THRESHOLD,

    linestyle="--",

    color="black",

    label="Classification threshold"
)

for bar, probability in zip(

    bars,

    session_df["mean_probability"]
):

    plt.text(

        bar.get_x() + bar.get_width() / 2,

        bar.get_height() + 0.025,

        f"{probability:.3f}",

        ha="center"
    )

plt.ylim(0, 1.12)

plt.xlabel("Test session")

plt.ylabel("Mean class 1 probability")

plt.title(

    "CogniView: Three-Modality Session Predictions"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    RESULTS_DIR /
    "test_session_probabilities.png",

    dpi=300
)

plt.close()

# =================================================
# COMPARE TWO-MODALITY AND THREE-MODALITY
# =================================================

comparison = pd.DataFrame({

    "metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-score",

        "ROC-AUC"
    ],

    "two_modality": [

        1.0, 1.0, 1.0, 1.0, 1.0
    ],

    "three_modality": [

        window_metrics["accuracy"],

        window_metrics["precision_macro"],

        window_metrics["recall_macro"],

        window_metrics["f1_macro"],

        window_metrics["roc_auc"]
    ]
})

comparison["difference_percentage_points"] = (

    comparison["three_modality"]

    - comparison["two_modality"]

) * 100

comparison.to_csv(

    RESULTS_DIR / "model_comparison.csv",

    index=False
)

x = np.arange(len(comparison))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(

    x - width / 2,

    comparison["two_modality"] * 100,

    width,

    label="Two-modality",

    color="#94A3B8"
)

plt.bar(

    x + width / 2,

    comparison["three_modality"] * 100,

    width,

    label="Three-modality",

    color="#4472C4"
)

plt.xticks(

    x,

    comparison["metric"]
)

plt.ylabel("Metric value (%)")

plt.ylim(0, 110)

plt.title(

    "Two-Modality vs Three-Modality "
    "Synthetic Test Performance"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "model_comparison.png",

    dpi=300
)

plt.close()

# =================================================
# SAVE EVALUATION SUMMARY
# =================================================

summary = {

    "experiment":
        "Three-modality retinal negative control",

    "model": "AdaptiveFusionModel",

    "num_features": NUM_FEATURES,

    "sequence_length": SEQ_LEN,

    "test_sessions": len(test_sessions),

    "test_windows": len(X_test),

    "classification_threshold": THRESHOLD,

    "window_level_metrics": window_metrics,

    "session_level_metrics": session_metrics,

    "window_confusion_matrix": window_cm,

    "session_confusion_matrix": session_cm,

    "session_leakage": False,

    "test_evaluated": True,

    "data_source": "synthetic",

    "retinal_negative_control": True,

    "real_retinal_used_for_training": False,

    "real_plr_used_for_training": False,

    "clinical_validation": False,

    "limitations": [

        "Only four synthetic test sessions",

        "Overlapping windows are not independent",

        "Random retinal features may introduce "
        "spurious associations",

        "Synthetic PLR distributions are "
        "class-dependent",

        "Test sessions were previously evaluated "
        "in the two-modality experiment",

        "No matched clinical multimodal subjects",

        "No clinical Alzheimer's validation"
    ]
}

with open(

    RESULTS_DIR / "evaluation_metrics.json",

    "w"

) as f:

    json.dump(summary, f, indent=4)

# =================================================
# PRINT FINAL RESULTS
# =================================================

print(
    "\n========== WINDOW-LEVEL METRICS =========="
)

for name, value in window_metrics.items():

    print(f"{name}: {value:.4f}")

print(
    "\n========== SESSION-LEVEL METRICS =========="
)

for name, value in session_metrics.items():

    print(f"{name}: {value:.4f}")

print(
    "\n========== SESSION PREDICTIONS =========="
)

print(
    session_df.to_string(index=False)
)

print(
    "\n========== MODEL COMPARISON =========="
)

print(
    comparison.to_string(index=False)
)

print("\nWindow confusion matrix:")

print(np.asarray(window_cm))

print("\nSession confusion matrix:")

print(np.asarray(session_cm))

print(
    "\n========== EVALUATION COMPLETE =========="
)

print("Test sessions:", len(test_sessions))

print("Test windows:", len(X_test))

print("Data source: SYNTHETIC")

print("Retinal negative control: True")

print("Clinical validation: False")

print("Results saved to:", RESULTS_DIR)
