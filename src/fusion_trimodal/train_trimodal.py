
from pathlib import Path
import sys
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 64
DROPOUT = 0.3
NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

# =================================================
# LOAD DATASET CONFIGURATION
# =================================================

with open(
    DATA_DIR / "dataset_metadata.json",
    "r"
) as f:

    metadata = json.load(f)

FEATURES = metadata["feature_columns"]

SEQ_LEN = metadata["sequence_length"]

NUM_FEATURES = len(FEATURES)

assert NUM_FEATURES == 25

assert SEQ_LEN == 30

print("Device:", DEVICE)

print("Input features:", NUM_FEATURES)

print("Sequence length:", SEQ_LEN)

# =================================================
# LOAD TRAINING AND VALIDATION DATA
# =================================================

train_df = pd.read_csv(
    DATA_DIR / "train.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "val.csv"
)

train_sessions = set(
    train_df["session_id"]
)

val_sessions = set(
    val_df["session_id"]
)

assert train_sessions.isdisjoint(
    val_sessions
)

print("Training sessions:", len(train_sessions))

print("Validation sessions:", len(val_sessions))

print("Train-validation session overlap: None")

# =================================================
# TRAINING-ONLY NORMALIZATION
# =================================================

scaler = StandardScaler()

scaler.fit(
    train_df[FEATURES]
)

joblib.dump(
    scaler,
    MODEL_DIR / "feature_scaler.pkl"
)

print("Feature scaler saved.")

# =================================================
# SESSION-AWARE WINDOW GENERATION
# =================================================

def create_windows(df):

    windows = []
    labels = []
    session_ids = []

    for session_id, group in df.groupby(
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

            window = values[
                start:start + SEQ_LEN
            ]

            windows.append(window)

            labels.append(label)

            session_ids.append(session_id)

    X = np.asarray(
        windows,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    assert np.isfinite(X).all()

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        session_ids
    )


X_train, y_train, train_ids = create_windows(
    train_df
)

X_val, y_val, val_ids = create_windows(
    val_df
)

print("Training windows:", len(X_train))

print("Validation windows:", len(X_val))

print("Training input shape:", X_train.shape)

print("Validation input shape:", X_val.shape)

train_loader = DataLoader(

    TensorDataset(X_train, y_train),

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)

val_loader = DataLoader(

    TensorDataset(X_val, y_val),

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)

# =================================================
# INITIALIZE MODEL
# =================================================

model = AdaptiveFusionModel(

    num_features=NUM_FEATURES,

    hidden_size=HIDDEN_SIZE,

    num_classes=NUM_CLASSES,

    dropout=DROPOUT

).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=4
)

CHECKPOINT = (
    MODEL_DIR / "fusion_trimodal_model.pth"
)

# =================================================
# TRAINING FUNCTION
# =================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for X, y in train_loader:

        X = X.to(DEVICE)

        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits, attention = model(X)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * len(y)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == y
        ).sum().item()

        total += len(y)

    return (

        total_loss / total,

        correct / total
    )

# =================================================
# VALIDATION FUNCTION
# =================================================

def validate():

    model.eval()

    total_loss = 0.0

    total = 0

    predictions = []

    targets = []

    probabilities = []

    with torch.inference_mode():

        for X, y in val_loader:

            X = X.to(DEVICE)

            y = y.to(DEVICE)

            logits, attention = model(X)

            loss = criterion(logits, y)

            total_loss += (
                loss.item() * len(y)
            )

            total += len(y)

            probs = torch.softmax(
                logits,
                dim=1
            )

            pred = probs.argmax(dim=1)

            predictions.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                y.cpu().numpy()
            )

            probabilities.extend(
                probs[:, 1].cpu().numpy()
            )

    metrics = {

        "loss": total_loss / total,

        "accuracy": accuracy_score(
            targets, predictions
        ),

        "precision": precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0
        )
    }

    prediction_df = pd.DataFrame({

        "session_id": val_ids,

        "true_label": targets,

        "class_1_probability": probabilities
    })

    session_df = prediction_df.groupby(
        "session_id",
        as_index=False
    ).agg(

        true_label=("true_label", "first"),

        mean_probability=(
            "class_1_probability", "mean"
        )
    )

    session_df["predicted_label"] = (

        session_df["mean_probability"] >= 0.5

    ).astype(int)

    metrics["session_accuracy"] = accuracy_score(

        session_df["true_label"],

        session_df["predicted_label"]
    )

    metrics["session_f1"] = f1_score(

        session_df["true_label"],

        session_df["predicted_label"],

        average="macro",

        zero_division=0
    )

    return metrics, session_df

# =================================================
# TRAINING LOOP
# =================================================

history = []

best_val_loss = float("inf")

best_epoch = 0

wait = 0

print(
    "\n========== TRIMODAL TRAINING STARTED =========="
)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = train_epoch()

    val_metrics, _ = validate()

    val_loss = val_metrics["loss"]

    val_acc = val_metrics["accuracy"]

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_acc,

        "val_loss": val_loss,

        "val_accuracy": val_acc,

        "val_session_accuracy":
            val_metrics["session_accuracy"],

        "learning_rate": current_lr
    })

    print(

        f"Epoch [{epoch:02d}/{EPOCHS}] | "

        f"Train Loss: {train_loss:.4f} | "

        f"Val Loss: {val_loss:.4f} | "

        f"Train Acc: {train_acc:.4f} | "

        f"Val Acc: {val_acc:.4f} | "

        f"Session Acc: "
        f"{val_metrics['session_accuracy']:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        wait = 0

        torch.save(
            model.state_dict(),
            CHECKPOINT
        )

        print("Best model saved.")

    else:

        wait += 1

    if wait >= PATIENCE:

        print("Early stopping triggered.")

        break

# =================================================
# RESTORE BEST MODEL
# =================================================

model.load_state_dict(

    torch.load(

        CHECKPOINT,

        map_location=DEVICE,

        weights_only=True
    )
)

final_metrics, session_predictions = validate()

session_predictions.to_csv(

    RESULTS_DIR /
    "validation_session_predictions.csv",

    index=False
)

# =================================================
# SAVE TRAINING HISTORY
# =================================================

history_df = pd.DataFrame(history)

history_df.to_csv(

    RESULTS_DIR / "training_history.csv",

    index=False
)

# =================================================
# CHART 1: TRAINING AND VALIDATION LOSS
# =================================================

plt.figure(figsize=(9, 5))

plt.plot(

    history_df["epoch"],

    history_df["train_loss"],

    label="Training loss",

    marker="o"
)

plt.plot(

    history_df["epoch"],

    history_df["val_loss"],

    label="Validation loss",

    marker="s"
)

plt.axvline(

    best_epoch,

    linestyle="--",

    color="gray",

    label=f"Best epoch: {best_epoch}"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "CogniView: Three-Modality Training Loss"
)

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "training_loss.png",

    dpi=300
)

plt.close()

# =================================================
# CHART 2: TRAINING AND VALIDATION ACCURACY
# =================================================

plt.figure(figsize=(9, 5))

plt.plot(

    history_df["epoch"],

    history_df["train_accuracy"],

    label="Training window accuracy",

    marker="o"
)

plt.plot(

    history_df["epoch"],

    history_df["val_accuracy"],

    label="Validation window accuracy",

    marker="s"
)

plt.plot(

    history_df["epoch"],

    history_df["val_session_accuracy"],

    label="Validation session accuracy",

    linestyle="--"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.ylim(0, 1.05)

plt.title(
    "CogniView: Three-Modality Training Accuracy"
)

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "training_accuracy.png",

    dpi=300
)

plt.close()

# =================================================
# SAVE TRAINING SUMMARY
# =================================================

summary = {

    "model": "AdaptiveFusionModel",

    "experiment":
        "Three-modality retinal negative control",

    "num_features": NUM_FEATURES,

    "num_classes": NUM_CLASSES,

    "hidden_size": HIDDEN_SIZE,

    "dropout": DROPOUT,

    "sequence_length": SEQ_LEN,

    "feature_columns": FEATURES,

    "modality_features":
        metadata["modality_features"],

    "best_epoch": best_epoch,

    "best_validation_loss": best_val_loss,

    "validation_window_metrics": {

        key: float(final_metrics[key])

        for key in [

            "loss",

            "accuracy",

            "precision",

            "recall",

            "f1"
        ]
    },

    "validation_session_metrics": {

        "accuracy": float(
            final_metrics["session_accuracy"]
        ),

        "f1": float(
            final_metrics["session_f1"]
        )
    },

    "training_sessions": len(train_sessions),

    "validation_sessions": len(val_sessions),

    "training_windows": len(X_train),

    "validation_windows": len(X_val),

    "seed": SEED,

    "data_source": "synthetic",

    "clinical_validation": False,

    "real_retinal_used_for_training": False,

    "real_plr_used_for_training": False,

    "retinal_modality_included": True,

    "retinal_negative_control": True,

    "test_evaluated": False,

    "limitations": [

        "Synthetic multimodal inputs",

        "Only 14 independent training sessions",

        "Overlapping sequence windows",

        "Random retinal features may exhibit "
        "accidental label correlations",

        "No matched clinical subjects",

        "No clinical Alzheimer's validation"
    ]
}

with open(

    RESULTS_DIR / "training_summary.json",

    "w"

) as f:

    json.dump(summary, f, indent=4)

# =================================================
# FINAL RESULTS
# =================================================

print(
    "\n========== TRIMODAL TRAINING COMPLETE =========="
)

print("Best epoch:", best_epoch)

print(
    "Best validation loss:",
    round(best_val_loss, 6)
)

for name, value in final_metrics.items():

    print(
        f"Validation {name}: {value:.4f}"
    )

print("\nValidation session predictions:")

print(
    session_predictions.to_string(index=False)
)

print("\nInput features:", NUM_FEATURES)

print("Retinal negative control: True")

print("Data source: SYNTHETIC")

print("Clinical validation: False")

print("Test evaluated: False")

print("Model saved to:", CHECKPOINT)

print("Results saved to:", RESULTS_DIR)
