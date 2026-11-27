
from pathlib import Path
import sys
import json
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------------------------
# Project configuration
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fusion.model import AdaptiveFusionModel

DATA_DIR = ROOT / "data" / "fusion_extended"
MODEL_DIR = ROOT / "models" / "fusion_extended"
OUT_DIR = ROOT / "results" / "xai_multimodal"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# -------------------------------------------------
# Load datasets
# -------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

with open(DATA_DIR / "dataset_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_columns"]

assert len(FEATURES) == 17

assert set(train_df["session_id"]).isdisjoint(
    set(val_df["session_id"])
)

print("Input features:", len(FEATURES))
print("Sequence length:", SEQ_LEN)

# -------------------------------------------------
# Feature normalization
# Fit using training data only
# -------------------------------------------------

scaler = StandardScaler()

scaler.fit(train_df[FEATURES])

joblib.dump(
    scaler,
    MODEL_DIR / "feature_scaler.pkl"
)

# -------------------------------------------------
# Session-aware dataset
# -------------------------------------------------

class ExtendedFusionDataset(Dataset):

    def __init__(self, df, scaler):

        self.samples = []
        self.labels = []
        self.session_ids = []

        for session_id, group in df.groupby(
            "session_id", sort=False
        ):

            assert group["label"].nunique() == 1

            x = scaler.transform(
                group[FEATURES]
            ).astype(np.float32)

            label = int(group["label"].iloc[0])

            for start in range(
                len(x) - SEQ_LEN + 1
            ):

                end = start + SEQ_LEN

                self.samples.append(
                    x[start:end]
                )

                self.labels.append(label)

                self.session_ids.append(
                    session_id
                )

        if not self.samples:
            raise ValueError("No valid windows")

        self.samples = torch.tensor(
            np.stack(self.samples),
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.samples[index],
            self.labels[index]
        )


train_dataset = ExtendedFusionDataset(
    train_df, scaler
)

val_dataset = ExtendedFusionDataset(
    val_df, scaler
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training windows:", len(train_dataset))
print("Validation windows:", len(val_dataset))

# -------------------------------------------------
# Extended fusion model
# -------------------------------------------------

model = AdaptiveFusionModel(
    num_features=17,
    hidden_size=64,
    num_classes=2,
    dropout=0.30
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4
)

# -------------------------------------------------
# Training and validation function
# -------------------------------------------------

def run_epoch(loader, training=False):

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    predictions = []
    labels = []

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            logits, attention = model(x)

            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(y)
        total_samples += len(y)

        pred = logits.argmax(dim=1)

        predictions.extend(
            pred.detach().cpu().numpy()
        )

        labels.extend(
            y.detach().cpu().numpy()
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(
            labels, predictions
        ),
        "precision": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }

# -------------------------------------------------
# Training loop
# -------------------------------------------------

history = []

best_loss = float("inf")
best_state = None
best_epoch = 0
counter = 0

print("\n========== EXTENDED FUSION TRAINING ==========")

for epoch in range(1, EPOCHS + 1):

    train_metrics = run_epoch(
        train_loader, training=True
    )

    val_metrics = run_epoch(
        val_loader, training=False
    )

    scheduler.step(val_metrics["loss"])

    history.append({
        "epoch": epoch,

        **{
            f"train_{k}": v
            for k, v in train_metrics.items()
        },

        **{
            f"val_{k}": v
            for k, v in val_metrics.items()
        }
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f}"
    )

    if val_metrics["loss"] < best_loss:

        best_loss = val_metrics["loss"]
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state,
            MODEL_DIR / "fusion_extended_model.pth"
        )

        counter = 0

        print("Best model saved.")

    else:

        counter += 1

        if counter >= PATIENCE:

            print("Early stopping triggered.")
            break

# -------------------------------------------------
# Restore best model
# -------------------------------------------------

model.load_state_dict(best_state)

final_val = run_epoch(
    val_loader,
    training=False
)

# -------------------------------------------------
# Save training history
# -------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    OUT_DIR / "training_history.csv",
    index=False
)

# -------------------------------------------------
# Training loss graph
# -------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("CogniView: Extended Fusion Training Loss")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT_DIR / "training_loss.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Training accuracy graph
# -------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"] * 100,
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"] * 100,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CogniView: Extended Fusion Accuracy")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT_DIR / "training_accuracy.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Save model configuration
# -------------------------------------------------

summary = {
    "model": "AdaptiveFusionModel",
    "num_features": 17,
    "num_classes": 2,
    "hidden_size": 64,
    "dropout": 0.30,
    "sequence_length": SEQ_LEN,
    "feature_columns": FEATURES,
    "best_epoch": best_epoch,
    "best_validation_loss": best_loss,
    "validation_window_metrics": final_val,
    "training_sessions": int(
        train_df["session_id"].nunique()
    ),
    "validation_sessions": int(
        val_df["session_id"].nunique()
    ),
    "seed": SEED,
    "data_source": "synthetic",
    "clinical_validation": False,
    "real_plr_used_for_training": False,
    "retinal_modality_included": False,
    "test_evaluated": False
}

with open(
    OUT_DIR / "training_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)

print("\n========== TRAINING COMPLETE ==========")

print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_loss, 6))

for key, value in final_val.items():
    print(f"Validation {key}: {value:.4f}")

print("\nModel saved to:", MODEL_DIR)
print("Results saved to:", OUT_DIR)
print("Data source: SYNTHETIC")
print("Clinical validation: False")
print("Test dataset evaluated: False")
