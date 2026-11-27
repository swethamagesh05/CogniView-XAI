
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
    roc_auc_score
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
# Load configuration
# -------------------------------------------------

with open(OUT_DIR / "training_summary.json") as f:
    config = json.load(f)

FEATURES = config["feature_columns"]
SEQ_LEN = config["sequence_length"]

ET_INDICES = [
    i for i, name in enumerate(FEATURES)
    if not name.startswith("plr_")
]

PLR_INDICES = [
    i for i, name in enumerate(FEATURES)
    if name.startswith("plr_")
]

assert len(ET_INDICES) == 11
assert len(PLR_INDICES) == 6

# -------------------------------------------------
# Load datasets and scaler
# -------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

assert set(train_df["session_id"]).isdisjoint(
    set(test_df["session_id"])
)

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

model.load_state_dict(
    torch.load(
        MODEL_DIR / "fusion_extended_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Extended model loaded successfully.")

# -------------------------------------------------
# Training-only reference baseline
# Same baseline as Integrated Gradients
# -------------------------------------------------

reference = train_df[
    train_df["label"] == 0
]

if reference.empty:
    raise ValueError("No class 0 reference sessions")

reference_means = reference.groupby(
    "session_id"
)[FEATURES].mean()

reference_frame = pd.DataFrame(
    [reference_means.mean(axis=0)],
    columns=FEATURES
)

reference_scaled = scaler.transform(
    reference_frame
).astype(np.float32)

baseline = torch.tensor(
    np.repeat(
        reference_scaled,
        SEQ_LEN,
        axis=0
    ),
    dtype=torch.float32,
    device=DEVICE
).unsqueeze(0)

# -------------------------------------------------
# Prepare all test windows
# -------------------------------------------------

samples = []
labels = []
session_ids = []
window_starts = []

for session_id, group in test_df.groupby(
    "session_id", sort=False
):

    assert group["label"].nunique() == 1

    values = scaler.transform(
        group[FEATURES]
    ).astype(np.float32)

    label = int(group["label"].iloc[0])

    if len(values) < SEQ_LEN:
        raise ValueError(
            f"Session {session_id} too short"
        )

    for start in range(
        len(values) - SEQ_LEN + 1
    ):

        samples.append(
            values[start:start + SEQ_LEN]
        )

        labels.append(label)
        session_ids.append(session_id)
        window_starts.append(start)

X = torch.tensor(
    np.stack(samples),
    dtype=torch.float32
)

print("Test sessions:", len(set(session_ids)))
print("Test windows:", len(X))

# -------------------------------------------------
# Define ablation experiments
# -------------------------------------------------

EXPERIMENTS = {
    "Full model": [],
    "Eye-tracking retained": PLR_INDICES,
    "PLR retained": ET_INDICES,
    "Reference only": ET_INDICES + PLR_INDICES
}

# -------------------------------------------------
# Predict under each condition
# -------------------------------------------------

@torch.no_grad()
def predict_condition(replace_indices):

    probabilities = []
    margins = []

    for start in range(0, len(X), 32):

        batch = X[start:start + 32].to(
            DEVICE
        ).clone()

        if replace_indices:

            batch[:, :, replace_indices] = (
                baseline[:, :, replace_indices]
                .expand(len(batch), -1, -1)
            )

        logits, _ = model(batch)

        prob = torch.softmax(
            logits, dim=1
        )[:, 1]

        margin = logits[:, 1] - logits[:, 0]

        probabilities.extend(
            prob.cpu().numpy()
        )

        margins.extend(
            margin.cpu().numpy()
        )

    return probabilities, margins


def calculate_metrics(y_true, y_prob):

    y_pred = (
        np.asarray(y_prob) >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        )
    }


metric_records = []
window_records = []
session_records = []

print("\n========== MODALITY ABLATION ==========")

for name, replace_indices in EXPERIMENTS.items():

    probabilities, margins = predict_condition(
        replace_indices
    )

    windows = pd.DataFrame({
        "experiment": name,
        "session_id": session_ids,
        "window_start": window_starts,
        "true_label": labels,
        "probability": probabilities,
        "logit_margin": margins
    })

    windows["predicted_label"] = (
        windows["probability"] >= 0.5
    ).astype(int)

    sessions = windows.groupby(
        "session_id", as_index=False
    ).agg(
        true_label=("true_label", "first"),
        mean_probability=("probability", "mean"),
        mean_logit_margin=("logit_margin", "mean")
    )

    sessions["experiment"] = name

    sessions["predicted_label"] = (
        sessions["mean_probability"] >= 0.5
    ).astype(int)

    wm = calculate_metrics(
        windows["true_label"],
        windows["probability"]
    )

    sm = calculate_metrics(
        sessions["true_label"],
        sessions["mean_probability"]
    )

    metric_records.append({
        "experiment": name,

        **{
            f"window_{k}": v
            for k, v in wm.items()
        },

        **{
            f"session_{k}": v
            for k, v in sm.items()
        }
    })

    window_records.append(windows)
    session_records.append(sessions)

    print(f"\n{name}")
    print(
        f"Window accuracy: "
        f"{wm['accuracy']:.4f}"
    )
    print(
        f"Session accuracy: "
        f"{sm['accuracy']:.4f}"
    )
    print(
        f"Session F1: {sm['f1']:.4f}"
    )
    print(
        f"Session ROC-AUC: "
        f"{sm['roc_auc']:.4f}"
    )

# -------------------------------------------------
# Consolidate results
# -------------------------------------------------

metrics = pd.DataFrame(metric_records)

all_windows = pd.concat(
    window_records,
    ignore_index=True
)

all_sessions = pd.concat(
    session_records,
    ignore_index=True
)

full_accuracy = float(
    metrics.loc[
        metrics["experiment"] == "Full model",
        "session_accuracy"
    ].iloc[0]
)

metrics["session_accuracy_drop_pp"] = (
    full_accuracy - metrics["session_accuracy"]
) * 100

metrics.to_csv(
    OUT_DIR / "modality_ablation.csv",
    index=False
)

all_windows.to_csv(
    OUT_DIR / "ablation_window_predictions.csv",
    index=False
)

all_sessions.to_csv(
    OUT_DIR / "ablation_session_predictions.csv",
    index=False
)

# -------------------------------------------------
# Prediction sensitivity analysis
# -------------------------------------------------

pivot = all_sessions.pivot(
    index="session_id",
    columns="experiment",
    values="mean_probability"
)

margin_pivot = all_sessions.pivot(
    index="session_id",
    columns="experiment",
    values="mean_logit_margin"
)

sensitivity = pd.DataFrame({
    "session_id": pivot.index,

    "plr_removal_probability_change":
        (
            pivot["Full model"]
            - pivot["Eye-tracking retained"]
        ).values,

    "eye_tracking_removal_probability_change":
        (
            pivot["Full model"]
            - pivot["PLR retained"]
        ).values,

    "plr_removal_margin_change":
        (
            margin_pivot["Full model"]
            - margin_pivot["Eye-tracking retained"]
        ).values,

    "eye_tracking_removal_margin_change":
        (
            margin_pivot["Full model"]
            - margin_pivot["PLR retained"]
        ).values
})

sensitivity.to_csv(
    OUT_DIR / "modality_sensitivity.csv",
    index=False
)

# -------------------------------------------------
# Chart 1: Ablation accuracy comparison
# -------------------------------------------------

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width / 2,
    metrics["window_accuracy"] * 100,
    width,
    label="Window-level",
    color="#4472C4"
)

bars2 = ax.bar(
    x + width / 2,
    metrics["session_accuracy"] * 100,
    width,
    label="Session-level",
    color="#2E8B57"
)

ax.bar_label(
    bars1,
    fmt="%.1f%%",
    padding=3
)

ax.bar_label(
    bars2,
    fmt="%.1f%%",
    padding=3
)

ax.set_xticks(x)

ax.set_xticklabels(
    metrics["experiment"],
    rotation=12
)

ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)

ax.set_title(
    "CogniView: Modality Ablation Analysis\n"
    "Synthetic Demonstration"
)

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "modality_ablation.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Chart 2: Session probability comparison
# -------------------------------------------------

ordered = list(EXPERIMENTS.keys())

pivot[ordered].plot(
    kind="bar",
    figsize=(12, 6),
    color=[
        "#4472C4",
        "#2E8B57",
        "#E69F00",
        "#808080"
    ]
)

plt.axhline(
    0.5,
    color="black",
    linestyle="--"
)

plt.xlabel("Synthetic test session")
plt.ylabel("Mean class 1 probability")
plt.ylim(0, 1.12)

plt.title(
    "CogniView: Session Predictions "
    "Under Modality Ablation"
)

plt.xticks(rotation=0)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig(
    OUT_DIR / "ablation_session_probabilities.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Chart 3: Logit margin sensitivity
# -------------------------------------------------

plt.figure(figsize=(10, 6))

x = np.arange(len(sensitivity))

plt.bar(
    x - 0.18,
    sensitivity["eye_tracking_removal_margin_change"],
    0.36,
    label="Eye-tracking replaced",
    color="#4472C4"
)

plt.bar(
    x + 0.18,
    sensitivity["plr_removal_margin_change"],
    0.36,
    label="PLR replaced",
    color="#2E8B57"
)

plt.axhline(
    0,
    color="black",
    linewidth=1
)

plt.xticks(
    x,
    sensitivity["session_id"]
)

plt.ylabel("Original minus ablated logit margin")

plt.title(
    "CogniView: Modality Prediction Sensitivity"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_DIR / "modality_sensitivity.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Save summary
# -------------------------------------------------

summary = {
    "experiment_type":
        "inference_time_modality_replacement",

    "reference":
        "training_only_class_0_session_mean",

    "test_sessions": len(set(session_ids)),
    "test_windows": len(X),

    "results": metrics.to_dict(
        orient="records"
    ),

    "mean_absolute_plr_margin_change": float(
        sensitivity[
            "plr_removal_margin_change"
        ].abs().mean()
    ),

    "mean_absolute_eye_tracking_margin_change": float(
        sensitivity[
            "eye_tracking_removal_margin_change"
        ].abs().mean()
    ),

    "data_source": "synthetic",
    "clinical_validation": False,
    "real_plr_used_for_training": False,
    "retinal_modality_included": False,

    "limitations": [
        "Only four synthetic test sessions",
        "Reference replacement may produce "
        "out-of-distribution combinations",
        "Not independently trained unimodal models",
        "Synthetic class-dependent PLR distributions",
        "Not clinical biomarker validation"
    ]
}

with open(
    OUT_DIR / "ablation_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)

# -------------------------------------------------
# Final output
# -------------------------------------------------

print("\n========== ABLATION COMPARISON ==========")

print(
    metrics[[
        "experiment",
        "window_accuracy",
        "session_accuracy",
        "session_f1",
        "session_roc_auc",
        "session_accuracy_drop_pp"
    ]].round(4).to_string(index=False)
)

print("\n========== SESSION PROBABILITIES ==========")

print(
    pivot[ordered].round(6).to_string()
)

print("\n========== MODALITY SENSITIVITY ==========")

print(
    sensitivity.round(6).to_string(index=False)
)

print(
    "\nMean absolute eye-tracking margin change:",
    summary[
        "mean_absolute_eye_tracking_margin_change"
    ]
)

print(
    "Mean absolute PLR margin change:",
    summary[
        "mean_absolute_plr_margin_change"
    ]
)

print("\n========== ABLATION COMPLETE ==========")

print("Test sessions:", len(set(session_ids)))
print("Test windows:", len(X))
print("Data source: SYNTHETIC")
print("Clinical validation: False")
print("Outputs saved to:", OUT_DIR)
