
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
# LOAD CONFIGURATION
# =================================================

with open(
    RESULTS_DIR / "training_summary.json",
    "r"
) as f:
    config = json.load(f)

FEATURES = config["feature_columns"]

SEQ_LEN = config["sequence_length"]

GROUPS = {
    modality: [
        FEATURES.index(feature)
        for feature in feature_names
    ]
    for modality, feature_names
    in config["modality_features"].items()
}

assert len(FEATURES) == 25

assert len(GROUPS["eye_tracking"]) == 11
assert len(GROUPS["plr"]) == 6
assert len(GROUPS["retinal"]) == 8

assert sorted(
    index
    for indices in GROUPS.values()
    for index in indices
) == list(range(len(FEATURES)))

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
# LOAD MODEL AND SCALER
# =================================================

scaler = joblib.load(
    MODEL_DIR / "feature_scaler.pkl"
)

assert list(
    scaler.feature_names_in_
) == FEATURES

model = AdaptiveFusionModel(
    num_features=len(FEATURES),
    hidden_size=config["hidden_size"],
    num_classes=config["num_classes"],
    dropout=config["dropout"]
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "fusion_trimodal_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Three-modality model loaded.")

# =================================================
# TRAINING-ONLY CLASS 0 REFERENCE
# =================================================

reference = train_df[
    train_df["label"] == 0
]

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

reference_vector = torch.tensor(
    reference_scaled[0],
    dtype=torch.float32
)

# =================================================
# CREATE TEST WINDOWS
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
# ABLATION CONFIGURATIONS
# =================================================

EXPERIMENTS = {
    "Full model": [
        "eye_tracking", "plr", "retinal"
    ],

    "Without retinal": [
        "eye_tracking", "plr"
    ],

    "Without PLR": [
        "eye_tracking", "retinal"
    ],

    "Without eye-tracking": [
        "plr", "retinal"
    ],

    "Eye-tracking only": [
        "eye_tracking"
    ],

    "PLR only": [
        "plr"
    ],

    "Retinal only": [
        "retinal"
    ],

    "Reference only": []
}

# =================================================
# MODEL INFERENCE
# =================================================

def predict(X):

    probabilities = []
    margins = []

    with torch.inference_mode():

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            batch = X[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            logits, _ = model(batch)

            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            margin = (
                logits[:, 1] - logits[:, 0]
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            margins.extend(
                margin.cpu().numpy()
            )

    return (
        np.asarray(probabilities),
        np.asarray(margins)
    )

# =================================================
# METRIC CALCULATION
# =================================================

def calculate_metrics(y_true, y_prob):

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "f1": float(
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

# =================================================
# EXECUTE ALL ABLATION EXPERIMENTS
# =================================================

experiment_records = []
all_window_predictions = []
all_session_predictions = []

print("\n========== MODALITY ABLATION ==========")

for experiment, retained in EXPERIMENTS.items():

    X_ablated = X_test.clone()

    removed = [
        modality
        for modality in GROUPS
        if modality not in retained
    ]

    for modality in removed:

        indices = GROUPS[modality]

        X_ablated[:, :, indices] = (
            reference_vector[indices]
            .view(1, 1, -1)
        )

    probabilities, margins = predict(
        X_ablated
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    window_df = pd.DataFrame({
        "experiment": experiment,
        "session_id": session_ids,
        "window_start": window_starts,
        "true_label": y_test,
        "class_1_probability": probabilities,
        "logit_margin": margins,
        "predicted_label": predictions
    })

    session_df = window_df.groupby(
        "session_id",
        as_index=False
    ).agg(
        true_label=("true_label", "first"),
        mean_probability=(
            "class_1_probability", "mean"
        ),
        mean_logit_margin=(
            "logit_margin", "mean"
        ),
        windows=("window_start", "count")
    )

    session_df["experiment"] = experiment

    session_df["predicted_label"] = (
        session_df["mean_probability"]
        >= THRESHOLD
    ).astype(int)

    session_df["correct"] = (
        session_df["predicted_label"]
        == session_df["true_label"]
    )

    window_metrics = calculate_metrics(
        y_test,
        probabilities
    )

    session_metrics = calculate_metrics(
        session_df["true_label"].to_numpy(),
        session_df["mean_probability"].to_numpy()
    )

    record = {
        "experiment": experiment,
        "retained_modalities": ",".join(retained),
        "removed_modalities": ",".join(removed)
    }

    for key, value in window_metrics.items():

        record[f"window_{key}"] = value

    for key, value in session_metrics.items():

        record[f"session_{key}"] = value

    experiment_records.append(record)

    all_window_predictions.append(window_df)

    all_session_predictions.append(session_df)

    print(
        f"{experiment:24s} | "
        f"Window Acc: "
        f"{window_metrics['accuracy']:.4f} | "
        f"Session Acc: "
        f"{session_metrics['accuracy']:.4f}"
    )

# =================================================
# SAVE EXPERIMENT RESULTS
# =================================================

results_df = pd.DataFrame(
    experiment_records
)

window_results = pd.concat(
    all_window_predictions,
    ignore_index=True
)

session_results = pd.concat(
    all_session_predictions,
    ignore_index=True
)

full_result = results_df.iloc[0]

results_df["window_accuracy_change_pp"] = (
    results_df["window_accuracy"]
    - full_result["window_accuracy"]
) * 100

results_df["session_accuracy_change_pp"] = (
    results_df["session_accuracy"]
    - full_result["session_accuracy"]
) * 100

results_df.to_csv(
    RESULTS_DIR / "modality_ablation.csv",
    index=False
)

window_results.to_csv(
    RESULTS_DIR / "ablation_window_predictions.csv",
    index=False
)

session_results.to_csv(
    RESULTS_DIR / "ablation_session_predictions.csv",
    index=False
)

# =================================================
# VERIFY FULL MODEL REPRODUCIBILITY
# =================================================

with open(
    RESULTS_DIR / "evaluation_metrics.json",
    "r"
) as f:

    previous_evaluation = json.load(f)

expected_accuracy = previous_evaluation[
    "window_level_metrics"
]["accuracy"]

assert np.isclose(
    full_result["window_accuracy"],
    expected_accuracy,
    atol=1e-6
)

expected_session_accuracy = previous_evaluation[
    "session_level_metrics"
]["accuracy"]

assert np.isclose(
    full_result["session_accuracy"],
    expected_session_accuracy,
    atol=1e-6
)

print("\nFull model evaluation reproduced.")

# =================================================
# SESSION PROBABILITY COMPARISON
# =================================================

probability_table = session_results.pivot(
    index="session_id",
    columns="experiment",
    values="mean_probability"
)

probability_table = probability_table[
    list(EXPERIMENTS)
]

probability_table.to_csv(
    RESULTS_DIR / "ablation_probability_table.csv"
)

margin_table = session_results.pivot(
    index="session_id",
    columns="experiment",
    values="mean_logit_margin"
)

# =================================================
# MODALITY SENSITIVITY
# =================================================

REMOVAL_NAMES = {
    "eye_tracking": "Without eye-tracking",
    "plr": "Without PLR",
    "retinal": "Without retinal"
}

sensitivity_records = []

for session_id in probability_table.index:

    for modality, experiment in REMOVAL_NAMES.items():

        full_probability = float(
            probability_table.loc[
                session_id, "Full model"
            ]
        )

        ablated_probability = float(
            probability_table.loc[
                session_id, experiment
            ]
        )

        full_margin = float(
            margin_table.loc[
                session_id, "Full model"
            ]
        )

        ablated_margin = float(
            margin_table.loc[
                session_id, experiment
            ]
        )

        sensitivity_records.append({
            "session_id": session_id,
            "removed_modality": modality,
            "full_probability": full_probability,
            "ablated_probability": ablated_probability,
            "probability_change": (
                ablated_probability - full_probability
            ),
            "full_margin": full_margin,
            "ablated_margin": ablated_margin,
            "margin_change": (
                ablated_margin - full_margin
            )
        })

sensitivity_df = pd.DataFrame(
    sensitivity_records
)

sensitivity_df.to_csv(
    RESULTS_DIR / "modality_sensitivity.csv",
    index=False
)

# =================================================
# TRIAL 24 CASE STUDY
# =================================================

trial24 = session_results[
    session_results["session_id"] == "trial24"
].copy()

trial24.to_csv(
    RESULTS_DIR / "trial24_ablation_analysis.csv",
    index=False
)

# =================================================
# CHART 1: ABLATION PERFORMANCE
# =================================================

x = np.arange(len(results_df))

width = 0.36

plt.figure(figsize=(13, 6))

plt.bar(
    x - width / 2,
    results_df["window_accuracy"] * 100,
    width,
    label="Window accuracy",
    color="#4472C4"
)

plt.bar(
    x + width / 2,
    results_df["session_accuracy"] * 100,
    width,
    label="Session accuracy",
    color="#27966B"
)

plt.xticks(
    x,
    results_df["experiment"],
    rotation=35,
    ha="right"
)

plt.ylabel("Accuracy (%)")

plt.ylim(0, 110)

plt.title(
    "CogniView: Three-Modality Ablation Analysis"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "modality_ablation.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 2: SESSION PROBABILITY HEATMAP
# =================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)

matrix = probability_table.to_numpy()

image = ax.imshow(
    matrix,
    cmap="viridis",
    vmin=0,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(
    np.arange(len(probability_table.columns))
)

ax.set_xticklabels(
    probability_table.columns,
    rotation=35,
    ha="right"
)

ax.set_yticks(
    np.arange(len(probability_table.index))
)

ax.set_yticklabels(
    probability_table.index
)

for i in range(matrix.shape[0]):

    for j in range(matrix.shape[1]):

        value = matrix[i, j]

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=(
                "black"
                if value > 0.65
                else "white"
            )
        )

fig.colorbar(
    image,
    ax=ax,
    label="Class 1 probability"
)

ax.set_title(
    "Session Predictions Under Modality Ablation"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "ablation_probability_heatmap.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 3: TRIAL 24 ERROR INVESTIGATION
# =================================================

plt.figure(figsize=(12, 6))

colors = [
    "#27966B" if correct else "#D97706"
    for correct in trial24["correct"]
]

bars = plt.bar(
    trial24["experiment"],
    trial24["mean_probability"],
    color=colors
)

plt.axhline(
    THRESHOLD,
    color="black",
    linestyle="--",
    label="Classification threshold"
)

for bar, probability in zip(
    bars,
    trial24["mean_probability"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{probability:.4f}",
        ha="center",
        fontsize=9
    )

plt.xticks(
    rotation=35,
    ha="right"
)

plt.ylim(0, 1.15)

plt.ylabel("Mean class 1 probability")

plt.title(
    "Trial 24: Prediction Under Modality Ablation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "trial24_ablation_analysis.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 4: MODALITY SENSITIVITY
# =================================================

sensitivity_pivot = sensitivity_df.pivot(
    index="session_id",
    columns="removed_modality",
    values="margin_change"
)

sensitivity_pivot = sensitivity_pivot[
    ["eye_tracking", "plr", "retinal"]
]

sensitivity_pivot.plot(
    kind="bar",
    figsize=(10, 6),
    color=[
        "#27966B",
        "#D6A23B",
        "#4472C4"
    ]
)

plt.axhline(
    0,
    color="black",
    linewidth=1
)

plt.ylabel(
    "Change in logit margin after removal"
)

plt.xlabel("Test session")

plt.title(
    "Sensitivity to Modality Reference Replacement"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "modality_sensitivity.png",
    dpi=300
)

plt.close()

# =================================================
# SAVE SUMMARY
# =================================================

trial24_full = trial24[
    trial24["experiment"] == "Full model"
].iloc[0]

trial24_without_retinal = trial24[
    trial24["experiment"] == "Without retinal"
].iloc[0]

mean_absolute_margin_changes = {
    modality: float(
        sensitivity_df.loc[
            sensitivity_df["removed_modality"]
            == modality,
            "margin_change"
        ].abs().mean()
    )
    for modality in GROUPS
}

summary = {
    "experiment":
        "Three-modality reference replacement",

    "model_retrained": False,

    "reference":
        "training_class_0_session_mean",

    "test_sessions": len(test_sessions),

    "test_windows": len(X_test),

    "classification_threshold": THRESHOLD,

    "experiments": len(EXPERIMENTS),

    "full_window_accuracy": float(
        full_result["window_accuracy"]
    ),

    "full_session_accuracy": float(
        full_result["session_accuracy"]
    ),

    "trial24_full_probability": float(
        trial24_full["mean_probability"]
    ),

    "trial24_without_retinal_probability": float(
        trial24_without_retinal["mean_probability"]
    ),

    "trial24_corrected_after_retinal_replacement":
        bool(
            not trial24_full["correct"]
            and trial24_without_retinal["correct"]
        ),

    "mean_absolute_margin_change":
        mean_absolute_margin_changes,

    "data_source": "synthetic",

    "retinal_negative_control": True,

    "clinical_validation": False,

    "limitations": [
        "Only four synthetic test sessions",
        "Reference replacement is not retraining",
        "Ablated inputs may be out of distribution",
        "Results depend on reference selection",
        "Retinal features are random negative controls",
        "Previously evaluated test sessions",
        "No clinical Alzheimer's validation"
    ]
}

with open(
    RESULTS_DIR / "ablation_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)

# =================================================
# PRINT FINAL RESULTS
# =================================================

print("\n========== ABLATION RESULTS ==========")

print(
    results_df[
        [
            "experiment",
            "window_accuracy",
            "session_accuracy",
            "session_f1",
            "session_roc_auc",
            "window_accuracy_change_pp",
            "session_accuracy_change_pp"
        ]
    ].to_string(index=False)
)

print(
    "\n========== SESSION PROBABILITIES =========="
)

print(
    probability_table.to_string()
)

print(
    "\n========== TRIAL 24 INVESTIGATION =========="
)

print(
    trial24[
        [
            "experiment",
            "true_label",
            "mean_probability",
            "mean_logit_margin",
            "predicted_label",
            "correct"
        ]
    ].to_string(index=False)
)

print(
    "\n========== MODALITY SENSITIVITY =========="
)

print(
    sensitivity_df.to_string(index=False)
)

print(
    "\nMean absolute margin changes:"
)

for modality, value in (
    mean_absolute_margin_changes.items()
):

    print(
        f"{modality}: {value:.6f}"
    )

print(
    "\nTrial 24 corrected after retinal replacement:",
    summary[
        "trial24_corrected_after_retinal_replacement"
    ]
)

print(
    "\n========== ABLATION COMPLETE =========="
)

print("Experiments:", len(EXPERIMENTS))

print("Test sessions:", len(test_sessions))

print("Test windows:", len(X_test))

print("Model retrained: False")

print("Data source: SYNTHETIC")

print("Clinical validation: False")

print("Results saved to:", RESULTS_DIR)
