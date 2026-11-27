
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch

from captum.attr import IntegratedGradients

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

WINDOWS_PER_SESSION = 5
IG_STEPS = 100

print("Device:", DEVICE)

# -------------------------------------------------
# Load model configuration
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
# Load data and scaler
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
# Load trained extended model
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
# -------------------------------------------------

reference = train_df[
    train_df["label"] == 0
]

if reference.empty:
    raise ValueError("No class 0 training data")

# Give every reference session equal weight.

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
# Explain the class 1 logit margin.
# Less affected by probability saturation.
# -------------------------------------------------

def forward_margin(x):

    logits, _ = model(x)

    return logits[:, 1] - logits[:, 0]


ig = IntegratedGradients(forward_margin)

with torch.no_grad():

    baseline_logits, _ = model(baseline)

    baseline_probability = torch.softmax(
        baseline_logits,
        dim=1
    )[0, 1].item()

print(
    "Training reference probability:",
    round(baseline_probability, 6)
)

if baseline_probability >= 0.5:
    print(
        "WARNING: Reference is classified as class 1."
    )

# -------------------------------------------------
# Select equal numbers of windows per session
# -------------------------------------------------

records = []
feature_records = []

all_heatmaps = []

print("\n========== MODALITY EXPLANATION ==========")

for session_id, group in test_df.groupby(
    "session_id", sort=False
):

    assert group["label"].nunique() == 1

    label = int(group["label"].iloc[0])

    values = scaler.transform(
        group[FEATURES]
    ).astype(np.float32)

    num_windows = len(values) - SEQ_LEN + 1

    if num_windows <= 0:
        raise ValueError(
            f"Session {session_id} too short"
        )

    starts = np.linspace(
        0,
        num_windows - 1,
        min(WINDOWS_PER_SESSION, num_windows),
        dtype=int
    )

    for start in np.unique(starts):

        x = torch.tensor(
            values[start:start + SEQ_LEN],
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():

            logits, attention = model(x)

            probability = torch.softmax(
                logits, dim=1
            )[0, 1].item()

            margin = forward_margin(x).item()

        attribution, delta = ig.attribute(
            x,
            baselines=baseline,
            n_steps=IG_STEPS,
            internal_batch_size=20,
            return_convergence_delta=True
        )

        attr = attribution.squeeze(
            0
        ).detach().cpu().numpy()

        # Absolute attribution aggregated
        # across temporal frames.

        feature_importance = np.abs(
            attr
        ).sum(axis=0)

        et_total = float(
            feature_importance[ET_INDICES].sum()
        )

        plr_total = float(
            feature_importance[PLR_INDICES].sum()
        )

        total = et_total + plr_total

        et_pct = (
            100 * et_total / total
            if total > 1e-10 else np.nan
        )

        plr_pct = (
            100 * plr_total / total
            if total > 1e-10 else np.nan
        )

        records.append({
            "session_id": session_id,
            "window_start": int(start),
            "true_label": label,
            "class_1_probability": probability,
            "logit_margin": margin,
            "eye_tracking_abs_ig": et_total,
            "plr_abs_ig": plr_total,
            "eye_tracking_contribution_pct": et_pct,
            "plr_contribution_pct": plr_pct,
            "eye_tracking_mean_per_feature":
                et_total / len(ET_INDICES),
            "plr_mean_per_feature":
                plr_total / len(PLR_INDICES),
            "eye_tracking_signed_ig":
                float(attr[:, ET_INDICES].sum()),
            "plr_signed_ig":
                float(attr[:, PLR_INDICES].sum()),
            "ig_convergence_delta":
                float(delta.abs().mean().item())
        })

        for i, feature in enumerate(FEATURES):

            feature_records.append({
                "session_id": session_id,
                "window_start": int(start),
                "feature": feature,
                "modality": (
                    "PLR"
                    if feature.startswith("plr_")
                    else "Eye-tracking"
                ),
                "absolute_ig":
                    float(feature_importance[i])
            })

        all_heatmaps.append(
            (session_id, int(start), attr)
        )

        print(
            f"{session_id} | "
            f"window={start} | "
            f"ET={et_pct:.2f}% | "
            f"PLR={plr_pct:.2f}% | "
            f"delta={delta.abs().mean().item():.6f}"
        )

# -------------------------------------------------
# Save per-window explanations
# -------------------------------------------------

results = pd.DataFrame(records)

results.to_csv(
    OUT_DIR / "modality_contribution_per_window.csv",
    index=False
)

feature_df = pd.DataFrame(feature_records)

# -------------------------------------------------
# Session-level contribution
# -------------------------------------------------

session_results = results.groupby(
    "session_id", as_index=False
).agg(
    true_label=("true_label", "first"),
    eye_tracking_pct=(
        "eye_tracking_contribution_pct", "mean"
    ),
    plr_pct=(
        "plr_contribution_pct", "mean"
    ),
    mean_probability=(
        "class_1_probability", "mean"
    ),
    mean_ig_delta=(
        "ig_convergence_delta", "mean"
    )
)

session_results.to_csv(
    OUT_DIR / "modality_contribution_per_session.csv",
    index=False
)

et_mean = float(
    session_results["eye_tracking_pct"].mean()
)

plr_mean = float(
    session_results["plr_pct"].mean()
)

if not np.isfinite([et_mean, plr_mean]).all():
    raise ValueError(
        "Modality attribution is undefined."
    )

# -------------------------------------------------
# Global feature ranking
# Equal weighting across sessions
# -------------------------------------------------

ranking = (
    feature_df.groupby(
        ["session_id", "feature", "modality"],
        as_index=False
    )["absolute_ig"].mean()
    .groupby(
        ["feature", "modality"],
        as_index=False
    )["absolute_ig"].mean()
    .sort_values(
        "absolute_ig", ascending=False
    )
)

ranking.to_csv(
    OUT_DIR / "global_feature_ranking.csv",
    index=False
)

# -------------------------------------------------
# Chart 1: Global modality contribution
# -------------------------------------------------

plt.figure(figsize=(8, 6))

plt.pie(
    [et_mean, plr_mean],
    labels=[
        "Eye-tracking",
        "Synthetic PLR"
    ],
    autopct="%1.2f%%",
    colors=["#4472C4", "#2E8B57"],
    startangle=90,
    wedgeprops={"edgecolor": "white"}
)

plt.title(
    "CogniView: Modality Attribution\n"
    "Synthetic Demonstration"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "modality_contribution.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Chart 2: Session-wise modality contribution
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    session_results["session_id"],
    session_results["eye_tracking_pct"],
    label="Eye-tracking",
    color="#4472C4"
)

plt.bar(
    session_results["session_id"],
    session_results["plr_pct"],
    bottom=session_results["eye_tracking_pct"],
    label="Synthetic PLR",
    color="#2E8B57"
)

plt.xlabel("Synthetic test session")
plt.ylabel("Absolute IG share (%)")
plt.ylim(0, 105)

plt.title(
    "CogniView: Session-wise Modality Contribution"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_DIR / "session_modality_contribution.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Chart 3: Global feature importance
# -------------------------------------------------

colors = ranking["modality"].map({
    "Eye-tracking": "#4472C4",
    "PLR": "#2E8B57"
})

plt.figure(figsize=(10, 8))

plt.barh(
    ranking["feature"],
    ranking["absolute_ig"],
    color=colors
)

plt.gca().invert_yaxis()

plt.xlabel("Mean absolute Integrated Gradients")

plt.title(
    "CogniView: Global Feature Importance"
)

from matplotlib.patches import Patch

plt.legend(handles=[
    Patch(
        color="#4472C4",
        label="Eye-tracking"
    ),
    Patch(
        color="#2E8B57",
        label="Synthetic PLR"
    )
])

plt.tight_layout()

plt.savefig(
    OUT_DIR / "global_feature_ranking.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# Chart 4: Temporal attribution heatmap
# One representative window per session
# -------------------------------------------------

seen = set()

for session_id, start, attr in all_heatmaps:

    if session_id in seen:
        continue

    seen.add(session_id)

    limit = max(
        float(np.abs(attr).max()),
        1e-10
    )

    plt.figure(figsize=(12, 8))

    plt.imshow(
        attr.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    plt.yticks(
        range(len(FEATURES)),
        FEATURES
    )

    plt.axhline(
        len(ET_INDICES) - 0.5,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Temporal frame")
    plt.ylabel("Input feature")

    plt.title(
        f"CogniView: IG Heatmap — {session_id}"
    )

    plt.colorbar(
        label="Signed attribution to class 1 margin"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"ig_heatmap_{session_id}.png",
        dpi=300
    )

    plt.close()

# -------------------------------------------------
# Save consolidated results
# -------------------------------------------------

summary = {
    "method": "Integrated Gradients",
    "target": "class_1_logit_minus_class_0_logit",
    "baseline": "training_only_class_0_session_mean",
    "baseline_class_1_probability":
        baseline_probability,
    "ig_steps": IG_STEPS,
    "sessions": int(
        results["session_id"].nunique()
    ),
    "explained_windows": len(results),
    "eye_tracking_features": len(ET_INDICES),
    "plr_features": len(PLR_INDICES),
    "eye_tracking_contribution_pct": et_mean,
    "plr_contribution_pct": plr_mean,
    "mean_ig_convergence_delta": float(
        results["ig_convergence_delta"].mean()
    ),
    "max_ig_convergence_delta": float(
        results["ig_convergence_delta"].max()
    ),
    "top_feature": str(
        ranking.iloc[0]["feature"]
    ),
    "data_source": "synthetic",
    "clinical_validation": False,
    "real_plr_used_for_training": False,
    "retinal_modality_included": False,
    "limitations": [
        "Only four synthetic test sessions",
        "Attribution depends on reference baseline",
        "Modalities have unequal feature counts",
        "Synthetic features depend on class labels",
        "Attribution is not clinical importance",
        "Ablation and faithfulness tests pending"
    ]
}

with open(
    OUT_DIR / "modality_contribution_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)

# -------------------------------------------------
# Print results
# -------------------------------------------------

print("\n========== GLOBAL MODALITY CONTRIBUTION ==========")

print(f"Eye-tracking: {et_mean:.2f}%")
print(f"Synthetic PLR: {plr_mean:.2f}%")

print("\n========== SESSION CONTRIBUTION ==========")

print(session_results.to_string(index=False))

print("\n========== GLOBAL FEATURE RANKING ==========")

print(ranking.to_string(index=False))

print("\n========== IG QUALITY ==========")

print(
    "Mean convergence delta:",
    summary["mean_ig_convergence_delta"]
)

print(
    "Maximum convergence delta:",
    summary["max_ig_convergence_delta"]
)

print("\n========== EXPLANATION COMPLETE ==========")

print("Sessions:", summary["sessions"])
print("Explained windows:", len(results))
print("Data source: SYNTHETIC")
print("Clinical validation: False")
print("Outputs saved to:", OUT_DIR)
