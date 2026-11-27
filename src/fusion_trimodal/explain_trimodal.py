
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch

from captum.attr import IntegratedGradients

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

WINDOWS_PER_SESSION = 5
IG_STEPS = 100

COLORS = {
    "eye_tracking": "#27966B",
    "plr": "#C58A27",
    "retinal": "#4472C4"
}

DISPLAY_NAMES = {
    "eye_tracking": "Eye-tracking",
    "plr": "Synthetic PLR",
    "retinal": "Synthetic retinal"
}

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

MODALITIES = config["modality_features"]

GROUPS = {
    modality: [
        FEATURES.index(feature)
        for feature in feature_names
    ]
    for modality, feature_names
    in MODALITIES.items()
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

print("Input features:", len(FEATURES))

print("Sequence length:", SEQ_LEN)

# =================================================
# LOAD DATA
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

assert set(
    train_df["session_id"]
).isdisjoint(test_df["session_id"])

assert set(
    train_df["session_id"]
).isdisjoint(val_df["session_id"])

assert set(
    val_df["session_id"]
).isdisjoint(test_df["session_id"])

# =================================================
# LOAD SCALER AND MODEL
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

print("Three-modality model loaded successfully.")

# =================================================
# TRAINING-ONLY CLASS 0 BASELINE
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

baseline = torch.tensor(
    np.repeat(
        reference_scaled,
        SEQ_LEN,
        axis=0
    ),
    dtype=torch.float32,
    device=DEVICE
).unsqueeze(0)

with torch.inference_mode():

    baseline_logits, _ = model(baseline)

    baseline_probability = float(
        torch.softmax(
            baseline_logits,
            dim=1
        )[0, 1].item()
    )

    baseline_margin = float(
        (
            baseline_logits[0, 1]
            - baseline_logits[0, 0]
        ).item()
    )

print(
    "Training reference probability:",
    baseline_probability
)

print(
    "Training reference logit margin:",
    baseline_margin
)

if baseline_probability >= 0.5:
    print(
        "WARNING: Reference is not classified "
        "as synthetic class 0."
    )

# =================================================
# INTEGRATED GRADIENTS TARGET
# =================================================

def forward_margin(x):

    logits, _ = model(x)

    return logits[:, 1] - logits[:, 0]


ig = IntegratedGradients(forward_margin)

window_records = []

feature_records = []

heatmaps = {}

# =================================================
# EXPLAIN SESSION WINDOWS
# =================================================

def explain_sessions(df, split):

    for session_id, group in df.groupby(
        "session_id",
        sort=True
    ):

        assert group["label"].nunique() == 1

        label = int(
            group["label"].iloc[0]
        )

        values = scaler.transform(
            group[FEATURES]
        ).astype(np.float32)

        num_windows = (
            len(values) - SEQ_LEN + 1
        )

        if num_windows <= 0:
            continue

        starts = np.unique(
            np.linspace(
                0,
                num_windows - 1,
                min(
                    WINDOWS_PER_SESSION,
                    num_windows
                ),
                dtype=int
            )
        )

        session_attributions = []

        for start in starts:

            x = torch.tensor(
                values[start:start + SEQ_LEN],
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():

                logits, _ = model(x)

                probability = float(
                    torch.softmax(
                        logits,
                        dim=1
                    )[0, 1].item()
                )

                margin = float(
                    (
                        logits[0, 1]
                        - logits[0, 0]
                    ).item()
                )

            attribution, delta = ig.attribute(
                x,
                baselines=baseline,
                n_steps=IG_STEPS,
                internal_batch_size=20,
                return_convergence_delta=True
            )

            attr = (
                attribution
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            assert np.isfinite(attr).all()

            session_attributions.append(attr)

            absolute_feature_ig = (
                np.abs(attr).sum(axis=0)
            )

            total_absolute_ig = float(
                absolute_feature_ig.sum()
            )

            margin_difference = (
                margin - baseline_margin
            )

            absolute_delta = float(
                abs(delta.item())
            )

            relative_delta = (
                absolute_delta
                / abs(margin_difference)
                if abs(margin_difference) > 1e-6
                else np.nan
            )

            record = {
                "split": split,
                "session_id": session_id,
                "window_start": int(start),
                "true_label": label,
                "class_1_probability": probability,
                "predicted_label": int(
                    probability >= 0.5
                ),
                "logit_margin": margin,
                "baseline_margin": baseline_margin,
                "margin_difference": margin_difference,
                "total_absolute_ig": total_absolute_ig,
                "ig_convergence_delta": absolute_delta,
                "relative_ig_delta": relative_delta
            }

            for modality, indices in GROUPS.items():

                abs_ig = float(
                    absolute_feature_ig[indices].sum()
                )

                signed_ig = float(
                    attr[:, indices].sum()
                )

                contribution = (
                    100 * abs_ig / total_absolute_ig
                    if total_absolute_ig > 1e-12
                    else np.nan
                )

                record[
                    f"{modality}_abs_ig"
                ] = abs_ig

                record[
                    f"{modality}_signed_ig"
                ] = signed_ig

                record[
                    f"{modality}_contribution_pct"
                ] = contribution

                record[
                    f"{modality}_mean_per_feature"
                ] = abs_ig / len(indices)

                for index in indices:

                    feature_records.append({
                        "split": split,
                        "session_id": session_id,
                        "window_start": int(start),
                        "true_label": label,
                        "feature": FEATURES[index],
                        "modality": modality,
                        "absolute_ig": float(
                            absolute_feature_ig[index]
                        ),
                        "signed_ig": float(
                            attr[:, index].sum()
                        )
                    })

            window_records.append(record)

        heatmaps[(split, session_id)] = np.mean(
            np.stack(session_attributions),
            axis=0
        )

        print(
            f"{split.upper()} | "
            f"{session_id} | "
            f"Class {label} | "
            f"Explained windows: {len(starts)}"
        )


print(
    "\n========== INTEGRATED GRADIENTS =========="
)

# Explain all four test sessions.

explain_sessions(test_df, "test")

# Separately investigate the validation error.
# This is excluded from global test statistics.

validation_error = val_df[
    val_df["session_id"] == "trial13"
]

if not validation_error.empty:

    explain_sessions(
        validation_error,
        "validation_case_study"
    )

# =================================================
# SAVE WINDOW AND FEATURE ATTRIBUTIONS
# =================================================

window_df = pd.DataFrame(window_records)

feature_df = pd.DataFrame(feature_records)

window_df.to_csv(
    RESULTS_DIR /
    "modality_contribution_per_window.csv",
    index=False
)

feature_df.to_csv(
    RESULTS_DIR /
    "feature_attributions_per_window.csv",
    index=False
)

test_windows = window_df[
    window_df["split"] == "test"
].copy()

test_features = feature_df[
    feature_df["split"] == "test"
].copy()

# =================================================
# SESSION-LEVEL CONTRIBUTION
# =================================================

aggregation = {
    "true_label": ("true_label", "first"),
    "mean_probability": (
        "class_1_probability", "mean"
    ),
    "mean_ig_delta": (
        "ig_convergence_delta", "mean"
    ),
    "mean_relative_ig_delta": (
        "relative_ig_delta", "mean"
    )
}

for modality in GROUPS:

    aggregation[f"{modality}_pct"] = (
        f"{modality}_contribution_pct",
        "mean"
    )

    aggregation[f"{modality}_signed_ig"] = (
        f"{modality}_signed_ig",
        "mean"
    )

    aggregation[f"{modality}_mean_per_feature"] = (
        f"{modality}_mean_per_feature",
        "mean"
    )

session_df = window_df.groupby(
    ["split", "session_id"],
    as_index=False
).agg(**aggregation)

session_df.to_csv(
    RESULTS_DIR /
    "modality_contribution_per_session.csv",
    index=False
)

test_sessions = session_df[
    session_df["split"] == "test"
].copy()

global_contributions = {
    modality: float(
        test_sessions[
            f"{modality}_pct"
        ].mean()
    )
    for modality in GROUPS
}

assert np.isclose(
    sum(global_contributions.values()),
    100.0,
    atol=1e-4
)

# =================================================
# GLOBAL FEATURE RANKING
# =================================================

# Equal weighting of test sessions.

session_feature_ig = test_features.groupby(
    ["session_id", "feature", "modality"],
    as_index=False
).agg(
    absolute_ig=("absolute_ig", "mean"),
    signed_ig=("signed_ig", "mean")
)

ranking = session_feature_ig.groupby(
    ["feature", "modality"],
    as_index=False
).agg(
    absolute_ig=("absolute_ig", "mean"),
    signed_ig=("signed_ig", "mean")
).sort_values(
    "absolute_ig",
    ascending=False
)

ranking.to_csv(
    RESULTS_DIR / "global_feature_ranking.csv",
    index=False
)

# =================================================
# CHART 1: GLOBAL MODALITY CONTRIBUTION
# =================================================

plt.figure(figsize=(8, 6))

plt.pie(
    list(global_contributions.values()),
    labels=[
        DISPLAY_NAMES[m]
        for m in global_contributions
    ],
    colors=[
        COLORS[m]
        for m in global_contributions
    ],
    autopct="%1.2f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white"}
)

plt.title(
    "CogniView: Three-Modality Absolute IG Contribution"
)

plt.figtext(
    0.5,
    0.02,
    "Synthetic test sessions only",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(
    RESULTS_DIR / "modality_contribution.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 2: SESSION MODALITY CONTRIBUTION
# =================================================

plt.figure(figsize=(10, 6))

bottom = np.zeros(len(test_sessions))

for modality in GROUPS:

    values = test_sessions[
        f"{modality}_pct"
    ].to_numpy()

    plt.bar(
        test_sessions["session_id"],
        values,
        bottom=bottom,
        color=COLORS[modality],
        label=DISPLAY_NAMES[modality]
    )

    bottom += values

plt.ylim(0, 100)

plt.ylabel("Absolute IG contribution (%)")

plt.xlabel("Synthetic test session")

plt.title(
    "CogniView: Session-Wise Modality Contribution"
)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "session_modality_contribution.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 3: GLOBAL FEATURE RANKING
# =================================================

plot_df = ranking.sort_values(
    "absolute_ig",
    ascending=True
)

plt.figure(figsize=(11, 10))

plt.barh(
    plot_df["feature"],
    plot_df["absolute_ig"],
    color=[
        COLORS[m]
        for m in plot_df["modality"]
    ]
)

plt.xlabel("Mean absolute Integrated Gradients")

plt.title(
    "CogniView: Global Feature Importance"
)

from matplotlib.patches import Patch

plt.legend(
    handles=[
        Patch(
            color=COLORS[m],
            label=DISPLAY_NAMES[m]
        )
        for m in GROUPS
    ],
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "global_feature_ranking.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 4: SIGNED MODALITY CONTRIBUTION
# =================================================

x = np.arange(len(test_sessions))

width = 0.25

plt.figure(figsize=(10, 6))

for index, modality in enumerate(GROUPS):

    plt.bar(
        x + (index - 1) * width,
        test_sessions[f"{modality}_signed_ig"],
        width,
        label=DISPLAY_NAMES[modality],
        color=COLORS[modality]
    )

plt.axhline(
    0,
    color="black",
    linewidth=1
)

plt.xticks(
    x,
    test_sessions["session_id"]
)

plt.ylabel("Signed IG contribution to logit margin")

plt.title(
    "Positive and Negative Modality Contributions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "signed_modality_contribution.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 5: FEATURE-NORMALIZED CONTRIBUTION
# =================================================

normalized = {
    modality: float(
        test_sessions[
            f"{modality}_mean_per_feature"
        ].mean()
    )
    for modality in GROUPS
}

plt.figure(figsize=(8, 5))

plt.bar(
    [DISPLAY_NAMES[m] for m in normalized],
    list(normalized.values()),
    color=[COLORS[m] for m in normalized]
)

plt.ylabel("Mean absolute IG per feature")

plt.title(
    "Feature-Count-Normalized Modality Importance"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "normalized_modality_importance.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 6: SESSION IG HEATMAPS
# =================================================

vmax = max(
    float(np.abs(attr).max())
    for attr in heatmaps.values()
)

vmax = max(vmax, 1e-8)

for (split, session_id), attr in heatmaps.items():

    plt.figure(figsize=(13, 10))

    plt.imshow(
        attr.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest"
    )

    plt.yticks(
        np.arange(len(FEATURES)),
        FEATURES
    )

    plt.axhline(
        10.5,
        color="black",
        linewidth=1.5
    )

    plt.axhline(
        16.5,
        color="black",
        linewidth=1.5
    )

    plt.colorbar(
        label="Signed Integrated Gradients"
    )

    plt.xlabel("Time step within window")

    plt.ylabel("Input feature")

    plt.title(
        f"Mean IG Heatmap: {session_id} ({split})"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        f"ig_heatmap_{session_id}.png",
        dpi=300
    )

    plt.close()

# =================================================
# CHART 7: TRIAL 24 ERROR ANALYSIS
# =================================================

trial24 = feature_df[
    (feature_df["session_id"] == "trial24")
    & (feature_df["split"] == "test")
]

if not trial24.empty:

    trial24_ranking = trial24.groupby(
        ["feature", "modality"],
        as_index=False
    ).agg(
        signed_ig=("signed_ig", "mean"),
        absolute_ig=("absolute_ig", "mean")
    ).sort_values("signed_ig")

    trial24_ranking.to_csv(
        RESULTS_DIR /
        "trial24_feature_attributions.csv",
        index=False
    )

    plt.figure(figsize=(11, 10))

    plt.barh(
        trial24_ranking["feature"],
        trial24_ranking["signed_ig"],
        color=[
            COLORS[m]
            for m in trial24_ranking["modality"]
        ]
    )

    plt.axvline(
        0,
        color="black",
        linewidth=1
    )

    plt.xlabel(
        "Signed IG relative to class 0 reference"
    )

    plt.title(
        "Trial 24: Feature-Level Error Investigation"
    )

    plt.legend(
        handles=[
            Patch(
                color=COLORS[m],
                label=DISPLAY_NAMES[m]
            )
            for m in GROUPS
        ]
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "trial24_error_explanation.png",
        dpi=300
    )

    plt.close()

# =================================================
# SAVE SUMMARY
# =================================================

relative_errors = test_windows[
    "relative_ig_delta"
].dropna()

summary = {
    "method": "Integrated Gradients",
    "target": "class_1_logit_minus_class_0_logit",
    "baseline": "training_class_0_session_mean",
    "baseline_probability": baseline_probability,
    "baseline_margin": baseline_margin,
    "ig_steps": IG_STEPS,
    "test_sessions": len(test_sessions),
    "explained_test_windows": len(test_windows),
    "validation_case_study_windows": int(
        (
            window_df["split"]
            == "validation_case_study"
        ).sum()
    ),
    "global_modality_contribution_pct":
        global_contributions,
    "mean_absolute_ig_per_feature": normalized,
    "mean_ig_convergence_delta": float(
        test_windows["ig_convergence_delta"].mean()
    ),
    "max_ig_convergence_delta": float(
        test_windows["ig_convergence_delta"].max()
    ),
    "mean_relative_ig_delta": (
        float(relative_errors.mean())
        if len(relative_errors)
        else None
    ),
    "max_relative_ig_delta": (
        float(relative_errors.max())
        if len(relative_errors)
        else None
    ),
    "top_feature": str(
        ranking.iloc[0]["feature"]
    ),
    "data_source": "synthetic",
    "retinal_negative_control": True,
    "real_retinal_used_for_training": False,
    "real_plr_used_for_training": False,
    "clinical_validation": False,
    "limitations": [
        "Only four synthetic test sessions",
        "Five sampled windows per test session",
        "Attributions depend on the baseline",
        "Absolute IG is not causal importance",
        "Modalities have unequal feature counts",
        "Random retinal inputs are negative controls",
        "Validation case study excluded from global results",
        "No clinical Alzheimer's validation"
    ]
}

with open(
    RESULTS_DIR /
    "modality_contribution_summary.json",
    "w"
) as f:

    json.dump(summary, f, indent=4)

# =================================================
# PRINT FINAL RESULTS
# =================================================

print(
    "\n========== GLOBAL MODALITY CONTRIBUTION =========="
)

for modality, value in global_contributions.items():

    print(
        f"{DISPLAY_NAMES[modality]}: "
        f"{value:.2f}%"
    )

print(
    "\n========== SESSION CONTRIBUTION =========="
)

print(
    test_sessions[
        [
            "session_id",
            "true_label",
            "eye_tracking_pct",
            "plr_pct",
            "retinal_pct",
            "mean_probability",
            "mean_ig_delta"
        ]
    ].to_string(index=False)
)

print(
    "\n========== SIGNED MODALITY CONTRIBUTION =========="
)

print(
    test_sessions[
        [
            "session_id",
            "eye_tracking_signed_ig",
            "plr_signed_ig",
            "retinal_signed_ig"
        ]
    ].to_string(index=False)
)

print(
    "\n========== GLOBAL FEATURE RANKING =========="
)

print(ranking.to_string(index=False))

print(
    "\n========== TRIAL 24 ERROR ANALYSIS =========="
)

print(
    test_sessions[
        test_sessions["session_id"] == "trial24"
    ].to_string(index=False)
)

if not trial24.empty:

    print(
        "\nTrial 24 feature attributions:"
    )

    print(
        trial24_ranking.to_string(index=False)
    )

print(
    "\n========== IG CONVERGENCE =========="
)

print(
    "Mean absolute convergence delta:",
    summary["mean_ig_convergence_delta"]
)

print(
    "Maximum absolute convergence delta:",
    summary["max_ig_convergence_delta"]
)

print(
    "Mean relative convergence delta:",
    summary["mean_relative_ig_delta"]
)

print(
    "Maximum relative convergence delta:",
    summary["max_relative_ig_delta"]
)

print(
    "\n========== XAI ANALYSIS COMPLETE =========="
)

print("Test sessions:", len(test_sessions))

print("Explained test windows:", len(test_windows))

print("Data source: SYNTHETIC")

print("Retinal negative control: True")

print("Clinical validation: False")

print("Results saved to:", RESULTS_DIR)
