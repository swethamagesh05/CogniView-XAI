
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================
# CONFIGURATION
# =================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = ROOT / "data" / "fusion_extended"

OUTPUT_DIR = ROOT / "data" / "fusion_trimodal"

RESULTS_DIR = ROOT / "results" / "xai_trimodal"

RETINAL_DIR = ROOT / "results" / "retinal_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQUENCE_LENGTH = 30

rng = np.random.default_rng(SEED)

# =================================================
# LOAD EXISTING DATASET CONFIGURATION
# =================================================

with open(
    SOURCE_DIR / "dataset_metadata.json",
    "r"
) as f:

    original_metadata = json.load(f)

BASE_FEATURES = original_metadata["feature_columns"]

EYE_FEATURES = [
    feature for feature in BASE_FEATURES
    if not feature.startswith("plr_")
]

PLR_FEATURES = [
    feature for feature in BASE_FEATURES
    if feature.startswith("plr_")
]

RETINAL_FEATURES = [
    f"retinal_sim_{i:02d}"
    for i in range(1, 9)
]

ALL_FEATURES = (
    EYE_FEATURES
    + PLR_FEATURES
    + RETINAL_FEATURES
)

assert len(EYE_FEATURES) == 11
assert len(PLR_FEATURES) == 6
assert len(RETINAL_FEATURES) == 8
assert len(ALL_FEATURES) == 25

# =================================================
# VERIFY REAL RETINAL EXTRACTION OUTPUTS
# =================================================

embeddings = np.load(
    RETINAL_DIR / "retinal_embeddings.npy"
)

probabilities = np.load(
    RETINAL_DIR / "retinal_probabilities.npy"
)

assert embeddings.shape == (3662, 1280)

assert probabilities.shape == (3662, 5)

assert np.isfinite(embeddings).all()

assert np.isfinite(probabilities).all()

assert np.allclose(
    probabilities.sum(axis=1),
    1.0,
    atol=1e-5
)

print("Real retinal embeddings:", embeddings.shape)

print("Real retinal probabilities:", probabilities.shape)

print(
    "Real retinal data preserved separately."
)

# =================================================
# LOAD EXISTING FUSION DATASETS
# =================================================

datasets = {}

for split in ["train", "val", "test"]:

    path = SOURCE_DIR / f"{split}.csv"

    if not path.exists():

        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = set(
        BASE_FEATURES + ["label", "session_id"]
    )

    if not required.issubset(df.columns):

        raise ValueError(
            f"Missing columns in {split}"
        )

    assert df["session_id"].notna().all()

    assert df["label"].isin([0, 1]).all()

    assert np.isfinite(
        df[BASE_FEATURES].to_numpy(dtype=float)
    ).all()

    assert (
        df.groupby("session_id")["label"]
        .nunique() == 1
    ).all()

    datasets[split] = df

# =================================================
# SESSION LEAKAGE VERIFICATION
# =================================================

train_sessions = set(
    datasets["train"]["session_id"]
)

val_sessions = set(
    datasets["val"]["session_id"]
)

test_sessions = set(
    datasets["test"]["session_id"]
)

assert train_sessions.isdisjoint(val_sessions)

assert train_sessions.isdisjoint(test_sessions)

assert val_sessions.isdisjoint(test_sessions)

print("\nSession overlap: None")

# =================================================
# GENERATE SYNTHETIC RETINAL FEATURES
# =================================================

# IMPORTANT:
# Features are generated without using class labels.
# They are not derived from real retinal embeddings.
# Each session receives one static retinal vector.

all_sessions = sorted(
    train_sessions
    | val_sessions
    | test_sessions
)

retinal_vectors = {}

for session_id in all_sessions:

    vector = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(RETINAL_FEATURES)
    )

    retinal_vectors[session_id] = vector

print(
    "Synthetic retinal vectors generated:",
    len(retinal_vectors)
)

# =================================================
# PREPARE THREE-MODALITY DATASETS
# =================================================

manifest = []
split_summary = []

for split, original_df in datasets.items():

    df = original_df.copy()

    for index, feature in enumerate(
        RETINAL_FEATURES
    ):

        df[feature] = df["session_id"].map(
            lambda session_id:
            retinal_vectors[session_id][index]
        )

    df = df[
        ALL_FEATURES + ["label", "session_id"]
    ]

    assert np.isfinite(
        df[ALL_FEATURES].to_numpy(dtype=float)
    ).all()

    assert len(df) == len(original_df)

    assert df["label"].equals(
        original_df["label"]
    )

    assert df["session_id"].equals(
        original_df["session_id"]
    )

    session_sizes = df.groupby(
        "session_id"
    ).size()

    assert (
        session_sizes >= SEQUENCE_LENGTH
    ).all()

    session_labels = df.groupby(
        "session_id"
    )["label"].first()

    num_windows = int(
        (
            session_sizes
            - SEQUENCE_LENGTH
            + 1
        ).sum()
    )

    df.to_csv(
        OUTPUT_DIR / f"{split}.csv",
        index=False
    )

    for session_id, label in session_labels.items():

        record = {
            "session_id": session_id,
            "split": split,
            "label": int(label),
            "retinal_source":
                "synthetic_label_independent",
            "real_retinal_image_used": False
        }

        for index, feature in enumerate(
            RETINAL_FEATURES
        ):

            record[feature] = float(
                retinal_vectors[session_id][index]
            )

        manifest.append(record)

    summary = {
        "split": split,
        "rows": len(df),
        "sessions": len(session_labels),
        "class_0_sessions": int(
            (session_labels == 0).sum()
        ),
        "class_1_sessions": int(
            (session_labels == 1).sum()
        ),
        "features": len(ALL_FEATURES),
        "windows": num_windows
    }

    split_summary.append(summary)

    print(f"\n========== {split.upper()} ==========")

    print("Rows:", len(df))

    print("Sessions:", len(session_labels))

    print("Features:", len(ALL_FEATURES))

    print("Windows:", num_windows)

    print(
        "Class distribution:",
        session_labels.value_counts()
        .sort_index().to_dict()
    )

# =================================================
# SAVE MANIFEST AND SPLIT SUMMARY
# =================================================

manifest_df = pd.DataFrame(manifest)

manifest_df.to_csv(
    OUTPUT_DIR / "synthetic_retinal_manifest.csv",
    index=False
)

summary_df = pd.DataFrame(split_summary)

summary_df.to_csv(
    RESULTS_DIR / "dataset_split_summary.csv",
    index=False
)

# =================================================
# RETINAL NEGATIVE-CONTROL DIAGNOSTICS
# =================================================

# Small samples can show accidental class
# associations even when generated independently.

train_manifest = manifest_df[
    manifest_df["split"] == "train"
]

correlations = []

for feature in RETINAL_FEATURES:

    correlation = train_manifest[
        [feature, "label"]
    ].corr().iloc[0, 1]

    correlations.append({
        "feature": feature,
        "training_label_correlation":
            float(correlation)
    })

correlation_df = pd.DataFrame(correlations)

correlation_df.to_csv(
    RESULTS_DIR /
    "retinal_control_correlations.csv",
    index=False
)

print("\nTraining retinal-label correlations:")

print(correlation_df.to_string(index=False))

# =================================================
# SAVE DATASET METADATA
# =================================================

metadata = {

    "experiment":
        "CogniView three-modality fusion",

    "data_source": "synthetic",

    "eye_tracking_source": "synthetic",

    "plr_source": "synthetic",

    "retinal_source":
        "synthetic_label_independent",

    "retinal_modality_included": True,

    "retinal_negative_control": True,

    "real_retinal_used_for_training": False,

    "real_plr_used_for_training": False,

    "matched_clinical_subjects": False,

    "clinical_validation": False,

    "sequence_length": SEQUENCE_LENGTH,

    "num_features": len(ALL_FEATURES),

    "num_classes": 2,

    "feature_columns": ALL_FEATURES,

    "modality_features": {

        "eye_tracking": EYE_FEATURES,

        "plr": PLR_FEATURES,

        "retinal": RETINAL_FEATURES
    },

    "real_retinal_embedding_shape":
        list(embeddings.shape),

    "real_retinal_embeddings_preserved": True,

    "seed": SEED,

    "session_leakage": False,

    "previously_evaluated_test_sessions": True,

    "dataset_splits": split_summary,

    "limitations": [

        "All fusion inputs are synthetic",

        "Retinal features are random negative "
        "controls, not clinical biomarkers",

        "Retinal features generated independently "
        "of class labels",

        "Accidental retinal-label correlations "
        "may occur due to small sample size",

        "Existing synthetic PLR distributions "
        "are class-dependent",

        "Retinal and PLR features are repeated "
        "within each session",

        "Real retinal embeddings represent "
        "diabetic retinopathy images",

        "No matched clinical multimodal subjects",

        "Previously evaluated test sessions are "
        "not independent external validation",

        "Software demonstration only"
    ]
}

with open(
    OUTPUT_DIR / "dataset_metadata.json",
    "w"
) as f:

    json.dump(metadata, f, indent=4)

# =================================================
# CHART 1: MODALITY FEATURE DISTRIBUTION
# =================================================

plt.figure(figsize=(8, 6))

plt.pie(

    [
        len(EYE_FEATURES),
        len(PLR_FEATURES),
        len(RETINAL_FEATURES)
    ],

    labels=[
        "Eye-tracking (11)",
        "PLR (6)",
        "Retinal (8)"
    ],

    autopct="%1.1f%%",

    startangle=90,

    colors=[
        "#27966B",
        "#C58A27",
        "#4472C4"
    ],

    wedgeprops={
        "edgecolor": "white"
    }
)

plt.title(
    "CogniView: Three-Modality Input Feature Distribution"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "modality_feature_distribution.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 2: DATASET SPLIT DISTRIBUTION
# =================================================

plt.figure(figsize=(8, 5))

plt.bar(

    summary_df["split"],

    summary_df["class_0_sessions"],

    label="Synthetic class 0",

    color="#4472C4"
)

plt.bar(

    summary_df["split"],

    summary_df["class_1_sessions"],

    bottom=summary_df["class_0_sessions"],

    label="Synthetic class 1",

    color="#ED7D31"
)

plt.xlabel("Dataset split")

plt.ylabel("Number of sessions")

plt.title(
    "CogniView: Session-Level Dataset Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "dataset_split_distribution.png",
    dpi=300
)

plt.close()

# =================================================
# CHART 3: NEGATIVE-CONTROL CORRELATIONS
# =================================================

plt.figure(figsize=(10, 5))

plt.bar(

    correlation_df["feature"],

    correlation_df["training_label_correlation"],

    color="#4472C4"
)

plt.axhline(0, color="black", linewidth=1)

plt.ylim(-1, 1)

plt.xticks(rotation=45)

plt.ylabel("Pearson correlation")

plt.title(
    "Synthetic Retinal Negative-Control "
    "Features vs Training Labels"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "retinal_control_correlations.png",
    dpi=300
)

plt.close()

# =================================================
# FINAL SUMMARY
# =================================================

print(
    "\n========== THREE-MODALITY DATASET COMPLETE =========="
)

print("Eye-tracking features:", len(EYE_FEATURES))

print("PLR features:", len(PLR_FEATURES))

print("Retinal features:", len(RETINAL_FEATURES))

print("Total features:", len(ALL_FEATURES))

print("Sequence length:", SEQUENCE_LENGTH)

print("Total sessions:", len(all_sessions))

print("Session leakage: None")

print("Data source: SYNTHETIC")

print("Retinal negative control: True")

print("Real retinal data used for training: False")

print("Clinical validation: False")

print("Dataset saved to:", OUTPUT_DIR)

print("Results saved to:", RESULTS_DIR)
