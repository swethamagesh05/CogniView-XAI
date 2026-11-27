from pathlib import Path

import numpy as np
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "fusion"
OUTPUT = ROOT / "data" / "fusion_extended"

OUTPUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

PLR_FEATURES = [
    "plr_baseline_ratio",
    "plr_constriction_pct",
    "plr_latency_s",
    "plr_peak_time_s",
    "plr_recovery_s",
    "plr_noise"
]

train = pd.read_csv(SOURCE / "train.csv")
val = pd.read_csv(SOURCE / "val.csv")

# Verify session separation

train_ids = set(train["session_id"])
val_ids = set(val["session_id"])

assert train_ids.isdisjoint(val_ids), (
    "Session leakage detected!"
)

# Create an independent held-out synthetic test split.
# Existing validation data remains validation data.

test_ids = []

for label, group in train.groupby("label"):

    ids = sorted(group["session_id"].unique())

    if len(ids) < 4:
        raise ValueError(
            "Insufficient sessions for test split"
        )

    selected = rng.choice(
        ids, size=2, replace=False
    )

    test_ids.extend(selected.tolist())

test = train[
    train["session_id"].isin(test_ids)
].copy()

train = train[
    ~train["session_id"].isin(test_ids)
].copy()


def generate_plr(label):

    # Artificial distributions for software testing.
    # These are NOT clinical reference values.

    baseline = rng.normal(0.50, 0.08)

    constriction = rng.normal(
        25 if label == 0 else 20, 6
    )

    latency = rng.normal(
        0.25 if label == 0 else 0.32, 0.07
    )

    peak = rng.normal(0.9, 0.2)

    recovery = rng.normal(
        1.2 if label == 0 else 1.5, 0.4
    )

    noise = rng.uniform(0.5, 3.0)

    return [
        np.clip(baseline, 0.2, 0.8),
        np.clip(constriction, 2, 50),
        np.clip(latency, 0.1, 0.7),
        np.clip(peak, 0.4, 2.0),
        np.clip(recovery, 0.3, 3.0),
        noise
    ]


def extend_dataset(df, split):

    df = df.copy()

    for col in PLR_FEATURES:
        df[col] = np.nan

    records = []

    for session_id, group in df.groupby(
        "session_id"
    ):

        labels = group["label"].unique()

        assert len(labels) == 1

        label = int(labels[0])

        values = generate_plr(label)

        df.loc[
            group.index, PLR_FEATURES
        ] = values

        records.append({
            "session_id": session_id,
            "label": label,
            "split": split,
            "plr_source": "synthetic",
            **dict(zip(PLR_FEATURES, values))
        })

    return df, records


train, train_records = extend_dataset(
    train, "train"
)

val, val_records = extend_dataset(
    val, "validation"
)

test, test_records = extend_dataset(
    test, "test"
)

# Verify that all three splits are independent.

train_ids = set(train["session_id"])
val_ids = set(val["session_id"])
test_ids = set(test["session_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

FEATURES = [
    col for col in train.columns
    if col not in ["label", "session_id"]
]

assert len(FEATURES) == 17

for name, df in [
    ("train", train),
    ("val", val),
    ("test", test)
]:

    assert np.isfinite(
        df[FEATURES].to_numpy()
    ).all()

    df.to_csv(
        OUTPUT / f"{name}.csv",
        index=False
    )

    print(f"\n{name.upper()}")
    print("Rows:", len(df))
    print("Sessions:", df["session_id"].nunique())
    print("Features:", len(FEATURES))

    print(
        df.groupby("label")["session_id"]
        .nunique()
    )

pd.DataFrame(
    train_records + val_records + test_records
).to_csv(
    OUTPUT / "synthetic_plr_manifest.csv",
    index=False
)

metadata = {
    "experiment": "CogniView extended fusion",
    "eye_tracking_source": "synthetic",
    "plr_source": "synthetic",
    "clinical_validation": False,
    "real_plr_used_for_training": False,
    "retinal_modality_included": False,
    "sequence_length": 30,
    "num_features": len(FEATURES),
    "feature_columns": FEATURES,
    "seed": 42,
    "limitations": [
        "Artificial class-dependent PLR distributions",
        "No matched clinical multimodal subjects",
        "Repeated PLR features within each session",
        "Software demonstration only"
    ]
}

with open(
    OUTPUT / "dataset_metadata.json",
    "w"
) as f:

    json.dump(metadata, f, indent=4)

print("\n========== DATASET PREPARATION COMPLETE ==========")

print("Total input features:", len(FEATURES))
print("Session leakage: None")
print("Data source: SYNTHETIC")
print("Clinical validation: False")
print("Saved to:", OUTPUT)