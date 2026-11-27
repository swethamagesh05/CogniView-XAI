from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------------------------
# Configuration
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

CHECKPOINT = (
    ROOT / "models" / "retina" / "best_model.pth"
)

OUT_DIR = ROOT / "results" / "retinal_features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 5
SEED = 42

# -------------------------------------------------
# Dataset location
# -------------------------------------------------

def find_dataset():

    candidates = [
        DATA_DIR / "retina",
        DATA_DIR
    ]

    for folder in candidates:

        csv_path = folder / "train.csv"
        image_dir = folder / "train_images"

        if csv_path.is_file() and image_dir.is_dir():
            return csv_path, image_dir

    raise FileNotFoundError(
        "Retinal dataset not found."
    )


# -------------------------------------------------
# Retinal model
# Same architecture as existing model.py
# -------------------------------------------------

class RetinaClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # No download required because the complete
        # trained checkpoint will be loaded.

        self.backbone = efficientnet_b0(
            weights=None
        )

        in_features = (
            self.backbone.classifier[1].in_features
        )

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):

        return self.backbone(x)

    def extract_features(self, x):

        x = self.backbone.features(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)

        return x


# -------------------------------------------------
# Validation preprocessing
# -------------------------------------------------

def get_transforms():

    return A.Compose([

        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


# -------------------------------------------------
# Retinal dataset
# -------------------------------------------------

class RetinalFeatureDataset(Dataset):

    def __init__(self, dataframe, image_dir):

        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = get_transforms()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = str(row["id_code"])
        label = int(row["diagnosis"])

        image_path = (
            self.image_dir / f"{image_id}.png"
        )

        with Image.open(image_path) as img:

            image = np.array(
                img.convert("RGB")
            )

        tensor = self.transform(
            image=image
        )["image"]

        return tensor, label, image_id


# -------------------------------------------------
# Main extraction pipeline
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum images; 0 means all images"
    )

    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("Limit cannot be negative")

    print("Device:", DEVICE)

    csv_path, image_dir = find_dataset()

    print("Retinal CSV:", csv_path)
    print("Image directory:", image_dir)

    df = pd.read_csv(
        csv_path,
        dtype={"id_code": str}
    )

    required = {"id_code", "diagnosis"}

    if not required.issubset(df.columns):

        raise ValueError(
            f"Unexpected CSV columns: {df.columns.tolist()}"
        )

    print("Total dataset rows:", len(df))

    print("\nOriginal class distribution:")

    print(
        df["diagnosis"].value_counts().sort_index()
    )

    assert not df["id_code"].duplicated().any()

    assert df["diagnosis"].isin(
        range(NUM_CLASSES)
    ).all()

    # Verify image availability.

    exists = df["id_code"].map(
        lambda image_id:
        (image_dir / f"{image_id}.png").is_file()
    )

    missing = df.loc[~exists, "id_code"]

    print("Missing images:", len(missing))

    if len(missing) > 0:

        missing.to_csv(
            OUT_DIR / "missing_images.csv",
            index=False
        )

        raise FileNotFoundError(
            "Some retinal images are missing."
        )

    # Optional smoke test.

    if args.limit > 0:

        df = df.sample(
            n=min(args.limit, len(df)),
            random_state=SEED
        ).reset_index(drop=True)

    print("Images selected:", len(df))

    # Load checkpoint.

    model = RetinaClassifier().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    print("Retinal checkpoint loaded successfully.")

    dataset = RetinalFeatureDataset(
        df,
        image_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    all_probabilities = []
    all_labels = []
    all_predictions = []
    all_ids = []

    processed = 0

    print(
        "\n========== RETINAL FEATURE EXTRACTION =========="
    )

    with torch.inference_mode():

        for images, labels, image_ids in loader:

            images = images.to(DEVICE)

            features = model.extract_features(
                images
            )

            logits = model.backbone.classifier(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_features.append(
                features.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            all_ids.extend(
                list(image_ids)
            )

            processed += len(images)

            if processed % 100 < BATCH_SIZE or \
                    processed == len(dataset):

                print(
                    f"Processed: "
                    f"{processed}/{len(dataset)}"
                )

    # -------------------------------------------------
    # Consolidate features
    # -------------------------------------------------

    features = np.concatenate(
        all_features,
        axis=0
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    assert features.shape[1] == 1280

    assert np.isfinite(features).all()

    assert np.isfinite(probabilities).all()

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5
    )

    metadata = pd.DataFrame({

        "image_id": all_ids,

        "dr_true_label": all_labels,

        "dr_predicted_label": all_predictions,

        "dr_prediction_confidence":
            probabilities.max(axis=1)
    })

    for i in range(NUM_CLASSES):

        metadata[f"dr_probability_{i}"] = (
            probabilities[:, i]
        )

    # Keep smoke-test outputs separate.

    save_dir = (
        OUT_DIR / "smoke_test"
        if args.limit > 0
        else OUT_DIR
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Save numerical outputs
    # -------------------------------------------------

    np.save(
        save_dir / "retinal_embeddings.npy",
        features
    )

    np.save(
        save_dir / "retinal_probabilities.npy",
        probabilities
    )

    metadata.to_csv(
        save_dir / "retinal_metadata.csv",
        index=False
    )

    feature_columns = [
        f"retinal_feature_{i:04d}"
        for i in range(features.shape[1])
    ]

    feature_df = pd.DataFrame(
        features,
        columns=feature_columns
    )

    feature_df.insert(
        0,
        "image_id",
        all_ids
    )

    feature_df.to_csv(
        save_dir / "retinal_embeddings.csv",
        index=False
    )

    # -------------------------------------------------
    # Chart 1: DR class distribution
    # -------------------------------------------------

    counts = (
        metadata["dr_true_label"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values,
        color="#4472C4"
    )

    plt.xlabel("Diabetic retinopathy class")
    plt.ylabel("Number of images")

    plt.title(
        "CogniView: Retinal Dataset Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        save_dir / "retinal_class_distribution.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # Chart 2: Embedding visualization
    # -------------------------------------------------

    sample_count = min(50, len(features))

    selected = features[:sample_count]

    plt.figure(figsize=(12, 6))

    plt.imshow(
        selected,
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(
        label="Feature activation"
    )

    plt.xlabel("Embedding dimension")
    plt.ylabel("Retinal image index")

    plt.title(
        "CogniView: EfficientNet-B0 "
        "Retinal Feature Representation"
    )

    plt.tight_layout()

    plt.savefig(
        save_dir / "retinal_embedding_heatmap.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # Save experiment summary
    # -------------------------------------------------

    summary = {

        "model": "EfficientNet-B0",

        "images_processed": len(features),

        "embedding_dimension":
            int(features.shape[1]),

        "probability_dimension":
            int(probabilities.shape[1]),

        "embedding_shape":
            list(features.shape),

        "dataset":
            "APTOS diabetic retinopathy",

        "label_type":
            "Diabetic retinopathy severity",

        "checkpoint":
            str(CHECKPOINT),

        "feature_extraction_only": True,

        "model_retrained": False,

        "smoke_test": args.limit > 0,

        "matched_multimodal_subjects": False,

        "clinical_alzheimers_validation": False,

        "limitations": [

            "Retinal labels represent diabetic "
            "retinopathy, not Alzheimer's disease",

            "No matched eye-tracking or PLR subjects",

            "Images may include original training samples",

            "Extracted representations do not "
            "establish Alzheimer's diagnostic validity"
        ]
    }

    with open(
        save_dir / "retinal_feature_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # -------------------------------------------------
    # Final output
    # -------------------------------------------------

    print(
        "\n========== FEATURE EXTRACTION COMPLETE =========="
    )

    print("Images processed:", len(features))

    print("Embedding shape:", features.shape)

    print(
        "Probability shape:",
        probabilities.shape
    )

    print("Feature dimension:", features.shape[1])

    print("Model retrained: False")

    print(
        "Retinal labels: Diabetic retinopathy"
    )

    print(
        "Clinical Alzheimer's validation: False"
    )

    print("Outputs saved to:", save_dir)


if __name__ == "__main__":

    main()
