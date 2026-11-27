import argparse
import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from dataset import ADEyeTrackingDataset
from model import ADClassifierModel


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Create reproducible stratified split
# ---------------------------------------------------------
def create_splits(dataset, seed=42):
    rng = np.random.default_rng(seed)

    labels = dataset.metadata["clinical_group"].astype(int).values

    class_0 = np.where(labels == 0)[0]
    class_1 = np.where(labels == 1)[0]

    rng.shuffle(class_0)
    rng.shuffle(class_1)

    # Approximately 80% train, 10% validation, 10% test
    def split_class(indices):
        n = len(indices)

        train_n = int(0.8 * n)
        val_n = int(0.1 * n)

        train_idx = indices[:train_n]
        val_idx = indices[train_n:train_n + val_n]
        test_idx = indices[train_n + val_n:]

        return train_idx, val_idx, test_idx

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train_indices = np.concatenate([train_0, train_1])
    val_indices = np.concatenate([val_0, val_1])
    test_indices = np.concatenate([test_0, test_1])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        train_indices.tolist(),
        val_indices.tolist(),
        test_indices.tolist()
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_model(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"🚀 Training AD Eye-Tracking Screening Model "
        f"on device: {device}"
    )

    dataset = ADEyeTrackingDataset(args.data_dir)

    print(f"Total subjects: {len(dataset)}")

    # -----------------------------------------------------
    # Reproducible train / validation / test split
    # -----------------------------------------------------
    train_indices, val_indices, test_indices = create_splits(
        dataset,
        seed=args.seed
    )

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    print(f"Training subjects:   {len(train_set)}")
    print(f"Validation subjects: {len(val_set)}")
    print(f"Test subjects:       {len(test_set)}")

    # -----------------------------------------------------
    # Save exact split
    # -----------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    split_path = os.path.join(
        args.save_dir,
        "data_split.json"
    )

    split_data = {
        "seed": args.seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }

    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"✅ Data split saved to: {split_path}")

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = ADClassifierModel().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=4,
        factor=0.5
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    for epoch in range(args.epochs):

        model.train()

        train_loss = 0.0

        for seqs, labels in train_loader:

            seqs = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda")
            ):

                preds = model(seqs)

                loss = criterion(
                    preds,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for seqs, labels in val_loader:

                seqs = seqs.to(device)
                labels = labels.to(device)

                outputs = model(seqs)

                val_loss += criterion(
                    outputs,
                    labels
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f}"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir,
                    "best_ad_tracker.pth"
                )
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= args.patience:

                print(
                    "🛑 Early stopping triggered."
                )

                break

    print(
        f"✅ Training completed. "
        f"Best model saved to "
        f"{args.save_dir}/best_ad_tracker.pth"
    )

    print(
        "⚠️ Test set was NOT used during training. "
        "It is reserved for final evaluation."
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ad_eyetracking/synthetic"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    train_model(args)