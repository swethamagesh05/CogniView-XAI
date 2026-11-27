import argparse
import os
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from dataset import ADEyeTrackingDataset
from model import ADClassifierModel


def evaluate_model(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"🔍 Evaluating AD Eye-Tracking Model "
        f"on device: {device}"
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    dataset = ADEyeTrackingDataset(
        args.data_dir
    )

    # -----------------------------------------------------
    # Load saved split
    # -----------------------------------------------------
    split_path = os.path.join(
        args.split_dir,
        "data_split.json"
    )

    if not os.path.exists(split_path):

        raise FileNotFoundError(
            f"Split file not found:\n{split_path}\n\n"
            "Run train.py first to create the "
            "reproducible train/validation/test split."
        )

    with open(split_path, "r") as f:
        split_data = json.load(f)

    test_indices = split_data["test_indices"]

    print(
        f"Total dataset: {len(dataset)} subjects"
    )

    print(
        f"Held-out test set: {len(test_indices)} subjects"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Evaluate ONLY the held-out test set
    # -----------------------------------------------------
    test_set = Subset(
        dataset,
        test_indices
    )

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False
    )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------
    model = ADClassifierModel().to(device)

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device
        )
    )

    model.eval()

    print(
        f"✅ Model loaded from:\n{args.model_path}"
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    true_labels = []
    pred_probabilities = []

    with torch.no_grad():

        for seqs, labels in loader:

            seqs = seqs.to(device)

            logits = model(seqs)

            probs = torch.sigmoid(
                logits
            ).cpu().numpy()

            true_labels.extend(
                labels.numpy()
            )

            pred_probabilities.extend(
                probs
            )

    true_labels = np.array(
        true_labels
    ).astype(int)

    pred_probabilities = np.array(
        pred_probabilities
    )

    pred_classes = (
        pred_probabilities >= 0.5
    ).astype(int)

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------
    print(
        "\n================ HELD-OUT TEST REPORT ================"
    )

    print(
        classification_report(
            true_labels,
            pred_classes,
            target_names=[
                "Healthy Control",
                "MCI / AD Risk"
            ],
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------
    auc = roc_auc_score(
        true_labels,
        pred_probabilities
    )

    print(
        f"Held-Out Test ROC-AUC: {auc:.4f}"
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    cm = confusion_matrix(
        true_labels,
        pred_classes
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    # -----------------------------------------------------
    # Save output directory
    # -----------------------------------------------------
    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------
    predictions_df = pd.DataFrame({

        "test_dataset_index": test_indices,

        "true_label": true_labels,

        "predicted_probability": pred_probabilities,

        "predicted_class": pred_classes

    })

    prediction_path = os.path.join(
        args.output_dir,
        "held_out_test_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"✅ Test predictions saved to:\n"
        f"{prediction_path}"
    )

    # -----------------------------------------------------
    # ROC curve
    # -----------------------------------------------------
    fpr, tpr, _ = roc_curve(
        true_labels,
        pred_probabilities
    )

    save_path = os.path.join(
        args.output_dir,
        "ad_screening_roc_test.png"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        lw=2.5,
        label=f"Oculomotor Marker (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        lw=1.5,
        linestyle="--"
    )

    plt.xlim(
        [0.0, 1.0]
    )

    plt.ylim(
        [0.0, 1.05]
    )

    plt.xlabel(
        "False Positive Rate (1 - Specificity)",
        fontsize=11
    )

    plt.ylabel(
        "True Positive Rate (Sensitivity)",
        fontsize=11
    )

    plt.title(
        "CogniView: AD/MCI Eye-Tracking Screening\n"
        "Held-Out Test Performance",
        fontsize=12
    )

    plt.legend(
        loc="lower right",
        fontsize=11
    )

    plt.grid(
        alpha=0.3
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✅ Test ROC curve saved to:\n"
        f"{save_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ad_eyetracking/synthetic"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_ad_tracker.pth"
    )

    parser.add_argument(
        "--split_dir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/plots"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    args = parser.parse_args()

    evaluate_model(args)