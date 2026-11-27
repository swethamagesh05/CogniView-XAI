import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from pathlib import Path
from model import ADClassifierModel
from dataset import ADEyeTrackingDataset

FEATURE_NAMES = ["saccadic_latency", "antisaccade_error",
                  "fixation_duration", "saccade_amplitude"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EYE_TRACKING_DIR = Path(__file__).resolve().parent

MODEL_PATH = EYE_TRACKING_DIR / "checkpoints" / "best_ad_tracker.pth"
DATA_DIR = EYE_TRACKING_DIR / "data" / "ad_eyetracking" / "synthetic"
OUT_DIR = PROJECT_ROOT / "results" / "xai_eyetracking"

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model checkpoint: {MODEL_PATH}")
    print(f"Data directory: {DATA_DIR}")

    model = ADClassifierModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("✅ ADClassifierModel loaded successfully")

    return model, device


def deletion_insertion_curves(model, device, input_seq, attr, num_steps=20):
    """
    input_seq: (seq_len, 4) tensor, the real sequence
    attr: (seq_len, 4) numpy array of IG attributions for this subject
    """
    seq_len, n_features = input_seq.shape
    n_cells = seq_len * n_features
    order = np.argsort(-np.abs(attr).flatten())  # most important first

    step_sizes = np.linspace(0, n_cells, num_steps + 1).astype(int)
    deletion_probs, insertion_probs = [], []

    with torch.no_grad():
        for k in step_sizes:
            mask_flat = np.zeros(n_cells, dtype=bool)
            mask_flat[order[:k]] = True
            mask = mask_flat.reshape(seq_len, n_features)
            mask_t = torch.from_numpy(mask)

            # Deletion: zero out top-k important cells in the real sequence
            del_seq = input_seq.clone()
            del_seq[mask_t] = 0.0
            prob = torch.sigmoid(model(del_seq.unsqueeze(0).to(device))).item()
            deletion_probs.append(prob)

            # Insertion: reveal top-k cells on an all-zero sequence
            ins_seq = torch.zeros_like(input_seq)
            ins_seq[mask_t] = input_seq[mask_t]
            prob = torch.sigmoid(model(ins_seq.unsqueeze(0).to(device))).item()
            insertion_probs.append(prob)

    fractions = step_sizes / n_cells
    deletion_auc = np.trapezoid(deletion_probs, fractions)
    insertion_auc = np.trapezoid(insertion_probs, fractions)
    return deletion_auc, insertion_auc


def run(num_subjects=20):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = ADEyeTrackingDataset(DATA_DIR)
    model, device = load_model()
    ig = IntegratedGradients(lambda x: torch.sigmoid(model(x)))

    n = min(num_subjects, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n).astype(int)

    rows = []
    for i in indices:
        features_tensor, label = dataset[i]
        subject_id = dataset.metadata.iloc[i]["subject_id"]
        true_label = int(label.item())

        input_tensor = features_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_()
        with torch.no_grad():
            pred_prob = torch.sigmoid(model(input_tensor)).item()

        baseline = torch.zeros_like(input_tensor)
        attributions = ig.attribute(input_tensor, baseline, n_steps=50)
        attr = attributions.squeeze(0).detach().cpu().numpy()

        del_auc, ins_auc = deletion_insertion_curves(
            model, device, features_tensor, attr)

        rows.append({"subject_id": subject_id, "true_label": true_label,
                      "predicted_prob": pred_prob,
                      "deletion_auc": del_auc, "insertion_auc": ins_auc})
        print(f"{subject_id}: true={true_label} pred={pred_prob:.3f} "
              f"deletion_auc={del_auc:.3f} insertion_auc={ins_auc:.3f}")

    results_df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "eyetracking_faithfulness_scores.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY (report these numbers in the paper) ===")
    print(f"Mean Deletion AUC:  {results_df['deletion_auc'].mean():.3f}  (lower = more faithful)")
    print(f"Mean Insertion AUC: {results_df['insertion_auc'].mean():.3f}  (higher = more faithful)")
    print(f"Saved -> {out_csv}")

    plt.figure(figsize=(6, 4))
    plt.bar(results_df["subject_id"], results_df["deletion_auc"], color="#c0392b", label="Deletion AUC (lower=better)")
    plt.bar(results_df["subject_id"], results_df["insertion_auc"], color="#27ae60", alpha=0.6, label="Insertion AUC (higher=better)")
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    plt.tight_layout()
    fig_path = OUT_DIR / "eyetracking_faithfulness_summary.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_subjects", type=int, default=20)
    args = parser.parse_args()
    run(num_subjects=args.num_subjects)













"""
STEP 7 - Deletion/Insertion faithfulness test for the oculomotor screening
model (ADClassifierModel), matching the methodology already used for the
retina Grad-CAM (step5a_faithfulness.py). Same logic, different input shape:
instead of ranking and masking PIXELS, this ranks and masks (timestep,
feature) CELLS in the (200, 4) sequence, using the same Integrated Gradients
attributions from step6.

DELETION: start from the real sequence, progressively zero out the cells IG
ranked most important first, track how fast the predicted risk score
collapses toward the "no signal" baseline. Fast collapse = faithful.
INSERTION: mirror test - start from all-zero, progressively reveal the
top-ranked cells, track how fast risk rises toward the real prediction.

Only run this AFTER retraining on matched data (the earlier broken
checkpoint would make these numbers meaningless, same as it made the
feature ranking untrustworthy).

Usage (run from src/eye_tracking/, same convention as step6):
    python step7_faithfulness_eyetracking.py --num_subjects 20
"""