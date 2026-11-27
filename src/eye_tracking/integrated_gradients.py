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
    model = ADClassifierModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def explain_patient(model, device, features_tensor, subject_id, true_label, save_dir):
    """
    features_tensor: (seq_len, 4) already normalized/padded, as returned by
    ADEyeTrackingDataset.__getitem__
    """
    input_tensor = features_tensor.unsqueeze(0).to(device)  # (1, seq_len, 4)
    input_tensor.requires_grad_()

    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor)).item()

    ig = IntegratedGradients(lambda x: torch.sigmoid(model(x)))
    baseline = torch.zeros_like(input_tensor)  # "no signal" baseline
    attributions = ig.attribute(input_tensor, baseline, n_steps=50)
    attr = attributions.squeeze(0).detach().cpu().numpy()  # (seq_len, 4)

    # ---- Heatmap: time (x) vs feature (y) ----
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(attr.T, aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(attr).max(), vmax=np.abs(attr).max())
    ax.set_yticks(range(4))
    ax.set_yticklabels(FEATURE_NAMES)
    ax.set_xlabel("Timestep in recording")
    ax.set_title(f"{subject_id} | true={true_label} | predicted risk={prob:.3f}")
    plt.colorbar(im, ax=ax, label="Attribution (red = raises risk score)")
    plt.tight_layout()
    fname = f"{save_dir}/{subject_id}_ig_heatmap.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    # Per-feature total contribution for this one patient (for the report)
    per_feature = np.abs(attr).sum(axis=0)
    return prob, per_feature, fname


def run(num_subjects=30, subject_ids=None):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = ADEyeTrackingDataset(DATA_DIR)
    model, device = load_model()

    # Select subjects
    if subject_ids:
        subject_to_index = {
            str(row["subject_id"]): i
            for i, row in dataset.metadata.iterrows()
        }

        indices = []
        for sid in subject_ids:
            if sid not in subject_to_index:
                print(f"⚠️ Subject not found: {sid}")
            else:
                indices.append(subject_to_index[sid])

        if not indices:
            print("❌ No valid subject IDs found.")
            return
    else:
        n = min(num_subjects, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, n).astype(int)

    rows = []
    all_feature_importance = np.zeros(4)

    for i in indices:
        features_tensor, label = dataset[i]
        subject_id = dataset.metadata.iloc[i]["subject_id"]
        true_label = int(label.item())

        prob, per_feature, heatmap_path = explain_patient(
            model, device, features_tensor, subject_id, true_label, OUT_DIR
        )

        all_feature_importance += per_feature

        rows.append({
            "subject_id": subject_id,
            "true_label": true_label,
            "predicted_prob": prob,
            **{
                f"importance_{f}": v
                for f, v in zip(FEATURE_NAMES, per_feature)
            }
        })

        print(
            f"{subject_id}: true={true_label} "
            f"risk={prob:.3f} "
            f"top_feature={FEATURE_NAMES[int(np.argmax(per_feature))]}"
        )

    # Only create/update summary files when running the normal batch mode
    if not subject_ids:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(
            OUT_DIR / "ig_per_patient.csv",
            index=False
        )

        ranking = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "mean_abs_attribution": all_feature_importance / len(indices)
        }).sort_values(
            "mean_abs_attribution",
            ascending=False
        )

        ranking.to_csv(
            OUT_DIR / "global_feature_ranking.csv",
            index=False
        )

        print("\n=== GLOBAL FEATURE IMPORTANCE ===")
        print(ranking.to_string(index=False))

        plt.figure(figsize=(6, 4))
        plt.barh(
            ranking["feature"],
            ranking["mean_abs_attribution"]
        )
        plt.xlabel("Mean |Integrated Gradients attribution|")
        plt.title("Which oculomotor biomarkers drive AD/MCI risk")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / "global_feature_ranking.png",
            dpi=150
        )
        plt.close()

    print(f"\nSaved heatmaps to {OUT_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_subjects",
        type=int,
        default=30
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific subject IDs, e.g. subject_001 subject_040"
    )

    args = parser.parse_args()

    run(
        num_subjects=args.num_subjects,
        subject_ids=args.subjects
    )
















"""
STEP 6 - Explainability for the REAL oculomotor screening model
(src/eye_tracking/model.py :: ADClassifierModel, checkpoints/best_ad_tracker.pth).

This is the model behind your AUC=0.853 "Oculomotor Marker" result - the
Conv1D+BiLSTM binary classifier over 4 features:
    saccadic_latency, antisaccade_error, fixation_duration, saccade_amplitude
These are standard, literature-recognized oculomotor biomarkers for AD/MCI,
which makes this explanation genuinely meaningful for the paper (not just
"feature 2 mattered" - you can say "antisaccade error was the dominant
driver", which reviewers/clinicians can sanity-check against prior work).

METHOD: Integrated Gradients (Captum) - attributes the model's output back
to every (timestep, feature) cell of the input sequence. Two outputs:
  1. PER-PATIENT: a heatmap over the (200 timesteps x 4 features) grid
     showing WHEN in the recording and WHICH feature drove that patient's
     risk score - your local/patient-level explanation.
  2. GLOBAL: aggregate |attribution| per feature across many patients -
     a feature-importance ranking for the "top predictive biomarkers"
     table in your results section.

Install once:
    pip install captum

Usage (run from src/eye_tracking/, same convention as Gradcam.py):
    python step6_integrated_gradients.py --num_subjects 30
"""