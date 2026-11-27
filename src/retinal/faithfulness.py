import argparse
from matplotlib.path import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from Gradcam import model, device, generate_gradcam, unnormalize
from transforms import get_valid_transforms
from config import GRADCAM_DIR, IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def deletion_insertion_curves(input_tensor, cam, class_idx, num_steps=20):
    """
    input_tensor: (C,H,W) normalized tensor, as fed to the model
    cam: (H,W) numpy array, 0-255, from generate_gradcam
    class_idx: which class's probability to track (use the predicted class)
    """
    H, W = cam.shape
    n_pixels = H * W
    order = np.argsort(-cam.flatten())  # most important first

    # A blurred version acts as the "neutral" baseline (grey-out target),
    # standard practice - preserves color statistics better than plain zeros.
    baseline = F.avg_pool2d(input_tensor.unsqueeze(0), kernel_size=11,
                             stride=1, padding=5, count_include_pad=False)[0]

    step_sizes = np.linspace(0, n_pixels, num_steps + 1).astype(int)

    deletion_probs = []
    insertion_probs = []

    with torch.no_grad():
        for i in range(num_steps + 1):
            k = step_sizes[i]
            mask_flat = np.zeros(n_pixels, dtype=bool)
            mask_flat[order[:k]] = True
            mask = torch.from_numpy(mask_flat.reshape(H, W)).to(device)

            # Deletion: remove top-k important pixels from the REAL image
            del_img = input_tensor.clone()
            for c in range(3):
                del_img[c][mask] = baseline[c][mask]
            prob = F.softmax(model(del_img.unsqueeze(0).to(device)), dim=1)[0, class_idx]
            deletion_probs.append(prob.item())

            # Insertion: reveal top-k important pixels on TOP of the baseline
            ins_img = baseline.clone()
            for c in range(3):
                ins_img[c][mask] = input_tensor[c][mask]
            prob = F.softmax(model(ins_img.unsqueeze(0).to(device)), dim=1)[0, class_idx]
            insertion_probs.append(prob.item())

    fractions = step_sizes / n_pixels
    deletion_auc = np.trapezoid(deletion_probs, fractions)
    insertion_auc = np.trapezoid(insertion_probs, fractions)
    return fractions, deletion_probs, insertion_probs, deletion_auc, insertion_auc


_RD = (DATA_DIR / "retina") if (DATA_DIR / "retina" / "train.csv").exists() else DATA_DIR  # accepts data/retina or data


def run(num_samples=20, seed=42):
    df = pd.read_csv(_RD / "train.csv")
    image_dir = _RD / "train_images"
    transform = get_valid_transforms()

    # Sample across classes, not just one-per-class, for a real average.
    # (Deliberately NOT using groupby().apply() here - pandas can silently
    # drop the grouping column depending on version when the applied
    # function returns a same-shaped result per group. A plain loop + concat
    # avoids that entirely.)
    per_class = max(1, num_samples // df["diagnosis"].nunique())
    groups = []
    for _, group in df.groupby("diagnosis"):
        n = min(len(group), per_class)
        groups.append(group.sample(n=n, random_state=seed))
    sample_df = pd.concat(groups, ignore_index=True)

    results = []
    for _, row in sample_df.iterrows():
        image_id, true_label = row["id_code"], int(row["diagnosis"])
        image_path = image_dir / f"{image_id}.png"
        image = np.array(Image.open(image_path).convert("RGB"))
        input_tensor = transform(image=image)["image"]

        cam, pred_class = generate_gradcam(input_tensor, class_idx=true_label)
        cam_norm = cam.astype(np.float32) / 255.0

        _, _, _, del_auc, ins_auc = deletion_insertion_curves(
            input_tensor, cam_norm, class_idx=pred_class)

        results.append({"image_id": image_id, "true": true_label,
                         "predicted": pred_class,
                         "deletion_auc": del_auc, "insertion_auc": ins_auc})
        print(f"{image_id}: true={true_label} pred={pred_class} "
              f"deletion_auc={del_auc:.3f} insertion_auc={ins_auc:.3f}")

    results_df = pd.DataFrame(results)
    out_csv = GRADCAM_DIR / "faithfulness_scores.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY (report these numbers in the paper) ===")
    print(f"Mean Deletion AUC:  {results_df['deletion_auc'].mean():.3f}  (lower = more faithful)")
    print(f"Mean Insertion AUC: {results_df['insertion_auc'].mean():.3f}  (higher = more faithful)")
    print(f"Saved per-image scores -> {out_csv}")

    # A quick reference bar so you can spot-check any single image against the average
    plt.figure(figsize=(6, 4))
    plt.bar(results_df["image_id"], results_df["deletion_auc"], color="#c0392b", label="Deletion AUC (lower=better)")
    plt.bar(results_df["image_id"], results_df["insertion_auc"], color="#27ae60", alpha=0.6, label="Insertion AUC (higher=better)")
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRADCAM_DIR / "faithfulness_summary.png", dpi=150)
    print(f"Saved summary chart -> {GRADCAM_DIR / 'faithfulness_summary.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    run(num_samples=args.num_samples)




















"""
STEP 5a - Grad-CAM faithfulness validation (deletion/insertion test).

Turns "this heatmap looks plausible to me" into a number. This directly
addresses cases like Image 3 and Image 5 from your sanity check - a blob
that isn't obviously anatomical might still be faithful (the model really
does rely on those pixels), or might not be. This test tells you which.

METHOD (Petsiuk et al., "RISE", 2018 - standard in XAI papers):
  DELETION: start from the real image, progressively grey out the pixels
  Grad-CAM ranked most important first, and track how fast the model's
  confidence in its predicted class collapses. A STEEP, FAST drop = the CAM
  found pixels the model actually depends on (faithful). A flat/slow drop =
  the CAM is highlighting pixels the model doesn't really use (unfaithful -
  possibly just texture/edge bias).
  Lower deletion-AUC = better (faster collapse = more faithful).

  INSERTION: the mirror test - start from a blurred/grey image and
  progressively reveal the top-ranked pixels, tracking how fast confidence
  RISES. Higher insertion-AUC = better.

This file lives alongside Gradcam.py in src/retinal/ and imports directly
from it, so it reuses your exact model, checkpoint, and CAM logic - no
architecture guessing.

Usage (run from src/retinal/, same as Gradcam.py):
    python step5a_faithfulness.py --num_samples 20
"""