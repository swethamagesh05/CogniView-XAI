
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.models import efficientnet_b0

from transforms import get_valid_transforms

# =================================================
# CONFIGURATION
# =================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

CHECKPOINT = (
    ROOT / "models" / "retina" / "best_model.pth"
)

OUTPUT_DIR = (
    ROOT / "results" / "gradcam_corrected"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
NUM_CLASSES = 5
BATCH_SIZE = 8

# =================================================
# MODEL ARCHITECTURE
# =================================================

class RetinaClassifier(nn.Module):

    def __init__(self):

        super().__init__()

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


# =================================================
# GRAD-CAM
# =================================================

def generate_gradcam(model, image):

    target_layer = model.backbone.features[-1]

    captured = {}

    def forward_hook(module, inputs, output):

        captured["activation"] = output

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        x = (
            image.unsqueeze(0)
            .to(DEVICE)
            .detach()
            .requires_grad_(True)
        )

        with torch.enable_grad():

            logits = model(x)

            predicted_class = int(
                logits.argmax(dim=1).item()
            )

            confidence = float(
                torch.softmax(logits, dim=1)[
                    0, predicted_class
                ].item()
            )

            activation = captured["activation"]

            gradients = torch.autograd.grad(
                logits[0, predicted_class],
                activation
            )[0]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = (
                weights * activation
            ).sum(dim=1, keepdim=True)

            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            cam = cam.detach().cpu().numpy()[0, 0]

            cam -= cam.min()

            maximum = float(cam.max())

            if maximum > 0:

                cam /= maximum

        return cam, predicted_class, confidence

    finally:

        handle.remove()


# =================================================
# MODEL PROBABILITY
# =================================================

def predict_probability(model, images, target):

    results = []

    with torch.inference_mode():

        for start in range(
            0, len(images), BATCH_SIZE
        ):

            batch = torch.stack(
                images[start:start + BATCH_SIZE]
            ).to(DEVICE)

            logits = model(batch)

            probs = torch.softmax(
                logits, dim=1
            )[:, target]

            results.extend(
                probs.cpu().numpy()
            )

    return np.asarray(results)


# =================================================
# DELETION AND INSERTION
# =================================================

def perturbation_curves(
    model,
    image,
    baseline,
    order,
    target,
    steps
):

    height, width = image.shape[-2:]

    n_pixels = height * width

    step_sizes = np.linspace(
        0,
        n_pixels,
        steps + 1
    ).astype(int)

    deletion_images = []
    insertion_images = []

    for k in step_sizes:

        mask = np.zeros(
            n_pixels,
            dtype=bool
        )

        mask[order[:k]] = True

        mask = torch.from_numpy(
            mask.reshape(height, width)
        ).unsqueeze(0)

        deletion_image = torch.where(
            mask,
            baseline,
            image
        )

        insertion_image = torch.where(
            mask,
            image,
            baseline
        )

        deletion_images.append(
            deletion_image
        )

        insertion_images.append(
            insertion_image
        )

    deletion = predict_probability(
        model,
        deletion_images,
        target
    )

    insertion = predict_probability(
        model,
        insertion_images,
        target
    )

    fractions = step_sizes / n_pixels

    integrate = (
        np.trapezoid
        if hasattr(np, "trapezoid")
        else np.trapz
    )

    deletion_auc = float(
        integrate(deletion, fractions)
    )

    insertion_auc = float(
        integrate(insertion, fractions)
    )

    return (
        fractions,
        deletion,
        insertion,
        deletion_auc,
        insertion_auc
    )


# =================================================
# DISPLAY IMAGE
# =================================================

def unnormalize(image):

    mean = np.array([
        0.485, 0.456, 0.406
    ])

    std = np.array([
        0.229, 0.224, 0.225
    ])

    image = image.numpy().transpose(1, 2, 0)

    return np.clip(
        image * std + mean,
        0,
        1
    )


# =================================================
# MAIN EXPERIMENT
# =================================================

def run(num_samples, steps, random_runs):

    if min(num_samples, steps, random_runs) < 1:

        raise ValueError(
            "All experiment counts must be positive."
        )

    print("Device:", DEVICE)

    rng = np.random.default_rng(SEED)

    torch.manual_seed(SEED)

    # ---------------------------------------------
    # DATASET VERIFICATION
    # ---------------------------------------------

    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    image_dir = DATA_DIR / "test_images"

    assert train_df["id_code"].is_unique
    assert test_df["id_code"].is_unique

    train_ids = set(train_df["id_code"])

    test_ids = set(test_df["id_code"])

    overlap = train_ids.intersection(test_ids)

    assert not overlap, "Image ID overlap detected."

    missing = [
        image_id
        for image_id in test_ids
        if not (
            image_dir / f"{image_id}.png"
        ).exists()
    ]

    if missing:

        raise FileNotFoundError(
            f"Missing test images: {len(missing)}"
        )

    print("Labeled dataset:", len(train_df))
    print("Official test dataset:", len(test_df))
    print("Overlapping image IDs:", len(overlap))
    print("Missing test images:", len(missing))

    # ---------------------------------------------
    # LOAD EXISTING CHECKPOINT
    # ---------------------------------------------

    model = RetinaClassifier().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True
        ),
        strict=True
    )

    model.eval()

    print("Retinal model loaded successfully.")

    transform = get_valid_transforms()

    # ---------------------------------------------
    # REPRODUCIBLE RANDOM SAMPLE
    # ---------------------------------------------

    sample_df = (
        test_df.sort_values("id_code")
        .sample(
            n=min(num_samples, len(test_df)),
            random_state=SEED
        )
        .reset_index(drop=True)
    )

    sample_df.to_csv(
        OUTPUT_DIR / "sample_manifest.csv",
        index=False
    )

    records = []
    curve_records = []
    random_records = []

    examples = []

    gradcam_deletion_curves = []
    gradcam_insertion_curves = []

    random_deletion_curves = []
    random_insertion_curves = []

    print("\n========== CORRECTED GRAD-CAM ==========")

    # ---------------------------------------------
    # PROCESS TEST IMAGES
    # ---------------------------------------------

    for index, row in sample_df.iterrows():

        image_id = row["id_code"]

        path = image_dir / f"{image_id}.png"

        with Image.open(path) as pil_image:

            raw_image = np.array(
                pil_image.convert("RGB")
            )

        image = transform(
            image=raw_image
        )["image"].cpu()

        cam, target, confidence = generate_gradcam(
            model,
            image
        )

        flat_cam = bool(
            np.ptp(cam) <= 1e-8
        )

        # Same blurred baseline for all methods.

        baseline = F.avg_pool2d(
            image.unsqueeze(0),
            kernel_size=11,
            stride=1,
            padding=5,
            count_include_pad=False
        )[0]

        # Random tie-breaking prevents arbitrary
        # spatial ordering for equal CAM values.

        tie_order = rng.permutation(cam.size)

        gradcam_order = tie_order[
            np.argsort(
                -cam.flatten()[tie_order],
                kind="stable"
            )
        ]

        (
            fractions,
            grad_del,
            grad_ins,
            grad_del_auc,
            grad_ins_auc
        ) = perturbation_curves(
            model,
            image,
            baseline,
            gradcam_order,
            target,
            steps
        )

        # -----------------------------------------
        # RANDOM ORDERING CONTROL
        # -----------------------------------------

        random_del = []
        random_ins = []

        random_del_auc = []
        random_ins_auc = []

        for repeat in range(random_runs):

            order = rng.permutation(cam.size)

            (
                _,
                deletion,
                insertion,
                deletion_auc,
                insertion_auc
            ) = perturbation_curves(
                model,
                image,
                baseline,
                order,
                target,
                steps
            )

            random_del.append(deletion)
            random_ins.append(insertion)

            random_del_auc.append(deletion_auc)
            random_ins_auc.append(insertion_auc)

            random_records.append({
                "image_id": image_id,
                "repeat": repeat + 1,
                "target_class": target,
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc
            })

        mean_random_del = np.mean(
            random_del,
            axis=0
        )

        mean_random_ins = np.mean(
            random_ins,
            axis=0
        )

        mean_random_del_auc = float(
            np.mean(random_del_auc)
        )

        mean_random_ins_auc = float(
            np.mean(random_ins_auc)
        )

        # Positive values favor Grad-CAM.

        deletion_advantage = (
            mean_random_del_auc - grad_del_auc
        )

        insertion_advantage = (
            grad_ins_auc - mean_random_ins_auc
        )

        records.append({
            "image_id": image_id,
            "predicted_class": target,
            "gradcam_target": target,
            "faithfulness_target": target,
            "prediction_confidence": confidence,
            "flat_gradcam": flat_cam,
            "gradcam_deletion_auc": grad_del_auc,
            "gradcam_insertion_auc": grad_ins_auc,
            "random_deletion_auc":
                mean_random_del_auc,
            "random_insertion_auc":
                mean_random_ins_auc,
            "deletion_advantage":
                deletion_advantage,
            "insertion_advantage":
                insertion_advantage
        })

        gradcam_deletion_curves.append(grad_del)
        gradcam_insertion_curves.append(grad_ins)

        random_deletion_curves.append(
            mean_random_del
        )

        random_insertion_curves.append(
            mean_random_ins
        )

        for i, fraction in enumerate(fractions):

            curve_records.append({
                "image_id": image_id,
                "fraction": float(fraction),
                "gradcam_deletion": grad_del[i],
                "gradcam_insertion": grad_ins[i],
                "random_deletion":
                    mean_random_del[i],
                "random_insertion":
                    mean_random_ins[i]
            })

        if len(examples) < 5:

            examples.append((
                image_id,
                image,
                cam,
                target,
                confidence
            ))

        print(
            f"[{index + 1}/{len(sample_df)}] "
            f"{image_id} | Class: {target} | "
            f"GradCAM Del: {grad_del_auc:.4f} | "
            f"Random Del: {mean_random_del_auc:.4f} | "
            f"GradCAM Ins: {grad_ins_auc:.4f} | "
            f"Random Ins: {mean_random_ins_auc:.4f}"
        )

    # ---------------------------------------------
    # SAVE NUMERICAL RESULTS
    # ---------------------------------------------

    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT_DIR / "faithfulness_scores.csv",
        index=False
    )

    pd.DataFrame(curve_records).to_csv(
        OUTPUT_DIR / "perturbation_curves.csv",
        index=False
    )

    pd.DataFrame(random_records).to_csv(
        OUTPUT_DIR / "random_baseline_runs.csv",
        index=False
    )

    # ---------------------------------------------
    # PLOT MEAN CURVES
    # ---------------------------------------------

    def plot_curves(
        gradcam_curves,
        random_curves,
        title,
        filename
    ):

        gradcam_curves = np.asarray(
            gradcam_curves
        )

        random_curves = np.asarray(
            random_curves
        )

        plt.figure(figsize=(8, 5))

        for curves, label, color in [
            (gradcam_curves, "Grad-CAM", "#4472C4"),
            (random_curves, "Random", "#D97706")
        ]:

            mean = curves.mean(axis=0)
            std = curves.std(axis=0)

            plt.plot(
                fractions * 100,
                mean,
                label=label,
                color=color,
                linewidth=2
            )

            plt.fill_between(
                fractions * 100,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=color,
                alpha=0.12
            )

        plt.xlabel("Perturbed pixels (%)")

        plt.ylabel(
            "Original predicted-class probability"
        )

        plt.title(title)

        plt.ylim(0, 1.05)

        plt.legend()

        plt.grid(alpha=0.2)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=300
        )

        plt.close()

    plot_curves(
        gradcam_deletion_curves,
        random_deletion_curves,
        "Grad-CAM vs Random: Deletion",
        "deletion_curve_comparison.png"
    )

    plot_curves(
        gradcam_insertion_curves,
        random_insertion_curves,
        "Grad-CAM vs Random: Insertion",
        "insertion_curve_comparison.png"
    )

    # ---------------------------------------------
    # AUC COMPARISON
    # ---------------------------------------------

    gradcam_means = [
        results["gradcam_deletion_auc"].mean(),
        results["gradcam_insertion_auc"].mean()
    ]

    random_means = [
        results["random_deletion_auc"].mean(),
        results["random_insertion_auc"].mean()
    ]

    x = np.arange(2)

    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2,
        gradcam_means,
        width,
        label="Grad-CAM",
        color="#4472C4"
    )

    bars2 = ax.bar(
        x + width / 2,
        random_means,
        width,
        label="Random",
        color="#D97706"
    )

    ax.bar_label(bars1, fmt="%.4f")
    ax.bar_label(bars2, fmt="%.4f")

    ax.set_xticks(x)

    ax.set_xticklabels([
        "Deletion\n(Lower is better)",
        "Insertion\n(Higher is better)"
    ])

    ax.set_ylabel("Mean AUC")

    ax.set_ylim(0, 1.15)

    ax.set_title(
        "Corrected Retinal Grad-CAM Faithfulness"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "faithfulness_auc_comparison.png",
        dpi=300
    )

    plt.close()

    # ---------------------------------------------
    # CORRECTED GRAD-CAM EXAMPLES
    # ---------------------------------------------

    fig, axes = plt.subplots(
        len(examples),
        2,
        figsize=(8, 3.5 * len(examples)),
        squeeze=False
    )

    for i, example in enumerate(examples):

        image_id, image, cam, target, conf = example

        display_image = unnormalize(image)

        axes[i, 0].imshow(display_image)

        axes[i, 0].set_title(
            f"Original: {image_id}"
        )

        axes[i, 1].imshow(display_image)

        if np.ptp(cam) > 1e-8:

            axes[i, 1].imshow(
                cam,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1
            )

        axes[i, 1].set_title(
            f"Grad-CAM target: Class {target}\n"
            f"Model probability: {conf:.3f}"
        )

        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gradcam_corrected_examples.png",
        dpi=300
    )

    plt.close()

    # ---------------------------------------------
    # SUMMARY
    # ---------------------------------------------

    summary = {
        "model": "EfficientNet-B0",
        "dataset": "APTOS 2019 official test images",
        "dataset_size": len(test_df),
        "evaluated_images": len(results),
        "overlapping_image_ids": len(overlap),
        "missing_images": len(missing),
        "target": "original_predicted_class",
        "target_consistency": True,
        "ground_truth_available": False,
        "model_retrained": False,
        "random_runs_per_image": random_runs,
        "perturbation_steps": steps,
        "baseline": "11x11 average pooling blur",
        "seed": SEED,
        "checkpoint_sha256": hashlib.sha256(
            CHECKPOINT.read_bytes()
        ).hexdigest(),
        "mean_gradcam_deletion_auc":
            float(gradcam_means[0]),
        "mean_gradcam_insertion_auc":
            float(gradcam_means[1]),
        "mean_random_deletion_auc":
            float(random_means[0]),
        "mean_random_insertion_auc":
            float(random_means[1]),
        "mean_deletion_advantage":
            float(results["deletion_advantage"].mean()),
        "mean_insertion_advantage":
            float(results["insertion_advantage"].mean()),
        "deletion_wins": int(
            (results["deletion_advantage"] > 0).sum()
        ),
        "insertion_wins": int(
            (results["insertion_advantage"] > 0).sum()
        ),
        "flat_gradcam_count": int(
            results["flat_gradcam"].sum()
        ),
        "predicted_class_distribution": {
            str(k): int(v)
            for k, v in results[
                "predicted_class"
            ].value_counts().items()
        },
        "clinical_alzheimers_validation": False,
        "limitations": [
            "Unlabeled official test images",
            "Classification accuracy cannot be computed",
            "Patient-level separation is unverified",
            "Image-content duplication is unverified",
            "Faithfulness depends on perturbation baseline",
            "Random pixel ordering is a limited control",
            "Small exploratory image sample",
            "Retinal model predicts diabetic retinopathy",
            "No clinical Alzheimer's validation"
        ]
    }

    with open(
        OUTPUT_DIR / "faithfulness_summary.json",
        "w"
    ) as file:

        json.dump(summary, file, indent=4)

    # ---------------------------------------------
    # TERMINAL OUTPUT
    # ---------------------------------------------

    print("\n========== FAITHFULNESS SUMMARY ==========")

    print(
        f"Grad-CAM Deletion AUC: "
        f"{gradcam_means[0]:.4f}"
    )

    print(
        f"Random Deletion AUC:   "
        f"{random_means[0]:.4f}"
    )

    print(
        f"Grad-CAM Insertion AUC: "
        f"{gradcam_means[1]:.4f}"
    )

    print(
        f"Random Insertion AUC:   "
        f"{random_means[1]:.4f}"
    )

    print(
        "Mean deletion advantage:",
        summary["mean_deletion_advantage"]
    )

    print(
        "Mean insertion advantage:",
        summary["mean_insertion_advantage"]
    )

    print(
        "Deletion wins:",
        summary["deletion_wins"],
        "/",
        len(results)
    )

    print(
        "Insertion wins:",
        summary["insertion_wins"],
        "/",
        len(results)
    )

    print(
        "Flat Grad-CAM maps:",
        summary["flat_gradcam_count"]
    )

    print(
        "Predicted class distribution:",
        summary["predicted_class_distribution"]
    )

    print("\n========== EXPERIMENT COMPLETE ==========")

    print("Images evaluated:", len(results))

    print("Target consistency: True")

    print("Model retrained: False")

    print("Ground truth available: False")

    print("Clinical validation: False")

    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20
    )

    parser.add_argument(
        "--random_runs",
        type=int,
        default=3
    )

    args = parser.parse_args()

    run(
        args.num_samples,
        args.steps,
        args.random_runs
    )
