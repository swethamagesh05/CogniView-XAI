from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EYE_TRACKING_DIR = Path(__file__).resolve().parent

DATA_DIR = EYE_TRACKING_DIR / "data" / "ad_eyetracking" / "synthetic"

MODEL_PATH = EYE_TRACKING_DIR / "checkpoints" / "best_ad_tracker.pth"

OUTPUT_DIR = PROJECT_ROOT / "results" / "eye_tracking_shap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Allow importing model.py and dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

#from eye_tracking.step6_integrated_gradients import EYE_TRACKING_DIR
from model import ADClassifierModel
from dataset import ADEyeTrackingDataset


FEATURE_NAMES = [
    "Saccadic Latency",
    "Antisaccade Error",
    "Fixation Duration",
    "Saccade Amplitude"
]


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_DIR}")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = ADClassifierModel().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("✅ ADClassifierModel loaded successfully")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
dataset = ADEyeTrackingDataset(
    str(DATA_DIR),
    max_seq_len=200
)

print(f"✅ Dataset loaded: {len(dataset)} subjects")


# ---------------------------------------------------------
# Prepare a small background set
# ---------------------------------------------------------
background_samples = []

for i in range(min(20, len(dataset))):
    sequence, _ = dataset[i]
    background_samples.append(sequence)

background = torch.stack(background_samples).to(device)

print(f"Background shape: {background.shape}")


# ---------------------------------------------------------
# Prepare samples to explain
# ---------------------------------------------------------
explain_samples = []

for i in range(min(20, len(dataset))):
    sequence, _ = dataset[i]
    explain_samples.append(sequence)

explain_data = torch.stack(explain_samples).to(device)

print(f"Explanation data shape: {explain_data.shape}")


# ---------------------------------------------------------
# ---------------------------------------------------------
# SHAP-compatible model wrapper
# ---------------------------------------------------------
class SHAPModelWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        output = self.base_model(x)

        # SHAP expects 2D output: [batch, 1]
        if output.ndim == 1:
            output = output.unsqueeze(1)

        return output


shap_model = SHAPModelWrapper(model).to(device)
shap_model.eval()


# ---------------------------------------------------------
# SHAP Gradient Explainer
# ---------------------------------------------------------
print("\n⏳ Running SHAP analysis...")

explainer = shap.GradientExplainer(
    shap_model,
    background
)
shap_values = explainer.shap_values(
    explain_data
)

# ---------------------------------------------------------
# Handle SHAP output formats
# ---------------------------------------------------------
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

print(f"Raw SHAP shape: {shap_values.shape}")


# ---------------------------------------------------------
# Normalize SHAP dimensions
# ---------------------------------------------------------
# Expected final shape:
# (samples, sequence_length, features)
#
# Depending on SHAP/PyTorch version, an extra dimension
# may appear for the single-output model.

if shap_values.ndim == 4:
    shap_values = np.squeeze(shap_values, axis=-1)

if shap_values.ndim != 3:
    raise RuntimeError(
        f"Unexpected SHAP shape: {shap_values.shape}"
    )


# ---------------------------------------------------------
# Aggregate importance across time
# ---------------------------------------------------------
#
# Mean absolute SHAP value over:
#   1. subjects
#   2. sequence/time steps
#
# Result = importance of each clinical biomarker.

feature_importance = np.mean(
    np.abs(shap_values),
    axis=(0, 1)
)

# Safety check
if len(feature_importance) != 4:
    raise RuntimeError(
        f"Expected 4 feature importance values, "
        f"got {len(feature_importance)}"
    )


# ---------------------------------------------------------
# Create results table
# ---------------------------------------------------------
results = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Mean_Absolute_SHAP": feature_importance
})

results = results.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

csv_path = OUTPUT_DIR / "shap_feature_importance.csv"

results.to_csv(
    csv_path,
    index=False
)

print("\n================ SHAP FEATURE IMPORTANCE ================")
print(results.to_string(index=False))

print(f"\n✅ CSV saved to:")
print(csv_path)


# ---------------------------------------------------------
# Publication-style feature importance plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.barh(
    results["Feature"],
    results["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Oculomotor Biomarker")
plt.title(
    "SHAP Feature Importance for AD/MCI Eye-Tracking Screening"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plot_path = OUTPUT_DIR / "shap_feature_importance.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"✅ SHAP plot saved to:")
print(plot_path)

print("\n🎉 SHAP analysis completed successfully!")