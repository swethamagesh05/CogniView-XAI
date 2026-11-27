from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "plr_features"

df = pd.read_csv(
    OUT_DIR / "plr_session_features.csv"
)

# 1. PLR feature distributions

features = {
    "baseline_ratio_pct": "Baseline Pupil-to-Iris Ratio (%)",
    "constriction_pct": "Relative Constriction (%)",
    "time_to_peak_s": "Time to Peak (s)",
    "baseline_noise_pct_points": "Baseline Noise (pp)"
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (column, label) in zip(axes.flat, features.items()):

    values = df[column].dropna()

    ax.hist(
        values,
        bins=8,
        color="#4472C4",
        edgecolor="black",
        alpha=0.85
    )

    ax.axvline(
        values.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {values.mean():.2f}"
    )

    ax.set_xlabel(label)
    ax.set_ylabel("Number of Sessions")
    ax.legend()
    ax.grid(alpha=0.2)

fig.suptitle(
    "CogniView: Exploratory PLR Feature Distributions",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "plr_feature_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 2. Constriction amplitude by session

colors = df["feature_status"].map({
    "EXPLORATORY_RESPONSE": "#2E8B57",
    "LOW_SIGNAL_TO_NOISE": "#E69F00"
}).fillna("gray")

plt.figure(figsize=(12, 6))

plt.bar(
    range(1, len(df) + 1),
    df["constriction_pct"],
    color=colors,
    edgecolor="black"
)

plt.xticks(range(1, len(df) + 1))
plt.xlabel("PLR Session")
plt.ylabel("Relative Constriction (%)")
plt.title("CogniView: Session-wise Constriction Amplitude")

from matplotlib.patches import Patch

plt.legend(handles=[
    Patch(
        color="#2E8B57",
        label="Exploratory response"
    ),
    Patch(
        color="#E69F00",
        label="Low signal-to-noise"
    )
])

plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUT_DIR / "plr_constriction_comparison.png",
    dpi=300
)

plt.close()

# 3. Response status distribution

counts = df["feature_status"].value_counts()

plt.figure(figsize=(7, 6))

plt.pie(
    counts.values,
    labels=[
        label.replace("_", " ").title()
        for label in counts.index
    ],
    autopct="%1.1f%%",
    colors=[
        "#2E8B57" if label == "EXPLORATORY_RESPONSE"
        else "#E69F00"
        for label in counts.index
    ],
    startangle=90
)

plt.title("CogniView: Exploratory PLR Response Status")

plt.tight_layout()

plt.savefig(
    OUT_DIR / "plr_response_distribution.png",
    dpi=300
)

plt.close()

print("\n========== PLR VISUALIZATION COMPLETE ==========")

print("Sessions:", len(df))

print("\nGenerated files:")

for name in [
    "plr_feature_distribution.png",
    "plr_constriction_comparison.png",
    "plr_response_distribution.png"
]:
    print(OUT_DIR / name)