from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "plr_raw"
OUT_DIR = ROOT / "results" / "plr_features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

quality = pd.read_csv(
    OUT_DIR / "plr_quality_report.csv"
).set_index("session")

files = sorted(RAW_DIR.glob("plr_session_*.csv"))

rows = []

plt.figure(figsize=(12, 7))

for file in files:

    df = pd.read_csv(file)

    df["timestamp_s"] = pd.to_numeric(
        df["timestamp_s"], errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp_s"])
        .sort_values("timestamp_s")
        .reset_index(drop=True)
    )

    left = pd.to_numeric(
        df["left_pupil_iris_ratio"],
        errors="coerce"
    )

    right = pd.to_numeric(
        df["right_pupil_iris_ratio"],
        errors="coerce"
    )

    left = left.where(
        left.between(0, 1, inclusive="neither")
    )

    right = right.where(
        right.between(0, 1, inclusive="neither")
    )

    ratio = pd.concat(
        [left, right], axis=1
    ).mean(axis=1) * 100

    baseline_mask = df["stimulus_state"].eq("baseline")
    flash_mask = df["stimulus_state"].eq("flash")
    recovery_mask = df["stimulus_state"].eq("recovery")

    if not (
        baseline_mask.any()
        and flash_mask.any()
        and recovery_mask.any()
    ):
        print(f"{file.stem}: missing phase")
        continue

    flash_onset = df.loc[
        flash_mask, "timestamp_s"
    ].min()

    recovery_onset = df.loc[
        recovery_mask, "timestamp_s"
    ].min()

    time = df["timestamp_s"] - flash_onset

    # Smooth without zero-padding.
    # Preserve missing observations.
    smooth = ratio.rolling(
        window=3,
        center=True,
        min_periods=2
    ).median().where(ratio.notna())

    baseline = smooth[
        baseline_mask & (time >= -2)
    ].median()

    baseline_noise = (
        smooth[baseline_mask] - baseline
    ).abs().median() * 1.4826

    flags = quality.loc[
        file.stem, "quality_flags"
    ]

    row = {
        "session": file.stem,
        "baseline_ratio_pct": baseline,
        "baseline_noise_pct_points": baseline_noise,
        "valid_pct": quality.loc[file.stem, "valid_pct"],
        "quality_flags": flags,
        "clinical_validation": False,
        "feature_status": "REVIEW"
    }

    # Search for peak constriction
    # within 2 seconds of observed flash onset.
    search = (
        (time >= 0)
        & (time <= 2.0)
        & smooth.notna()
    )

    if (
        pd.isna(baseline)
        or search.sum() < 5
    ):
        row["feature_status"] = "INSUFFICIENT_DATA"
        rows.append(row)
        continue

    peak_idx = smooth[search].idxmin()

    minimum = smooth.loc[peak_idx]
    peak_time = time.loc[peak_idx]

    amplitude = baseline - minimum

    relative_constriction = (
        amplitude / baseline * 100
        if baseline > 0 else np.nan
    )

    row.update({
        "min_ratio_pct": minimum,
        "constriction_amplitude_pct_points": amplitude,
        "constriction_pct": relative_constriction,
        "time_to_peak_s": peak_time
    })

    latency = np.nan
    recovery_time = np.nan

    # Provisional noise-aware response check.
    response_detected = (
        pd.notna(baseline_noise)
        and amplitude > max(0.5, 2 * baseline_noise)
    )

    if response_detected:

        threshold = baseline - 0.1 * amplitude

        latency_mask = (
            (time >= 0)
            & (time <= peak_time)
            & (smooth <= threshold)
        )

        if latency_mask.any():
            latency = time[latency_mask].iloc[0]

        recovery_target = minimum + 0.75 * amplitude

        recovered = (
            (time > peak_time)
            & (df["timestamp_s"] >= recovery_onset)
            & (smooth >= recovery_target)
        )

        if recovered.any():
            recovery_time = (
                time[recovered].iloc[0] - peak_time
            )

        row["feature_status"] = "EXPLORATORY_RESPONSE"

    else:
        row["feature_status"] = "LOW_SIGNAL_TO_NOISE"

    row["constriction_latency_s"] = latency
    row["recovery_time_75pct_s"] = recovery_time
    row["response_detected"] = response_detected

    rows.append(row)

    # Baseline-relative response curve
    normalized = (
        (smooth - baseline) / baseline * 100
    )

    plt.plot(
        time,
        normalized,
        alpha=0.35,
        linewidth=1
    )

    print(
        f"{file.stem}: "
        f"constriction={relative_constriction:.2f}% | "
        f"latency={latency:.3f}s | "
        f"recovery={recovery_time:.3f}s | "
        f"{row['feature_status']}"
    )

if not rows:
    raise RuntimeError("No sessions processed.")

results = pd.DataFrame(rows)

results.to_csv(
    OUT_DIR / "plr_session_features.csv",
    index=False
)

numeric = results.select_dtypes(include="number")

numeric.describe().T.to_csv(
    OUT_DIR / "plr_feature_summary.csv"
)

plt.axvline(
    0,
    linestyle="--",
    label="Observed flash onset"
)

plt.axhline(0, linestyle=":")

plt.xlabel("Time relative to flash onset (s)")
plt.ylabel("Change from baseline (%)")
plt.title("CogniView: Exploratory PLR Response Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_DIR / "plr_response_curves.png",
    dpi=300
)

plt.close()

print("\n========== PLR FEATURE SUMMARY ==========")

print(f"Sessions processed: {len(results)}")

print(results["feature_status"].value_counts())

print("\nNumerical summary:")
print(numeric.describe().round(4))

print(f"\nOutputs saved to: {OUT_DIR}")