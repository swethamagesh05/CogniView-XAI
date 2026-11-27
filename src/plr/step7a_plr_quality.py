from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "plr_raw"
OUT_DIR = ROOT / "results" / "plr_features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("plr_session_*.csv"))
rows = []

for file in files:

    df = pd.read_csv(file)

    left = pd.to_numeric(
        df["left_pupil_iris_ratio"], errors="coerce"
    )

    right = pd.to_numeric(
        df["right_pupil_iris_ratio"], errors="coerce"
    )

    left = left.where(left.between(0, 1, inclusive="neither"))
    right = right.where(right.between(0, 1, inclusive="neither"))

    ratio = pd.concat([left, right], axis=1).mean(axis=1)
    valid = ratio.notna()

    baseline = df["stimulus_state"].eq("baseline")
    flash = df["stimulus_state"].eq("flash")
    recovery = df["stimulus_state"].eq("recovery")

    def coverage(mask):
        if mask.sum() == 0:
            return np.nan

        return round(100 * valid[mask].mean(), 2)

    baseline_ratio = ratio[baseline].median()

    flash_onset = (
        df.loc[flash, "timestamp_s"].min()
        if flash.any() else np.nan
    )

    recovery_onset = (
        df.loc[recovery, "timestamp_s"].min()
        if recovery.any() else np.nan
    )

    flags = []

    if not (baseline.any() and flash.any() and recovery.any()):
        flags.append("missing_phase")

    if valid.mean() < 0.70:
        flags.append("low_valid_coverage")

    if baseline.sum() == 0 or coverage(baseline) < 70:
        flags.append("poor_baseline")

    if flash.sum() == 0 or coverage(flash) < 50:
        flags.append("poor_flash_tracking")

    if recovery.sum() == 0 or coverage(recovery) < 70:
        flags.append("poor_recovery_tracking")

    if pd.notna(baseline_ratio) and baseline_ratio > 0.80:
        flags.append("high_baseline_ratio")

    if not df["timestamp_s"].is_monotonic_increasing:
        flags.append("invalid_timestamp_order")

    rows.append({
        "session": file.stem,
        "total_frames": len(df),
        "baseline_frames": int(baseline.sum()),
        "flash_frames": int(flash.sum()),
        "recovery_frames": int(recovery.sum()),
        "valid_pct": round(100 * valid.mean(), 2),
        "baseline_valid_pct": coverage(baseline),
        "flash_valid_pct": coverage(flash),
        "recovery_valid_pct": coverage(recovery),
        "baseline_ratio_pct": round(
            baseline_ratio * 100, 2
        ),
        "flash_onset_s": flash_onset,
        "recovery_onset_s": recovery_onset,
        "quality_flags": "; ".join(flags) if flags else "none",
        "status": "REVIEW" if flags else "PASS"
    })

    print(
        f"{file.stem}: "
        f"valid={100 * valid.mean():.1f}% | "
        f"baseline={baseline_ratio * 100:.2f}% | "
        f"status={'REVIEW' if flags else 'PASS'}"
    )

if not rows:
    raise FileNotFoundError(
        f"No PLR session CSV files found in {RAW_DIR}"
    )

results = pd.DataFrame(rows)

output = OUT_DIR / "plr_quality_report.csv"
results.to_csv(output, index=False)

print("\n========== PLR QUALITY SUMMARY ==========")
print(f"Total sessions: {len(results)}")
print(f"Passed: {(results.status == 'PASS').sum()}")
print(f"Needs review: {(results.status == 'REVIEW').sum()}")

print("\n", results.to_string(index=False))
print(f"\nSaved report: {output}")