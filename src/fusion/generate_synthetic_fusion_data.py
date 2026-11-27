"""Synthetic fusion sessions in the exact format aggregate.py expects.

Writes data/fusion/sessions/trial*.csv  (one label per session, 11 feature columns + label).
SYNTHETIC: use only to exercise/demonstrate the XAI pipeline until real sessions exist.
Class 1 ("elevated risk") shifts fixation stability down, blink rate up, saccades up, with noise
and overlap so the task is not trivially separable.

    python src/fusion/generate_synthetic_fusion_data.py [--per-class 12] [--seed 7]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ["ear", "blink_count", "blink_rate", "fixation_stability", "fixation_count", "saccade_count",
        "smooth_pursuit_count", "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"]
OUT = Path(__file__).resolve().parents[2] / "data" / "fusion" / "sessions"

def session(rng, label, n):
    base = np.array([0.30, 1.0, 0.25, 0.80, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5]) + rng.normal(0, 0.05, 11)
    X = base + rng.normal(0, 0.12, (n, 11)) * np.array([1, 4, 1, 1, 4, 4, 3, 1, 1, 1, 1])
    X += np.cumsum(rng.normal(0, 0.01, (n, 11)), axis=0)           # slow drift within a session
    if label:
        X[:, 3] -= 0.18; X[:, 2] += 0.08; X[:, 5] += 1.0; X[:, 0] -= 0.02
    X = np.clip(X, 0, None)
    d = pd.DataFrame(X, columns=COLS); d["label"] = label
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
    rng = np.random.default_rng(a.seed); OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("trial*.csv"): old.unlink()
    k = 0
    for lab in (0, 1):
        for _ in range(a.per_class):
            k += 1; session(rng, lab, int(rng.integers(150, 260))).to_csv(OUT / f"trial{k:02d}.csv", index=False)
    print(f"Wrote {k} synthetic sessions to {OUT}")
