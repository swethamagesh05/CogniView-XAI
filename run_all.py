import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ET = ROOT / "src" / "eye_tracking"

def has_retina():
    d = ROOT / "data" / "retina"; d = d if (d / "train.csv").exists() else ROOT / "data"
    return (d / "train.csv").exists() and (d / "train_images").exists() and (ROOT / "models/retina/best_model.pth").exists()
def has_et():
    return (ET / "data/ad_eyetracking/synthetic/metadata.csv").exists() and (ET / "checkpoints/best_ad_tracker.pth").exists()
def has_fusion():
    return (ROOT / "data/fusion/val.csv").exists() and (ROOT / "models/fusion/fusion_model.pth").exists()

NEED_RET = "data/retina/{train.csv,train_images/} and models/retina/best_model.pth (from the model owner)"
NEED_ET = "src/eye_tracking/data/ad_eyetracking/synthetic and checkpoints/best_ad_tracker.pth (run --retrain or generate_ad_synthetic_data.py + train.py)"
NEED_FUS = "data/fusion/val.csv and models/fusion/fusion_model.pth (from the model owner)"

# (id, name, cwd, script, args, check, what is missing)
STEPS = [
 (1, "Retina Grad-CAM",              "src/retinal",      "Gradcam.py",                         [], has_retina, NEED_RET),
 (2, "Retina Grad-CAM faithfulness", "src/retinal",      "faithfulness.py",             ["--num_samples", "20"], has_retina, NEED_RET),
 (3, "Eye-tracking SHAP",            "src/eye_tracking", "shap_ad_explainer.py",               [], has_et, NEED_ET),
 (4, "Eye-tracking Integrated Gradients", "src/eye_tracking", "integrated_gradients.py", ["--num_subjects", "30"], has_et, NEED_ET),
 (5, "Eye-tracking faithfulness",    "src/eye_tracking", "faithfulness_eyetracking.py", ["--num_subjects", "20"], has_et, NEED_ET),
 (6, "Fusion XAI (IG+attention+faithfulness)", "src/xai", "fusion_xai.py",               [], has_fusion, NEED_FUS),
]

def run(cmd, cwd):
    t = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode, time.time() - t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--retrain", action="store_true", help="regenerate synthetic data + retrain eye-tracking model first")
    ap.add_argument("--prepare-fusion", action="store_true", help="make SYNTHETIC fusion sessions, aggregate, train fusion model (use when no real fusion data exists)")
    a = ap.parse_args()

    if a.list:
        for i, n, _, _, _, chk, need in STEPS:
            print(f"[{i}] {n:45s} {'READY' if chk() else 'MISSING: ' + need}")
        return

    summary = []
    if a.prepare_fusion:
        for cmd in ([sys.executable, "src/fusion/generate_synthetic_fusion_data.py"],
                    [sys.executable, "-m", "src.fusion.aggregate", "--val-sessions-per-class", "3"],
                    [sys.executable, "-m", "src.fusion.train"]):
            rc, dt = run(cmd, ROOT); summary.append(("prepare-fusion: " + " ".join(cmd[1:3]), "OK" if rc == 0 else f"FAILED({rc})", dt))
            if rc: break
    if a.retrain:
        for s in (["generate_ad_synthetic_data.py"], ["train.py"], ["evaluate.py"]):
            rc, dt = run([sys.executable] + s, ET); summary.append((f"retrain: {s[0]}", "OK" if rc == 0 else f"FAILED({rc})", dt))
            if rc: break

    for i, n, cwd, script, args, chk, need in STEPS:
        if a.only and i not in a.only: continue
        if not chk():
            print(f"\n=== [{i}] {n}: SKIPPED. Missing {need}"); summary.append((f"[{i}] {n}", "SKIPPED", 0)); continue
        print(f"\n=== [{i}] {n} ===")
        rc, dt = run([sys.executable, script] + args, ROOT / cwd)
        summary.append((f"[{i}] {n}", "OK" if rc == 0 else f"FAILED({rc})", dt))

    print("\n================ SUMMARY ================")
    for n, st, dt in summary: print(f"{st:10s} {dt:6.0f}s  {n}")
    print("Outputs are in ./results/")
    sys.exit(1 if any(s.startswith("FAILED") for _, s, _ in summary) else 0)

if __name__ == "__main__":
    main()



















"""Run the whole CogniView XAI pipeline, step by step, with clear pre-checks.

    python run_all.py              # run every step that has its inputs
    python run_all.py --list       # show steps and whether inputs are present
    python run_all.py --only 3 4   # run selected steps
    python run_all.py --retrain    # also retrain the eye-tracking model first (step 0)

Steps that lack inputs are SKIPPED with an explanation, never crash the rest.
"""