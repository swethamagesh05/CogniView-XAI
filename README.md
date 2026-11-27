# CogniView-XAI

Explainability (XAI) for CogniView: Grad-CAM on the retina model, SHAP / Integrated Gradients on the
eye-tracking model, and attention + IG + faithfulness tests on the fusion model.
This is a separate repo. The model code under `src/` is copied from the CogniView models repo
(credit: its authors) so the XAI scripts run unchanged. Nothing here writes back to that repo.

## Setup (Windows PowerShell)
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Keep the project OFF OneDrive (e.g. F:\CogniView-XAI).

## Inputs you must supply (not in git)
| Step | Needs | From |
|---|---|---|
| 1-2 Retina | `data/retina/train.csv`, `data/retina/train_images/`, `models/retina/best_model.pth` | model owner |
| 3-5 Eye-tracking | already included (`src/eye_tracking/data/...` synthetic, `checkpoints/best_ad_tracker.pth`) | included |
| 6 Fusion | `data/fusion/val.csv`, `models/fusion/fusion_model.pth` | real sessions if they exist; otherwise `python run_all.py --prepare-fusion` builds SYNTHETIC ones |

## Run
    python run_all.py --list      # shows READY / MISSING per step
    python run_all.py             # runs everything that is ready, skips the rest with a reason
    python run_all.py --only 3 4 5
    python run_all.py --prepare-fusion   # no real fusion data? make synthetic sessions, aggregate, train fusion model, then run step 6
    python run_all.py --retrain   # regenerate synthetic data, retrain + evaluate eye-tracking first

Steps: 1 Grad-CAM, 2 Grad-CAM faithfulness, 3 SHAP, 4 Integrated Gradients, 5 eye-tracking faithfulness,
6 fusion XAI. Outputs go to `results/`.

## Notes
- Eye-tracking data is synthetic: AUC ~1.0 shows clean separability, not clinical accuracy.
- Fusion currently sees only the 11 eye-tracking features; PLR and retina are not fusion inputs yet.
- If you retrain, retrain and explain with the same data/checkpoint pair (stale checkpoint => flat 0.5 results).
- To retrain fusion: `python -m src.fusion.train` from the repo root (needs tensorboard).
