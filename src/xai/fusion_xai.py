import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fusion"))

from model import AdaptiveFusionModel, FUSION_FEATURE_NAMES  # pyright: ignore[reportMissingImports] # noqa: E402
from dataset import FusionDataset  # pyright: ignore[reportMissingImports] # noqa: E402

CLASS_NAMES = {0: "low risk", 1: "elevated risk"}   # 2-class model
TARGET_CLASS = 1
SEQ_LEN = 30


def load_model(checkpoint, device):
    model = AdaptiveFusionModel(num_features=len(FUSION_FEATURE_NAMES),
                                num_classes=2).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")
    return model


def risk_prob(model, x):
    """Probability of the target class, shape (batch,). IG needs one scalar per sample."""
    logits, _ = model(x)
    return F.softmax(logits, dim=1)[:, TARGET_CLASS]


def pick_samples(dataset, n):
    """Spread samples across the dataset and across labels."""
    labels = np.array([int(dataset.labels[end - 1]) for _, end in dataset.indices])
    chosen = []
    per_label = max(1, n // max(1, len(np.unique(labels))))
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        chosen.extend(idx[np.linspace(0, len(idx) - 1, min(per_label, len(idx))).astype(int)])
    return sorted(set(int(i) for i in chosen))


def deletion_insertion(model, device, x, attr, baseline, steps=20):
    """x, attr, baseline: (30, F). Rank cells by |attr|; hide/reveal them."""
    T, Fn = x.shape
    n = T * Fn
    order = np.argsort(-np.abs(attr).flatten())
    sizes = np.linspace(0, n, steps + 1).astype(int)
    del_p, ins_p = [], []
    with torch.no_grad():
        for k in sizes:
            m = np.zeros(n, dtype=bool)
            m[order[:k]] = True
            m = torch.from_numpy(m.reshape(T, Fn))
            d = x.clone(); d[m] = baseline[m]
            i = baseline.clone(); i[m] = x[m]
            del_p.append(risk_prob(model, d.unsqueeze(0).to(device)).item())
            ins_p.append(risk_prob(model, i.unsqueeze(0).to(device)).item())
    frac = sizes / n
    # np.trapz was renamed np.trapezoid in NumPy 2.x; support both.
    area = getattr(np, "trapezoid", None) or np.trapz
    return area(del_p, frac), area(ins_p, frac)


def run(csv_path, checkpoint, out_dir, num_samples):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint, device)
    dataset = FusionDataset(str(csv_path), sequence_length=SEQ_LEN)
    print(f"Windows available: {len(dataset)} from {csv_path}")

    # Baseline = average frame of the LOW-RISK class (label 0): the "nothing unusual" reference.
    # (A dataset-wide mean can itself be scored high-risk by the model, which makes
    # deletion meaningless.) Falls back to the global mean if label 0 is absent.
    feats = np.asarray(dataset.features, dtype=np.float32)
    lab = np.asarray(dataset.labels)
    ref = feats[lab == 0] if (lab == 0).any() else feats
    baseline_window = torch.tensor(ref.mean(axis=0), dtype=torch.float32).unsqueeze(0).repeat(SEQ_LEN, 1)
    with torch.no_grad():
        bp = risk_prob(model, baseline_window.unsqueeze(0).to(device)).item()
    print(f"Baseline (low-risk mean frame) risk = {bp:.3f}  (should be low, else IG/deletion are unreliable)")

    ig = IntegratedGradients(lambda z: risk_prob(model, z))
    rows, all_attr, all_att = [], [], []

    for i in pick_samples(dataset, num_samples):
        x, y = dataset[i]
        xb = x.unsqueeze(0).to(device).requires_grad_()
        with torch.no_grad():
            logits, att = model(xb)
            prob = F.softmax(logits, dim=1)[0, TARGET_CLASS].item()
        att = att.squeeze(0).squeeze(-1).cpu().numpy()          # (30,)

        attr = ig.attribute(xb, baseline_window.unsqueeze(0).to(device),
                            n_steps=50).squeeze(0).detach().cpu().numpy()   # (30, F)

        del_auc, ins_auc = deletion_insertion(model, device, x, attr, baseline_window)
        rho = spearmanr(att, np.abs(attr).sum(axis=1)).correlation

        all_attr.append(np.abs(attr).sum(axis=0))
        all_att.append(att)
        rows.append(dict(window_index=i, true_label=int(y), risk_prob=prob,
                         deletion_auc=del_auc, insertion_auc=ins_auc,
                         attention_vs_ig_spearman=rho,
                         top_feature=FUSION_FEATURE_NAMES[int(np.argmax(np.abs(attr).sum(axis=0)))]))
        print(f"window {i}: true={int(y)} risk={prob:.3f} del={del_auc:.3f} "
              f"ins={ins_auc:.3f} attn~IG rho={rho:.2f}")

        # per-window figure: IG heatmap + attention curve
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
        lim = np.abs(attr).max() + 1e-12
        im = a1.imshow(attr.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a1.set_yticks(range(len(FUSION_FEATURE_NAMES)))
        a1.set_yticklabels(FUSION_FEATURE_NAMES, fontsize=7)
        a1.set_title(f"window {i} | true={int(y)} | risk={prob:.3f}")
        plt.colorbar(im, ax=a1, label="IG (red raises risk)")
        a2.plot(att, color="#2c6e91"); a2.set_ylabel("attention")
        a2.set_xlabel("frame in 30-frame window")
        plt.tight_layout(); plt.savefig(out_dir / f"window_{i}_explanation.png", dpi=150)
        plt.close()

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "fusion_xai_per_window.csv", index=False)

    ranking = (pd.DataFrame({"feature": FUSION_FEATURE_NAMES,
                             "mean_abs_IG": np.mean(all_attr, axis=0)})
               .sort_values("mean_abs_IG", ascending=False))
    ranking.to_csv(out_dir / "fusion_global_feature_ranking.csv", index=False)

    plt.figure(figsize=(6, 4.5))
    plt.barh(ranking["feature"], ranking["mean_abs_IG"], color="#2c6e91")
    plt.gca().invert_yaxis(); plt.xlabel("Mean |Integrated Gradients|")
    plt.title("Fusion model: which features drive risk")
    plt.tight_layout(); plt.savefig(out_dir / "fusion_global_feature_ranking.png", dpi=150)
    plt.close()

    # ---- sanity guard: explaining a model that isn't discriminating is meaningless ----
    spread = res["risk_prob"].std()
    tiny_ig = float(np.sum(all_attr)) < 1e-6
    if spread < 0.05 or tiny_ig:
        print("\n" + "!" * 70)
        print("WARNING: predictions barely vary across windows "
              f"(std of risk = {spread:.3f})" + (" and attributions are ~0" if tiny_ig else ""))
        print("Do NOT trust the rankings below. Likely causes: wrong csv/checkpoint pair,")
        print("data on a different scale than training, or a model that saturated.")
        print("Check that risk_prob spreads from near 0 to near 1 across true labels first.")
        print("!" * 70)

    print("\n=== GLOBAL FEATURE RANKING ===")
    print(ranking.to_string(index=False))
    print("\n=== FAITHFULNESS ===")
    print(f"Mean deletion AUC : {res.deletion_auc.mean():.3f}  (lower = more faithful)")
    print(f"Mean insertion AUC: {res.insertion_auc.mean():.3f}  (higher = more faithful)")
    print(f"Mean attention-vs-IG Spearman: {res.attention_vs_ig_spearman.mean():.2f}")
    print(f"\nEverything saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "fusion" / "val.csv"))
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "models" / "fusion" / "fusion_model.pth"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "xai_fusion"))
    p.add_argument("--num_samples", type=int, default=20)
    a = p.parse_args()
    run(a.csv, a.checkpoint, a.out, a.num_samples)









"""
STEP 8 (Phase II) - Explainability for the fusion model (AdaptiveFusionModel).

Place this file in:  CogniView/src/xai/step8_fusion_xai.py
Run from anywhere:   python src/xai/step8_fusion_xai.py

What it produces, for the model your friend trained in src/fusion/train.py:
  1. ATTENTION: the model's own temporal-attention weights over the 30-frame
     window (which moments in the window the model leaned on).
  2. INTEGRATED GRADIENTS: attribution for every (frame, feature) cell, so you
     can say which of the 11 features pushed the risk score up or down.
  3. GLOBAL RANKING: mean |IG| per feature across many windows (paper table).
  4. FAITHFULNESS: deletion/insertion AUC, same method as the retina (step5a)
     and eye-tracking (step7) tests, so all modalities use one methodology.
  5. ATTENTION vs IG AGREEMENT: Spearman correlation between where attention
     is high and where IG is high. Attention alone is not proof of reasoning,
     so this check tells you how far to trust the attention plot.

IMPORTANT FACTS ABOUT THE CODE (read from the repo):
  * train.py builds AdaptiveFusionModel(num_classes=2). model.py defaults to 3,
    so FusionEngine() cannot load the checkpoint. This script builds the
    2-class model itself.
  * Input window = 30 consecutive rows x 11 features, label = last row's label.
  * Data comes from data/fusion/val.csv, which is NOT in GitHub (data/ is
    git-ignored). You need your friend's data/fusion folder (train.csv, val.csv).
  * Today the model only sees the 11 eye-tracking features. Retina and PLR are
    not inputs yet, so this explains the eye-tracking part of the fusion model.
    When your friend appends PLR/retina features, add their names to
    FUSION_FEATURE_NAMES in model.py and everything here picks them up.

Install once:  pip install captum scipy
"""