#!/usr/bin/env python3
"""Generate additional figures for the final report improvements."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs(config.FIGURES_DIR, exist_ok=True)
LOGS = config.LOGS_DIR

CLASS_NAMES = [
    "Actinic\nKeratosis", "Basal Cell\nCarcinoma", "Dermatofibroma",
    "Melanoma", "Nevus", "Pigm. Benign\nKeratosis",
    "Seborrheic\nKeratosis", "Squamous Cell\nCarcinoma", "Vascular\nLesion"
]
SHORT = ["AK","BCC","DF","MEL","NV","PBK","SK","SCC","VASC"]

# class counts (approx, from dataset section)
CLASS_COUNTS = [327, 514, 115, 438, 1341, 462, 767, 181, 142]

# ── 1. Dataset Distribution Bar Chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.Set2(np.linspace(0, 1, 9))
bars = ax.bar(SHORT, CLASS_COUNTS, color=colors, edgecolor="black", linewidth=0.6)
ax.set_xlabel("Lesion Class", fontsize=11)
ax.set_ylabel("Image Count", fontsize=11)
ax.set_title("ISIC 2018 Dataset Class Distribution (9 Classes)", fontsize=12, fontweight="bold")
ax.axhline(np.mean(CLASS_COUNTS), color="red", linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(CLASS_COUNTS):.0f}")
for bar, cnt in zip(bars, CLASS_COUNTS):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, str(cnt),
            ha="center", va="bottom", fontsize=8)
ax.legend(fontsize=9)
ax.set_ylim(0, 1600)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = os.path.join(config.FIGURES_DIR, "dataset_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 2. Per-class metrics from best model (CNN+Trans Hybrid v2) ─────────────────
cm_path = os.path.join(LOGS, "cnn_transformer_hybrid_v2_v2_cm.npy")
if os.path.exists(cm_path):
    cm = np.load(cm_path)
    n = cm.shape[0]

    # Per-class Precision, Recall, F1
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)
    support   = cm.sum(axis=1)  # actual counts per class

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i]        = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
                       if (precision[i] + recall[i]) > 0 else 0.0

    # Save as JSON for reference
    per_class = {SHORT[i]: {
        "precision": float(round(precision[i], 3)),
        "recall":    float(round(recall[i], 3)),
        "f1":        float(round(f1[i], 3)),
        "support":   int(support[i])
    } for i in range(n)}
    with open(os.path.join(LOGS, "hybrid_v2_per_class_metrics.json"), "w") as f:
        json.dump(per_class, f, indent=2)
    print("Saved hybrid_v2_per_class_metrics.json")
    for cls, v in per_class.items():
        print(f"  {cls}: P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}  n={v['support']}")

    # Per-class grouped bar chart
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w,   precision, w, label="Precision", color="#3498db", edgecolor="black", lw=0.5)
    ax.bar(x,       recall,    w, label="Recall",    color="#2ecc71", edgecolor="black", lw=0.5)
    ax.bar(x + w,   f1,        w, label="F1",        color="#e74c3c", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class Precision, Recall, and F1 — CNN+Transformer Hybrid v2", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    # Annotate recall for MEL specifically
    mel_idx = SHORT.index("MEL")
    ax.annotate(f"MEL recall={recall[mel_idx]:.2f}",
                xy=(mel_idx, recall[mel_idx] + 0.01),
                xytext=(mel_idx + 0.5, 0.55),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9, color="darkred")
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "per_class_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ── 3. Per-class sensitivity plot for all 5 models ────────────────────────
    model_cms = {
        "Custom CNN":    os.path.join(LOGS, "custom_cnn_v2_v2_cm.npy"),
        "EfficientNetB0":os.path.join(LOGS, "efficientnetb0_v2_v2_cm.npy"),
        "MobileNetV2":   os.path.join(LOGS, "mobilenetv2_confusion_matrix.npy"),
        "ViT":           os.path.join(LOGS, "vit_classifier_confusion_matrix.npy"),
        "CNN+Trans":     os.path.join(LOGS, "cnn_transformer_hybrid_v2_v2_cm.npy"),
    }
    # Heatmap: rows = models, cols = classes, value = per-class recall
    recall_matrix = np.zeros((5, n))
    model_labels = list(model_cms.keys())
    for mi, (mname, mpath) in enumerate(model_cms.items()):
        if os.path.exists(mpath):
            c = np.load(mpath)
            for i in range(min(n, c.shape[0])):
                denom = c[i, :].sum()
                recall_matrix[mi, i] = c[i, i] / denom if denom > 0 else 0.0

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(recall_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(SHORT, fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(model_labels, fontsize=10)
    for r in range(5):
        for c_idx in range(n):
            val = recall_matrix[r, c_idx]
            ax.text(c_idx, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if val > 0.3 else "white")
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label="Per-Class Recall")
    ax.set_title("Per-Class Recall Across All Models (green=good, red=poor)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "per_class_recall_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# ── 4. Balanced Accuracy and Sensitivity/Specificity macro ────────────────────
# Compute per-model balanced accuracy from confusion matrices
model_cms_ordered = {
    "Custom CNN":    "custom_cnn_v2_v2_cm.npy",
    "EfficientNetB0":"efficientnetb0_v2_v2_cm.npy",
    "MobileNetV2":   "mobilenetv2_confusion_matrix.npy",
    "ViT":           "vit_classifier_confusion_matrix.npy",
    "CNN+Trans":     "cnn_transformer_hybrid_v2_v2_cm.npy",
}
bal_accs = {}
for mname, fname in model_cms_ordered.items():
    p = os.path.join(LOGS, fname)
    if os.path.exists(p):
        c = np.load(p)
        recalls = [c[i,i]/c[i,:].sum() if c[i,:].sum()>0 else 0 for i in range(c.shape[0])]
        bal_accs[mname] = np.mean(recalls)
        print(f"  Balanced Accuracy {mname}: {np.mean(recalls)*100:.2f}%")

if bal_accs:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_b = ["#e74c3c","#2ecc71","#3498db","#9b59b6","#f39c12"]
    bars2 = ax.barh(list(bal_accs.keys()), [v*100 for v in bal_accs.values()],
                    color=colors_b, edgecolor="black", lw=0.6)
    ax.set_xlabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_title("Balanced Accuracy (Mean Per-Class Recall) — All Models", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 80)
    for bar, val in zip(bars2, bal_accs.values()):
        ax.text(val*100 + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "balanced_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

print("\nAll extra figures saved.")
