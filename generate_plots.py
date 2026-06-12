#!/usr/bin/env python3
"""
Generate all visualizations for the final report.
Uses the latest v2 training results where available.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs(config.FIGURES_DIR, exist_ok=True)

CLASS_NAMES = [
    "actinic keratosis", "basal cell carcinoma", "dermatofibroma",
    "melanoma", "nevus", "pigmented benign keratosis",
    "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion"
]
SHORT_NAMES = ["AK", "BCC", "DF", "MEL", "NV", "PBK", "SK", "SCC", "VASC"]

# ── Merged metrics: use v2 where available, fallback to original ──────────────
# Load base metrics for params / timing info
base_path = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
with open(base_path) as f:
    base_metrics = json.load(f)

# Load v2 improved metrics
imp_path = os.path.join(config.LOGS_DIR, "improved_metrics.json")
with open(imp_path) as f:
    imp_metrics = json.load(f)

# Build merged metrics dict (ordered: CNN, EfficientNet, MobileNet, ViT, Hybrid)
merged = {
    "custom_cnn_v2": {
        "accuracy":         imp_metrics["custom_cnn_v2"]["accuracy"],
        "macro_f1":         imp_metrics["custom_cnn_v2"]["macro_f1"],
        "macro_precision":  imp_metrics["custom_cnn_v2"]["macro_precision"],
        "macro_recall":     imp_metrics["custom_cnn_v2"]["macro_recall"],
        "params_M":         base_metrics["custom_cnn"]["params_M"],
        "time_per_epoch_s": base_metrics["custom_cnn"]["time_per_epoch_s"],
    },
    "efficientnetb0_v2": {
        "accuracy":         imp_metrics["efficientnetb0_v2"]["accuracy"],
        "macro_f1":         imp_metrics["efficientnetb0_v2"]["macro_f1"],
        "macro_precision":  imp_metrics["efficientnetb0_v2"]["macro_precision"],
        "macro_recall":     imp_metrics["efficientnetb0_v2"]["macro_recall"],
        "params_M":         base_metrics["efficientnetb0"]["params_M"],
        "time_per_epoch_s": base_metrics["efficientnetb0"]["time_per_epoch_s"],
    },
    "mobilenetv2": {
        "accuracy":         base_metrics["mobilenetv2"]["accuracy"],
        "macro_f1":         base_metrics["mobilenetv2"]["macro_f1"],
        "macro_precision":  base_metrics["mobilenetv2"]["macro_precision"],
        "macro_recall":     base_metrics["mobilenetv2"]["macro_recall"],
        "params_M":         base_metrics["mobilenetv2"]["params_M"],
        "time_per_epoch_s": base_metrics["mobilenetv2"]["time_per_epoch_s"],
    },
    "vit_classifier": {
        "accuracy":         base_metrics["vit_classifier"]["accuracy"],
        "macro_f1":         base_metrics["vit_classifier"]["macro_f1"],
        "macro_precision":  base_metrics["vit_classifier"]["macro_precision"],
        "macro_recall":     base_metrics["vit_classifier"]["macro_recall"],
        "params_M":         base_metrics["vit_classifier"]["params_M"],
        "time_per_epoch_s": base_metrics["vit_classifier"]["time_per_epoch_s"],
    },
    "cnn_transformer_hybrid_v2": {
        "accuracy":         imp_metrics["cnn_transformer_hybrid_v2"]["accuracy"],
        "macro_f1":         imp_metrics["cnn_transformer_hybrid_v2"]["macro_f1"],
        "macro_precision":  imp_metrics["cnn_transformer_hybrid_v2"]["macro_precision"],
        "macro_recall":     imp_metrics["cnn_transformer_hybrid_v2"]["macro_recall"],
        "params_M":         base_metrics["cnn_transformer_hybrid"]["params_M"],
        "time_per_epoch_s": base_metrics["cnn_transformer_hybrid"]["time_per_epoch_s"],
    },
}

models = list(merged.keys())
labels = ["Custom CNN", "EfficientNetB0", "MobileNetV2", "ViT", "CNN+Trans"]
accs   = [merged[m]["accuracy"] * 100 for m in models]
f1s    = [merged[m]["macro_f1"]        for m in models]
precs  = [merged[m]["macro_precision"] for m in models]
params = [merged[m]["params_M"]        for m in models]
colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

# ── 1. Accuracy / F1 / Precision bar chart ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Classification Model Comparison (Best v2 Results)", fontsize=11, fontweight="bold")

axes[0].barh(labels, accs, color=colors)
axes[0].set_xlabel("Accuracy (%)")
axes[0].set_title("Test Accuracy")
axes[0].set_xlim(0, 70)
for i, v in enumerate(accs):
    axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

axes[1].barh(labels, f1s, color=colors)
axes[1].set_xlabel("Macro F1 Score")
axes[1].set_title("Macro F1")
axes[1].set_xlim(0, 0.75)
for i, v in enumerate(f1s):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

axes[2].barh(labels, precs, color=colors)
axes[2].set_xlabel("Macro Precision")
axes[2].set_title("Macro Precision")
axes[2].set_xlim(0, 0.75)
for i, v in enumerate(precs):
    axes[2].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
out = os.path.join(config.FIGURES_DIR, "classification_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 2. Confusion matrices (use v2 where available) ────────────────────────────
cm_map = {
    "custom_cnn_v2":            "custom_cnn_v2_v2_cm.npy",
    "efficientnetb0_v2":        "efficientnetb0_v2_v2_cm.npy",
    "mobilenetv2":              "mobilenetv2_confusion_matrix.npy",
    "vit_classifier":           "vit_classifier_confusion_matrix.npy",
    "cnn_transformer_hybrid_v2":"cnn_transformer_hybrid_v2_v2_cm.npy",
}

fig, axes = plt.subplots(1, 5, figsize=(26, 5))
fig.suptitle("Confusion Matrices – All Models (9-class ISIC 2018 test set)", fontsize=10)

for idx, (mkey, label) in enumerate(zip(models, labels)):
    fname = cm_map[mkey]
    cm_path = os.path.join(config.LOGS_DIR, fname)
    ax = axes[idx]
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                    ax=ax, cbar=False, annot_kws={"size": 7})
        ax.set_title(f"{label}\nAcc={merged[mkey]['accuracy']*100:.1f}%", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.set_ylabel("True", fontsize=8)
    else:
        ax.text(0.5, 0.5, f"N/A\n({fname})", ha="center", va="center", fontsize=8)
        ax.set_title(label)

plt.tight_layout()
out = os.path.join(config.FIGURES_DIR, "confusion_matrices.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 3. Training / validation curves (v2 where available) ─────────────────────
history_map = {
    "Custom CNN v2":       [("custom_cnn_v2_history.csv", None)],
    "EfficientNetB0 v2":   [("efficientnetb0_v2_phase1_history.csv", "P1"),
                             ("efficientnetb0_v2_phase2_history.csv", "P2")],
    "MobileNetV2":         [("mobilenetv2_phase1_history.csv", "P1"),
                             ("mobilenetv2_phase2_history.csv", "P2")],
    "ViT":                 [("vit_classifier_history.csv", None)],
    "CNN+Transformer v2":  [("cnn_transformer_hybrid_v2_history.csv", None)],
}
curve_colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Validation Accuracy & Loss During Training (Best v2 Runs)", fontsize=10, fontweight="bold")

for ci, (label, phases) in enumerate(history_map.items()):
    all_acc, all_loss = [], []
    for (fname, phase_tag) in phases:
        path = os.path.join(config.LOGS_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "val_accuracy" in df.columns:
                all_acc.extend(df["val_accuracy"].tolist())
            if "val_loss" in df.columns:
                all_loss.extend(df["val_loss"].tolist())
    if all_acc:
        axes[0].plot(all_acc, label=label, color=curve_colors[ci])
    if all_loss:
        axes[1].plot(all_loss, label=label, color=curve_colors[ci])

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(config.FIGURES_DIR, "training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 4. Segmentation training curves ──────────────────────────────────────────
seg_history_map = {
    "U-Net":        "unet_seg_history.csv",
    "TransUNet":    "transunet_seg_history.csv",
    "Instance Seg": "instance_seg_seg_history.csv",
}
seg_colors = ["#2980b9", "#e67e22", "#27ae60"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Segmentation Model Training Curves", fontsize=10, fontweight="bold")

for ci, (label, fname) in enumerate(seg_history_map.items()):
    path = os.path.join(config.LOGS_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"  WARNING: {fname} is empty, skipping")
                continue
            # Find dice / val_dice columns
            dice_col = next((c for c in df.columns if "dice" in c.lower() and "val" in c.lower()), None)
            loss_col = next((c for c in df.columns if "loss" in c.lower() and "val" in c.lower()), None)
            if dice_col:
                axes[0].plot(df[dice_col], label=label, color=seg_colors[ci])
            if loss_col:
                axes[1].plot(df[loss_col], label=label, color=seg_colors[ci])
        except Exception as e:
            print(f"  WARNING: Could not read {fname}: {e}")
    else:
        print(f"  WARNING: {path} not found or empty")

axes[0].set_title("Validation Dice Score")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Dice")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(config.FIGURES_DIR, "segmentation_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 5. Model complexity vs accuracy scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for i, (label, acc, p) in enumerate(zip(labels, accs, params)):
    ax.scatter(p, acc, s=200, c=colors[i], zorder=5, edgecolors="black", linewidth=0.8)
    ax.annotate(label, (p, acc), textcoords="offset points", xytext=(8, 6), fontsize=9)

ax.set_xlabel("Parameters (Millions)", fontsize=11)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Model Complexity vs. Accuracy (Best v2 Results)", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 70)
plt.tight_layout()
out = os.path.join(config.FIGURES_DIR, "complexity_vs_accuracy.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 6. Update all_model_metrics.json with merged best results ─────────────────
updated_all = {}
for mkey, label in zip(models, labels):
    updated_all[mkey] = merged[mkey]
with open(base_path, "w") as f:
    json.dump(updated_all, f, indent=2)
print(f"Updated all_model_metrics.json with v2 best results")

print(f"\nAll figures saved to: {config.FIGURES_DIR}")
print("\nSummary of metrics used:")
for mkey, label in zip(models, labels):
    print(f"  {label:20s}: Acc={merged[mkey]['accuracy']*100:.2f}%  F1={merged[mkey]['macro_f1']:.3f}  Params={merged[mkey]['params_M']}M")
