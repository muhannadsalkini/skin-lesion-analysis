"""
src/visualize.py
----------------
Generates all figures for the final report:

  1. Training curves (accuracy/loss) per model
  2. Confusion matrix heatmaps (classification)
  3. Model comparison bar charts
  4. Segmentation prediction overlays
  5. Architecture comparison (params vs accuracy)

Usage:
    python src/visualize.py
    python src/visualize.py --figure curves
    python src/visualize.py --figure confusion
    python src/visualize.py --figure comparison
    python src/visualize.py --figure segmentation
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
DPI = 200


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load CSV training history
# ─────────────────────────────────────────────────────────────────────────────

def _load_history(model_name: str, phase: str = "") -> pd.DataFrame | None:
    tag = f"{model_name}_{phase}" if phase else model_name
    csv_path = os.path.join(config.LOGS_DIR, f"{tag}_history.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _merge_phases(model_name: str) -> pd.DataFrame | None:
    p1 = _load_history(model_name, "phase1")
    p2 = _load_history(model_name, "phase2")
    if p1 is not None and p2 is not None:
        p2 = p2.copy()
        p2["epoch"] = p2["epoch"] + len(p1)
        return pd.concat([p1, p2], ignore_index=True)
    if p1 is not None:
        return p1
    return _load_history(model_name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(model_name: str):
    df = _merge_phases(model_name)
    if df is None:
        print(f"[Visualize] No history for {model_name} — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    title = model_name.replace("_", " ").title()
    fig.suptitle(f"Training History — {title}", fontsize=13)

    # Determine metric columns
    acc_col = "accuracy" if "accuracy" in df.columns else None
    val_acc_col = "val_accuracy" if "val_accuracy" in df.columns else None
    dice_col = "dice_coefficient" if "dice_coefficient" in df.columns else None
    val_dice_col = "val_dice_coefficient" if "val_dice_coefficient" in df.columns else None

    if acc_col and val_acc_col:
        axes[0].plot(df["epoch"] + 1, df[acc_col], label="Train Acc", linewidth=1.8)
        axes[0].plot(df["epoch"] + 1, df[val_acc_col], label="Val Acc", linewidth=1.8, ls="--")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy")
    elif dice_col and val_dice_col:
        axes[0].plot(df["epoch"] + 1, df[dice_col], label="Train Dice", linewidth=1.8)
        axes[0].plot(df["epoch"] + 1, df[val_dice_col], label="Val Dice", linewidth=1.8, ls="--")
        axes[0].set_ylabel("Dice Score")
        axes[0].set_title("Dice Score")

    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(df["epoch"] + 1, df["loss"], label="Train Loss", linewidth=1.8)
    axes[1].plot(df["epoch"] + 1, df["val_loss"], label="Val Loss", linewidth=1.8, ls="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, f"{model_name}_training_curves.png")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[Visualize] Saved → {out_path}")


def plot_all_training_curves():
    all_models = [
        config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE,
        config.MODEL_VIT, config.MODEL_HYBRID_CLF,
        config.MODEL_UNET, config.MODEL_TRANSUNET, config.MODEL_INSTANCE,
    ]
    for name in all_models:
        plot_training_curves(name)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model_name: str, class_names: list | None = None):
    cm_path = os.path.join(config.LOGS_DIR, f"{model_name}_confusion_matrix.npy")
    if not os.path.exists(cm_path):
        print(f"[Visualize] No confusion matrix for {model_name} — skipping.")
        return

    cm = np.load(cm_path)
    n = cm.shape[0]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig_size = max(8, n // 2 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    tick_labels = class_names if class_names else list(range(n))
    short_labels = [l.replace("_", " ") for l in tick_labels] if class_names else tick_labels

    sns.heatmap(
        cm_norm, ax=ax, cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        linewidths=0.3, linecolor="lightgrey",
        annot=True, fmt=".1f",
        cbar_kws={"label": "Row %"},
    )
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    out_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[Visualize] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model comparison charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison():
    metrics_path = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
    if not os.path.exists(metrics_path):
        print("[Visualize] No all_model_metrics.json — run evaluate.py first.")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # ── Classification comparison ──────────────────────────────────────────
    clf_names = [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE,
                 config.MODEL_VIT, config.MODEL_HYBRID_CLF]
    clf_data = {k: v for k, v in all_metrics.items() if k in clf_names and "accuracy" in v}

    if clf_data:
        names = list(clf_data.keys())
        accs = [clf_data[n]["accuracy"] * 100 for n in names]
        f1s = [clf_data[n]["macro_f1"] for n in names]
        labels = [n.replace("_", "\n") for n in names]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(names)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Classification Model Comparison — ISIC 2018", fontsize=13)

        bars = axes[0].bar(labels, accs, color=colors, edgecolor="white")
        axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
        axes[0].set_ylabel("Test Accuracy (%)")
        axes[0].set_title("Test Accuracy")
        axes[0].set_ylim([max(0, min(accs) - 15), 105])

        bars2 = axes[1].bar(labels, f1s, color=colors, edgecolor="white")
        axes[1].bar_label(bars2, fmt="%.4f", padding=3)
        axes[1].set_ylabel("Macro F1 Score")
        axes[1].set_title("Macro F1 Score")
        axes[1].set_ylim([max(0, min(f1s) - 0.15), 1.05])

        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, "classification_comparison.png")
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        plt.savefig(out, dpi=DPI)
        plt.close()
        print(f"[Visualize] Saved → {out}")

        # Parameter efficiency
        params = [clf_data[n].get("params_M", 0) for n in names]
        if any(p > 0 for p in params):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(params, accs, s=200, c=colors, zorder=5, edgecolors="black")
            for i, name in enumerate(names):
                ax.annotate(name.replace("_", " "), (params[i], accs[i]),
                           textcoords="offset points", xytext=(10, 5), fontsize=9)
            ax.set_xlabel("Parameters (Millions)")
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_title("Parameter Efficiency — Accuracy vs. Model Size")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = os.path.join(config.FIGURES_DIR, "param_efficiency.png")
            plt.savefig(out, dpi=DPI)
            plt.close()
            print(f"[Visualize] Saved → {out}")

    # ── Segmentation comparison ────────────────────────────────────────────
    seg_names = [config.MODEL_UNET, config.MODEL_TRANSUNET, config.MODEL_INSTANCE]
    seg_data = {k: v for k, v in all_metrics.items() if k in seg_names and "dice_score" in v}

    if seg_data:
        names = list(seg_data.keys())
        dice_scores = [seg_data[n]["dice_score"] for n in names]
        ious = [seg_data[n]["iou"] for n in names]
        labels = [n.replace("_", "\n") for n in names]
        colors = ["#4C72B0", "#C44E52", "#55A868"][:len(names)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Segmentation Model Comparison — ISIC 2018", fontsize=13)

        bars = axes[0].bar(labels, dice_scores, color=colors, edgecolor="white")
        axes[0].bar_label(bars, fmt="%.4f", padding=3)
        axes[0].set_ylabel("Dice Score")
        axes[0].set_title("Dice Score")
        axes[0].set_ylim([max(0, min(dice_scores) - 0.15), 1.05])

        bars2 = axes[1].bar(labels, ious, color=colors, edgecolor="white")
        axes[1].bar_label(bars2, fmt="%.4f", padding=3)
        axes[1].set_ylabel("IoU (Jaccard)")
        axes[1].set_title("Intersection over Union")
        axes[1].set_ylim([max(0, min(ious) - 0.15), 1.05])

        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, "segmentation_comparison.png")
        plt.savefig(out, dpi=DPI)
        plt.close()
        print(f"[Visualize] Saved → {out}")

    # ── Combined overview chart ────────────────────────────────────────────
    if clf_data and seg_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("All Models Overview — ISIC 2018 Skin Lesion Dataset", fontsize=13)

        # All params
        all_names = list(clf_data.keys()) + list(seg_data.keys())
        all_params = [clf_data.get(n, seg_data.get(n, {})).get("params_M", 0) for n in all_names]
        all_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                      "#937860", "#DA8BC3", "#8C8C8C"][:len(all_names)]

        bars = axes[0].barh([n.replace("_", " ") for n in all_names],
                            all_params, color=all_colors, edgecolor="white")
        axes[0].bar_label(bars, fmt="%.1fM", padding=3)
        axes[0].set_xlabel("Parameters (Millions)")
        axes[0].set_title("Model Complexity")

        # Training time
        all_times = []
        for n in all_names:
            summary_path = os.path.join(config.LOGS_DIR, f"{n}_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    s = json.load(f)
                all_times.append(s.get("time_per_epoch_s", 0))
            else:
                all_times.append(0)

        if any(t > 0 for t in all_times):
            bars2 = axes[1].barh([n.replace("_", " ") for n in all_names],
                                all_times, color=all_colors, edgecolor="white")
            axes[1].bar_label(bars2, fmt="%.1fs", padding=3)
            axes[1].set_xlabel("Time per Epoch (seconds)")
            axes[1].set_title("Training Time")

        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, "all_models_overview.png")
        plt.savefig(out, dpi=DPI)
        plt.close()
        print(f"[Visualize] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Segmentation prediction samples
# ─────────────────────────────────────────────────────────────────────────────

def plot_segmentation_predictions():
    """Generate sample segmentation prediction overlays."""
    import tensorflow as tf
    from src.data_pipeline import load_segmentation_data
    from src.segmentation_models import bce_dice_loss, dice_coefficient, iou_metric

    _, _, test_ds, _ = load_segmentation_data()
    if test_ds is None:
        print("[Visualize] No segmentation data — skipping prediction plots.")
        return

    seg_models = [config.MODEL_UNET, config.MODEL_TRANSUNET]
    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
    }

    # Get a batch of test images
    for images, masks in test_ds.take(1):
        sample_images = images[:4].numpy()
        sample_masks = masks[:4].numpy()

    for model_name in seg_models:
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}_final.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")
        if not os.path.exists(model_path):
            continue

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        preds = model.predict(sample_images, verbose=0)
        if isinstance(preds, dict):
            preds = preds.get("segmentation", preds)

        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        fig.suptitle(f"Segmentation Results — {model_name.replace('_', ' ').title()}", fontsize=14)

        for i in range(4):
            axes[i, 0].imshow(sample_images[i])
            axes[i, 0].set_title("Input Image" if i == 0 else "")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(sample_masks[i, :, :, 0], cmap="gray")
            axes[i, 1].set_title("Ground Truth" if i == 0 else "")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(preds[i, :, :, 0], cmap="gray")
            axes[i, 2].set_title("Prediction" if i == 0 else "")
            axes[i, 2].axis("off")

        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, f"{model_name}_predictions.png")
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        plt.savefig(out, dpi=DPI)
        plt.close()
        print(f"[Visualize] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure", type=str, default="all",
        choices=["curves", "confusion", "comparison", "segmentation", "all"],
    )
    args = parser.parse_args()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # Load class names
    class_names = None
    try:
        from src.data_pipeline import build_classification_dataframe
        _, class_names = build_classification_dataframe()
    except Exception:
        class_names = config.CLASS_NAMES

    if args.figure in ("curves", "all"):
        plot_all_training_curves()

    if args.figure in ("confusion", "all"):
        clf_models = [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE,
                      config.MODEL_VIT, config.MODEL_HYBRID_CLF]
        for name in clf_models:
            plot_confusion_matrix(name, class_names)

    if args.figure in ("comparison", "all"):
        plot_model_comparison()

    if args.figure in ("segmentation", "all"):
        plot_segmentation_predictions()

    print(f"\n✅ All figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
