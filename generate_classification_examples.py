#!/usr/bin/env python3
"""
generate_classification_examples.py
-------------------------------------
Generate correct and incorrect classification example figures
from the best trained CNN+Transformer Hybrid v2 model.

Outputs (saved to results/figures/):
  correct_examples.png   — 3×3 grid, one correctly classified image per class
  incorrect_examples.png — 3×3 grid, one incorrectly classified image per class

Run:
  cd final_project
  python3 generate_classification_examples.py
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from PIL import Image

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.classification_models import (
    PatchExtract,
    PatchEmbedding,
    TransformerEncoderBlock,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(config.MODELS_DIR, "cnn_transformer_hybrid_v2_best.keras")
FALLBACK_PATH = os.path.join(config.MODELS_DIR, "cnn_transformer_hybrid_v2_final.keras")
FIGURES_DIR   = config.FIGURES_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

# Short display labels for the plots
CLASS_DISPLAY = {
    "actinic keratosis":        "Actinic Keratosis",
    "basal cell carcinoma":     "Basal Cell Carc.",
    "dermatofibroma":           "Dermatofibroma",
    "melanoma":                 "Melanoma",
    "nevus":                    "Nevus",
    "pigmented benign keratosis": "Pigm. Benign Ker.",
    "seborrheic keratosis":     "Seborrheic Ker.",
    "squamous cell carcinoma":  "Squam. Cell Carc.",
    "vascular lesion":          "Vascular Lesion",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    custom_objects = {
        "PatchExtract":           PatchExtract,
        "PatchEmbedding":         PatchEmbedding,
        "TransformerEncoderBlock": TransformerEncoderBlock,
    }
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at:\n  {MODEL_PATH}\n  {FALLBACK_PATH}\n"
            "Run  python3 train_improved.py  first."
        )
    print(f"[LoadModel] {path}")
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Collect correct & incorrect examples per class
# ─────────────────────────────────────────────────────────────────────────────
def collect_examples(model, test_dir, class_names, n_per_class=5):
    """
    Walk through test images class-by-class.
    Collects up to `n_per_class` correct and incorrect predictions for each class.
    Returns dicts keyed by class name:
        {class_name: [(img_array, true_label, pred_label, confidence), ...]}
    """
    correct   = {c: [] for c in class_names}
    incorrect = {c: [] for c in class_names}

    print("\n[Scan] Collecting examples from test set …")

    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(test_dir) / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found — skipping")
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        np.random.shuffle(images)

        for img_path in images:
            if (len(correct[class_name])   >= n_per_class and
                len(incorrect[class_name]) >= n_per_class):
                break

            try:
                # Load + preprocess
                img = Image.open(img_path).convert("RGB").resize(config.IMAGE_SIZE)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                batch   = np.expand_dims(img_arr, 0)

                # Predict
                logits     = model.predict(batch, verbose=0)[0]
                pred_idx   = int(np.argmax(logits))
                confidence = float(logits[pred_idx])
                pred_name  = class_names[pred_idx]

                entry = (img_arr, class_name, pred_name, confidence)
                if pred_idx == class_idx:
                    if len(correct[class_name]) < n_per_class:
                        correct[class_name].append(entry)
                else:
                    if len(incorrect[class_name]) < n_per_class:
                        incorrect[class_name].append(entry)

            except Exception:
                continue

        nc = len(correct[class_name])
        ni = len(incorrect[class_name])
        print(f"  [{class_idx}] {class_name:<35}  correct={nc}  incorrect={ni}")

    return correct, incorrect


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot 3×3 grid figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_grid(examples_dict, class_names, title, filename, border_color):
    """
    3 rows × 3 cols grid (9 classes).
    Green border = correctly classified  |  Red border = incorrectly classified
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 13.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)

    for idx, class_name in enumerate(class_names):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        entries = examples_dict.get(class_name, [])

        if entries:
            img_arr, true_lbl, pred_lbl, conf = entries[0]
            ax.imshow(img_arr, interpolation="lanczos")

            # Coloured border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4.0)

            true_d = CLASS_DISPLAY.get(true_lbl, true_lbl)
            pred_d = CLASS_DISPLAY.get(pred_lbl, pred_lbl)

            if border_color == "green":
                t_str  = f"True: {true_d}\nPred: {pred_d}  ✓\nConf: {conf:.1%}"
                t_col  = "#1a7a1a"
            else:
                t_str  = f"True:  {true_d}\nPred: {pred_d}  ✗\nConf: {conf:.1%}"
                t_col  = "#c0392b"

            ax.set_title(t_str, fontsize=9, color=t_col, pad=4,
                         fontdict={"linespacing": 1.4})
        else:
            ax.text(0.5, 0.5,
                    f"No example found\nfor class:\n{CLASS_DISPLAY.get(class_name, class_name)}",
                    ha="center", va="center", fontsize=9, color="gray",
                    transform=ax.transAxes)
            ax.set_facecolor("#f5f5f5")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Two-panel combined figure (for report)
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined(correct_dict, incorrect_dict, class_names):
    """
    Single wide figure with two 3×3 panels side-by-side.
    Left panel: correct examples  |  Right panel: incorrect examples
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "CNN+Transformer Hybrid v2 — Classification Examples on ISIC 2018 Test Set\n"
        "(Left: Correctly Classified  |  Right: Incorrectly Classified)",
        fontsize=12, fontweight="bold"
    )

    panels = [
        (correct_dict,   "Correctly Classified ✓",   "green",  1),
        (incorrect_dict, "Incorrectly Classified ✗",  "red",   2),
    ]

    for panel_data, panel_title, bcolor, panel_num in panels:
        for idx, class_name in enumerate(class_names):
            row, col   = divmod(idx, 3)
            subplot_num = (panel_num - 1) * 9 + row * 3 + col + 1

            ax = fig.add_subplot(6, 3, (panel_num - 1) * 9 + idx + 1)

            entries = panel_data.get(class_name, [])

            if entries:
                img_arr, true_lbl, pred_lbl, conf = entries[0]
                ax.imshow(img_arr, interpolation="lanczos")

                for spine in ax.spines.values():
                    spine.set_edgecolor(bcolor)
                    spine.set_linewidth(3.5)

                true_d = CLASS_DISPLAY.get(true_lbl, true_lbl)
                pred_d = CLASS_DISPLAY.get(pred_lbl, pred_lbl)

                if bcolor == "green":
                    t_str = f"True: {true_d}\nPred: {pred_d} ✓  {conf:.0%}"
                    tc    = "#1a7a1a"
                else:
                    t_str = f"True:  {true_d}\nPred: {pred_d} ✗  {conf:.0%}"
                    tc    = "#c0392b"

                ax.set_title(t_str, fontsize=8, color=tc, pad=3,
                             fontdict={"linespacing": 1.4})
            else:
                ax.text(0.5, 0.5, "No example\nfound",
                        ha="center", va="center", fontsize=9, color="gray",
                        transform=ax.transAxes)
                ax.set_facecolor("#f5f5f5")

            ax.set_xticks([])
            ax.set_yticks([])

        # Panel label
        fig.text(0.25 if panel_num == 1 else 0.75, 0.99,
                 panel_title,
                 ha="center", va="top", fontsize=11, fontweight="bold",
                 color="#1a7a1a" if bcolor == "green" else "#c0392b")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGURES_DIR, "classification_examples_combined.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 62)
    print("  GENERATE CLASSIFICATION EXAMPLES")
    print("  Model : CNN+Transformer Hybrid v2")
    print("=" * 62)

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_model()
    print(f"[Model] Input shape: {model.input_shape}")

    # ── Get class names from test directory (must match train order) ─────────
    test_dir_path = Path(config.TEST_DIR)
    if not test_dir_path.exists():
        raise FileNotFoundError(
            f"Test directory not found: {config.TEST_DIR}\n"
            "Run  python3 download_dataset.py  first."
        )
    class_names = sorted([d.name for d in test_dir_path.iterdir() if d.is_dir()])
    print(f"[Classes] {len(class_names)}: {class_names}\n")

    # ── Collect examples ─────────────────────────────────────────────────────
    correct_ex, incorrect_ex = collect_examples(
        model, test_dir_path, class_names, n_per_class=10
    )

    n_c = sum(len(v) for v in correct_ex.values())
    n_i = sum(len(v) for v in incorrect_ex.values())
    print(f"\n[Summary] Correct examples: {n_c} | Incorrect examples: {n_i}")

    # ── Generate figures ─────────────────────────────────────────────────────
    plot_grid(
        correct_ex, class_names,
        title="Correctly Classified Skin Lesion Examples\n"
              "(CNN+Transformer Hybrid v2 — ISIC 2018 Test Set)",
        filename="correct_examples.png",
        border_color="green",
    )

    plot_grid(
        incorrect_ex, class_names,
        title="Incorrectly Classified Skin Lesion Examples\n"
              "(CNN+Transformer Hybrid v2 — ISIC 2018 Test Set)",
        filename="incorrect_examples.png",
        border_color="red",
    )

    plot_combined(correct_ex, incorrect_ex, class_names)

    print("\n" + "=" * 62)
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("  ✅  Done")
    print("=" * 62)
