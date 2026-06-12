"""
src/evaluate.py
---------------
Evaluation pipeline for all models:

  Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
  Segmentation:   Dice Score, IoU, Pixel Accuracy, Sensitivity, Specificity

Usage:
    python src/evaluate.py --task classification
    python src/evaluate.py --task segmentation
    python src/evaluate.py --task all
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_pipeline import load_classification_data, load_segmentation_data
from src.segmentation_models import dice_coefficient, iou_metric, bce_dice_loss


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str) -> tf.keras.Model:
    """Load a saved classification model."""
    paths_to_try = [
        os.path.join(config.MODELS_DIR, f"{model_name}_final.keras"),
        os.path.join(config.MODELS_DIR, f"{model_name}_phase2_best.keras"),
        os.path.join(config.MODELS_DIR, f"{model_name}_best.keras"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"[Evaluate] Loading model from {path}")
            return tf.keras.models.load_model(path)

    raise FileNotFoundError(f"No saved model found for '{model_name}'.")


def get_predictions(model, test_gen) -> tuple:
    """Run inference on classification test set."""
    test_gen.reset()
    steps = len(test_gen)
    y_prob = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes[:len(y_pred)]
    return y_true, y_pred


def compute_clf_metrics(y_true, y_pred) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_classification_report(y_true, y_pred, class_names, model_name):
    """Save per-class classification report."""
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    df = pd.DataFrame(report).transpose()
    out_path = os.path.join(config.LOGS_DIR, f"{model_name}_classification_report.csv")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    df.to_csv(out_path, float_format="%.4f")
    print(f"[Evaluate] Classification report → {out_path}")
    return df


def count_params(model) -> float:
    """Return total parameter count in millions."""
    total = sum(int(np.prod(w.shape)) for w in model.weights)
    return round(total / 1e6, 2)


def evaluate_classification_model(model_name, test_gen, class_names) -> dict:
    """Full evaluation for one classification model."""
    print(f"\n{'═' * 60}")
    print(f"  Evaluating: {model_name} (Classification)")
    print(f"{'═' * 60}")

    model = load_model(model_name)
    params = count_params(model)
    y_true, y_pred = get_predictions(model, test_gen)
    metrics = compute_clf_metrics(y_true, y_pred)
    metrics["params_M"] = params

    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  Precision : {metrics['macro_precision']:.4f}")
    print(f"  Recall    : {metrics['macro_recall']:.4f}")
    print(f"  Params    : {params}M")

    save_classification_report(y_true, y_pred, class_names, model_name)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(config.LOGS_DIR, f"{model_name}_confusion_matrix.npy")
    np.save(cm_path, cm)

    # Merge timing info
    timing_path = os.path.join(config.LOGS_DIR, f"{model_name}_summary.json")
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)
        metrics["time_per_epoch_s"] = timing.get("time_per_epoch_s")
        metrics["total_epochs"] = timing.get("total_epochs")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_seg_model(model_name: str) -> tf.keras.Model:
    """Load a saved segmentation model."""
    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
    }
    paths_to_try = [
        os.path.join(config.MODELS_DIR, f"{model_name}_final.keras"),
        os.path.join(config.MODELS_DIR, f"{model_name}_best.keras"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"[Evaluate] Loading segmentation model from {path}")
            return tf.keras.models.load_model(path, custom_objects=custom_objects)

    raise FileNotFoundError(f"No saved model found for '{model_name}'.")


def compute_seg_metrics(model, test_ds) -> dict:
    """Compute segmentation metrics on test set."""
    all_dice = []
    all_iou = []
    all_pixel_acc = []
    all_sensitivity = []
    all_specificity = []

    for images, masks in test_ds:
        predictions = model.predict(images, verbose=0)

        # Handle instance seg model with dual outputs
        if isinstance(predictions, dict):
            predictions = predictions["segmentation"]
        elif isinstance(predictions, (list, tuple)):
            # Find the segmentation output (4D tensor)
            for p in predictions:
                if len(p.shape) == 4:
                    predictions = p
                    break

        pred_binary = (predictions > 0.5).astype(np.float32)

        for i in range(len(images)):
            gt = masks[i].numpy().flatten()
            pr = pred_binary[i].flatten()

            # Dice
            intersection = np.sum(gt * pr)
            dice = (2.0 * intersection + 1e-7) / (np.sum(gt) + np.sum(pr) + 1e-7)
            all_dice.append(dice)

            # IoU
            union = np.sum(gt) + np.sum(pr) - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            all_iou.append(iou)

            # Pixel accuracy
            pixel_acc = np.mean(gt == pr)
            all_pixel_acc.append(pixel_acc)

            # Sensitivity (True Positive Rate)
            tp = np.sum(gt * pr)
            fn = np.sum(gt * (1 - pr))
            sensitivity = (tp + 1e-7) / (tp + fn + 1e-7)
            all_sensitivity.append(sensitivity)

            # Specificity (True Negative Rate)
            tn = np.sum((1 - gt) * (1 - pr))
            fp = np.sum((1 - gt) * pr)
            specificity = (tn + 1e-7) / (tn + fp + 1e-7)
            all_specificity.append(specificity)

    return {
        "dice_score": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "iou": float(np.mean(all_iou)),
        "iou_std": float(np.std(all_iou)),
        "pixel_accuracy": float(np.mean(all_pixel_acc)),
        "sensitivity": float(np.mean(all_sensitivity)),
        "specificity": float(np.mean(all_specificity)),
    }


def evaluate_segmentation_model(model_name, test_ds) -> dict:
    """Full evaluation for one segmentation model."""
    print(f"\n{'═' * 60}")
    print(f"  Evaluating: {model_name} (Segmentation)")
    print(f"{'═' * 60}")

    model = load_seg_model(model_name)
    params = count_params(model)
    metrics = compute_seg_metrics(model, test_ds)
    metrics["params_M"] = params

    print(f"  Dice Score     : {metrics['dice_score']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  IoU            : {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"  Pixel Accuracy : {metrics['pixel_accuracy']:.4f}")
    print(f"  Sensitivity    : {metrics['sensitivity']:.4f}")
    print(f"  Specificity    : {metrics['specificity']:.4f}")
    print(f"  Params         : {params}M")

    # Merge timing info
    timing_path = os.path.join(config.LOGS_DIR, f"{model_name}_summary.json")
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)
        metrics["time_per_epoch_s"] = timing.get("time_per_epoch_s")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all",
                        choices=["classification", "segmentation", "all"])
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    all_metrics = {}

    # ── Classification Evaluation ──────────────────────────────────────────
    if args.task in ("classification", "all"):
        print("\n[Evaluate] Loading classification data...")
        _, _, test_gen, class_names, _ = load_classification_data()

        clf_models = [
            config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE,
            config.MODEL_VIT, config.MODEL_HYBRID_CLF,
        ]
        for name in clf_models:
            try:
                m = evaluate_classification_model(name, test_gen, class_names)
                all_metrics[name] = m
            except FileNotFoundError as e:
                print(f"⚠️  {e}")

    # ── Segmentation Evaluation ────────────────────────────────────────────
    if args.task in ("segmentation", "all"):
        print("\n[Evaluate] Loading segmentation data...")
        _, _, test_ds, _ = load_segmentation_data()

        if test_ds is not None:
            seg_models = [config.MODEL_UNET, config.MODEL_TRANSUNET, config.MODEL_INSTANCE]
            for name in seg_models:
                try:
                    m = evaluate_segmentation_model(name, test_ds)
                    all_metrics[name] = m
                except FileNotFoundError as e:
                    print(f"⚠️  {e}")

    # ── Save combined results ──────────────────────────────────────────────
    out_path = os.path.join(config.LOGS_DIR, "all_model_metrics.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Evaluate] All metrics saved → {out_path}")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("═" * 80)

    # Classification
    clf_names = [config.MODEL_CUSTOM, config.MODEL_EFFICIENT, config.MODEL_MOBILE,
                 config.MODEL_VIT, config.MODEL_HYBRID_CLF]
    clf_results = {k: v for k, v in all_metrics.items() if k in clf_names}
    if clf_results:
        print(f"\n  {'Model':<25} {'Acc(%)':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Params(M)':>10}")
        print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")
        for name, m in clf_results.items():
            print(f"  {name:<25} {m['accuracy'] * 100:>8.2f} {m['macro_f1']:>8.4f} "
                  f"{m['macro_precision']:>8.4f} {m['macro_recall']:>8.4f} {m['params_M']:>10.2f}")

    # Segmentation
    seg_names = [config.MODEL_UNET, config.MODEL_TRANSUNET, config.MODEL_INSTANCE]
    seg_results = {k: v for k, v in all_metrics.items() if k in seg_names}
    if seg_results:
        print(f"\n  {'Model':<25} {'Dice':>8} {'IoU':>8} {'PixAcc':>8} {'Sens':>8} {'Spec':>8} {'Params(M)':>10}")
        print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")
        for name, m in seg_results.items():
            print(f"  {name:<25} {m['dice_score']:>8.4f} {m['iou']:>8.4f} "
                  f"{m['pixel_accuracy']:>8.4f} {m['sensitivity']:>8.4f} "
                  f"{m['specificity']:>8.4f} {m['params_M']:>10.2f}")


if __name__ == "__main__":
    main()
