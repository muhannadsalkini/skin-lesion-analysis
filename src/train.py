"""
src/train.py
------------
Training pipeline for all models:

  Classification: Custom CNN, EfficientNetB0, MobileNetV2, ViT, CNN+Transformer Hybrid
  Segmentation:   U-Net, TransUNet, Instance Segmentation

Usage:
    python src/train.py --task classification --model all
    python src/train.py --task classification --model vit_classifier
    python src/train.py --task segmentation --model unet
    python src/train.py --task all
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_pipeline import load_classification_data, load_segmentation_data
from src.classification_models import (
    build_custom_cnn,
    build_efficientnetb0,
    build_mobilenetv2,
    build_vit_classifier,
    build_cnn_transformer_hybrid,
    prepare_fine_tuning,
    print_model_summary,
)
from src.segmentation_models import (
    build_unet,
    build_transunet,
    build_instance_segmentation,
    print_model_summary as seg_print_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, phase: str = "", monitor: str = "val_accuracy"):
    """Standard set of Keras callbacks."""
    tag = f"{model_name}_{phase}" if phase else model_name
    ckpt_path = os.path.join(config.MODELS_DIR, f"{tag}_best.keras")
    log_path = os.path.join(config.LOGS_DIR, f"{tag}_history.csv")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor=monitor,
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=7,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]
    return callbacks


def get_seg_callbacks(model_name: str, monitor: str = "val_dice_coefficient"):
    """Callbacks for segmentation models."""
    ckpt_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")
    log_path = os.path.join(config.LOGS_DIR, f"{model_name}_history.csv")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor=monitor,
            save_best_only=True, verbose=1, mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=10,
            restore_best_weights=True, verbose=1, mode="max",
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Save training summary
# ─────────────────────────────────────────────────────────────────────────────

def save_training_summary(model_name: str, history_dict: dict, time_per_ep: float):
    """Save training summary to JSON."""
    # Find best val metric
    if "val_accuracy" in history_dict:
        best_val = max(history_dict["val_accuracy"])
        metric_name = "best_val_accuracy"
    elif "val_dice_coefficient" in history_dict:
        best_val = max(history_dict["val_dice_coefficient"])
        metric_name = "best_val_dice"
    else:
        best_val = None
        metric_name = "unknown"

    summary = {
        "model": model_name,
        metric_name: float(best_val) if best_val else None,
        "time_per_epoch_s": float(time_per_ep),
        "total_epochs": len(history_dict.get("loss", [])),
    }
    out_path = os.path.join(config.LOGS_DIR, f"{model_name}_summary.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Train] Summary saved → {out_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_custom_cnn(train_gen, val_gen, class_weights):
    """Train the custom CNN from scratch."""
    print("\n" + "═" * 60)
    print("  Training: Custom CNN (from scratch)")
    print("═" * 60)

    model = build_custom_cnn()
    print_model_summary(model)

    t0 = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=config.EPOCHS, class_weight=class_weights,
        callbacks=get_callbacks(config.MODEL_CUSTOM),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_CUSTOM}_final.keras"))
    print(f"\n✅ Custom CNN done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history.history, time_per_ep


def _train_transfer(model_name, build_fn, train_gen, val_gen, class_weights):
    """Generic two-phase transfer learning trainer."""
    print("\n" + "═" * 60)
    print(f"  Training: {model_name.upper()} (Transfer Learning)")
    print("═" * 60)

    # Phase 1: Feature Extraction
    print("\n[Phase 1] Feature Extraction — base frozen")
    model, base_model = build_fn()
    print_model_summary(model)

    t0 = time.time()
    history_p1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=config.EPOCHS, class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase1"),
        verbose=1,
    )
    ep1 = len(history_p1.history["loss"])

    # Phase 2: Fine-Tuning
    print("\n[Phase 2] Fine-Tuning — top layers unfrozen")
    model = prepare_fine_tuning(model, base_model)

    history_p2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=config.FINE_TUNE_EPOCHS, class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase2"),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep2 = len(history_p2.history["loss"])

    model.save(os.path.join(config.MODELS_DIR, f"{model_name}_final.keras"))

    total_epochs = ep1 + ep2
    time_per_ep = elapsed / total_epochs

    print(f"\n✅ {model_name} done — phase1={ep1} + phase2={ep2}, {time_per_ep:.1f}s/epoch")

    merged = {k: history_p1.history[k] + history_p2.history[k] for k in history_p1.history}
    return model, merged, time_per_ep


def train_vit(train_gen, val_gen, class_weights):
    """Train the Vision Transformer classifier."""
    print("\n" + "═" * 60)
    print("  Training: Vision Transformer (ViT)")
    print("═" * 60)

    model = build_vit_classifier()
    print_model_summary(model)

    t0 = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=config.EPOCHS, class_weight=class_weights,
        callbacks=get_callbacks(config.MODEL_VIT),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_VIT}_final.keras"))
    print(f"\n✅ ViT done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history.history, time_per_ep


def train_hybrid(train_gen, val_gen, class_weights):
    """Train the CNN + Transformer hybrid classifier."""
    print("\n" + "═" * 60)
    print("  Training: CNN + Transformer Hybrid")
    print("═" * 60)

    model = build_cnn_transformer_hybrid()
    print_model_summary(model)

    t0 = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=config.EPOCHS, class_weight=class_weights,
        callbacks=get_callbacks(config.MODEL_HYBRID_CLF),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_HYBRID_CLF}_final.keras"))
    print(f"\n✅ Hybrid done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history.history, time_per_ep


def run_classification_training(model_name: str = "all"):
    """Run classification model training."""
    print("\n[Train] Loading classification data...")
    train_gen, val_gen, _, class_names, class_weights = load_classification_data()

    results = {}

    if model_name in ("custom_cnn", "all"):
        _, hist, tpe = train_custom_cnn(train_gen, val_gen, class_weights)
        results[config.MODEL_CUSTOM] = save_training_summary(config.MODEL_CUSTOM, hist, tpe)

    if model_name in ("efficientnetb0", "all"):
        _, hist, tpe = _train_transfer(
            config.MODEL_EFFICIENT, build_efficientnetb0, train_gen, val_gen, class_weights
        )
        results[config.MODEL_EFFICIENT] = save_training_summary(config.MODEL_EFFICIENT, hist, tpe)

    if model_name in ("mobilenetv2", "all"):
        _, hist, tpe = _train_transfer(
            config.MODEL_MOBILE, build_mobilenetv2, train_gen, val_gen, class_weights
        )
        results[config.MODEL_MOBILE] = save_training_summary(config.MODEL_MOBILE, hist, tpe)

    if model_name in ("vit_classifier", "all"):
        _, hist, tpe = train_vit(train_gen, val_gen, class_weights)
        results[config.MODEL_VIT] = save_training_summary(config.MODEL_VIT, hist, tpe)

    if model_name in ("cnn_transformer_hybrid", "all"):
        _, hist, tpe = train_hybrid(train_gen, val_gen, class_weights)
        results[config.MODEL_HYBRID_CLF] = save_training_summary(config.MODEL_HYBRID_CLF, hist, tpe)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_unet(train_ds, val_ds):
    """Train U-Net for segmentation."""
    print("\n" + "═" * 60)
    print("  Training: U-Net (Semantic Segmentation)")
    print("═" * 60)

    model = build_unet()
    seg_print_summary(model)

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.SEG_EPOCHS,
        callbacks=get_seg_callbacks(config.MODEL_UNET),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_UNET}_final.keras"))
    print(f"\n✅ U-Net done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history.history, time_per_ep


def train_transunet(train_ds, val_ds):
    """Train TransUNet for segmentation."""
    print("\n" + "═" * 60)
    print("  Training: TransUNet (Transformer Segmentation)")
    print("═" * 60)

    model = build_transunet()
    seg_print_summary(model)

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.SEG_EPOCHS,
        callbacks=get_seg_callbacks(config.MODEL_TRANSUNET),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_TRANSUNET}_final.keras"))
    print(f"\n✅ TransUNet done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history.history, time_per_ep


def train_instance_seg(train_ds, val_ds, classification_data=None):
    """
    Train instance segmentation model.
    Requires both segmentation masks and classification labels.
    """
    print("\n" + "═" * 60)
    print("  Training: Instance-Based Segmentation")
    print("═" * 60)

    model = build_instance_segmentation()
    seg_print_summary(model)

    # For the instance model, we need paired (image, {class_label, mask}) data.
    # If we don't have paired classification+segmentation labels,
    # we train segmentation-only by duplicating the mask as a dummy class.
    # In a full implementation, this would use matched labels.

    t0 = time.time()

    # Use the segmentation dataset but adapt for dual output
    # Create a wrapper that provides the correct output format
    def adapt_for_instance(image, mask):
        # Create a dummy classification label (uniform distribution)
        # In production, this would use actual class labels
        batch_size = tf.shape(image)[0]
        dummy_cls = tf.ones([batch_size, config.NUM_CLASSES]) / config.NUM_CLASSES
        return image, {"classification": dummy_cls, "segmentation": mask}

    train_adapted = train_ds.map(adapt_for_instance)
    val_adapted = val_ds.map(adapt_for_instance)

    history = model.fit(
        train_adapted, validation_data=val_adapted,
        epochs=config.SEG_EPOCHS,
        callbacks=get_seg_callbacks(
            config.MODEL_INSTANCE,
            monitor="val_segmentation_dice_coefficient"
        ),
        verbose=1,
    )
    elapsed = time.time() - t0
    ep_count = len(history.history["loss"])
    time_per_ep = elapsed / ep_count

    model.save(os.path.join(config.MODELS_DIR, f"{config.MODEL_INSTANCE}_final.keras"))
    print(f"\n✅ Instance Seg done — {ep_count} epochs, {time_per_ep:.1f}s/epoch")
    return model, history.history, time_per_ep


def run_segmentation_training(model_name: str = "all"):
    """Run segmentation model training."""
    print("\n[Train] Loading segmentation data...")
    train_ds, val_ds, test_ds, df_test = load_segmentation_data()

    if train_ds is None:
        print("[Train] No segmentation data available. Skipping.")
        return {}

    results = {}

    if model_name in ("unet", "all"):
        _, hist, tpe = train_unet(train_ds, val_ds)
        results[config.MODEL_UNET] = save_training_summary(config.MODEL_UNET, hist, tpe)

    if model_name in ("transunet", "all"):
        _, hist, tpe = train_transunet(train_ds, val_ds)
        results[config.MODEL_TRANSUNET] = save_training_summary(config.MODEL_TRANSUNET, hist, tpe)

    if model_name in ("instance_seg", "all"):
        _, hist, tpe = train_instance_seg(train_ds, val_ds)
        results[config.MODEL_INSTANCE] = save_training_summary(config.MODEL_INSTANCE, hist, tpe)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train skin lesion models")
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["classification", "segmentation", "all"],
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=[
            "custom_cnn", "efficientnetb0", "mobilenetv2",
            "vit_classifier", "cnn_transformer_hybrid",
            "unet", "transunet", "instance_seg", "all",
        ],
    )
    args = parser.parse_args()

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU(s): {[g.name for g in gpus]}")
    else:
        print("⚠️  No GPU — training on CPU")

    all_results = {}

    if args.task in ("classification", "all"):
        clf_results = run_classification_training(args.model)
        all_results.update(clf_results)

    if args.task in ("segmentation", "all"):
        seg_results = run_segmentation_training(args.model)
        all_results.update(seg_results)

    # Print summary
    print("\n" + "═" * 60)
    print("  Training Complete — Summary")
    print("═" * 60)
    for name, s in all_results.items():
        metrics = {k: v for k, v in s.items() if k not in ("model",)}
        print(f"  {name:25s}  {metrics}")


if __name__ == "__main__":
    main()
