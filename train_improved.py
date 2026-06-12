#!/usr/bin/env python3
"""
train_improved.py
-----------------
Improved training pipeline with:
  1. Minority class oversampling (balance classes by repeat-sampling)
  2. Stronger augmentation (shear, channel shift, wider rotation)
  3. Cosine annealing LR + Label smoothing for EfficientNetB0
  4. Extended epochs (60 for EfficientNetB0, 30 for Custom CNN)
  5. Test-Time Augmentation (TTA) at evaluation

Run from terminal:
  python3 train_improved.py
"""

import os, sys, time, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.classification_models import (
    build_custom_cnn,
    build_efficientnetb0,
    build_mobilenetv2,
    build_cnn_transformer_hybrid,
    PatchExtract, PatchEmbedding, TransformerEncoderBlock,
    prepare_fine_tuning, print_model_summary,
)

os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# GPU setup
# ─────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s): {[g.name for g in gpus]}")
else:
    print("No GPU — training on CPU")


# ─────────────────────────────────────────────────────────────────────
# Improved Data Pipeline with Oversampling
# ─────────────────────────────────────────────────────────────────────

def load_data_with_oversampling(batch_size=16):
    """
    Load data and oversample minority classes so every class has
    at least `target_per_class` samples, then create generators.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import shutil

    TRAIN_DIR = config.TRAIN_DIR
    TEST_DIR = config.TEST_DIR

    # Count class sizes
    class_dirs = sorted([d for d in Path(TRAIN_DIR).iterdir() if d.is_dir()])
    class_counts = {d.name: len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) for d in class_dirs}
    target = max(class_counts.values())  # oversample to match largest class
    print("\n[DataPipeline] Original class counts:")
    for cls, cnt in class_counts.items():
        print(f"  {cls[:35]:35s}: {cnt:4d}  {'✓' if cnt == target else f'→ will oversample to {target}'}")

    # Create a temporary oversampled directory
    oversampled_dir = os.path.join(config.DATA_DIR, "oversampled_train")
    if os.path.exists(oversampled_dir):
        shutil.rmtree(oversampled_dir)
    os.makedirs(oversampled_dir)

    for d in class_dirs:
        cls = d.name
        images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        dst_dir = Path(oversampled_dir) / cls
        dst_dir.mkdir()
        n = len(images)
        # Copy all originals
        for img in images:
            shutil.copy(str(img), str(dst_dir / img.name))
        # Repeat-copy to reach target
        i = 0
        extra = target - n
        while extra > 0:
            img = images[i % n]
            new_name = f"os_{i:05d}{img.suffix}"
            shutil.copy(str(img), str(dst_dir / new_name))
            i += 1
            extra -= 1

    new_counts = {
        d.name: len(list((Path(oversampled_dir) / d.name).glob("*")))
        for d in class_dirs
    }
    total = sum(new_counts.values())
    print(f"\n[DataPipeline] After oversampling: {total} images, {target} per class")

    # Strong augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        zoom_range=0.25,
        shear_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.75, 1.25],
        channel_shift_range=30.0,
        fill_mode="reflect",
        validation_split=0.15,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        oversampled_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
        subset="training",
    )
    val_gen = train_datagen.flow_from_directory(
        oversampled_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=config.RANDOM_SEED,
        subset="validation",
    )
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"[DataPipeline] Train: {train_gen.samples}  Val: {val_gen.samples}  Test: {test_gen.samples}")
    class_names = list(train_gen.class_indices.keys())

    return train_gen, val_gen, test_gen, class_names


# ─────────────────────────────────────────────────────────────────────
# Cosine LR Schedule
# ─────────────────────────────────────────────────────────────────────

class CosineAnnealingLR(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate with warm restarts."""
    def __init__(self, base_lr=5e-4, min_lr=1e-7, total_epochs=60):
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        cos = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        lr = self.min_lr + (self.base_lr - self.min_lr) * cos
        self.model.optimizer.learning_rate.assign(lr)
        print(f"  [CosineAnnealingLR] Epoch {epoch+1}: lr = {lr:.2e}")


# ─────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA)
# ─────────────────────────────────────────────────────────────────────

def predict_with_tta(model, test_gen, tta_steps=5):
    """Apply TTA: predict N times with random augmentations, average logits."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    tta_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
    )

    all_preds = []
    # First pass: original (no augmentation)
    test_gen.reset()
    preds_orig = model.predict(test_gen, verbose=0)
    all_preds.append(preds_orig)
    y_true = test_gen.classes[:len(preds_orig)]

    # TTA passes
    for t in range(tta_steps - 1):
        tta_gen = tta_datagen.flow_from_directory(
            config.TEST_DIR,
            target_size=config.IMAGE_SIZE,
            batch_size=16,
            class_mode="categorical",
            shuffle=False,
            seed=t * 7,
        )
        preds = model.predict(tta_gen, verbose=0)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_preds, axis=1)
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────
# Compute metrics
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(config.LOGS_DIR, f"{model_name}_v2_cm.npy"), cm)
    print(f"\n{'='*50}")
    print(f"  {model_name} TEST RESULTS (with TTA)")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"{'='*50}\n")
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": prec, "macro_recall": rec}


# ─────────────────────────────────────────────────────────────────────
# Standard callbacks (no ReduceLROnPlateau — cosine handles LR)
# ─────────────────────────────────────────────────────────────────────

def get_callbacks(model_name, phase="", total_epochs=30, use_cosine=False, base_lr=5e-4):
    tag = f"{model_name}_v2_{phase}" if phase else f"{model_name}_v2"
    ckpt = os.path.join(config.MODELS_DIR, f"{tag}_best.keras")
    log = os.path.join(config.LOGS_DIR, f"{tag}_history.csv")
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(log, append=False),
    ]
    if use_cosine:
        cbs.append(CosineAnnealingLR(base_lr=base_lr, total_epochs=total_epochs))
    else:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1))
    return cbs, ckpt


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Custom CNN — 30 epochs + strong augmentation
# ─────────────────────────────────────────────────────────────────────

def train_custom_cnn_v2(train_gen, val_gen, test_gen):
    print("\n" + "█"*60)
    print("  EXPERIMENT 1: Custom CNN — 30 epochs, strong augmentation")
    print("█"*60)
    model = build_custom_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    print_model_summary(model)
    cbs, ckpt = get_callbacks("custom_cnn", total_epochs=30, use_cosine=True, base_lr=5e-4)
    t0 = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=30, callbacks=cbs, verbose=1,
    )
    elapsed = time.time() - t0
    model.save(os.path.join(config.MODELS_DIR, "custom_cnn_v2_final.keras"))
    print(f"\n✅ Custom CNN v2 done — {len(history.history['loss'])} epochs, {elapsed/len(history.history['loss']):.1f}s/ep")
    print("\n[Evaluating Custom CNN v2 with TTA...]")
    y_true, y_pred = predict_with_tta(model, test_gen, tta_steps=5)
    return compute_metrics(y_true, y_pred, "custom_cnn_v2"), history.history


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: EfficientNetB0 — 60 epochs, cosine LR, label smoothing
# ─────────────────────────────────────────────────────────────────────

def train_efficientnetb0_v2(train_gen, val_gen, test_gen):
    print("\n" + "█"*60)
    print("  EXPERIMENT 2: EfficientNetB0 — 60 epochs, cosine LR + label smoothing + TTA")
    print("█"*60)
    model, base_model = build_efficientnetb0()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    print_model_summary(model)

    # Phase 1: Feature extraction — 30 epochs, cosine LR
    print("\n[Phase 1] Feature Extraction (frozen base) — 30 epochs")
    cbs, ckpt = get_callbacks("efficientnetb0", "phase1", total_epochs=30, use_cosine=True, base_lr=5e-4)
    t0 = time.time()
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=30, callbacks=cbs, verbose=1,
    )
    ep1 = len(h1.history["loss"])
    print(f"\n✅ Phase 1 done — {ep1} epochs | best val_acc={max(h1.history['val_accuracy']):.4f}")

    # Phase 2: Fine-tuning — 30 more epochs, lower cosine LR
    print("\n[Phase 2] Fine-Tuning (top 30 layers unfrozen) — 30 epochs")
    model = prepare_fine_tuning(model, base_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    cbs2, _ = get_callbacks("efficientnetb0", "phase2", total_epochs=30, use_cosine=True, base_lr=1e-5)
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=30, callbacks=cbs2, verbose=1,
    )
    ep2 = len(h2.history["loss"])
    elapsed = time.time() - t0
    total_ep = ep1 + ep2
    print(f"\n✅ EfficientNetB0 v2 done — phase1={ep1} + phase2={ep2} epochs | {elapsed/total_ep:.1f}s/ep")
    model.save(os.path.join(config.MODELS_DIR, "efficientnetb0_v2_final.keras"))

    # Evaluate with TTA
    print("\n[Evaluating EfficientNetB0 v2 with TTA (5 passes)...]")
    y_true, y_pred = predict_with_tta(model, test_gen, tta_steps=5)
    merged = {k: h1.history[k] + h2.history[k] for k in h1.history}
    return compute_metrics(y_true, y_pred, "efficientnetb0_v2"), merged


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: CNN+Transformer Hybrid — 30 epochs, cosine LR
# ─────────────────────────────────────────────────────────────────────

def train_hybrid_v2(train_gen, val_gen, test_gen):
    print("\n" + "█"*60)
    print("  EXPERIMENT 3: CNN+Transformer Hybrid — 30 epochs, cosine LR + TTA")
    print("█"*60)
    model = build_cnn_transformer_hybrid()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    print_model_summary(model)
    cbs, _ = get_callbacks("cnn_transformer_hybrid", total_epochs=30, use_cosine=True, base_lr=5e-4)
    t0 = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=30, callbacks=cbs, verbose=1,
    )
    elapsed = time.time() - t0
    ep = len(history.history["loss"])
    model.save(os.path.join(config.MODELS_DIR, "cnn_transformer_hybrid_v2_final.keras"))
    print(f"\n✅ Hybrid v2 done — {ep} epochs, {elapsed/ep:.1f}s/ep")

    print("\n[Evaluating Hybrid v2 with TTA (5 passes)...]")
    y_true, y_pred = predict_with_tta(model, test_gen, tta_steps=5)
    return compute_metrics(y_true, y_pred, "cnn_transformer_hybrid_v2"), history.history


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  IMPROVED TRAINING PIPELINE")
    print("  EfficientNetB0 + Custom CNN + CNN+Transformer Hybrid")
    print("  Oversampling + Strong Augmentation + Cosine LR + TTA")
    print("="*60 + "\n")

    # Load data with oversampling
    train_gen, val_gen, test_gen, class_names = load_data_with_oversampling(batch_size=16)

    all_results = {}

    # Run experiments
    m1, h1 = train_custom_cnn_v2(train_gen, val_gen, test_gen)
    all_results["custom_cnn_v2"] = m1

    m2, h2 = train_efficientnetb0_v2(train_gen, val_gen, test_gen)
    all_results["efficientnetb0_v2"] = m2

    m3, h3 = train_hybrid_v2(train_gen, val_gen, test_gen)
    all_results["cnn_transformer_hybrid_v2"] = m3

    # Save combined results
    out = os.path.join(config.LOGS_DIR, "improved_metrics.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison
    print("\n" + "="*60)
    print("  FINAL COMPARISON: Original vs Improved")
    print("="*60)
    print(f"{'Model':<35} {'Acc (v1)':>10} {'Acc (v2+TTA)':>14}")
    print("-"*62)
    originals = {"custom_cnn": 31.4, "efficientnetb0": 44.9, "cnn_transformer_hybrid": 41.5}
    names_map = {
        "custom_cnn_v2": "custom_cnn",
        "efficientnetb0_v2": "efficientnetb0",
        "cnn_transformer_hybrid_v2": "cnn_transformer_hybrid",
    }
    for name, metrics in all_results.items():
        orig_name = names_map.get(name, name)
        orig_acc = originals.get(orig_name, "--")
        print(f"  {name:<33} {orig_acc:>8.1f}%  →  {metrics['accuracy']*100:>8.2f}%")

    print(f"\nAll results saved to: {out}")
