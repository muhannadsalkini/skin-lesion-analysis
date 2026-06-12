"""
src/data_pipeline.py
--------------------
Data loading using directory-based structure (flow_from_directory).
The ISIC 2018 Kaggle dataset is organized as:
  Train/<class_name>/*.jpg
  Test/<class_name>/*.jpg

We take the Train folder, split it into train/val, and use Test as the held-out test set.
"""

import os
import sys
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_classification_data(batch_size: int = config.BATCH_SIZE):
    """
    Load classification data using flow_from_directory.

    The Kaggle dataset has Train/ and Test/ folders with class subfolders.
    We split Train/ into train (85%) and val (15%), and use Test/ as test set.

    Returns: train_gen, val_gen, test_gen, class_names, class_weights
    """
    print("[DataPipeline] Loading classification data from directory structure...")

    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found: {config.TRAIN_DIR}\n"
            "Run  python download_dataset.py  first."
        )

    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.15,  # 15% of Train for validation
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=config.RANDOM_SEED,
        subset="validation",
    )

    test_gen = eval_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    # Compute class weights (inverse frequency)
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_gen.classes
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=labels,
    )
    class_weights = {i: w for i, w in enumerate(weights)}

    print(f"[DataPipeline] Classes ({num_classes}): {class_names}")
    print(f"[DataPipeline] Train: {train_gen.samples}  Val: {val_gen.samples}  Test: {test_gen.samples}")
    print(f"[DataPipeline] Class weights: { {class_names[i]: round(w, 2) for i, w in class_weights.items()} }")

    return train_gen, val_gen, test_gen, class_names, class_weights


def load_segmentation_data(batch_size: int = config.SEG_BATCH_SIZE):
    """
    Segmentation data loader.
    Returns None if no masks are available (this dataset has no masks).
    """
    masks_dir = config.MASKS_DIR
    if not os.path.exists(masks_dir) or len(os.listdir(masks_dir)) == 0:
        print("[DataPipeline] No segmentation masks available. Skipping segmentation.")
        return None, None, None, None

    return None, None, None, None


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing Classification Pipeline")
    print("=" * 60)
    try:
        train_gen, val_gen, test_gen, class_names, cw = load_classification_data()
        batch_x, batch_y = next(iter(train_gen))
        print(f"  Batch X: {batch_x.shape}  Y: {batch_y.shape}")
        print(f"  Classes: {class_names}")
    except Exception as e:
        print(f"  Error: {e}")
