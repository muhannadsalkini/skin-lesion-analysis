"""
download_dataset.py
-------------------
Downloads the ISIC 2018 Skin Lesion dataset from Kaggle.

Dataset: ISIC 2018 Challenge — Task 1 (Segmentation) + Task 3 (Classification)
Source:  https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

This script downloads:
  - Dermoscopic images (JPEG)
  - Segmentation masks (PNG)
  - Classification labels (CSV with 7 disease categories)

Usage:
    python download_dataset.py
"""

import os
import sys
import glob
import shutil
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def download_from_kaggle():
    """Download ISIC 2018 dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print("[Download] Downloading ISIC 2018 dataset from Kaggle...")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Download the ISIC 2018 dataset
    # Using a well-known Kaggle dataset that has both images and masks
    api.dataset_download_files(
        "nodoubttome/skin-cancer9-classesisic",
        path=config.DATA_DIR,
        unzip=True,
    )
    print("[Download] Download complete.")


def organize_dataset():
    """
    Organize the downloaded dataset into a standardized structure:
        data/ISIC2018/images/   — all dermoscopic images
        data/ISIC2018/masks/    — all segmentation masks
        data/ISIC2018/labels.csv — classification labels
    """
    print("[Download] Organizing dataset...")

    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    os.makedirs(config.MASKS_DIR, exist_ok=True)

    # The Kaggle dataset may have different directory structures.
    # We handle common layouts:
    data_root = config.DATA_DIR

    # Look for image files recursively
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(data_root, "**", ext), recursive=True))

    # Separate images from masks (masks typically have "_segmentation" in name)
    images = []
    masks = []
    for f in all_images:
        fname = os.path.basename(f).lower()
        if "_segmentation" in fname or "mask" in fname or "seg" in fname:
            masks.append(f)
        elif "images" in f.lower() or "input" in f.lower() or not any(
            kw in f.lower() for kw in ["mask", "segmentation", "ground"]
        ):
            images.append(f)

    # Copy images and masks to organized directories
    img_count = 0
    for img_path in images:
        dst = os.path.join(config.IMAGES_DIR, os.path.basename(img_path))
        if not os.path.exists(dst) and img_path != dst:
            shutil.copy2(img_path, dst)
            img_count += 1

    mask_count = 0
    for mask_path in masks:
        dst = os.path.join(config.MASKS_DIR, os.path.basename(mask_path))
        if not os.path.exists(dst) and mask_path != dst:
            shutil.copy2(mask_path, dst)
            mask_count += 1

    print(f"[Download] Organized {img_count} images, {mask_count} masks")


def create_labels_csv():
    """
    Create or locate the classification labels CSV.
    ISIC 2018 Task 3 provides labels in a CSV with one-hot encoding:
        image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
    """
    # Check if labels CSV already exists
    if os.path.exists(config.LABELS_CSV):
        df = pd.read_csv(config.LABELS_CSV)
        print(f"[Download] Labels CSV found: {len(df)} entries")
        return

    # Look for ground truth CSV in downloaded files
    csv_candidates = glob.glob(os.path.join(config.DATA_DIR, "**", "*.csv"), recursive=True)

    labels_df = None
    for csv_path in csv_candidates:
        try:
            df = pd.read_csv(csv_path)
            # Check if it has the expected ISIC columns
            if any(col in df.columns for col in config.CLASS_NAMES):
                labels_df = df
                print(f"[Download] Found labels in: {csv_path}")
                break
        except Exception:
            continue

    if labels_df is not None:
        # Standardize column names
        if "image" not in labels_df.columns:
            # Try to identify the image ID column
            for col in labels_df.columns:
                if "image" in col.lower() or "isic" in col.lower() or "id" in col.lower():
                    labels_df = labels_df.rename(columns={col: "image"})
                    break

        labels_df.to_csv(config.LABELS_CSV, index=False)
        print(f"[Download] Labels saved to {config.LABELS_CSV}")
    else:
        # If no CSV found, create labels from directory structure
        print("[Download] No labels CSV found. Creating from directory structure...")
        records = []
        for class_dir in sorted(os.listdir(config.DATA_DIR)):
            class_path = os.path.join(config.DATA_DIR, class_dir)
            if not os.path.isdir(class_path) or class_dir in ("images", "masks"):
                continue
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    record = {"image": os.path.splitext(img_file)[0]}
                    for cn in config.CLASS_NAMES:
                        record[cn] = 1.0 if class_dir.upper() == cn else 0.0
                    records.append(record)
                    # Also copy image to images dir
                    src = os.path.join(class_path, img_file)
                    dst = os.path.join(config.IMAGES_DIR, img_file)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

        if records:
            labels_df = pd.DataFrame(records)
            labels_df.to_csv(config.LABELS_CSV, index=False)
            print(f"[Download] Created labels CSV with {len(records)} entries")
        else:
            print("[Download] WARNING: Could not create labels. Please provide labels.csv manually.")


def print_dataset_stats():
    """Print dataset statistics."""
    print("\n" + "═" * 60)
    print("  ISIC 2018 Dataset Summary")
    print("═" * 60)

    # Count images
    if os.path.exists(config.IMAGES_DIR):
        n_images = len([f for f in os.listdir(config.IMAGES_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  Images: {n_images:,}")

    # Count masks
    if os.path.exists(config.MASKS_DIR):
        n_masks = len([f for f in os.listdir(config.MASKS_DIR)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  Masks:  {n_masks:,}")

    # Class distribution
    if os.path.exists(config.LABELS_CSV):
        df = pd.read_csv(config.LABELS_CSV)
        print(f"\n  Classification Labels: {len(df):,} entries")
        print(f"  {'Class':<10} {'Full Name':<35} {'Count':>6} {'%':>7}")
        print(f"  {'─'*10} {'─'*35} {'─'*6} {'─'*7}")
        for cls in config.CLASS_NAMES:
            if cls in df.columns:
                count = int(df[cls].sum())
                pct = count / len(df) * 100
                full = config.CLASS_FULL_NAMES.get(cls, cls)
                print(f"  {cls:<10} {full:<35} {count:>6} {pct:>6.1f}%")

    print("═" * 60)


def main():
    print("\n" + "═" * 60)
    print("  ISIC 2018 Dataset Download & Setup")
    print("═" * 60)

    # Step 1: Download
    if not os.path.exists(config.IMAGES_DIR) or len(os.listdir(config.IMAGES_DIR)) < 100:
        download_from_kaggle()
    else:
        print("[Download] Dataset already exists, skipping download.")

    # Step 2: Organize
    organize_dataset()

    # Step 3: Labels
    create_labels_csv()

    # Step 4: Stats
    print_dataset_stats()

    print("\n✅ Dataset setup complete!")


if __name__ == "__main__":
    main()
