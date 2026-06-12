"""
main.py
-------
Master entry point for the BIS539 Final Project:
Skin Lesion Classification & Segmentation using ISIC 2018 Dataset.

Pipeline stages:
  1. download    — Download ISIC 2018 dataset from Kaggle
  2. train_clf   — Train classification models (CNN, EfficientNetB0, MobileNetV2, ViT, Hybrid)
  3. train_seg   — Train segmentation models (U-Net, TransUNet, Instance Seg)
  4. evaluate    — Evaluate all models on test set
  5. visualize   — Generate all figures for the report

Usage:
  python main.py                              # run all stages
  python main.py --stage download
  python main.py --stage train_clf
  python main.py --stage train_seg
  python main.py --stage evaluate
  python main.py --stage visualize
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import config


def _setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU(s) available: {[g.name for g in gpus]}")
    else:
        print("⚠️  No GPU found — running on CPU.")


def stage_download():
    print("\n" + "═" * 60)
    print("  STAGE 1 — Download ISIC 2018 Dataset")
    print("═" * 60)
    import download_dataset
    download_dataset.main()


def stage_train_clf(model: str = "all"):
    print("\n" + "═" * 60)
    print(f"  STAGE 2 — Train Classification Models: {model}")
    print("═" * 60)
    from src.train import run_classification_training
    run_classification_training(model)


def stage_train_seg(model: str = "all"):
    print("\n" + "═" * 60)
    print(f"  STAGE 3 — Train Segmentation Models: {model}")
    print("═" * 60)
    from src.train import run_segmentation_training
    run_segmentation_training(model)


def stage_evaluate():
    print("\n" + "═" * 60)
    print("  STAGE 4 — Evaluate All Models")
    print("═" * 60)
    import subprocess
    subprocess.run([sys.executable, "src/evaluate.py", "--task", "all"], check=True)


def stage_visualize():
    print("\n" + "═" * 60)
    print("  STAGE 5 — Generate Figures")
    print("═" * 60)
    import subprocess
    subprocess.run([sys.executable, "src/visualize.py", "--figure", "all"], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="BIS539 Final Project — Skin Lesion Classification & Segmentation"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["download", "train_clf", "train_seg", "evaluate", "visualize", "all"],
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

    _setup_gpu()

    if args.stage in ("download", "all"):
        stage_download()

    if args.stage in ("train_clf", "all"):
        stage_train_clf(args.model)

    if args.stage in ("train_seg", "all"):
        stage_train_seg(args.model)

    if args.stage in ("evaluate", "all"):
        stage_evaluate()

    if args.stage in ("visualize", "all"):
        stage_visualize()

    print("\n" + "═" * 60)
    print("  ✅  Pipeline complete!")
    print(f"  Models   → {config.MODELS_DIR}")
    print(f"  Logs     → {config.LOGS_DIR}")
    print(f"  Figures  → {config.FIGURES_DIR}")
    print("═" * 60)


if __name__ == "__main__":
    main()
