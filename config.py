"""
config.py
---------
Central configuration for the Final Project:
Skin Lesion Detection & Segmentation using ISIC 2018 Dataset.

Models:
  - Classification: Custom CNN, EfficientNetB0, MobileNetV2, ViT, CNN+Transformer Hybrid
  - Segmentation:   U-Net, TransUNet (Transformer-based), Instance-based (Mask R-CNN style)
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "ISIC2018")
IMAGES_DIR    = os.path.join(DATA_DIR, "images")
MASKS_DIR     = os.path.join(DATA_DIR, "masks")
LABELS_CSV    = os.path.join(DATA_DIR, "labels.csv")
DATASET_ROOT  = os.path.join(DATA_DIR, "Skin cancer ISIC The International Skin Imaging Collaboration")
TRAIN_DIR     = os.path.join(DATASET_ROOT, "Train")
TEST_DIR      = os.path.join(DATASET_ROOT, "Test")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
MODELS_DIR    = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR      = os.path.join(RESULTS_DIR, "logs")

# ─────────────────────────────────────────────
# DATASET - ISIC 2018
# ─────────────────────────────────────────────
IMAGE_SIZE       = (224, 224)         # For classification models
SEG_IMAGE_SIZE   = (256, 256)         # For segmentation models
NUM_CLASSES      = 9
CLASS_NAMES      = [
    "actinic keratosis", "basal cell carcinoma", "dermatofibroma",
    "melanoma", "nevus", "pigmented benign keratosis",
    "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion",
]
CLASS_FULL_NAMES = {
    "actinic keratosis": "Actinic Keratosis",
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "dermatofibroma": "Dermatofibroma",
    "melanoma": "Melanoma",
    "nevus": "Melanocytic Nevus",
    "pigmented benign keratosis": "Pigmented Benign Keratosis",
    "seborrheic keratosis": "Seborrheic Keratosis",
    "squamous cell carcinoma": "Squamous Cell Carcinoma",
    "vascular lesion": "Vascular Lesion",
}
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42

# ─────────────────────────────────────────────
# DEFAULT TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE       = 16
EPOCHS           = 15          # Feature-extraction / from-scratch phase
FINE_TUNE_EPOCHS = 10          # Fine-tuning phase (transfer models)
LEARNING_RATE    = 5e-4
FINE_TUNE_LR     = 1e-5
DROPOUT_RATE     = 0.4
OPTIMIZER        = "adam"      # "adam" | "sgd" | "rmsprop"
FINE_TUNE_LAYERS = 20          # Number of top base-model layers to unfreeze

# ─────────────────────────────────────────────
# SEGMENTATION HYPERPARAMETERS
# ─────────────────────────────────────────────
SEG_BATCH_SIZE   = 8
SEG_EPOCHS       = 50
SEG_LEARNING_RATE = 1e-4

# ─────────────────────────────────────────────
# HYPERPARAMETER SWEEP (EfficientNetB0)
# ─────────────────────────────────────────────
SWEEP_LR         = [1e-3, 5e-4, 1e-4]
SWEEP_DROPOUT    = [0.2, 0.4, 0.5]
SWEEP_OPTIMIZERS = ["adam", "sgd", "rmsprop"]
SWEEP_BATCH      = [8, 16, 32]
SWEEP_EPOCHS     = 15

# ─────────────────────────────────────────────
# MODEL NAMES
# ─────────────────────────────────────────────
# Classification
MODEL_CUSTOM      = "custom_cnn"
MODEL_EFFICIENT   = "efficientnetb0"
MODEL_MOBILE      = "mobilenetv2"
MODEL_VIT         = "vit_classifier"
MODEL_HYBRID_CLF  = "cnn_transformer_hybrid"

# Segmentation
MODEL_UNET        = "unet"
MODEL_TRANSUNET   = "transunet"
MODEL_INSTANCE    = "instance_seg"
