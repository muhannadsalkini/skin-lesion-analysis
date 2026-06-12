"""
src/segmentation_models.py
--------------------------
All segmentation model architectures:

  Model 6 — U-Net (classic semantic segmentation baseline)
  Model 7 — TransUNet (Transformer-based segmentation)
  Model 8 — Instance Segmentation head (Mask R-CNN inspired approach)

Each builder returns a compiled tf.keras.Model for binary lesion segmentation.
"""

import sys
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper: optimizer builder
# ─────────────────────────────────────────────────────────────────────────────

def _get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions for segmentation
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss for binary segmentation."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy + Dice loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def iou_metric(y_true, y_pred, smooth=1.0):
    """Intersection over Union (IoU / Jaccard) metric."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 6 — U-Net (Classic Semantic Segmentation)
# ═══════════════════════════════════════════════════════════════════════════════

def _conv_block(x, filters, kernel_size=3):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU"""
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet(
    input_shape: tuple = (*config.SEG_IMAGE_SIZE, 3),
    lr: float = config.SEG_LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    U-Net architecture for binary lesion segmentation.

    Architecture:
      Encoder: 4 levels of double-conv + maxpool (64→128→256→512)
      Bottleneck: 1024 filters
      Decoder: 4 levels of upconv + skip-concat + double-conv
      Output: 1×1 Conv → Sigmoid (binary mask)

    Reference: Ronneberger et al., "U-Net: Convolutional Networks for
               Biomedical Image Segmentation," MICCAI 2015.
    """
    inputs = layers.Input(shape=input_shape)

    # ── Encoder ───────────────────────────────────────────────────────────
    # Level 1
    c1 = _conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Level 2
    c2 = _conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Level 3
    c3 = _conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Level 4
    c4 = _conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # ── Bottleneck ────────────────────────────────────────────────────────
    bn = _conv_block(p4, 1024)

    # ── Decoder ───────────────────────────────────────────────────────────
    # Level 4
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])
    d4 = _conv_block(u4, 512)

    # Level 3
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(d4)
    u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, 256)

    # Level 2
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, 128)

    # Level 1
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, 64)

    # ── Output ────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name="unet")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient, iou_metric],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 7 — TransUNet (Transformer-based Segmentation)
# ═══════════════════════════════════════════════════════════════════════════════

class SegTransformerBlock(layers.Layer):
    """Transformer encoder block for segmentation."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate_val = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn = self.att(x, x, training=training)
        attn = self.drop(attn, training=training)
        x = self.ln1(x + attn)
        ffn_out = self.ffn(x, training=training)
        x = self.ln2(x + ffn_out)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate_val,
        })
        return cfg


def build_transunet(
    input_shape: tuple = (*config.SEG_IMAGE_SIZE, 3),
    embed_dim: int = 256,
    num_heads: int = 4,
    ff_dim: int = 512,
    num_transformer_blocks: int = 4,
    dropout_rate: float = 0.1,
    lr: float = config.SEG_LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    TransUNet: CNN encoder + Transformer bottleneck + CNN decoder.

    Architecture:
      Encoder: CNN (3 conv blocks) with skip connections
      Bottleneck: Flatten spatial → Transformer Encoder → Reshape back
      Decoder: Upsampling + skip concatenation + conv blocks
      Output: 1×1 Conv → Sigmoid

    Reference: Chen et al., "TransUNet: Transformers Make Strong Encoders
               for Medical Image Segmentation," arXiv 2021.
    """
    inputs = layers.Input(shape=input_shape)

    # ── CNN Encoder (with skip connections) ────────────────────────────────
    # Level 1: 256→128
    c1 = _conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Level 2: 128→64
    c2 = _conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Level 3: 64→32
    c3 = _conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # ── Transformer Bottleneck ─────────────────────────────────────────────
    # Flatten spatial dimensions to sequence
    feat_h, feat_w = p3.shape[1], p3.shape[2]
    feat_channels = p3.shape[3]
    num_tokens = feat_h * feat_w

    x = layers.Reshape((num_tokens, feat_channels))(p3)
    x = layers.Dense(embed_dim)(x)

    # Add positional encoding
    positions = tf.range(start=0, limit=num_tokens, delta=1)
    pos_embed = layers.Embedding(input_dim=num_tokens, output_dim=embed_dim)(positions)
    x = x + pos_embed

    # Transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = SegTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # Reshape back to spatial
    x = layers.Dense(feat_channels)(x)
    x = layers.Reshape((feat_h, feat_w, feat_channels))(x)

    # ── CNN Decoder ────────────────────────────────────────────────────────
    # Level 3: 32→64
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(x)
    u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, 256)

    # Level 2: 64→128
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, 128)

    # Level 1: 128→256
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, 64)

    # ── Output ─────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name="transunet")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient, iou_metric],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 8 — Instance-Based Segmentation (Lightweight Mask R-CNN inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def build_instance_segmentation(
    input_shape: tuple = (*config.SEG_IMAGE_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
    lr: float = config.SEG_LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    Instance-based segmentation model with dual output heads:
      - Classification head: predicts disease class
      - Segmentation head: predicts binary lesion mask

    This is a simplified Mask R-CNN-inspired architecture that performs
    both instance classification and pixel-level segmentation simultaneously.

    Architecture:
      Shared CNN Backbone (EfficientNetB0-like feature extractor)
      ├── Classification Branch: GAP → Dense → Softmax
      └── Segmentation Branch: FPN-style decoder → 1×1 Conv → Sigmoid

    This demonstrates instance-based methods where each detected object
    (lesion) gets both a class label and a segmentation mask.
    """
    inputs = layers.Input(shape=input_shape)

    # ── Shared Backbone ───────────────────────────────────────────────────
    # Block 1: 256→128
    x = layers.Conv2D(64, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    f1 = x  # skip connection
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2: 128→64
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    f2 = x  # skip connection
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3: 64→32
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    f3 = x  # skip connection
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4: 32→16
    x = layers.Conv2D(512, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    shared_features = x

    # ── Classification Branch ─────────────────────────────────────────────
    cls_x = layers.GlobalAveragePooling2D()(shared_features)
    cls_x = layers.Dense(256, activation="relu")(cls_x)
    cls_x = layers.Dropout(0.4)(cls_x)
    cls_output = layers.Dense(num_classes, activation="softmax", name="classification")(cls_x)

    # ── Segmentation Branch (FPN-style decoder) ───────────────────────────
    # Upsample from 16→32
    seg_x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(shared_features)
    seg_x = layers.Concatenate()([seg_x, f3])
    seg_x = _conv_block(seg_x, 256)

    # 32→64
    seg_x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(seg_x)
    seg_x = layers.Concatenate()([seg_x, f2])
    seg_x = _conv_block(seg_x, 128)

    # 64→128
    seg_x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(seg_x)
    seg_x = layers.Concatenate()([seg_x, f1])
    seg_x = _conv_block(seg_x, 64)

    # 128→256 (original resolution)
    seg_x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(seg_x)
    seg_x = _conv_block(seg_x, 32)

    seg_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="segmentation")(seg_x)

    # ── Multi-task Model ──────────────────────────────────────────────────
    model = models.Model(
        inputs=inputs,
        outputs={"classification": cls_output, "segmentation": seg_output},
        name="instance_seg",
    )
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss={
            "classification": "categorical_crossentropy",
            "segmentation": bce_dice_loss,
        },
        loss_weights={"classification": 0.3, "segmentation": 0.7},
        metrics={
            "classification": ["accuracy"],
            "segmentation": [dice_coefficient, iou_metric],
        },
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model summary helper
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model: tf.keras.Model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    total = trainable + non_trainable
    print(f"\n{'─' * 50}")
    print(f"  Model       : {model.name}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable   : {trainable:,}")
    print(f"  Frozen      : {non_trainable:,}")
    print(f"{'─' * 50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building all segmentation models...\n")

    unet = build_unet()
    print_model_summary(unet)

    transunet = build_transunet()
    print_model_summary(transunet)

    instance = build_instance_segmentation()
    print_model_summary(instance)

    print("✅ All segmentation models built successfully!")
