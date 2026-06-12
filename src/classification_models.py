"""
src/classification_models.py
----------------------------
All classification model architectures:

  Model 1 — Custom CNN (trained from scratch)
  Model 2 — EfficientNetB0 (transfer learning)
  Model 3 — MobileNetV2 (transfer learning)
  Model 4 — Vision Transformer (ViT) classifier
  Model 5 — CNN + Transformer Hybrid classifier

Each builder returns a compiled tf.keras.Model ready for training.
"""

import sys
import math
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build optimizer from string name
# ─────────────────────────────────────────────────────────────────────────────

def _get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Custom CNN (Baseline, from scratch)
# ═══════════════════════════════════════════════════════════════════════════════

def build_custom_cnn(
    input_shape: tuple = (*config.IMAGE_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float = config.LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    Three convolutional blocks (32 → 64 → 128 filters):
        Conv2D(3×3) → BatchNorm → ReLU → MaxPool(2×2)
    Head: GAP → Dense(256) → Dropout → Softmax
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — EfficientNetB0 (Transfer Learning)
# ═══════════════════════════════════════════════════════════════════════════════

def build_efficientnetb0(
    input_shape: tuple = (*config.IMAGE_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float = config.LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
    trainable_base: bool = False,
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
) -> tuple:
    """
    EfficientNetB0 with ImageNet weights.
    Phase 1: base frozen.  Phase 2: top N layers unfrozen.
    """
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape,
    )

    if not trainable_base:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=255.0)(inputs)  # [0,1] → [0,255]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="efficientnetb0")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — MobileNetV2 (Transfer Learning)
# ═══════════════════════════════════════════════════════════════════════════════

def build_mobilenetv2(
    input_shape: tuple = (*config.IMAGE_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float = config.LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
    trainable_base: bool = False,
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
) -> tuple:
    """MobileNetV2 with ImageNet weights. Same two-phase strategy."""
    base_model = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape,
    )

    if not trainable_base:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — Vision Transformer (ViT) Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class PatchExtract(layers.Layer):
    """Extract non-overlapping patches from an image."""
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        num_patches = patches.shape[1] * patches.shape[2]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        return patches

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEmbedding(layers.Layer):
    """Project patches to embedding dimension + add positional encoding."""
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.projection = layers.Dense(embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, output_dim=embed_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        # Project patches
        projected = self.projection(patches)
        # Add [CLS] token
        batch_size = tf.shape(patches)[0]
        cls_token = tf.zeros([batch_size, 1, self.embed_dim])
        projected = tf.concat([cls_token, projected], axis=1)
        # Add positional embeddings
        encoded = projected + self.position_embedding(positions)
        return encoded

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
        })
        return cfg


class TransformerEncoderBlock(layers.Layer):
    """Single Transformer encoder block with multi-head self-attention."""
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
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-head self-attention
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate_val,
        })
        return cfg


def build_vit_classifier(
    input_shape: tuple = (*config.IMAGE_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
    patch_size: int = 16,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_transformer_blocks: int = 4,
    dropout_rate: float = 0.1,
    lr: float = config.LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    Vision Transformer (ViT) for image classification.

    Architecture:
      Image → Patch Extract → Patch Embedding + Positional Encoding
      → N × Transformer Encoder Blocks → [CLS] token → MLP Head → Softmax
    """
    h, w, c = input_shape
    num_patches = (h // patch_size) * (w // patch_size)

    inputs = layers.Input(shape=input_shape)

    # Extract and embed patches
    patches = PatchExtract(patch_size)(inputs)
    encoded = PatchEmbedding(num_patches, embed_dim)(patches)

    # Transformer encoder blocks
    for _ in range(num_transformer_blocks):
        encoded = TransformerEncoderBlock(
            embed_dim, num_heads, ff_dim, dropout_rate
        )(encoded)

    # Use [CLS] token output for classification
    cls_output = layers.LayerNormalization(epsilon=1e-6)(encoded[:, 0, :])

    # MLP classification head
    x = layers.Dense(256, activation="gelu")(cls_output)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="vit_classifier")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — CNN + Transformer Hybrid Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def build_cnn_transformer_hybrid(
    input_shape: tuple = (*config.IMAGE_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_transformer_blocks: int = 2,
    dropout_rate: float = config.DROPOUT_RATE,
    lr: float = config.LEARNING_RATE,
    optimizer: str = config.OPTIMIZER,
) -> tf.keras.Model:
    """
    CNN + Transformer Hybrid: CNN extracts spatial features, then Transformer
    encoder captures global context via self-attention.

    Architecture:
      Image → CNN Feature Extractor (3 conv blocks) → Reshape to sequence
      → Positional Encoding → N × Transformer Encoder → GAP → MLP → Softmax

    This demonstrates the key advantage of combining:
      - CNN's local feature extraction (edges, textures, patterns)
      - Transformer's global context modeling (long-range dependencies)
    """
    inputs = layers.Input(shape=input_shape)

    # ── CNN Feature Extractor ─────────────────────────────────────────────
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # ── Reshape CNN features to sequence for Transformer ──────────────────
    # After 3 MaxPool on 224×224: spatial dims = 28×28
    feat_h, feat_w = x.shape[1], x.shape[2]
    feat_channels = x.shape[3]
    num_tokens = feat_h * feat_w

    x = layers.Reshape((num_tokens, feat_channels))(x)

    # Project to embedding dimension
    x = layers.Dense(embed_dim)(x)

    # Add learnable positional encoding
    positions = tf.range(start=0, limit=num_tokens, delta=1)
    pos_embed = layers.Embedding(input_dim=num_tokens, output_dim=embed_dim)(positions)
    x = x + pos_embed

    # ── Transformer Encoder ───────────────────────────────────────────────
    for _ in range(num_transformer_blocks):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # ── Classification Head ───────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cnn_transformer_hybrid")
    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: prepare fine-tuning (unfreeze top N layers)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_fine_tuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_layers: int = config.FINE_TUNE_LAYERS,
    fine_tune_lr: float = config.FINE_TUNE_LR,
    optimizer: str = config.OPTIMIZER,
):
    """Unfreeze top layers and recompile for fine-tuning phase."""
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=_get_optimizer(optimizer, fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[Models] Fine-tuning: {fine_tune_layers} layers unfrozen, lr={fine_tune_lr}")
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
    print("Building all classification models...\n")

    cnn = build_custom_cnn()
    print_model_summary(cnn)

    eff, eff_base = build_efficientnetb0()
    print_model_summary(eff)

    mob, mob_base = build_mobilenetv2()
    print_model_summary(mob)

    vit = build_vit_classifier()
    print_model_summary(vit)

    hybrid = build_cnn_transformer_hybrid()
    print_model_summary(hybrid)

    print("✅ All classification models built successfully!")
