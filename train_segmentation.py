#!/usr/bin/env python3
"""
train_segmentation.py
---------------------
Full segmentation pipeline — fast (CPU-friendly) version:
  Step 1  Generate binary pseudo-masks (Otsu + morphology) → masks/
  Step 2  Train lightweight U-Net, TransUNet, Instance-Seg  (64×64 input)
  Step 3  Evaluate  →  Dice, IoU, PixAcc, Sensitivity, Specificity
  Step 4  Save  results/logs/seg_results.json

Run:
  cd final_project && python3 train_segmentation.py
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import layers, models, optimizers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-params  (small for fast CPU training)
# ─────────────────────────────────────────────────────────────────────────────
H, W         = 64, 64       # image size
BATCH        = 16
EPOCHS       = 15
PATIENCE     = 5
LR           = 1e-3
SEED         = 42
TRAIN_F      = 0.70
VAL_F        = 0.15
# TEST_F     = 0.15

IMAGES_DIR   = config.IMAGES_DIR
MASKS_DIR    = config.MASKS_DIR
MODELS_DIR   = config.MODELS_DIR
LOGS_DIR     = config.LOGS_DIR

for d in [MASKS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Generate pseudo-masks (Otsu + morphology)
# ─────────────────────────────────────────────────────────────────────────────

def gen_mask(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((H, W), dtype=np.uint8)
    img = cv2.resize(img, (W, H))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_eq = cv2.equalizeHist(lab[:, :, 0])
    blurred = cv2.GaussianBlur(l_eq, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    return mask


def generate_all_masks():
    imgs = sorted(Path(IMAGES_DIR).glob("*.jpg"))
    print(f"\n[Masks] Generating {len(imgs)} pseudo-masks…")
    done = 0
    for ip in imgs:
        op = Path(MASKS_DIR) / (ip.stem + "_mask.png")
        if op.exists():
            done += 1
            continue
        cv2.imwrite(str(op), gen_mask(str(ip)))
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(imgs)}")
    print(f"[Masks] Done – {done} masks")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs():
    imgs = sorted(Path(IMAGES_DIR).glob("*.jpg"))
    pairs = [(ip, Path(MASKS_DIR) / (ip.stem + "_mask.png"))
             for ip in imgs
             if (Path(MASKS_DIR) / (ip.stem + "_mask.png")).exists()]
    np.random.shuffle(pairs)
    n = len(pairs)
    ntr, nval = int(n * TRAIN_F), int(n * VAL_F)
    return pairs[:ntr], pairs[ntr:ntr + nval], pairs[ntr + nval:]


def make_ds(pairs, augment=False):
    img_p = tf.constant([str(p[0]) for p in pairs])
    msk_p = tf.constant([str(p[1]) for p in pairs])

    def _load(ip, mp):
        img = tf.cast(
            tf.image.resize(
                tf.image.decode_jpeg(tf.io.read_file(ip), channels=3),
                [H, W]), tf.float32) / 255.0
        msk = tf.cast(
            tf.image.resize(
                tf.image.decode_png(tf.io.read_file(mp), channels=1),
                [H, W], method="nearest") > 127, tf.float32)
        return img, msk

    def _aug(img, msk):
        if tf.random.uniform(()) > .5:
            img = tf.image.flip_left_right(img)
            msk = tf.image.flip_left_right(msk)
        if tf.random.uniform(()) > .5:
            img = tf.image.flip_up_down(img)
            msk = tf.image.flip_up_down(msk)
        img = tf.clip_by_value(
            tf.image.random_brightness(img, 0.15), 0., 1.)
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices((img_p, msk_p))
    ds = ds.shuffle(len(pairs), seed=SEED).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(yt, yp, s=1.):
    yt = tf.reshape(yt, [-1])
    yp = tf.reshape(yp, [-1])
    return 1 - (2 * tf.reduce_sum(yt * yp) + s) / (
        tf.reduce_sum(yt) + tf.reduce_sum(yp) + s)


def bce_dice(yt, yp):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(yt, yp)) + dice_loss(yt, yp)


def dice_coef(yt, yp, s=1.):
    yt = tf.reshape(yt, [-1])
    yp = tf.reshape(tf.cast(yp > .5, tf.float32), [-1])
    return (2 * tf.reduce_sum(yt * yp) + s) / (
        tf.reduce_sum(yt) + tf.reduce_sum(yp) + s)


def iou_m(yt, yp, s=1.):
    yt = tf.reshape(yt, [-1])
    yp = tf.reshape(tf.cast(yp > .5, tf.float32), [-1])
    inter = tf.reduce_sum(yt * yp)
    return (inter + s) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) - inter + s)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Lightweight model builders
# ─────────────────────────────────────────────────────────────────────────────

def _cb(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_light_unet():
    """Compact U-Net  (3 levels: 32→64→128)  for fast CPU training."""
    inp = layers.Input((H, W, 3))
    c1 = _cb(inp, 32);  p1 = layers.MaxPooling2D()(c1)
    c2 = _cb(p1, 64);   p2 = layers.MaxPooling2D()(c2)
    c3 = _cb(p2, 128);  p3 = layers.MaxPooling2D()(c3)
    bn = _cb(p3, 256)
    u3 = layers.concatenate([layers.Conv2DTranspose(128, 2, 2, padding="same")(bn), c3])
    d3 = _cb(u3, 128)
    u2 = layers.concatenate([layers.Conv2DTranspose(64,  2, 2, padding="same")(d3), c2])
    d2 = _cb(u2, 64)
    u1 = layers.concatenate([layers.Conv2DTranspose(32,  2, 2, padding="same")(d2), c1])
    d1 = _cb(u1, 32)
    out = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    m = models.Model(inp, out, name="unet")
    m.compile(optimizers.Adam(LR), loss=bce_dice,
              metrics=[dice_coef, iou_m])
    return m


class TBlock(layers.Layer):
    def __init__(self, d, h, ff, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(h, d // h)
        self.ff  = tf.keras.Sequential([
            layers.Dense(ff, activation="gelu"),
            layers.Dense(d)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        x = self.ln1(x + self.att(x, x, training=training))
        x = self.ln2(x + self.ff(x, training=training))
        return x


def build_light_transunet():
    """Compact TransUNet (2 CNN levels + 2 Transformer blocks)."""
    inp = layers.Input((H, W, 3))
    c1 = _cb(inp, 32);  p1 = layers.MaxPooling2D()(c1)
    c2 = _cb(p1, 64);   p2 = layers.MaxPooling2D()(c2)
    c3 = _cb(p2, 128);  p3 = layers.MaxPooling2D()(c3)

    fh, fw, fc = p3.shape[1], p3.shape[2], p3.shape[3]
    x = layers.Reshape((fh * fw, fc))(p3)
    x = layers.Dense(128)(x)
    pos = layers.Embedding(fh * fw, 128)(tf.range(fh * fw))
    x = x + pos
    x = TBlock(128, 4, 256, name="tb1")(x)
    x = TBlock(128, 4, 256, name="tb2")(x)
    x = layers.Dense(fc)(x)
    x = layers.Reshape((fh, fw, fc))(x)

    u3 = layers.concatenate([layers.Conv2DTranspose(128, 2, 2, padding="same")(x), c3])
    d3 = _cb(u3, 128)
    u2 = layers.concatenate([layers.Conv2DTranspose(64,  2, 2, padding="same")(d3), c2])
    d2 = _cb(u2, 64)
    u1 = layers.concatenate([layers.Conv2DTranspose(32,  2, 2, padding="same")(d2), c1])
    d1 = _cb(u1, 32)
    out = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    m = models.Model(inp, out, name="transunet")
    m.compile(optimizers.Adam(LR), loss=bce_dice,
              metrics=[dice_coef, iou_m])
    return m


def build_light_instance():
    """Compact instance-seg model (shared backbone + clf + seg heads)."""
    inp  = layers.Input((H, W, 3))
    x    = inp
    skips = []
    for f in [32, 64, 128]:
        x = _cb(x, f)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = _cb(x, 256)

    # Classification head
    cls = layers.GlobalAveragePooling2D()(x)
    cls = layers.Dense(128, activation="relu")(cls)
    cls = layers.Dropout(0.4)(cls)
    cls = layers.Dense(config.NUM_CLASSES, activation="softmax",
                       name="classification")(cls)

    # Segmentation head (FPN decoder)
    seg = x
    for f, sk in zip([128, 64, 32], reversed(skips)):
        seg = layers.concatenate([
            layers.Conv2DTranspose(f, 2, 2, padding="same")(seg), sk])
        seg = _cb(seg, f)
    seg = layers.Conv2DTranspose(16, 2, 2, padding="same")(seg)
    seg = _cb(seg, 16)
    seg = layers.Conv2D(1, 1, activation="sigmoid", name="segmentation")(seg)

    m = models.Model(inp, {"classification": cls, "segmentation": seg},
                     name="instance_seg")
    m.compile(
        optimizers.Adam(LR),
        loss={"classification": "categorical_crossentropy",
              "segmentation":   bce_dice},
        loss_weights={"classification": 0.3, "segmentation": 0.7},
        metrics={"classification": ["accuracy"],
                 "segmentation":   [dice_coef, iou_m]},
    )
    return m


def make_ds_inst(pairs):
    n = len(pairs)
    dummy = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
    dummy[:, 0] = 1.
    ip_t = tf.constant([str(p[0]) for p in pairs])
    mp_t = tf.constant([str(p[1]) for p in pairs])
    cl_t = tf.constant(dummy)

    def _load(ip, mp, cv):
        img = tf.cast(
            tf.image.resize(
                tf.image.decode_jpeg(tf.io.read_file(ip), channels=3),
                [H, W]), tf.float32) / 255.0
        msk = tf.cast(
            tf.image.resize(
                tf.image.decode_png(tf.io.read_file(mp), channels=1),
                [H, W], method="nearest") > 127, tf.float32)
        return img, {"classification": cv, "segmentation": msk}

    ds = tf.data.Dataset.from_tensor_slices((ip_t, mp_t, cl_t))
    ds = ds.shuffle(n, seed=SEED).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_pairs, inst=False):
    ds = make_ds_inst(test_pairs) if inst else make_ds(test_pairs)
    dices, ious, accs, senss, specs = [], [], [], [], []
    for batch in ds:
        imgs, targets = batch
        if inst:
            masks_np = targets["segmentation"].numpy()
        else:
            masks_np = targets.numpy()
        preds = model.predict(imgs, verbose=0)
        if isinstance(preds, dict):
            preds = preds["segmentation"]
        preds_bin = (preds > .5).astype(np.float32)
        for p, m in zip(preds_bin, masks_np):
            p, m = p.flatten(), m.flatten()
            tp = np.sum((p == 1) & (m == 1))
            tn = np.sum((p == 0) & (m == 0))
            fp = np.sum((p == 1) & (m == 0))
            fn = np.sum((p == 0) & (m == 1))
            s = 1e-6
            dices.append((2*tp+s)/(2*tp+fp+fn+s))
            ious.append((tp+s)/(tp+fp+fn+s))
            accs.append((tp+tn)/(tp+tn+fp+fn+s))
            senss.append((tp+s)/(tp+fn+s))
            specs.append((tn+s)/(tn+fp+s))
    return {k: float(np.mean(v)) for k, v in
            zip(["dice","iou","pix_acc","sens","spec"],
                [dices, ious, accs, senss, specs])}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def cbs(name, monitor="val_dice_coef"):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=PATIENCE,
            restore_best_weights=True, mode="max", verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=.5, patience=3, min_lr=1e-7, verbose=0),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOGS_DIR, f"{name}_seg_history.csv")),
    ]


if __name__ == "__main__":
    print("="*55)
    print(f"  SEGMENTATION PIPELINE  {H}×{W}  bs={BATCH}  ep={EPOCHS}")
    print("="*55)

    # 1. Masks
    generate_all_masks()

    # 2. Split
    tr, va, te = load_pairs()
    print(f"[Split] Train={len(tr)}  Val={len(va)}  Test={len(te)}")

    tr_ds = make_ds(tr, augment=True)
    va_ds = make_ds(va)

    results = {}

    # 3. U-Net
    print("\n── MODEL 6: U-Net ──────────────────────────────────")
    um = build_light_unet()
    t0 = time.time()
    um.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS,
           callbacks=cbs("unet"), verbose=1)
    print(f"  Trained in {time.time()-t0:.0f}s")
    results["unet"] = evaluate(um, te)
    r = results["unet"]
    print(f"  Dice={r['dice']:.4f}  IoU={r['iou']:.4f}  "
          f"Acc={r['pix_acc']:.4f}  Sens={r['sens']:.4f}  Spec={r['spec']:.4f}")

    # 4. TransUNet
    print("\n── MODEL 7: TransUNet ──────────────────────────────")
    tm = build_light_transunet()
    t0 = time.time()
    tm.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS,
           callbacks=cbs("transunet"), verbose=1)
    print(f"  Trained in {time.time()-t0:.0f}s")
    results["transunet"] = evaluate(tm, te)
    r = results["transunet"]
    print(f"  Dice={r['dice']:.4f}  IoU={r['iou']:.4f}  "
          f"Acc={r['pix_acc']:.4f}  Sens={r['sens']:.4f}  Spec={r['spec']:.4f}")

    # 5. Instance Seg
    print("\n── MODEL 8: Instance Segmentation ─────────────────")
    im = build_light_instance()
    tr_inst = make_ds_inst(tr)
    va_inst = make_ds_inst(va)
    t0 = time.time()
    im.fit(tr_inst, validation_data=va_inst, epochs=EPOCHS,
           callbacks=cbs("instance_seg", monitor="val_loss"), verbose=1)
    print(f"  Trained in {time.time()-t0:.0f}s")
    results["instance_seg"] = evaluate(im, te, inst=True)
    r = results["instance_seg"]
    print(f"  Dice={r['dice']:.4f}  IoU={r['iou']:.4f}  "
          f"Acc={r['pix_acc']:.4f}  Sens={r['sens']:.4f}  Spec={r['spec']:.4f}")

    # 6. Save
    out = os.path.join(LOGS_DIR, "seg_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # 7. Summary
    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    print(f"  {'Model':<20} {'Dice':>7} {'IoU':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7}")
    print("  " + "-"*51)
    for nm, res in results.items():
        print(f"  {nm:<20} "
              f"{res['dice']:>7.4f} {res['iou']:>7.4f} "
              f"{res['pix_acc']:>7.4f} {res['sens']:>7.4f} {res['spec']:>7.4f}")
    print(f"\n  Saved → {out}")
    print("  ✅  Done!")
