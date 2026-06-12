#!/usr/bin/env python3
"""Run Instance Segmentation only and merge into seg_results.json."""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import layers, models, optimizers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

H, W = 64, 64; BATCH = 16; EPOCHS = 15; PATIENCE = 5; LR = 1e-3; SEED = 42
TRAIN_F = 0.70; VAL_F = 0.15
np.random.seed(SEED); tf.random.set_seed(SEED)
IMAGES_DIR = config.IMAGES_DIR; MASKS_DIR = config.MASKS_DIR
LOGS_DIR = config.LOGS_DIR; os.makedirs(LOGS_DIR, exist_ok=True)

def load_pairs():
    imgs = sorted(Path(IMAGES_DIR).glob("*.jpg"))
    pairs = [(ip, Path(MASKS_DIR)/(ip.stem+"_mask.png"))
             for ip in imgs if (Path(MASKS_DIR)/(ip.stem+"_mask.png")).exists()]
    np.random.shuffle(pairs); n = len(pairs)
    ntr, nval = int(n*TRAIN_F), int(n*VAL_F)
    return pairs[:ntr], pairs[ntr:ntr+nval], pairs[ntr+nval:]

def dice_loss(yt,yp,s=1.):
    yt=tf.reshape(yt,[-1]); yp=tf.reshape(yp,[-1])
    return 1-(2*tf.reduce_sum(yt*yp)+s)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+s)
def bce_dice(yt,yp):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(yt,yp))+dice_loss(yt,yp)
def dice_coef(yt,yp,s=1.):
    yt=tf.reshape(yt,[-1]); yp=tf.reshape(tf.cast(yp>.5,tf.float32),[-1])
    return (2*tf.reduce_sum(yt*yp)+s)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+s)
def iou_m(yt,yp,s=1.):
    yt=tf.reshape(yt,[-1]); yp=tf.reshape(tf.cast(yp>.5,tf.float32),[-1])
    inter=tf.reduce_sum(yt*yp)
    return (inter+s)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)-inter+s)

def _cb(x,f):
    for _ in range(2):
        x=layers.Conv2D(f,3,padding="same")(x)
        x=layers.BatchNormalization()(x)
        x=layers.Activation("relu")(x)
    return x

def build_instance():
    """Fixed: FPN decoder brings 8→16→32→64, no extra upsampling."""
    inp=layers.Input((H,W,3)); x=inp; skips=[]
    for f in [32,64,128]:
        x=_cb(x,f); skips.append(x); x=layers.MaxPooling2D()(x)
    x=_cb(x,256)
    # Classification head
    cls=layers.GlobalAveragePooling2D()(x)
    cls=layers.Dense(128,activation="relu")(cls)
    cls=layers.Dropout(0.4)(cls)
    cls=layers.Dense(config.NUM_CLASSES,activation="softmax",name="classification")(cls)
    # Segmentation head — 3 upsamples: 8→16→32→64 (matches input H,W)
    seg=x
    for f,sk in zip([128,64,32],reversed(skips)):
        seg=layers.concatenate([layers.Conv2DTranspose(f,2,2,padding="same")(seg),sk])
        seg=_cb(seg,f)
    # Final 1×1 conv — output is already 64×64, NO extra upsample
    seg=layers.Conv2D(1,1,activation="sigmoid",name="segmentation")(seg)
    m=models.Model(inp,{"classification":cls,"segmentation":seg},name="instance_seg")
    m.compile(optimizers.Adam(LR),
              loss={"classification":"categorical_crossentropy","segmentation":bce_dice},
              loss_weights={"classification":0.3,"segmentation":0.7},
              metrics={"classification":["accuracy"],"segmentation":[dice_coef,iou_m]})
    return m

def make_ds_inst(pairs):
    n=len(pairs); dummy=np.zeros((n,config.NUM_CLASSES),dtype=np.float32); dummy[:,0]=1.
    ip_t=tf.constant([str(p[0]) for p in pairs])
    mp_t=tf.constant([str(p[1]) for p in pairs])
    cl_t=tf.constant(dummy)
    def _load(ip,mp,cv):
        img=tf.cast(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(ip),channels=3),[H,W]),tf.float32)/255.0
        msk=tf.cast(tf.image.resize(tf.image.decode_png(tf.io.read_file(mp),channels=1),[H,W],method="nearest")>127,tf.float32)
        return img,{"classification":cv,"segmentation":msk}
    ds=tf.data.Dataset.from_tensor_slices((ip_t,mp_t,cl_t)).shuffle(n,seed=SEED)
    ds=ds.map(_load,num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def evaluate_inst(model,test_pairs):
    ds=make_ds_inst(test_pairs)
    dices,ious,accs,senss,specs=[],[],[],[],[]
    for imgs,targets in ds:
        masks_np=targets["segmentation"].numpy()
        preds=model.predict(imgs,verbose=0)
        pb=(preds["segmentation"]>.5).astype(np.float32)
        for p,m in zip(pb,masks_np):
            p,m=p.flatten(),m.flatten()
            tp=np.sum((p==1)&(m==1)); tn=np.sum((p==0)&(m==0))
            fp=np.sum((p==1)&(m==0)); fn=np.sum((p==0)&(m==1)); s=1e-6
            dices.append((2*tp+s)/(2*tp+fp+fn+s)); ious.append((tp+s)/(tp+fp+fn+s))
            accs.append((tp+tn)/(tp+tn+fp+fn+s)); senss.append((tp+s)/(tp+fn+s))
            specs.append((tn+s)/(tn+fp+s))
    return {k:float(np.mean(v)) for k,v in zip(["dice","iou","pix_acc","sens","spec"],
            [dices,ious,accs,senss,specs])}

if __name__=="__main__":
    tr,va,te=load_pairs()
    print(f"Train={len(tr)} Val={len(va)} Test={len(te)}")

    print("\n── Instance Seg (fixed) ────────────────────────────")
    im=build_instance()
    im.summary()
    tr_i=make_ds_inst(tr); va_i=make_ds_inst(va)
    cbs=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=PATIENCE,
                                         restore_best_weights=True,verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=3,
                                             min_lr=1e-7,verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(LOGS_DIR,"instance_seg_history.csv")),
    ]
    t0=time.time()
    im.fit(tr_i,validation_data=va_i,epochs=EPOCHS,callbacks=cbs,verbose=1)
    print(f"  {time.time()-t0:.0f}s")
    results_inst=evaluate_inst(im,te)
    r=results_inst
    print(f"  Dice={r['dice']:.4f} IoU={r['iou']:.4f} Acc={r['pix_acc']:.4f} "
          f"Sens={r['sens']:.4f} Spec={r['spec']:.4f}")

    # Merge into seg_results.json
    out=os.path.join(LOGS_DIR,"seg_results.json")
    existing={}
    if os.path.exists(out):
        with open(out) as f: existing=json.load(f)
    existing["instance_seg"]=results_inst
    with open(out,"w") as f: json.dump(existing,f,indent=2)
    print(f"\nSaved → {out}")
    print("\n=== ALL RESULTS ===")
    for nm,res in existing.items():
        print(f"  {nm:<20} Dice={res['dice']:.4f} IoU={res['iou']:.4f} "
              f"Acc={res['pix_acc']:.4f} Sens={res['sens']:.4f} Spec={res['spec']:.4f}")
    print("✅ Done!")
