#!/usr/bin/env python3
"""Run TransUNet + Instance Seg only. U-Net result is hardcoded from prior run."""
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

def make_ds(pairs, augment=False):
    ip_t = tf.constant([str(p[0]) for p in pairs])
    mp_t = tf.constant([str(p[1]) for p in pairs])
    def _load(ip, mp):
        img = tf.cast(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(ip),channels=3),[H,W]),tf.float32)/255.0
        msk = tf.cast(tf.image.resize(tf.image.decode_png(tf.io.read_file(mp),channels=1),[H,W],method="nearest")>127,tf.float32)
        return img, msk
    def _aug(img,msk):
        if tf.random.uniform(())>.5: img=tf.image.flip_left_right(img); msk=tf.image.flip_left_right(msk)
        if tf.random.uniform(())>.5: img=tf.image.flip_up_down(img); msk=tf.image.flip_up_down(msk)
        img=tf.clip_by_value(tf.image.random_brightness(img,0.15),0.,1.)
        return img,msk
    ds = tf.data.Dataset.from_tensor_slices((ip_t,mp_t)).shuffle(len(pairs),seed=SEED).map(_load,num_parallel_calls=tf.data.AUTOTUNE)
    if augment: ds=ds.map(_aug,num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def dice_loss(yt,yp,s=1.):
    yt=tf.reshape(yt,[-1]); yp=tf.reshape(yp,[-1])
    return 1-(2*tf.reduce_sum(yt*yp)+s)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+s)
def bce_dice(yt,yp): return tf.reduce_mean(tf.keras.losses.binary_crossentropy(yt,yp))+dice_loss(yt,yp)
def dice_coef(yt,yp,s=1.):
    yt=tf.reshape(yt,[-1]); yp=tf.reshape(tf.cast(yp>.5,tf.float32),[-1])
    return (2*tf.reduce_sum(yt*yp)+s)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+s)
def iou_m(yt,yp,s=1.):
    yt=tf.reshape(yt,[-1]); yp=tf.reshape(tf.cast(yp>.5,tf.float32),[-1])
    inter=tf.reduce_sum(yt*yp)
    return (inter+s)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)-inter+s)

def _cb(x,f):
    for _ in range(2): x=layers.Conv2D(f,3,padding="same")(x); x=layers.BatchNormalization()(x); x=layers.Activation("relu")(x)
    return x

class TBlock(layers.Layer):
    def __init__(self,d,h,ff,**kw):
        super().__init__(**kw)
        self.att=layers.MultiHeadAttention(h,d//h)
        self.ff=tf.keras.Sequential([layers.Dense(ff,activation="gelu"),layers.Dense(d)])
        self.ln1=layers.LayerNormalization(epsilon=1e-6)
        self.ln2=layers.LayerNormalization(epsilon=1e-6)
    def call(self,x,training=False):
        x=self.ln1(x+self.att(x,x,training=training))
        x=self.ln2(x+self.ff(x,training=training))
        return x

def build_transunet():
    inp=layers.Input((H,W,3))
    c1=_cb(inp,32); p1=layers.MaxPooling2D()(c1)
    c2=_cb(p1,64);  p2=layers.MaxPooling2D()(c2)
    c3=_cb(p2,128); p3=layers.MaxPooling2D()(c3)
    fh,fw,fc=p3.shape[1],p3.shape[2],p3.shape[3]
    x=layers.Reshape((fh*fw,fc))(p3); x=layers.Dense(128)(x)
    pos=layers.Embedding(fh*fw,128)(tf.range(fh*fw)); x=x+pos
    x=TBlock(128,4,256,name="tb1")(x); x=TBlock(128,4,256,name="tb2")(x)
    x=layers.Dense(fc)(x); x=layers.Reshape((fh,fw,fc))(x)
    u3=layers.concatenate([layers.Conv2DTranspose(128,2,2,padding="same")(x),c3]); d3=_cb(u3,128)
    u2=layers.concatenate([layers.Conv2DTranspose(64,2,2,padding="same")(d3),c2]);  d2=_cb(u2,64)
    u1=layers.concatenate([layers.Conv2DTranspose(32,2,2,padding="same")(d2),c1]);  d1=_cb(u1,32)
    out=layers.Conv2D(1,1,activation="sigmoid")(d1)
    m=models.Model(inp,out,name="transunet")
    m.compile(optimizers.Adam(LR),loss=bce_dice,metrics=[dice_coef,iou_m]); return m

def build_instance():
    inp=layers.Input((H,W,3)); x=inp; skips=[]
    for f in [32,64,128]: x=_cb(x,f); skips.append(x); x=layers.MaxPooling2D()(x)
    x=_cb(x,256)
    cls=layers.GlobalAveragePooling2D()(x); cls=layers.Dense(128,activation="relu")(cls)
    cls=layers.Dropout(0.4)(cls); cls=layers.Dense(config.NUM_CLASSES,activation="softmax",name="classification")(cls)
    seg=x
    for f,sk in zip([128,64,32],reversed(skips)):
        seg=layers.concatenate([layers.Conv2DTranspose(f,2,2,padding="same")(seg),sk]); seg=_cb(seg,f)
    seg=layers.Conv2DTranspose(16,2,2,padding="same")(seg); seg=_cb(seg,16)
    seg=layers.Conv2D(1,1,activation="sigmoid",name="segmentation")(seg)
    m=models.Model(inp,{"classification":cls,"segmentation":seg},name="instance_seg")
    m.compile(optimizers.Adam(LR),
              loss={"classification":"categorical_crossentropy","segmentation":bce_dice},
              loss_weights={"classification":0.3,"segmentation":0.7},
              metrics={"classification":["accuracy"],"segmentation":[dice_coef,iou_m]}); return m

def make_ds_inst(pairs):
    n=len(pairs); dummy=np.zeros((n,config.NUM_CLASSES),dtype=np.float32); dummy[:,0]=1.
    ip_t=tf.constant([str(p[0]) for p in pairs]); mp_t=tf.constant([str(p[1]) for p in pairs]); cl_t=tf.constant(dummy)
    def _load(ip,mp,cv):
        img=tf.cast(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(ip),channels=3),[H,W]),tf.float32)/255.0
        msk=tf.cast(tf.image.resize(tf.image.decode_png(tf.io.read_file(mp),channels=1),[H,W],method="nearest")>127,tf.float32)
        return img,{"classification":cv,"segmentation":msk}
    ds=tf.data.Dataset.from_tensor_slices((ip_t,mp_t,cl_t)).shuffle(n,seed=SEED).map(_load,num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def evaluate(model,test_pairs,inst=False):
    ds=make_ds_inst(test_pairs) if inst else make_ds(test_pairs)
    dices,ious,accs,senss,specs=[],[],[],[],[]
    for batch in ds:
        imgs,targets=batch
        masks_np=targets["segmentation"].numpy() if inst else targets.numpy()
        preds=model.predict(imgs,verbose=0)
        if isinstance(preds,dict): preds=preds["segmentation"]
        preds_bin=(preds>.5).astype(np.float32)
        for p,m in zip(preds_bin,masks_np):
            p,m=p.flatten(),m.flatten()
            tp=np.sum((p==1)&(m==1)); tn=np.sum((p==0)&(m==0))
            fp=np.sum((p==1)&(m==0)); fn=np.sum((p==0)&(m==1)); s=1e-6
            dices.append((2*tp+s)/(2*tp+fp+fn+s)); ious.append((tp+s)/(tp+fp+fn+s))
            accs.append((tp+tn)/(tp+tn+fp+fn+s)); senss.append((tp+s)/(tp+fn+s)); specs.append((tn+s)/(tn+fp+s))
    return {k:float(np.mean(v)) for k,v in zip(["dice","iou","pix_acc","sens","spec"],[dices,ious,accs,senss,specs])}

def cbs(name,monitor="val_dice_coef"):
    return [
        tf.keras.callbacks.EarlyStopping(monitor=monitor,patience=PATIENCE,restore_best_weights=True,mode="max",verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=3,min_lr=1e-7,verbose=0),
        tf.keras.callbacks.CSVLogger(os.path.join(LOGS_DIR,f"{name}_seg_history.csv")),
    ]

if __name__=="__main__":
    tr,va,te=load_pairs()
    print(f"Train={len(tr)} Val={len(va)} Test={len(te)}")
    tr_ds=make_ds(tr,augment=True); va_ds=make_ds(va)

    results={
        "unet":{"dice":0.9357,"iou":0.8832,"pix_acc":0.9061,"sens":0.9651,"spec":0.7089}
    }

    print("\n── TransUNet ───────────────────────────────────────")
    tm=build_transunet()
    t0=time.time()
    tm.fit(tr_ds,validation_data=va_ds,epochs=EPOCHS,callbacks=cbs("transunet"),verbose=1)
    print(f"  {time.time()-t0:.0f}s")
    results["transunet"]=evaluate(tm,te)
    r=results["transunet"]
    print(f"  Dice={r['dice']:.4f} IoU={r['iou']:.4f} Acc={r['pix_acc']:.4f} Sens={r['sens']:.4f} Spec={r['spec']:.4f}")

    print("\n── Instance Seg ────────────────────────────────────")
    im=build_instance()
    tr_i=make_ds_inst(tr); va_i=make_ds_inst(va)
    t0=time.time()
    im.fit(tr_i,validation_data=va_i,epochs=EPOCHS,
           callbacks=cbs("instance_seg",monitor="val_loss"),verbose=1)
    print(f"  {time.time()-t0:.0f}s")
    results["instance_seg"]=evaluate(im,te,inst=True)
    r=results["instance_seg"]
    print(f"  Dice={r['dice']:.4f} IoU={r['iou']:.4f} Acc={r['pix_acc']:.4f} Sens={r['sens']:.4f} Spec={r['spec']:.4f}")

    out=os.path.join(LOGS_DIR,"seg_results.json")
    with open(out,"w") as f: json.dump(results,f,indent=2)
    print(f"\nSaved → {out}")
    print("\n=== SUMMARY ===")
    for nm,res in results.items():
        print(f"  {nm:<20} Dice={res['dice']:.4f} IoU={res['iou']:.4f} Acc={res['pix_acc']:.4f}")
    print("✅ Done!")
