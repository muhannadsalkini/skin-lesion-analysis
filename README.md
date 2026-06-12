# 🔬 Skin Lesion Analysis — BIS539 Final Project

**Author:** Muhannad Salkini  
**Student ID:** 251129910  
**Course:** BIS539 — Pattern Recognition and Computer Vision  
**University:** Biruni University, Department of Computer Engineering  
**Semester:** 2025–2026 Spring  

---

## 📋 Project Overview

This project implements and evaluates **eight deep learning models** across **three paradigms** for automated skin lesion classification and segmentation using the **ISIC 2018 dermoscopic image dataset** (9-class variant, ~4,287 images).

The project fulfills the BIS539 final requirements:
- ✅ Segmentation (semantic + instance-based)
- ✅ Instance-based methods
- ✅ Transformer architectures (ViT, CNN+Transformer Hybrid, TransUNet)
- ✅ Literature review of 10 state-of-the-art papers
- ✅ Comparative analysis across all models

---

## 🏆 Results Summary

### Classification (5 models, 9-class ISIC 2018)

| # | Model | Type | Accuracy | Bal. Acc | F1 | Precision | Recall | Params |
|---|-------|------|:--------:|:--------:|:--:|:---------:|:------:|:------:|
| 1 | Custom CNN | From scratch | 44.07% | 54.17% | 0.454 | 0.427 | 0.542 | 0.13M |
| 2 | MobileNetV2 | Transfer Learning | 40.68% | 45.37% | 0.393 | 0.457 | 0.454 | 2.59M |
| 3 | ViT Classifier | Transformer (scratch) | 29.66% | 36.34% | 0.331 | 0.393 | 0.363 | 0.69M |
| 4 | EfficientNetB0 | Transfer Learning | 52.54% | 52.08% | 0.488 | 0.479 | 0.521 | 4.38M |
| 5 | **CNN+Transformer Hybrid** | **Hybrid** | **54.24%** | **59.49%** | **0.553** | **0.567** | **0.595** | **0.41M** |

> **Best classifier:** CNN+Transformer Hybrid — highest accuracy AND lowest parameter count among accurate models (10× fewer params than EfficientNetB0 for better performance).

### Segmentation (3 models, binary lesion masks)

| # | Model | Type | Dice ↑ | IoU ↑ | Pixel Acc | Sensitivity | Specificity |
|---|-------|------|:------:|:-----:|:---------:|:-----------:|:-----------:|
| 6 | U-Net | Encoder-Decoder | 0.9357 | 0.8832 | 0.9061 | 0.9651 | 0.7089 |
| 7 | **TransUNet** | **CNN + Transformer** | **0.9444** | **0.8977** | **0.9197** | 0.9320 | **0.8867** |
| 8 | Instance Seg | Multi-task (Clf + Seg) | 0.9282 | 0.8702 | 0.8940 | 0.9538 | 0.7071 |

> **All three segmentation models outperform published benchmarks** (Dice 80–87% \[Attention U-Net: 86.4%\]) on ISIC 2018.

### Training Efficiency

| Model | s/epoch | Total (min) | Accuracy |
|-------|:-------:|:-----------:|:--------:|
| ViT Classifier | 22.4 | 5.6 | 29.66% |
| Custom CNN | 34.3 | 7.4 | 44.07% |
| MobileNetV2 | 19.7 | 8.2 | 40.68% |
| EfficientNetB0 | 22.5 | 9.4 | 52.54% |
| CNN+Transformer Hybrid | 94.0 | 23.5 | **54.24%** |

*Measured on Apple M-series chip, 18GB unified memory, TensorFlow 2.16.*

---

## 📊 Dataset — ISIC 2018 (9-class variant)

- **Source:** [ISIC Challenge 2018](https://challenge.isic-archive.com/) via Kaggle ([Skin Cancer: ISIC](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000))
- **Total images:** ~4,287 dermoscopic images
- **Segmentation masks:** Binary lesion boundary masks
- **Split:** Train 70% (2,357) | Val 15% | Test 15% (stratified)
- **Class balancing:** Minority classes oversampled to 462 images each on training set *only*

| Code | Disease | Count | % |
|------|---------|------:|--:|
| NV | Melanocytic Nevus | 1,341 | 31.3% |
| SBK | Seborrheic Keratosis | 767 | 17.9% |
| BCC | Basal Cell Carcinoma | 514 | 12.0% |
| PBK | Pigmented Benign Keratosis | 462 | 10.8% |
| MEL | Melanoma | 438 | 10.2% |
| AKIEC | Actinic Keratosis | 327 | 7.6% |
| SCC | Squamous Cell Carcinoma | 181 | 4.2% |
| VASC | Vascular Lesion | 142 | 3.3% |
| DF | Dermatofibroma | 115 | 2.7% |

> ⚠️ Severe class imbalance: NV (31%) vs. DF (2.7%) — a 12× imbalance ratio.

---

## 🏗️ Model Architectures

### Classification Models

**Model 1 — Custom CNN (Baseline)**
- 3 × Conv blocks: `Conv2D(3×3) → BatchNorm → ReLU → MaxPool`
- Filter sizes: 32 → 64 → 128
- Head: `GAP → Dense(256) → Dropout(0.4) → Softmax(9)`
- **0.13M parameters** — trained from scratch

**Model 2 — EfficientNetB0 (Transfer Learning)**
- ImageNet pre-trained; `Rescaling(255.0)` layer for pixel range correction
- Two-phase: feature extraction (frozen base) → fine-tuning (top 20 layers)
- **4.38M parameters**

**Model 3 — MobileNetV2 (Transfer Learning)**
- Inverted residuals with linear bottlenecks, depthwise separable convolutions
- Same two-phase strategy; accepts `[0,1]` inputs natively
- **2.59M parameters** — best deployment trade-off among transfer models

**Model 4 — Vision Transformer (ViT)**
- Image split into 16×16 patches → 128-dim embeddings + positional encoding
- 4 Transformer encoder blocks (4 heads), [CLS] token → MLP → Softmax(9)
- **Trained from scratch** (no JFT pre-training) → confirms data-hunger of pure ViT
- **0.69M parameters**

**Model 5 — CNN+Transformer Hybrid** ⭐ *Best classifier*
- CNN feature extractor (3 conv blocks) → reshape to sequence → positional encoding
- 2 Transformer encoder blocks → GAP → MLP → Softmax(9)
- Combines CNN local features with Transformer global attention
- **0.41M parameters** — 10× more efficient than EfficientNetB0 at higher accuracy

### Segmentation Models

**Model 6 — U-Net (Baseline Segmentation)**
- 4-level encoder-decoder: 64 → 128 → 256 → 512 filters
- 1024-filter bottleneck, skip connections, sigmoid output
- Loss: BCE + Dice
- Standard architecture; limited receptive field (no global attention)

**Model 7 — TransUNet** ⭐ *Best segmentation*
- CNN encoder (3 levels) → **Transformer bottleneck** (4 blocks, 4 heads, 256-dim)
- CNN decoder with skip connections
- Global self-attention at bottleneck captures long-range boundary context
- Loss: BCE + Dice

**Model 8 — Instance Segmentation (Multi-task)** 
- Shared CNN backbone with **dual output heads**:
  - Classification branch: GAP → Dense → Softmax(9)
  - Segmentation branch: FPN-style decoder → sigmoid mask
- Multi-task loss: `0.3 × classification_loss + 0.7 × segmentation_loss`
- Enables concurrent disease classification AND lesion boundary delineation

---

## 📁 Project Structure

```
BIS539_Final_Skin_Lesion/
├── config.py                          # All hyperparameters and path constants
├── main.py                            # One-command full pipeline runner
├── download_dataset.py                # ISIC 2018 Kaggle dataset downloader
├── requirements.txt                   # Python dependencies
├── BIS539_Final_Report.tex            # Full LaTeX report (IEEE two-column format)
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py               # ISIC data loading, splits, oversampling, augmentation
│   ├── classification_models.py       # CNN, EfficientNetB0, MobileNetV2, ViT, Hybrid
│   ├── segmentation_models.py         # U-Net, TransUNet, Instance Segmentation
│   ├── train.py                       # Training pipeline for all models
│   ├── evaluate.py                    # Test-set evaluation, metrics, reports
│   └── visualize.py                   # Training curves, confusion matrices, comparison plots
│
├── train_improved.py                  # v2 training: cosine annealing, TTA, oversampling
├── train_segmentation.py              # Segmentation-specific training pipeline
├── run_eval.py                        # Standalone evaluation runner
├── run_transunet_inst.py              # TransUNet + Instance Seg combined runner
├── run_instance_only.py               # Instance segmentation standalone runner
├── generate_plots.py                  # Generate classification comparison figures
├── generate_extra_figures.py          # Per-class metrics, recall heatmap, complexity plot
├── generate_classification_examples.py # Correct/incorrect example images
│
├── data/                              # Dataset (downloaded at runtime — not tracked)
│   └── ISIC2018/
│       ├── images/                    # Dermoscopic images (.jpg)
│       ├── masks/                     # Binary segmentation masks
│       └── labels.csv
│
└── results/                           # Generated outputs (not tracked)
    ├── models/                        # Saved .keras weights
    ├── logs/                          # Training histories, metrics JSON/CSV
    └── figures/                       # Generated PNG plots
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/muhannadsalkini/BIS539-skin-lesion-analysis.git
cd BIS539-skin-lesion-analysis
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API
```bash
# Go to kaggle.com → Account → Create API Token → download kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Download the dataset
```bash
python3 download_dataset.py
```
Downloads the ISIC 2018 skin lesion dataset (~1.5GB) into `data/ISIC2018/`.

---

## 🚀 Running the Project

### Option A — Full pipeline (recommended)
```bash
python3 main.py
```
Runs: download → train all models → evaluate → generate figures.

### Option B — Classification only (v2, improved)
```bash
python3 train_improved.py
```
Trains all 5 classification models with:
- Minority class oversampling (applied on training set only)
- Cosine annealing LR schedule
- Test-Time Augmentation (5 TTA passes)
- Strong augmentation (shear, channel shift, reflect-fill)

### Option C — Segmentation only
```bash
python3 train_segmentation.py
```
Trains U-Net and TransUNet.

### Option D — Instance segmentation
```bash
python3 run_instance_only.py
```

### Option E — TransUNet + Instance Seg combined
```bash
python3 run_transunet_inst.py
```

### Option F — Evaluate all trained models
```bash
python3 run_eval.py
```

### Step by step via main.py
```bash
python3 main.py --stage download
python3 main.py --stage train_clf
python3 main.py --stage train_clf --model cnn_transformer_hybrid
python3 main.py --stage train_seg --model transunet
python3 main.py --stage evaluate
python3 main.py --stage visualize
```

### Generate figures
```bash
python3 generate_plots.py                    # Classification comparison bar charts
python3 generate_extra_figures.py            # Per-class metrics, complexity plot, recall heatmap
python3 generate_classification_examples.py  # Correct/incorrect example grids
```

---

## 🔧 Configuration (`config.py`)

Key parameters:

```python
# Dataset
IMAGE_SIZE          = (224, 224)   # Classification input size
SEG_IMAGE_SIZE      = (256, 256)   # Segmentation input size
BATCH_SIZE          = 16
NUM_CLASSES         = 9

# Training
LEARNING_RATE       = 5e-4         # Initial LR (all models)
SEG_LEARNING_RATE   = 1e-3         # Segmentation LR
EPOCHS              = 60           # Classification epochs (early stopping)
SEG_EPOCHS          = 30           # Segmentation epochs
PATIENCE            = 7            # Early stopping patience
DROPOUT_RATE        = 0.4

# Transfer learning
FINE_TUNE_LAYERS    = 20           # Top N layers to unfreeze (Phase 2)
FINE_TUNE_LR        = 1e-4

# ViT
PATCH_SIZE          = 16
EMBED_DIM           = 128
NUM_HEADS           = 4
NUM_TRANSFORMER_BLOCKS = 4

# Oversampling
OVERSAMPLE_TO       = 462          # Balance all classes to this count
```

---

## 📈 Key Findings

### 1. Transfer Learning vs. Scratch Training
Transfer learning (EfficientNetB0: 52.54%) outperforms scratch training (Custom CNN: 44.07%) by ~8.5 percentage points, confirming that ImageNet features transfer effectively to dermoscopy.

### 2. Transformer Attention Advantage
The CNN+Transformer Hybrid (54.24%) outperforms all other classifiers — including EfficientNetB0 — with **10× fewer parameters** (0.41M vs. 4.38M). The balanced accuracy gap is even larger: 59.49% vs. 52.08% (+7.4 pp), confirming better handling of class imbalance.

### 3. Why ViT Underperforms
ViT trained from scratch (29.66%) is the weakest model, confirming its data-hungry nature. Without large-scale pre-training (JFT-300M), ViT cannot learn effective representations from 2,357 training images.

### 4. TransUNet Beats U-Net
TransUNet (Dice=0.9444) outperforms U-Net (Dice=0.9357) by +0.87% Dice and +1.45% IoU. The Transformer bottleneck's global self-attention provides richer boundary context that pure convolutional encoders miss.

### 5. Instance Segmentation Value
The Instance Segmentation model achieves Dice=0.9282 while **simultaneously** predicting lesion class — demonstrating multi-task learning at a modest Dice cost compared to TransUNet.

### 6. Critical Clinical Failure: MEL Recall = 0.00
**All classification models achieve 0% recall on Melanoma** — every MEL test image is misclassified as Nevus. This is the most critical finding: the models are **not clinically viable** as standalone diagnostic tools. Root cause: MEL and NV share similar visual features; the model defaults to NV due to class dominance (NV: 31% vs MEL: 10%) despite oversampling.

---

## 📚 Literature Review Summary

The report includes a **10-paper literature review** with detailed per-paper analysis (problem, method, results, strengths, limitations, comparison with our work):

| Paper | Architecture | Task | Best Result |
|-------|-------------|------|------------|
| Esteva et al., 2017 | InceptionV3 (129K clinical imgs) | Binary classification | AUC=0.96 |
| Ronneberger et al., 2015 | U-Net | Biomedical segmentation | ISBI 2015 winner |
| Chen et al., 2021 | TransUNet | Multi-organ segmentation | Dice=77.5% |
| Dosovitskiy et al., 2021 | ViT-L/16 (JFT-300M) | ImageNet classification | 88.55% top-1 |
| Codella et al., 2019 | Ensemble (40+) | ISIC 2018 challenge | Acc=88.5%, IoU=80.2% |
| Vesal et al., 2018 | SkinNet (DenseNet121) | ISIC 2017 segmentation | Dice=84.9% |
| Abraham & Khan, 2019 | Attention U-Net | ISIC 2018 segmentation | Dice=86.4% |
| Tan & Le, 2019 | EfficientNetB0 | ImageNet | 77.3% (5.3M params) |
| Sandler et al., 2018 | MobileNetV2 | ImageNet | 72.0% (3.4M params) |
| Kassem et al., 2024 | Systematic review (150+ papers) | Survey | Single model: 82–92% |

> **Our segmentation models (93–94% Dice) exceed all published ISIC segmentation benchmarks.**

---

## 📄 Report

The full academic report is in **`BIS539_Final_Report.tex`** (IEEE two-column format, ~812 lines):

```bash
pdflatex BIS539_Final_Report.tex
pdflatex BIS539_Final_Report.tex   # run twice for references
```

Or upload to [Overleaf](https://www.overleaf.com).

**Report sections:**
1. Introduction & problem definition
2. Literature review (10 papers, detailed analysis)
3. Dataset (ISIC 2018, 9-class)
4. Methodology (research hypotheses, all 8 model architectures, training config)
5. Experimental results & discussion
6. Threats to validity
7. Conclusion & future work
8. References (15 citations, IEEE format)

---

## ⚠️ Ethical Considerations

- **MEL recall = 0.00**: System misses 100% of melanomas — **must only be used as decision-support with mandatory clinician oversight**
- **Dataset bias**: ISIC 2018 underrepresents darker skin tones (Fitzpatrick IV–VI); fairness analysis required before clinical deployment
- **Privacy**: Dermoscopic images are biometric data — GDPR and HIPAA compliance required
- **Transparency**: Known limitations must be disclosed to clinicians and patients

---

## 📦 Dependencies

```
tensorflow>=2.16
keras>=3.0
numpy
pandas
scikit-learn
matplotlib
seaborn
kaggle
Pillow
tqdm
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Work

1. **Pre-trained ViT** (DeiT, Swin Transformer) — fix ViT's data-efficiency gap
2. **Grad-CAM + attention maps** — clinical explainability visualization
3. **Confidence intervals** — mean ± std over 5 runs for statistical validity
4. **External validation** — HAM10000, ISIC 2019, BCN20000, smartphone images
5. **Ablation study** — without oversampling / without augmentation / TL-only comparison
6. **Federated learning** — privacy-preserving multi-institutional training
7. **Model compression** — quantization + knowledge distillation for smartphone deployment
8. **Focal loss / class-weighted training** — explicitly target MEL recall = 0.00

---

## 📜 License

This project is for academic purposes (BIS539 Final Project, Biruni University, 2025–2026).  
The ISIC 2018 dataset is available under CC-BY-NC license from the [ISIC Archive](https://www.isic-archive.com/).
