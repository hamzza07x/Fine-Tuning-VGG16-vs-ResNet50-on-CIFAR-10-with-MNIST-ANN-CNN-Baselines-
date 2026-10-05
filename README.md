# Fine-Tuning VGG16 vs ResNet50 on CIFAR-10 (with MNIST ANN/CNN Baselines)


A deep learning notebook that compares **ANNs, CNNs, and ImageNet-pretrained networks** on image classification. The main experiment fine-tunes **VGG16** and **ResNet50** on **CIFAR-10** with an identical two-phase recipe, so the two architectures can be compared fairly.

---

## Table of Contents

1. [Overview](#overview)
2. [What the Notebook Covers](#what-the-notebook-covers)
3. [Datasets](#datasets)
4. [Method](#method)
5. [Model Architectures](#model-architectures)
6. [Results](#results)
7. [Observations](#observations)
8. [Limitations](#limitations)
9. [Getting Started](#getting-started)
10. [Repository Structure](#repository-structure)
11. [Author](#author)

---

## Overview

The notebook has two parts:

- **Part 1: MNIST baselines.** A fully connected ANN, a small CNN, and a frozen InceptionV3 transfer-learning model, to show how architecture choice changes accuracy and compute cost on the same data.
- **Part 2: CIFAR-10 fine-tuning.** VGG16 and ResNet50 are each trained in two phases: first only a new classifier head on a frozen backbone, then the last convolutional block is unfrozen and trained at a low learning rate.

The goal of Part 2 is to measure **how much fine-tuning actually helps, and how the two backbones differ in accuracy, overfitting, and training cost.**

---

## What the Notebook Covers

| Section | Model | Dataset | Technique |
|---|---|---|---|
| Part 1 | ANN (3000-1000-10 dense) | MNIST (5,000 train subset) | Plain fully connected network |
| Part 1 | CNN (2 conv + 2 pooling blocks) | MNIST (5,000 train subset) | Convolution + max pooling |
| Part 1 | InceptionV3 | MNIST, converted to 3-channel 224×224 | Frozen backbone + data augmentation |
| Part 2 | VGG16 | CIFAR-10 (10,000 train subset) | Two-phase fine-tuning |
| Part 2 | ResNet50 | CIFAR-10 (10,000 train subset) | Two-phase fine-tuning |

---

## Datasets

**MNIST**: 28×28 grayscale handwritten digits, 10 classes. A 5,000-image training subset and a 1,000-image test subset are used.

**CIFAR-10**: 32×32 RGB images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Split used:

| Split | Images | Source |
|---|---|---|
| Train | 10,000 | First 10,000 of the official training set |
| Validation | 2,000 | Next 2,000 of the official training set |
| Test | 2,000 | First 2,000 of the official test set |

The validation split is carved out of the training set so that the test set is never used for tuning decisions.

---

## Method

### Input pipeline (CIFAR-10)

- Images are **resized on the fly to 128×128** inside a `tf.data` pipeline, instead of resizing the full dataset up front. This keeps memory usage flat.
- Each model uses its **own `preprocess_input`** (`vgg16` and `resnet50` scale inputs differently), so pixel values are kept in the 0–255 range until that function is applied.
- Only the training set is shuffled; validation and test sets keep their order so predictions align with labels.

### Shared model recipe

Both backbones use the same head, so the comparison isolates the backbone:

```
Input (128×128×3)
  → RandomFlip (horizontal)
  → Pretrained backbone (ImageNet weights, include_top=False)
  → GlobalAveragePooling2D
  → Dense(256, ReLU)
  → Dropout(0.5)
  → Dense(10, Softmax)
```

Loss: `sparse_categorical_crossentropy` · Optimizer: Adam · Batch size: 64

### Two-phase training

| Phase | What is trainable | Learning rate | Epochs |
|---|---|---|---|
| 1. Feature extraction | New classifier head only | 1e-3 | 5 |
| 2. Fine-tuning | Head + last block of backbone | 1e-5 | 5 |

- **Why two phases:** the randomly initialized head would otherwise send large, noisy gradients into the pretrained layers.
- **Why a low learning rate in phase 2:** a high rate would overwrite useful pretrained features.
- **Layers unfrozen in phase 2:**
  - VGG16 from `block5_conv1` onward (the last conv block)
  - ResNet50 from `conv5_block1_1_conv` onward (the last residual stage)
- **BatchNorm layers stay frozen** during fine-tuning. Updating BatchNorm statistics with small batches on a small dataset tends to destabilize training, especially in ResNet.

---

## Model Architectures

| Backbone | Parameters (backbone, no top) | Final feature map at 128×128 input |
|---|---|---|
| VGG16 | ~14.7M | 4×4×512 |
| ResNet50 | ~23.6M | 4×4×2048 |

For reference, the MNIST models:

| Model | Parameters |
|---|---|
| ANN | 5,366,010 |
| VGG16 head (224×224, defined but not trained in this notebook) | 14,848,586 total / 133,898 trainable |
| InceptionV3 + head | 22,857,002 total / 1,054,218 trainable |

---

## Results

### Part 2: CIFAR-10 (validation accuracy)

| Model | End of Phase 1 (head only) | End of Phase 2 (fine-tuned) | Gain from fine-tuning | Final train acc |
|---|---|---|---|---|
| VGG16 | 82.60% | 88.05% | +5.45 pts | 93.37% |
| ResNet50 | 87.55% | **89.85%** | +2.30 pts | 97.05% |

Per-epoch validation accuracy:

| Epoch | VGG16 (P1) | VGG16 (P2) | ResNet50 (P1) | ResNet50 (P2) |
|---|---|---|---|---|
| 1 | 76.80% | 85.25% | 85.50% | 89.10% |
| 2 | 80.05% | 85.70% | 87.05% | 88.95% |
| 3 | 82.40% | 86.35% | 86.50% | 89.05% |
| 4 | 83.60% | 86.30% | 88.00% | 89.40% |
| 5 | 82.60% | 88.05% | 87.55% | 89.85% |

Training cost, measured on **CPU** (no GPU was available):

| Model | Approx. time per epoch |
|---|---|
| VGG16 | ~9–13 minutes |
| ResNet50 | ~3–4 minutes |

### Part 1: MNIST baselines

| Model | Training | Result |
|---|---|---|
| ANN | 3 epochs, final train acc 96.76% | 94% test accuracy (macro F1 0.93) on 1,000 test images |
| CNN | 3 epochs, final train acc 97.48% | 97.10% test accuracy, loss 0.1003 |
| InceptionV3 (frozen) | 3 epochs with augmentation, train acc 80.50% | 85.90% validation accuracy |

---

## Observations

- **ResNet50 wins on accuracy and on speed.** It ends about 1.8 points higher on validation accuracy than VGG16 and trains roughly 3× faster per epoch on CPU.
- **Fine-tuning helped VGG16 much more than ResNet50** (+5.45 vs +2.30 points). ResNet50's frozen features were already strong after phase 1, leaving less room to gain.
- **ResNet50 is starting to overfit.** Its train accuracy (97.05%) is far above validation (89.85%), and its validation loss rose from 0.358 to 0.394 across phase 2 even while accuracy crept up. VGG16's validation loss was still falling at the last epoch, so it likely had more headroom with additional epochs.
- **On MNIST, the simple CNN beat everything else.** It reached 97.10% test accuracy with a tiny parameter count. The frozen InceptionV3 scored lowest (85.90%) after 3 epochs, because ImageNet features on upscaled grayscale digits are a poor fit, and the frozen backbone cannot adapt. Bigger and pretrained does not automatically mean better.

---

## Limitations

Read these before drawing conclusions from the numbers:

- **Subsets, not the full datasets.** CIFAR-10 uses 10,000 of 50,000 training images. Absolute accuracies would be higher on the full set.
- **128×128 inputs.** CIFAR-10 images are natively 32×32, and upscaling does not add information. Pretrained networks usually do better at their native 224×224 resolution, but that is far more expensive on CPU.
- **Single run, no seeds fixed beyond the shuffle.** A 1–2 point difference between models is within what run-to-run variance could produce, so the VGG16 vs ResNet50 gap should be treated as indicative, not definitive.
- **Short training.** Five epochs per phase is not a convergence study.
- **Validation accuracy is the headline metric.** Evaluate on the held-out test set (the final evaluation cells) before quoting final performance.
- **CPU-only training** is why the epoch times are long. Timings will differ on a GPU.

---

## Getting Started

### Requirements

- Python 3.10+ (developed on 3.12)
- TensorFlow 2.16+ with Keras 3
- NumPy, Matplotlib, scikit-learn, Jupyter

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install tensorflow numpy matplotlib scikit-learn jupyter
```

### Run

```bash
jupyter notebook p230577_CP3_DLP_7E.ipynb
```

Run the cells top to bottom. MNIST, CIFAR-10, and the ImageNet pretrained weights are downloaded automatically on first use.

### Tips

- A **GPU is strongly recommended**. On CPU, the full CIFAR-10 section takes several hours.
- To speed things up, lower `imgSize` (for example to 96) or `trainSize` in the CIFAR-10 setup cell.
- If memory is tight, reduce `batchSize`.

---

## Repository Structure

```
.
├── p230577_CP3_DLP_7E.ipynb   # Full notebook: MNIST baselines + CIFAR-10 fine-tuning
└── README.md
```

---

## Author

**Muhammad Hamza**
Computer Science, FAST NUCES
