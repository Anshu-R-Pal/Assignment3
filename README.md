# CS6886W – Assignment 3: MobileNet‑V2 Compression on CIFAR‑10

**Course:** System Engineering for Deep Learning (CS6886W)
**Institute:** IIT Madras
**Assignment:** 3 – Model Compression

This repository contains a complete, reproducible implementation for **training MobileNet‑V2 on CIFAR‑10** and applying **post‑training quantization–based compression** without using any external compression libraries.

---

## 1. Environment Setup

### Python Version

```bash
Python >= 3.9
```

### Required Packages

```bash
pip install torch torchvision wandb matplotlib numpy
```

Recommended (CUDA system):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Repository Structure

```
.
├── dataloader.py              # CIFAR‑10 dataloaders with augmentation
├── mobilenet.py               # MobileNet‑V2 adapted for CIFAR‑10 (32×32)
├── train.py                   # Training loop + WandB logging
├── eval.py                    # Evaluation (Top‑1 accuracy)
├── utils.py                   # Quantization + activation hooks + size analysis
├── compress.py                # Compression sweep (Q3)
├── run_all.py                 # End‑to‑end pipeline
├── models/
│   └── mobilenet_custom.py    # Wrapper (optional)
└── README.md
```

---

## 3. Baseline Training (Question 1)

### CIFAR‑10 Preprocessing

* **Training:** RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
* **Testing:** Normalize only

Normalization:

```text
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]
```

### Model Configuration

* Architecture: **MobileNet‑V2 (CIFAR‑10 adapted)**
* Width multiplier: `1.0`
* Dropout: `0.2`
* BatchNorm: enabled

### Training Strategy

* Optimizer: **SGD**
* Momentum: `0.9`
* Weight decay: `5e‑4`
* Learning rate: `0.001`
* Scheduler: StepLR (step=30, gamma=0.1)
* Epochs: `50`
* Batch size: `128`

### Run Baseline Training

```bash
python run_all.py
```

This logs:

* Training / test loss curves
* Training / test accuracy curves
* Final FP32 baseline accuracy

---

## 4. Activation Memory Measurement (Q1c)

Activation memory is measured using **forward hooks** registered only on `Conv2d` layers.

Metric:

```text
Activation memory (MB) = num_elements × bits / 8 / 1024²
```

* FP32 activations assumed (32 bits)
* Single forward pass on one test batch

---

## 5. Compression Method (Question 2)

### Technique

* Symmetric, per‑tensor, uniform quantization
* No PyTorch quantization APIs used

### What is Quantized

* All **weight tensors**
* Biases remain FP32
* Activations analyzed analytically (not quantized during inference)

### Quantization Formula

```text
scale = max(|xmin|, |xmax|) / (2^(bits‑1) − 1)
q = clamp(round(x / scale)) × scale
```

---

## 6. Compression Sweep & WandB Logging (Question 3)

### Sweep Configuration

* Weight bit‑widths:

```python
(2, 3, 4, 5, 6, 7, 8, 12, 16, 32)
```

* Activation bits (analytical): `8`
* Total runs: **10 (meets assignment requirement)**

Each bit‑width is logged as a **separate WandB run**.

### Run Compression Sweep Only

```bash
python compress.py
```

(Usually triggered automatically from `run_all.py`.)

---

## 7. Parallel Coordinates Plot (Required Output)

In WandB UI:

1. Open project: `Asignment3_mobilenev2_compression`
2. Go to **Charts → Parallel Coordinates**
3. Select axes:

   * `weight_quant_bits`
   * `activation_quant_bits`
   * `quantized_acc`
   * `model_size_mb`
   * `compression_ratio`

This plot answers **Question 3(b)**.

---

## 8. Compression Analysis (Question 4)

Reported automatically:

* **Model compression ratio**
  ( CR = \frac{\text{FP32 model size}}{\text{quantized model size}} )

* **Weight compression ratio**
  Based on analytical bit‑count of parameters

* **Activation compression**
  Measured using forward hooks (Conv layers only)

* **Final model size (MB)**
  Includes weights + FP32 biases

Best trade‑off example:

```text
5‑bit weights → ~6.15× compression with ~73.5% accuracy
```

---

## 9. Reproducibility

* Fixed random seed (`seed=42`)
* Deterministic dataloaders
* All hyperparameters logged in WandB
* Single command reproduction

```bash
python run_all.py
```

---

## 10. Notes

* No compression libraries or APIs were used
* All quantization logic implemented from scratch
* MobileNet‑V2 adapted explicitly for CIFAR‑10 resolution

---

## 11. WandB Projects

* **Training:** `Assignment3_training`
* **Compression:** `Asignment3_mobilenev2_compression`

---

## Author

**Anshu Pal**
CS6886W – IIT Madras

