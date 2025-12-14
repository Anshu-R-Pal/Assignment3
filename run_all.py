

# run_all.py
import torch
import wandb

from dataloader import get_cifar10_loaders
from mobilenetv2 import get_mobilenet
from train import train_model, plot_curves
from eval import evaluate_model
from utils import (
   register_activation_hooks,
   activation_memory,
   get_model_size
)
from compress import compression_sweep


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Hyperparameters (Q1b)
# ------------------------------------------------------------
batch_size = 128
epochs = 50
learning_rate = 0.001

wandb.login()

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
train_loader, test_loader = get_cifar10_loaders(
    batch_size=batch_size,
    seed=42
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
model = get_mobilenet(num_classes=10).to(device)

# Q1(c): activation hooks
register_activation_hooks(model)

# Q1(b): print training configuration
print("\n=== Q1(b): Training Configuration ===")
print("Optimizer: SGD")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print("====================================\n")

# ------------------------------------------------------------
# Train baseline
# ------------------------------------------------------------
train_loss, train_acc, test_loss, test_acc = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=epochs,
    batch_size=batch_size,
    lr=learning_rate,
    device=device
)

plot_curves(train_loss, test_loss, train_acc, test_acc)

baseline_acc = evaluate_model(model, test_loader, device)
print(f"\nBaseline Accuracy (FP32): {baseline_acc:.4f}")

# ------------------------------------------------------------
# Q1(c): Activation memory
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    x, _ = next(iter(test_loader))
    model(x.to(device))

baseline_activation_mb = sum(
    v["num_elements"] * 32 / 8 / (1024 ** 2)
    for v in activation_memory.values()
)

print(f"Baseline Activation Memory: {baseline_activation_mb:.4f} MB")

# ------------------------------------------------------------
# Q2: Model size (FP32)
# ------------------------------------------------------------
fp32_size = get_model_size(model, bits=32)
print(f"FP32 Model Size: {fp32_size:.4f} MB")

# ------------------------------------------------------------
# Q3: Compression sweep (≥10 WandB runs)
# ------------------------------------------------------------
print("\n=== Q3: Compression Sweep ===")

weight_bits_list = (2, 3, 4, 5, 6, 7, 8, 12, 16, 32)
all_results = []

for w_bits in weight_bits_list:
    print(f"\n--- Running compression for {w_bits}-bit weights ---")

    results = compression_sweep(
        model=model,
        test_loader=test_loader,
        device=device,
        weight_bitwidths=(w_bits,),   # ONE BIT PER RUN
        act_bits=8
    )

    all_results.extend(results)

# ------------------------------------------------------------
# Best Accuracy–Compression Tradeoff
# ------------------------------------------------------------
best = max(
    all_results,
    key=lambda x: x["quantized_acc"] / x["model_size_mb"]
)

print("\n=== Best Accuracy–Compression Tradeoff ===")
print(
    f"W{best['weight_quant_bits']} | "
    f"Acc: {best['quantized_acc']:.2f}% | "
    f"Size: {best['model_size_mb']:.2f} MB | "
    f"CR: {best['compression_ratio']:.2f}x"
)



