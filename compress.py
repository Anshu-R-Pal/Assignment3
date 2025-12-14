import sys
drive_path = '/content/drive/MyDrive/Python/Assignment3'
sys.path.append(drive_path)


# compress.py
import copy
import torch
import wandb
from utils import quantize_model_weights, get_model_size  # removed model_size_bytes_fp32

def compression_sweep(model, test_loader, device, weight_bitwidths=(2,3,4,5,6,7,8,12,16,32), act_bits=8):
    """
    Perform compression sweeps over weight bitwidths (and optionally activation bits),
    evaluate accuracy, log results to WandB for parallel coordinates.
    """
    wandb.init(project="Asignment3_mobilenev2_compression", reinit=True)

    results = []

    # Use get_model_size instead of undefined model_size_bytes_fp32
    fp32_size = get_model_size(model, bits=32)

    for w_bits in weight_bitwidths:
        q_model = copy.deepcopy(model).to(device)
        q_model.eval()

        # Quantize weights
        quant_info = quantize_model_weights(q_model, bits=w_bits)

        # Evaluate accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = q_model(images)
                preds = outputs.argmax(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100. * correct / total

        # Compute size and compression ratio
        size_mb = get_model_size(q_model, bits=w_bits)
        compression_ratio = fp32_size / size_mb

        results.append({
            "weight_quant_bits": w_bits,
            "activation_quant_bits": act_bits,
            "compression_ratio": compression_ratio,
            "model_size_mb": size_mb,
            "quantized_acc": acc
        })

        # Log to WandB for Parallel Coordinates
        wandb.log({
            "weight_quant_bits": w_bits,
            "activation_quant_bits": act_bits,
            "compression_ratio": compression_ratio,
            "model_size_mb": size_mb,
            "quantized_acc": acc
        })

        print(f"[{w_bits}-bit] Acc: {acc:.2f}% | Size: {size_mb:.2f} MB | CR: {compression_ratio:.2f}x")

    return results



from google.colab import drive
drive.mount('/content/drive')

