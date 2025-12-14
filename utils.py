


# utils.py
import torch
import torch.nn as nn

# ============================================================
# Q1(c): Activation memory (Conv layers only)
# ============================================================
activation_memory = {}

def activation_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_memory[name] = {
                "num_elements": output.numel(),
                "shape": tuple(output.shape)
            }
    return hook


def register_activation_hooks(model):
    """
    Register hooks ONLY on Conv2d layers (as expected in Q1c).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(activation_hook(name))


# ============================================================
# Professor-style n-bit Post-Training Quantization (PTQ)
# ============================================================
def qparams_from_minmax(xmin, xmax, n_bits=8, eps=1e-12):
    """
    Symmetric per-tensor quantization parameters.
    """
    qmax = (1 << (n_bits - 1)) - 1
    max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
    scale = max_abs / qmax
    return scale, qmax


def quantize_tensor(x, scale, qmax):
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax, qmax)
    return q * scale


def quantize_model_weights(model, bits=8):
    """
    Post-training fake quantization of weights.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                xmin, xmax = param.min(), param.max()
                scale, qmax = qparams_from_minmax(xmin, xmax, bits)
                param.copy_(quantize_tensor(param, scale, qmax))


# ============================================================
# Model size (analytical — required for Q2/Q3)
# ============================================================
def get_model_size(model, bits=32):
    """
    Analytical model size in MB.
    Weights use 'bits', biases remain FP32.
    """
    total_bits = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            total_bits += p.numel() * bits
        elif "bias" in name:
            total_bits += p.numel() * 32
    return total_bits / 8 / (1024 ** 2)


