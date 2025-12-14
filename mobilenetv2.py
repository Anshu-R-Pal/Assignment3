

  # mobilenet.py
# MobileNetV2 adapted for CIFAR-10 (32x32)

import torch
from torch import nn
from typing import List, Optional, Callable


# -----------------------------------------------------------
# Basic building blocks
# -----------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        norm_layer=None,
        activation_layer=None,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []

        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
            )

        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer,
            )
        )

        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(norm_layer(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------
def _make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor * divisor)


# -----------------------------------------------------------
# MobileNetV2
# -----------------------------------------------------------
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=10,
        width_mult=1.0,
        dropout=0.2,
    ):
        super().__init__()

        block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * width_mult)

        features: List[nn.Module] = []

        # First conv (stride=1 for CIFAR-10)
        features.append(
            ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)
        )

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        features.append(
            ConvBNReLU(input_channel, last_channel, kernel_size=1,
                       norm_layer=norm_layer)
        )

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# -----------------------------------------------------------
# Factory function (IMPORTANT)
# -----------------------------------------------------------
def get_mobilenet(num_classes=10, width_mult=1.0, dropout=0.2):
    """
    ✅ NO register_hooks parameter
    ✅ Hooks are added externally (Q1c)
    """
    return MobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
    )


