"""
Copyright (c) 2024 Julien Posso
"""
import torch
import torch.nn as nn
import brevitas.nn as qnn
from src.modeling.common.quantizers import *


class URSONetHead(nn.Module):
    """A PyTorch module that defines a simple FP32 pose estimation head with two branches, one for position and the
    other for orientation."""
    def __init__(self, n_feature_maps=1280, n_ori_outputs=512, n_pos_outputs=3, bias=True, dropout_rate=0.2):
        super().__init__()

        # Position branch
        self.pos = nn.Sequential(
            nn.Linear(in_features=n_feature_maps, out_features=n_pos_outputs, bias=bias)
        )

        # Orientation branch
        self.ori = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=n_feature_maps, out_features=n_ori_outputs, bias=bias)
        )

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(x.size(0), x.size(1))
        ori = self.ori(x)
        pos = self.pos(x)
        return ori, pos


class QURSONetHead(nn.Module):
    """A PyTorch module that defines a simple quantized pose estimation head with Brevitas with two branches,
    one for position and the other for orientation."""
    def __init__(self, n_feature_maps=1280, n_ori_outputs=512, n_pos_outputs=3, bias=True, dropout_rate=0.2,
                 pool_kernel=(8, 12), quantization=True, bit_width=None):
        super().__init__()

        if bit_width is None:
            bit_width = {
                'fully_connected': (8, 8),
                'shared_act': 8,
                'pooling': 8,
            }

        return_quant = quantization
        weight_quant = quantization
        bias_quant = quantization
        pool_quant = quantization

        # Quantizers
        bias_quantizer = BiasQuant if bias_quant else None
        pool_quantizer = PoolQuant if pool_quant else None
        weight_quantizer = select_quantizer(weight_quant, bit_width=bit_width['fully_connected'][0],
                                            layer='convolution')

        self.pool = qnn.QuantAvgPool2d(kernel_size=pool_kernel, trunc_quant=pool_quantizer, bit_width=bit_width['pooling'],
                                       return_quant_tensor=return_quant)

        # Position branch
        self.pos = nn.Sequential(
            qnn.QuantLinear(in_features=n_feature_maps, out_features=n_pos_outputs, bias=bias,
                            weight_quant=weight_quantizer, weight_bit_width=bit_width['fully_connected'][0],
                            bias_quant=bias_quantizer, bias_bit_width=bit_width['fully_connected'][1],
                            return_quant_tensor=False)
        )

        # Orientation branch
        self.ori = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            qnn.QuantLinear(in_features=n_feature_maps, out_features=n_ori_outputs, bias=bias,
                            weight_quant=weight_quantizer, weight_bit_width=bit_width['fully_connected'][0],
                            bias_quant=bias_quantizer, bias_bit_width=bit_width['fully_connected'][1],
                            return_quant_tensor=False)
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), x.size(1))
        pos = self.pos(x)
        ori = self.ori(x)
        return ori, pos