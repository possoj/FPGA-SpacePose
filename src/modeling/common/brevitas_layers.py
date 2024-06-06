"""
Copyright (c) 2024 Julien Posso
"""

import torch.nn as nn
import brevitas.nn as qnn
from src.modeling.common.quantizers import *


class QConvBnAct(nn.Sequential):
    """A Pytorch Sequential container that allows for easy creation of a quantized Conv-BN-ReLU block using Brevitas
    with configurable bit-widths"""
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        batchnorm=True,
        activation=True,
        weight_quant=True,
        weight_bit_width=8,
        return_conv_quant=False,
        act_quant=True,
        act_bit_width=8,
        return_act_quant=False,
        act_scaling_per_channel=False,
    ):

        weight_quantizer = select_quantizer(weight_quant, weight_bit_width, layer='convolution')
        act_quantizer = select_quantizer(act_quant, act_bit_width, layer='activation')

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers = [qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False,
                                  weight_quant=weight_quantizer, weight_bit_width=weight_bit_width,
                                  return_quant_tensor=return_conv_quant)]

        if batchnorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation:
            layers.append(qnn.QuantReLU(act_quant=act_quantizer, bit_width=act_bit_width,
                                        return_quant_tensor=return_act_quant,
                                        per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                        scaling_stats_permute_dims=(1, 0, 2, 3),
                                        scaling_per_output_channel=act_scaling_per_channel
                                        ))

        super().__init__(*layers)


class QInvertedResidual(nn.Module):
    """A module for creating quantized inverted residual blocks (the building blocks of MobileNet-V2 backbone among
    others) with PyTorch and Brevitas"""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        batchnorm=True,
        weight_quant=True,
        act_quant=True,
        bit_width=None,
        shared_act_bw=8,
        return_quant=False,
        use_residual=False,
        input_quant=False,
    ):

        super().__init__()

        if weight_quant or act_quant:
            assert bit_width is not None, "bit_width must be specified"
        assert stride in [1, 2], "Only stride of 1 and 2 supported"

        self.use_residual = use_residual
        self.input_quant = input_quant

        hidden_channels = int(round(in_channels * expand_ratio))

        # convolutions/activations bit-widths
        (c1_w_bw, c1_a_bw), (c2_w_bw, c2_a_bw), (c3_w_bw,) = bit_width

        layers = []

        if expand_ratio != 1:
            layers.append(
                # 1x1 conv: expansion layer
                QConvBnAct(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, batchnorm=batchnorm,
                           weight_quant=weight_quant, weight_bit_width=c1_w_bw,
                           act_quant=act_quant, act_bit_width=c1_a_bw,
                           return_conv_quant=False, return_act_quant=return_quant,
                           act_scaling_per_channel=False)
            )

        layers.extend([
            # 3x3 depthwise conv
            QConvBnAct(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                       batchnorm=batchnorm, stride=stride, groups=hidden_channels,
                       weight_quant=weight_quant, weight_bit_width=c2_w_bw,
                       act_quant=act_quant, act_bit_width=c2_a_bw,
                       return_conv_quant=False, return_act_quant=return_quant,
                       act_scaling_per_channel=False),

            # 1x1 linear conv: projection layer
            QConvBnAct(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, batchnorm=batchnorm,
                       weight_quant=weight_quant, weight_bit_width=c3_w_bw,
                       activation=False, return_conv_quant=False)
        ])
        self.conv = nn.Sequential(*layers)

        # Shared activations and input quantization
        if self.input_quant or self.use_residual:
            shared_quantizer = IntActQuant if act_quant else None
            self.quant = qnn.QuantIdentity(shared_quantizer, bit_width=shared_act_bw,
                                           return_quant_tensor=return_quant,
                                           )

    def forward(self, x):
        if self.input_quant or self.use_residual:
            x = self.quant(x)

        residual = x
        x = self.conv(x)

        if self.use_residual:
            x = self.quant(x)
            x = x + residual

        return x
