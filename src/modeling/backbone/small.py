"""
Copyright (c) 2024 Julien Posso
"""

import torch.nn as nn
import brevitas.nn as qnn
from src.modeling.common.brevitas_layers import *


class QSmallBackbone(nn.Module):
    """Small backbone ideal to understand and debug the FINN build flow"""
    def __init__(self, in_channels=3, out_channels=32, batchnorm=True, residual_connections=True, quantization=True,
                 bit_width=None, last_return_quant=False):
        super().__init__()

        if bit_width is None:
            bit_width = {
                "image": 8,
                "conv1": (4, 4),
                "conv2": (4, 4),
                "shared_act": 4,
                "inverted_residual": [
                    [(4, 4), (4, 4), (4,)]
                ]
            }

        self.input_layer = qnn.QuantIdentity(bit_width=bit_width["image"], return_quant_tensor=True)

        activ = False if residual_connections else True
        self.conv1 = QConvBnAct(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0,
            batchnorm=batchnorm, activation=activ, return_conv_quant=False, return_act_quant=quantization,
            weight_bit_width=bit_width["conv1"][0], act_bit_width=bit_width["conv1"][1]
        )
        self.bloc = QInvertedResidual(
            in_channels=out_channels, out_channels=out_channels, stride=1, expand_ratio=6, batchnorm=batchnorm,
            return_quant=quantization, use_residual=residual_connections,
            bit_width=bit_width["inverted_residual"][0], shared_act_bw=bit_width["shared_act"]
        )
        self.quant = qnn.QuantIdentity(bit_width=bit_width["shared_act"], return_quant_tensor=quantization)
        self.conv2 = QConvBnAct(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0,
            batchnorm=batchnorm, activation=True, return_conv_quant=False, return_act_quant=last_return_quant,
            weight_bit_width=bit_width["conv2"][0], act_bit_width=bit_width["conv2"][1]
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.conv1(x)
        x = self.bloc(x)
        x = self.quant(x)
        x = self.conv2(x)
        return x
