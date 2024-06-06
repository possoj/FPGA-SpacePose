"""
Copyright (c) 2024 Julien Posso
"""

import torch.nn as nn
import brevitas.nn as qnn
from src.modeling.common.quantizers import *
from src.modeling.common.brevitas_layers import QConvBnAct, QInvertedResidual
from src.modeling.common.pytorch_layers import ConvBnAct, InvertedResidual


class QSmallMobile(nn.Module):
    """ Small MobileNet-V2 backbone with residual layers compatible with the FINN dataflow build"""
    def __init__(self, in_channels=3, out_channels=1280, batchnorm=True, residual_connections=True, quantization=True,
                 bit_width=None, last_return_quant=False, inverted_residual_settings=None):
        super().__init__()

        img_channel = in_channels  # Number of channels in the input image
        input_channel = 32  # Number of channels after the 1st convolution

        if inverted_residual_settings is None:
            inverted_residual_settings = [
                # t, c, n, s
                [6, 32, 1, 1],
                [6, 32, 1, 2],
            ]

        if bit_width is None:
            inverted_residual_bit_width = [
                # [(conv1_weight_bw, conv1_act_bw), (conv2_w_bw, conv2_act_bw), (conv3_w_bw, conv3_act_bw)]
                [(None, None), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
            ]

            bit_width = {
                'image': 8,
                'first_conv': (3, 3),
                'last_conv': (3, 3),
                'shared_act': 4,
                'inverted_residual': inverted_residual_bit_width
            }

        return_quant = quantization
        img_quant = quantization
        weight_quant = quantization
        act_quant = quantization
        shared_act_quant = quantization

        # Quantizers
        img_quantizer = InputQuant if img_quant else None

        # Building input quantization and first convolution layer
        layers = [
            # TODO: difference 1
            # qnn.QuantIdentity(bit_width=bit_width['image'], return_quant_tensor=return_quant),
            # Original:
            qnn.QuantIdentity(act_quant=img_quantizer, bit_width=bit_width['image'],
                              return_quant_tensor=return_quant),

            QConvBnAct(in_channels=img_channel, out_channels=input_channel, stride=2, padding=1, batchnorm=batchnorm,
                       weight_quant=weight_quant, weight_bit_width=bit_width['first_conv'][0],
                       act_quant=act_quant, act_bit_width=bit_width['first_conv'][1],
                       return_act_quant=return_quant)
        ]
        in_ch = input_channel

        # Building all residual layers (by block)
        prev_layer_use_residual = False
        block_number = 0
        for t, c, n, s in inverted_residual_settings:
            for i in range(n):
                stride = s if i == 0 else 1
                use_residual = stride == 1 and in_ch == c and residual_connections
                if residual_connections:
                    input_quant = use_residual or prev_layer_use_residual or (block_number == 1 and i==0)
                else:
                    input_quant = False if block_number == 0 and i == 0 else True
                layers.append(
                    QInvertedResidual(in_channels=in_ch, out_channels=c, stride=stride, expand_ratio=t,
                                      batchnorm=batchnorm, weight_quant=weight_quant, act_quant=act_quant,
                                      bit_width=bit_width['inverted_residual'][block_number],
                                      shared_act_bw=bit_width['shared_act'], return_quant=return_quant,
                                      use_residual=use_residual, input_quant=input_quant)
                )
                in_ch = c
                prev_layer_use_residual = use_residual
                block_number += 1

        # Quantize the output of the QInvertedResidual blocks that has no activation function
        # TODO: diff 2
        # quant = select_quantizer(shared_act_quant, bit_width=bit_width['shared_act'], layer='activation')
        # layers.append(qnn.QuantIdentity(bit_width=bit_width['shared_act'],
        #                                 return_quant_tensor=return_quant))

        # Original
        quant = select_quantizer(shared_act_quant, bit_width=bit_width['shared_act'], layer='activation')
        layers.append(qnn.QuantIdentity(act_quant=quant, bit_width=bit_width['shared_act'],
                                        return_quant_tensor=return_quant))

        # Build last convolutional layer
        layers.append(QConvBnAct(in_ch, out_channels, kernel_size=1, batchnorm=batchnorm,
                                 weight_quant=weight_quant, weight_bit_width=bit_width['last_conv'][0],
                                 act_quant=act_quant, act_bit_width=bit_width['last_conv'][1],
                                 return_act_quant=last_return_quant))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class QMobileNetV2(nn.Module):
    """MobileNet-V2 backbone with residual layers compatible with the FINN dataflow build"""
    def __init__(self, in_channels=3, out_channels=1280, batchnorm=True, residual_connections=True, quantization=True,
                 bit_width=None, last_return_quant=False, inverted_residual_settings=None):
        super().__init__()

        img_channel = in_channels  # Number of channels in the input image
        input_channel = 32  # Number of channels after the 1st convolution

        if inverted_residual_settings is None:
            inverted_residual_settings = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if bit_width is None:
            inverted_residual_bit_width = [
                # [(conv1_weight_bw, conv1_act_bw), (conv2_w_bw, conv2_act_bw), (conv3_w_bw, conv3_act_bw)]
                [(None, None), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
                [(3, 3), (3, 3), (3,)],
            ]

            bit_width = {
                'image': 8,
                'first_conv': (3, 3),
                'last_conv': (3, 3),
                'shared_act': 4,
                'inverted_residual': inverted_residual_bit_width
            }

        return_quant = quantization
        img_quant = quantization
        weight_quant = quantization
        act_quant = quantization
        shared_act_quant = quantization

        # Quantizers
        img_quantizer = InputQuant if img_quant else None

        # TODO: signed necessary? comment faiore pour que les multithreshold ne soient pas à moitié composés de zeros?
        #  avec zero point? quel impact sur les transformations?
        # Building input quantization and first convolution layer
        layers = [
            qnn.QuantIdentity(act_quant=img_quantizer, bit_width=bit_width['image'],
                              return_quant_tensor=return_quant, signed=True),
            QConvBnAct(in_channels=img_channel, out_channels=input_channel, stride=2, padding=1, batchnorm=batchnorm,
                       weight_quant=weight_quant, weight_bit_width=bit_width['first_conv'][0],
                       act_quant=act_quant, act_bit_width=bit_width['first_conv'][1],
                       return_act_quant=return_quant)
        ]
        in_ch = input_channel

        # Building all residual layers (by block)
        prev_layer_use_residual = False
        block_number = 0
        for t, c, n, s in inverted_residual_settings:
            for i in range(n):
                stride = s if i == 0 else 1
                use_residual = stride == 1 and in_ch == c and residual_connections
                if residual_connections:
                    input_quant = use_residual or prev_layer_use_residual or (block_number == 1 and i==0)
                else:
                    input_quant = False if block_number == 0 and i == 0 else True
                layers.append(
                    QInvertedResidual(in_channels=in_ch, out_channels=c, stride=stride, expand_ratio=t,
                                      batchnorm=batchnorm, weight_quant=weight_quant, act_quant=act_quant,
                                      bit_width=bit_width['inverted_residual'][block_number],
                                      shared_act_bw=bit_width['shared_act'], return_quant=return_quant,
                                      use_residual=use_residual, input_quant=input_quant)
                )
                in_ch = c
                prev_layer_use_residual = use_residual
                block_number += 1

        # Quantize the output of the QInvertedResidual blocks that has no activation function
        quant = select_quantizer(shared_act_quant, bit_width=bit_width['shared_act'], layer='activation', signed=True)
        layers.append(qnn.QuantIdentity(act_quant=quant, bit_width=bit_width['shared_act'],
                                        return_quant_tensor=return_quant))

        # Build last convolutional layer
        layers.append(QConvBnAct(in_ch, out_channels, kernel_size=1, batchnorm=batchnorm,
                                 weight_quant=weight_quant, weight_bit_width=bit_width['last_conv'][0],
                                 act_quant=act_quant, act_bit_width=bit_width['last_conv'][1],
                                 return_act_quant=last_return_quant))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1280, batchnorm=True, residual_connections=True,
                 inverted_residual_settings=None):
        super().__init__()
        img_channel = in_channels  # Number of channels in the input image
        input_channel = 32  # Number of channels after the 1st convolution

        if inverted_residual_settings is None:
            inverted_residual_settings = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Building input quantization and first convolution layer
        layers = [
            ConvBnAct(in_channels=img_channel, out_channels=input_channel, stride=2, padding=1, batchnorm=batchnorm)
        ]
        in_ch = input_channel

        for t, c, n, s in inverted_residual_settings:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels=in_ch, out_channels=c, stride=stride, expand_ratio=t,
                                               batchnorm=batchnorm, residual=residual_connections))
                in_ch = c

        layers.append(ConvBnAct(in_ch, out_channels, kernel_size=1, batchnorm=batchnorm))

        self.features = nn.Sequential(*layers)


    def forward(self, x):
        x = self.features(x)
        return x
