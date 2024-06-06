"""
Copyright (c) 2024 Julien Posso
"""

import torch.nn as nn
import brevitas.nn as qnn


class ModelWrapper(nn.Module):
    """A wrapper class that assembles a neural network feature extractor (backbone) and a head"""
    def __init__(self, features, head):
        super().__init__()
        self.features = features
        self.head = head

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) or isinstance(m, qnn.QuantLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        batchnorm=True,
        activation=True
    ):

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]

        if batchnorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation:
            # Replaced ReLU6 by ReLU as ReLU6 is not supported by layer fusion
            layers.append(nn.ReLU())

        super(ConvBnAct, self).__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, batchnorm=True, residual=True):
        super(InvertedResidual, self).__init__()
        # Only stride of 1 and 2 in Mobilenet-v2
        assert stride in [1, 2]

        self.use_residual = stride == 1 and in_channels == out_channels and residual

        hidden_channels = int(round(in_channels * expand_ratio))

        layers = []

        if expand_ratio != 1:
            layers.append(ConvBnAct(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                                    batchnorm=batchnorm))

        layers.extend([
            ConvBnAct(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      batchnorm=batchnorm, stride=stride, groups=hidden_channels),

            ConvBnAct(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1,
                      batchnorm=batchnorm, activation=False),
        ])

        self.skip_add = nn.quantized.FloatFunctional()

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            x = self.skip_add.add(x, self.conv(x))
        else:
            x = self.conv(x)
        return x
