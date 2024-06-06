"""
Copyright (c) 2024 Julien Posso
"""

from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat, ShiftedUint8WeightPerTensorFloat
from brevitas.quant.binary import SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst
from brevitas.quant.ternary import SignedTernaryWeightPerTensorConst, SignedTernaryActPerTensorConst
from brevitas.inject.defaults import TruncTo8bit
from brevitas.core.restrict_val import RestrictValueType


per_channel_scaling = True
scaling_type = RestrictValueType.LOG_FP

class IntWeightQuant(Int8WeightPerTensorFloat):
    """Signed integer Weight quantizer. Bit-width must be specified at layer creation (qnn.Conv2d(bit_width=X, ...)"""
    scaling_min_val = 2e-16
    bit_width = None
    scaling_per_output_channel = per_channel_scaling


class TernWeightQuant(SignedTernaryWeightPerTensorConst):
    """Ternary weight quantizer. Bit-width = 2, can be overwritten at layer creation"""
    scaling_min_val = 2e-16
    scaling_per_output_channel = per_channel_scaling


class BinWeightQuant(SignedBinaryWeightPerTensorConst):
    """Ternary weight quantizer. Bit-width must be specified at layer creation (qnn.Conv2d(bit_width=X, ...)"""
    scaling_min_val = 2e-16
    scaling_per_output_channel = per_channel_scaling


class UintActQuant(Uint8ActPerTensorFloat):
    """Unsigned integer activation function quantizer. Bit-width must be specified at layer creation"""
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = scaling_type


class IntActQuant(Int8ActPerTensorFloat):
    """Signed integer activation function quantizer. Bit-width must be specified at layer creation"""
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = scaling_type


class TernActQuant(SignedTernaryActPerTensorConst):
    """Signed ternary activation function quantizer"""
    scaling_min_val = 2e-16
    restrict_scaling_type = scaling_type


class BinActQuant(SignedBinaryActPerTensorConst):
    """Signed binary activation function quantizer"""
    scaling_min_val = 2e-16
    restrict_scaling_type = scaling_type


class InputQuant(UintActQuant):
    """Unsigned integer quantizer for input image"""
    pass


class ShiftedInputQuant(ShiftedUint8ActPerTensorFloat):
    """Unsigned integer quantizer for input image with zero point"""
    pass


class BiasQuant(Int8Bias):
    pass


class PoolQuant(TruncTo8bit):
    pass


def select_quantizer(quantization=False, bit_width=8, layer='convolution', signed=False):
    """Automatically switch quantizer according to bit_width.
    layer should be 'convolution' or 'activation'"""
    assert layer in ('convolution', 'activation')
    if quantization:
        # Automatically switch quantizer according to bit_width
        if bit_width == 1:
            quantizer = BinWeightQuant if layer == 'convolution' else BinActQuant
        elif bit_width == 2:
            quantizer = TernWeightQuant if layer == 'convolution' else TernActQuant
        else:
            act_quant = IntActQuant if signed else UintActQuant
            quantizer = IntWeightQuant if layer == 'convolution' else act_quant
    else:
        quantizer = None

    return quantizer

