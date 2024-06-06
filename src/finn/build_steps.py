"""
Copyright (c) 2024 Julien Posso
"""

from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.general import (
    ConvertDivToMul,
    RemoveStaticGraphInputs,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    ApplyConfig,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)

from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import (
    MoveLinearPastFork,
    MoveLinearPastEltwiseAdd,
    MoveAddPastConv,
    MoveAddPastMul,
    MoveMulPastMaxPool,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
    MoveMulPastDWConv,
)

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    ShellFlowType,
)

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
#
# from src.finn_transforms import (
#     AbsorbConsecutiveTransposes,
# )

from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes

def step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig=None):
    """Tidy-up operations: prepare the model for the streamlining step.
    More details: https://finn.readthedocs.io/en/latest/nw_prep.html#tidy-up-transformations"""
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """ Apply Streamlining transformations to get rif of floating point operations in the model (mul/div,
    add/sub, batchnorm, etc...). Additional steps required if the model contains skip connections (residual).
    More details about streamlining: https://finn.readthedocs.io/en/latest/nw_prep.html#streamlining-transformations"""

    streamline_transformations = [
        # To understand the comments:
        # OP_A -> OP_B => OP_C: Operation A followed by operation B is transformed into operation C
        BatchNormToAffine(),  # convert Batchnorm to Mul-Add layers
        AbsorbAddIntoMultiThreshold(),  # Add -> MultiThreshold => MultiThreshold
        AbsorbSignBiasIntoMultiThreshold(),  # MultiThreshold -> Add => MultiThreshold
        ConvertDivToMul(),  # Convert division to multiplication
        MoveScalarMulPastConv(),  # Mul (scalar) -> conv => conv -> mul (scalar)
        # MoveMulPastDWConv(),  # Used only when act_scaling_per_channel=True on the activation right before the DW conv
        CollapseRepeatedMul(),  # Absorb consecutive multiplications
        AbsorbMulIntoMultiThreshold(),  # Mul -> MultiThreshold => MultiThreshold
        RoundAndClipThresholds(),  # Round threshold to integer if the input is integer
    ]

    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(RemoveIdentityOps())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    return model


def step_streamline_residual(model: ModelWrapper, cfg: DataflowBuildConfig):
    """ Apply Streamlining transformations to a model that contains skip connections.
    step_streamline must be executed before"""

    streamline_residual_transformations = [
        # To understand the comments:
        # OP_A -> OP_B => OP_C: Operation A followed by operation B is transformed into operation C
        MoveLinearPastFork(),  # Move Mul operations past fork
        MoveLinearPastEltwiseAdd(),  # Move Mul operations past Add node
        MoveScalarMulPastConv(),  # Mul -> Conv => Conv -> Mul
        AbsorbMulIntoMultiThreshold(),  # Mul -> MultiThreshold => MultiThreshold
        RoundAndClipThresholds(),  # Round threshold to integer if the input is integer
    ]

    for trn in streamline_residual_transformations:
        model = model.transform(trn)
        model = model.transform(RemoveIdentityOps())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    return model


def step_convert_to_hls_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    """ Convert network layers to HLS layers. Additional steps required if the model contains skip connections.
    More details here: https://finn.readthedocs.io/en/latest/nw_prep.html#convert-to-hls-layers
    """
    mem_mode = cfg.default_mem_mode.value
    # Transform Conv into Im2Col + Matmul and insert Transpose nodes
    model = model.transform(LowerConvsToMatMul())
    # Insert Transpose layers before and after (as the Threshold node input was in NCHW)
    model = model.transform(to_hls.InferThresholdingLayer())
    # Fuse consecutive transposes inserted by InferThresholdingLayer and LowerConvsToMatMul
    model = model.transform(AbsorbConsecutiveTransposes())
    # Infer Matrix multiplications (convolutions)
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
    # Infer Depthwise convolutions
    model = model.transform(to_hls.InferVectorVectorActivation())
    # Infer IM2Col
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def step_convert_to_hls_residual_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    """ Convert residual network layers to HLS layers. step_convert_to_hls_layers must be executed before"""

    # Converts Add to Stream Add. Also converts output ADD from Float32 to UINT9 if adding two UINT8
    model = model.transform(to_hls.InferAddStreamsLayer())
    # Fuse the Transpose fork node to the Transpose nodes in the branches
    model = model.transform(AbsorbConsecutiveTransposes())
    # Add InferDuplicateStreamsLayer in the graph (branch node)
    model = model.transform(to_hls.InferDuplicateStreamsLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model
