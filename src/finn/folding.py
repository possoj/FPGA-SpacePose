"""
Copyright (c) 2024 Julien Posso
"""

from math import ceil
from qonnx.custom_op.registry import getCustomOp
from collections import Counter
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from qonnx.transformation.general import GiveUniqueNodeNames
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
)


def print_node(node_instance, res_estim):

    simd_nodes = (
        "ConvolutionInputGenerator", "FMPadding_Batch", "MatrixVectorActivation"
    )

    pe_nodes = (
        "Thresholding_Batch", "DuplicateStreams_Batch", "StreamingMaxPool_Batch",
        "MatrixVectorActivation", "VectorVectorActivation"
    )

    print(f"\nNode {node_instance.onnx_node.op_type}")

    if node_instance.onnx_node.op_type in simd_nodes:
        print(f"Set SIMD to {node_instance.get_nodeattr('SIMD')}")

    if node_instance.onnx_node.op_type in pe_nodes:
        print(f"Set PE to {node_instance.get_nodeattr('PE')}")

    print(f"Latency = {node_instance.get_exp_cycles():,} cycles")
    print(f"Estimated resources:\n{res_estim}")


def folding_3x3_conv(model, node, previous_parallelism, target_cycles_per_frame):
    """TODO improve this:
    Automatically finds the SW-MVU (3x3 normal conv) folding that respect the target_cycles_per_frame and the
    data-width constraints while minimizing the hardware resources used (minimizes SIMD_SW + PE_CONV * SIMD_CONV).
    Only works with squared images, feature maps and kernels for now"""

    next_node = model.find_consumer(node.output[0])
    conv_inst = getCustomOp(next_node)
    sw_inst = getCustomOp(node)

    ifm_c = sw_inst.get_nodeattr("IFMChannels")
    ifm_d = sw_inst.get_nodeattr("IFMDim")[0]
    ofm_c = conv_inst.get_nodeattr("MH")
    # Output feature maps dimension of both SW and MVU
    ofm_d = sw_inst.get_nodeattr("OFMDim")[0]
    k = sw_inst.get_nodeattr("ConvKernelDim")[0]

    # CONVOLUTION INPUT GENERATOR

    # Compute minimum SIMD in the Sliding Window to reach target_cycles_per_frame
    simd_sw_min_write = ceil((ifm_d * k * ifm_c + ofm_d * ofm_d * k * k * ifm_c) / target_cycles_per_frame)
    simd_sw_min_read = ceil((ifm_d * k * ifm_c + ofm_d * ifm_d * ifm_c) / target_cycles_per_frame)
    simd_sw_min = max(simd_sw_min_read, simd_sw_min_write)
    simd_sw_max = ifm_c

    assert simd_sw_min <= ifm_c, f"No SIMD value found to reach target_cycles_per_frame for node {node.name}:\n" \
                                 f"The number of SIMD lines cannot be greater than the number of input channels"

    candidates_sw_simd = []
    for simd_sw_candidate in range(simd_sw_min, simd_sw_max + 1, 1):

        # Input feature maps constraint
        if ifm_c % simd_sw_candidate == 0:

            # Previous layer constraint
            if (simd_sw_candidate % previous_parallelism == 0) or (previous_parallelism % simd_sw_candidate == 0):
                candidates_sw_simd.append(simd_sw_candidate)

    if not candidates_sw_simd:
        raise Exception(
            f"No SIMD value found for node {node.name} to satisfy the three constraints:\n"
            f"- target_cycles_per_frame\n"
            f"- (IFMChannels % SIMD = 0)\n"
            f"- DataWidthConverter: SIMD % previous_parallelism = 0 or previous_parallelism % SIMD  = 0"
        )

    # MATRIX VECTOR ACTIVATION

    pe_list = []
    pe_min = 1
    pe_max = ofm_c
    # Constraint =  MH % PE = 0
    for pe_candidate in range(pe_min, pe_max + 1, 1):
        if ofm_c % pe_candidate == 0:
            pe_list.append(pe_candidate)

    if not pe_list:
        raise Exception(f"No PE value found to satisfy MH % PE = 0 for node {node.name}/{next_node.name}.\n"
                        f"MH is the number of output feature maps")

    simd_min = 1
    mw = k * k * ifm_c
    simd_max = mw
    simd_list = []
    # Constraint =  MW % SIMD = 0
    for simd_candidate in range(simd_min, simd_max + 1, 1):
        if mw % simd_candidate == 0:
            simd_list.append(simd_candidate)

    if not simd_list:
        raise Exception(f"No SIMD value found to satisfy MW % SIMD = 0 for node {node.name}")

    # Check latency constraint on matrix vector unit
    pe_simd_candidates = []
    for pe_candidate in pe_list:
        for simd_candidate in simd_list:
            if pe_candidate * simd_candidate > (ifm_c * k * k * ofm_c * ofm_d * ofm_d) / target_cycles_per_frame:
                pe_simd_candidates.append((pe_candidate, simd_candidate))

    if not pe_simd_candidates:
        raise Exception(f"No PE and SIMD value found in the MVU to satisfy the latency constraint for node {node.name}")

    # Select folding config that respect simd_mvu % simd_sw = 0 or simd_sw % simd_mvu = 0
    sw_simd_mvu_pe_simd = []
    for simd_sw_candidate in candidates_sw_simd:
        for pe_mvu_candidate, simd_mvu_candidate in pe_simd_candidates:
            if (simd_mvu_candidate % simd_sw_candidate == 0) or (simd_sw_candidate % simd_mvu_candidate == 0):
                sw_simd_mvu_pe_simd.append((simd_sw_candidate, pe_mvu_candidate, simd_mvu_candidate))

    if not sw_simd_mvu_pe_simd:
        raise Exception(
            f"No PE and SIMD value found in the SW and MVU to satisfy the MVU_SW % PE_SW = 0 for node {node.name}")

    # TODO: Find Multiple solutions that minimizes the parallelism

    # Step 2: in the found solutions, choose the one for each layer that minimizes the hardware resources
    # In the solutions found, there are configurations that latency is way below target_cycles_per_frame
    lowest_parallelism = 1000000
    best_config = (None, None, None)
    for j, (simd_sw, pe_conv, simd_conv) in enumerate(sw_simd_mvu_pe_simd):
        parallelism = simd_sw + pe_conv * simd_conv
        if parallelism < lowest_parallelism:
            best_config = (simd_sw, pe_conv, simd_conv)
            lowest_parallelism = parallelism

    assert best_config != (None, None, None), f"No best_config found for the folding of node {node.name}"

    return best_config


def folding_1x1_conv(model, node, previous_parallelism, target_cycles_per_frame):
    """
    TODO: update this
    Finds Optimal 1x1 Convolution Folding. The folding configuration meets target_cycles_per_frame and the data-width
    constraints while minimizing the hardware resources used (minimizes PE_CONV * SIMD_CONV).
    Only works with squared images, feature maps and kernels for now
    1x1 convolutions => MatrixVectorActivation alone
    Parameters:
    - model (CustomModel): model object
    - target_cycles_per_frame (int): target cycles per frame for computation

    Returns:
    - list: list of tuples of optimal (PE, SIMD) values for each 1x1 convolution layer
    """

    conv_inst = getCustomOp(node)
    # MH = number of output feature maps (after the current MatrixVectorActivation node)
    mh = conv_inst.get_nodeattr("MH")
    # MW = number of input feature maps (before the current MatrixVectorActivation node)
    mw = conv_inst.get_nodeattr("MW")

    # Get the OFM dimensions
    # A 1x1 convolution should not modify the feature maps width and height, except padding is used.
    successor = model.find_consumer(node.output[0])
    successor_inst = getCustomOp(successor)
    ofm_d = successor_inst.get_nodeattr("numInputVectors")[1]

    pe_list = []
    pe_min = 1
    pe_max = mh
    # Constraint =  MH % PE = 0
    for pe_candidate in range(pe_min, pe_max + 1, 1):
        if mh % pe_candidate == 0:
            pe_list.append(pe_candidate)

    if not pe_list:
        raise Exception(f"No PE value found to satisfy MH % PE = 0 for node {node.name}")

    simd_min = 1
    simd_max = mw
    simd_list = []
    for simd_candidate in range(simd_min, simd_max + 1, 1):
        # Constraint =  MW % SIMD = 0
        if mw % simd_candidate == 0:
            # Previous layer constraint (DWC)
            if (simd_candidate % previous_parallelism == 0) or (previous_parallelism % simd_candidate == 0):
                simd_list.append(simd_candidate)

    if not simd_list:
        raise Exception(f"No SIMD value found to satisfy MW % SIMD = 0 and DataWidthConverter constraints "
                        f"for node {node.name}")

    # Check latency constraint on matrix vector unit
    pe_simd_candidates = []
    for pe_candidate in pe_list:
        for simd_candidate in simd_list:
            if pe_candidate * simd_candidate > (mw * mh * ofm_d * ofm_d) / target_cycles_per_frame:
                pe_simd_candidates.append((pe_candidate, simd_candidate))

    if not pe_simd_candidates:
        raise Exception(f"No PE and SIMD value found in the MVU to satisfy the latency constraint of node {node.name}")

    # In the found solutions, choose the one for each layer that minimizes the hardware resources
    # In the solutions found, there are configurations that latency is way below target_cycles_per_frame
    lowest_parallelism = 1000000
    best_config = (None, None)
    for j, (pe, simd) in enumerate(pe_simd_candidates):
        parallelism = pe * simd
        if parallelism < lowest_parallelism:
            best_config = (pe, simd)
            lowest_parallelism = parallelism

    assert best_config != (None, None), f"No best_config found for the folding of node {node.name}"

    return best_config


def folding_3x3dw_conv(model, node, previous_parallelism, target_cycles_per_frame):
    """ TODO: update this
    Automatically finds the SW-VVU (3x3 depthwise conv) folding that respect the target_cycles_per_frame and the
    data-width constraints while minimizing the hardware resources used (minimizes SIMD_SW + PE_CONV).
    Only works with squared images, feature maps and kernels for now"""

    next_node = model.find_consumer(node.output[0])
    sw_inst = getCustomOp(node)
    conv_inst = getCustomOp(next_node)

    ifm_c = sw_inst.get_nodeattr("IFMChannels")
    ifm_d = sw_inst.get_nodeattr("IFMDim")[0]
    ofm_c = conv_inst.get_nodeattr("Channels")
    # mh = ofm_c
    # Output feature maps dimension of both SW and MVU
    ofm_d = sw_inst.get_nodeattr("OFMDim")[0]
    k = sw_inst.get_nodeattr("ConvKernelDim")[0]

    # Compute minimum SIMD in the Sliding Window to reach target_cycles_per_frame
    simd_sw_min_write = ceil((ifm_d * k * ifm_c + ofm_d * ofm_d * k * k * ifm_c) / target_cycles_per_frame)
    simd_sw_min_read = ceil((ifm_d * k * ifm_c + ofm_d * ifm_d * ifm_c) / target_cycles_per_frame)
    simd_sw_min = max(simd_sw_min_read, simd_sw_min_write)
    simd_sw_max = ifm_c

    candidates_sw_simd = []
    for simd_sw_candidate in range(simd_sw_min, simd_sw_max + 1, 1):
        # IFMc % SIMD constraint
        if ifm_c % simd_sw_candidate == 0:
            # Previous layer constraint (DWC)
            if (simd_sw_candidate % previous_parallelism == 0) or (previous_parallelism % simd_sw_candidate == 0):
                candidates_sw_simd.append(simd_sw_candidate)

    if not candidates_sw_simd:
        raise Exception(
            f"No SIMD value found to satisfy both (ifm_c % simd = 0) and target_cycles_per_frame for node {node.name}")

    # VVU-PE
    pe_list = []
    pe_min = 1
    pe_max = ofm_c
    for pe_candidate in range(pe_min, pe_max + 1, 1):
        # Constraint Channels % PE = 0
        if ofm_c % pe_candidate == 0:
            pe_list.append(pe_candidate)

    if not pe_list:
        raise Exception(f"No PE value found to satisfy Channels % PE = 0 for node {node.name}")

    # Check latency constraint on matrix vector unit
    pe_candidates = []
    for pe_candidate in pe_list:
        # exp_cycles = ((ch * k_h * k_w) / pe) * batch_size * (dim_h * dim_w) / mmv
        # And mmv = 1, batch_size = 1, dim_h = dim_w for squared images
        if pe_candidate > (ifm_c * k * k * ofm_d * ofm_d) / target_cycles_per_frame:
            pe_candidates.append(pe_candidate)

    if not pe_candidates:
        raise Exception(f"No PE value found in the VVU to satisfy the latency constraint for node {node.name}")

    sw_simd_vvu_pe_simd = []
    for simd_sw_candidate in candidates_sw_simd:
        for pe_mvu_candidate in pe_candidates:
            # Previous layer constraint (DWC)
            if (simd_sw_candidate % pe_mvu_candidate == 0) or (pe_mvu_candidate % simd_sw_candidate == 0):
                sw_simd_vvu_pe_simd.append((simd_sw_candidate, pe_mvu_candidate))

    if not sw_simd_vvu_pe_simd:
        raise Exception(
            f"No PE and SIMD value found in the SW and MVU to satisfy the VVU_PE % SIMD_SW = 0 or SIMD_SW % VVU_PE = 0"
            f"for node {node.name}")

    # Step 2: in the found solutions, choose the one for each layer that minimizes the hardware resources
    # In the solutions found, there are configurations that latency is way below target_cycles_per_frame
    lowest_parallelism = 1000000
    best_config = (None, None)
    for j, (simd_sw, pe_conv) in enumerate(sw_simd_vvu_pe_simd):
        parallelism = simd_sw + pe_conv
        if parallelism < lowest_parallelism:
            best_config = (simd_sw, pe_conv)
            lowest_parallelism = parallelism

    assert best_config != (None, None), f"No best_config found for the folding of node {node.name}"

    return best_config


def step_set_folding(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Set the folding for all layers"""

    target_cycles_per_frame = cfg._resolve_cycles_per_frame()
    verbose = True
    non_finn_nodes = model.get_non_finn_nodes()
    assert not non_finn_nodes, f"Error there are still non FINN nodes in the graph: {non_finn_nodes}"

    latency_estim = {}

    # Name of the number of input channels by op_type
    in_channels = {
        "Thresholding_Batch": "NumChannels",
        "ConvolutionInputGenerator": "IFMChannels",
        "FMPadding_Batch": "NumChannels",
        "VectorVectorActivation": "Channels"
    }

    # Other than convolution nodes (ConvolutionInputGenerator,
    pe_or_simd = {
        "FMPadding_Batch": "SIMD",
        "Thresholding_Batch": "PE",
        "DuplicateStreams_Batch": "PE",
        "StreamingMaxPool_Batch": "PE",
        "AddStreams_Batch": "PE"
    }

    # Neither PE nor SIMD: UpsampleNearestNeighbour_Batch, StreamingConcat

    total_res = {'BRAM_18K': 0, 'BRAM_efficiency': 0, 'LUT': 0, 'URAM': 0, 'URAM_efficiency': 0, 'DSP': 0}

    res_type = {
        "MatrixVectorActivation": "auto",  # lut (default), auto, dsp
        "VectorVectorActivation": "auto",  # lut, auto (default), dsp
    }

    ram_style = {
        "ConvolutionInputGenerator": "distributed",  # distributed (default), auto, block, ultra
        "Thresholding_Batch": "distributed",  # distributed (default), block
        "MatrixVectorActivation": "auto",  # auto (default), block, distributed, ultra
    }

    folding_3x3 = (None, None, None)
    folding_3x3_dw = (None, None)
    previous_node_type = None
    previous_parallelism = 1
    n_channels = None

    for idx, node in enumerate(model.graph.node):
        instance = getCustomOp(node)

        # For now only works for networks with branch
        next_node = model.find_consumer(node.output[0])

        # Get number of input channels of the first node. Then update it at each layer
        if idx == 0:
            n_channels = instance.get_nodeattr(in_channels[node.op_type])

        # Convolution 3x3 (ConvolutionInputGenerator)
        if node.op_type == "ConvolutionInputGenerator" and next_node.op_type == "MatrixVectorActivation":
            # Get the folding for ConvolutionInputGenerator and MatrixVectorActivation
            folding_3x3 = folding_3x3_conv(model, node, previous_parallelism, target_cycles_per_frame)
            instance.set_nodeattr("SIMD", folding_3x3[0])
            # previous_parallelism = folding_3x3[0]  # Not useful

        # Convolution 3x3 (MatrixVectorActivation)
        elif node.op_type == "MatrixVectorActivation" and previous_node_type == "ConvolutionInputGenerator":
            instance.set_nodeattr("PE", folding_3x3[1])
            instance.set_nodeattr("SIMD", folding_3x3[2])
            # Update the number of channels
            n_channels = instance.get_nodeattr("MH")
            previous_parallelism = folding_3x3[1]

        # Conv 3x3 DW conv (ConvolutionInputGenerator)
        elif node.op_type == "ConvolutionInputGenerator" and next_node.op_type == "VectorVectorActivation":
            folding_3x3_dw = folding_3x3dw_conv(model, node, previous_parallelism, target_cycles_per_frame)
            instance.set_nodeattr("SIMD", folding_3x3_dw[0])

        # Conv 3x3 DW conv (VectorVectorActivation)
        elif node.op_type == "VectorVectorActivation" and previous_node_type == "ConvolutionInputGenerator":
            instance.set_nodeattr("PE", folding_3x3_dw[1])
            previous_parallelism = folding_3x3_dw[1]
            # n_channels = instance.get_nodeattr("Channels")  # A depthwise conv does not change the number of channels

        # 1x1 conv
        elif node.op_type == "MatrixVectorActivation":
            folding_1x1 = folding_1x1_conv(model, node, previous_parallelism, target_cycles_per_frame)
            instance.set_nodeattr("PE", folding_1x1[0])
            instance.set_nodeattr("SIMD", folding_1x1[1])
            n_channels = instance.get_nodeattr("MH")
            previous_parallelism = folding_1x1[0]

        elif node.op_type in list(pe_or_simd.keys()):
            folding_successful = False
            for i in range(1, n_channels+1):
                # Channel constraint
                if n_channels % i == 0:
                    instance.set_nodeattr(pe_or_simd[node.op_type], i)
                    # Latency constraint
                    if instance.get_exp_cycles() <= target_cycles_per_frame:
                        # DWC constraint
                        if i % previous_parallelism == 0 or previous_parallelism % i == 0:
                            folding_successful = True
                            previous_parallelism = i
                            break

            if not folding_successful:
                raise Exception(f"Failed to find a valid folding for node {node.name}")

        previous_node_type = node.op_type
        if node.op_type in list(res_type.keys()):
            instance.set_nodeattr("resType", res_type[node.op_type])

        if node.op_type in list(ram_style.keys()):
            instance.set_nodeattr("ram_style", ram_style[node.op_type])

        # latency and resource estimation
        latency_estim[node.name] = instance.get_exp_cycles()
        res_estim = instance.node_res_estimation()
        total_res = Counter(total_res) + Counter(res_estim)

        if verbose:
            print_node(instance, res_estim)

    if verbose:
        print(f"\nEstimated total resources:\n:{total_res}")

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateCycles())

    return model
