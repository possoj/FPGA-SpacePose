import os
import logging
import pdb

import torch
from qonnx.core.modelwrapper import ModelWrapper
from finn.util.pytorch import ToTensor
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import brevitas.onnx as bo

# from src.tools.evaluation import evaluation
from src.config.build.config import load_config, save_config
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model, save_model
from src.finn.folding import step_set_folding
from src.tools.utils import prepare_directories
from src.finn.build_steps import (
    step_tidy_up,
    step_streamline,
    step_streamline_residual,
    step_convert_to_hls_layers,
    step_convert_to_hls_residual_layers,
)


def noop(*args, **kwargs):
    """Define a no-op function to replace pdb.set_trace()"""
    pass


def check_model(model):
    """Check the Brevitas model"""
    for layer in model.modules():
       # Detect batchnorm layers
       if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
           # Detect multiplications in Batchnorm (scale parameter)
           negative_params = list(layer.parameters())[0] < 0
           if True in negative_params:
               raise ValueError(f"Detected negative weight in model batchnorm weights."
                                f"These parameters cannot be merged into MultiThreshold. "
                                f"Thus multiplications stay in the graph after Streamlining which will "
                                f"cause an error later")


def add_pre_processing(model):
    """Adds pre-processing (divide UINT8 input by 255) to the model"""
    global_inp_name = model.graph.input[0].name
    in_shape = model.get_tensor_shape(global_inp_name)
    # preprocessing: ToTensor divides uint8 inputs by 255
    pre_processing = ToTensor()
    # Export pre-processing to ONNX model
    pre_model = ModelWrapper(bo.export_finn_onnx(pre_processing, in_shape))
    # join preprocessing and core model
    model = model.transform(MergeONNXModels(pre_model))
    # add input quantization annotation: UINT8
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    return model


def select_build_steps(use_residual=False):
    """Select the build steps. Additional steps required when using residual layers"""
    if use_residual:
        return [
            step_tidy_up,
            step_streamline,
            step_streamline_residual,
            step_convert_to_hls_layers,
            step_convert_to_hls_residual_layers,
            "step_create_dataflow_partition",
            step_set_folding,
            "step_generate_estimate_reports",
            "step_hls_codegen",
            "step_hls_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
    else:
        return [
            step_tidy_up,
            step_streamline,
            step_convert_to_hls_layers,
            "step_create_dataflow_partition",
            step_set_folding,
            "step_generate_estimate_reports",
            "step_hls_codegen",
            "step_hls_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]


def main():
    # Monkey patch pdb.set_trace() with our no-op function
    pdb.set_trace = noop

    seed = 1001
    cfg_root = os.path.join('src', 'config', 'build')
    cfg_template = 'exp_'
    # List experiments to run (both files and directories that start with config_ in the config folder)
    experiment_list = [x for x in os.listdir(cfg_root) if x.startswith(cfg_template)]
    assert len(experiment_list) > 0, f'found no YAML config file in {cfg_root} that starts with {cfg_template}'

    # Run multiple experiments
    for exp in experiment_list:
        config = load_config(os.path.join(cfg_root, exp))
        assert 'brevitas' in config.MODEL.BACKBONE.NAME, f'At least the backbone of the model should be quantized with Brevitas'
        exp_id, _ = os.path.splitext(exp.replace(cfg_template, ''))
        save_folder = prepare_directories('experiments', 'build', f'{cfg_template}{exp_id}',
                                          ('model', 'export', 'finn_output'))
        print(f"\nLoad config file {os.path.join(cfg_root, exp)}")
        print(f"Results will be saved to {save_folder}\n")

        camera = load_camera(config.DATA.PATH)

        spe_utils = SPEUtils(camera, config.MODEL.HEAD.ORI, config.MODEL.HEAD.N_ORI_BINS_PER_DIM,
                             config.DATA.ORI_SMOOTH_FACTOR, config.MODEL.HEAD.POS)

        rot_augment = False
        other_augment = False
        data_shuffle = False

        data, split = load_dataset(spe_utils, config.DATA.PATH, config.DATA.BATCH_SIZE, config.DATA.IMG_SIZE,
                                   rot_augment, other_augment, data_shuffle, seed)

        params_path = os.path.join(config.MODEL.PATH, 'parameters.pt')
        bit_width_path = os.path.join(config.MODEL.PATH, 'bit_width.json')

        model, bit_width = import_model(
            data, config.MODEL.BACKBONE.NAME, config.MODEL.HEAD.NAME, params_path, bit_width_path,
            config.MODEL.MANUAL_COPY, residual=config.MODEL.BACKBONE.RESIDUAL, quantization=config.MODEL.QUANTIZATION,
            ori_mode=config.MODEL.HEAD.ORI, n_ori_bins_per_dim=config.MODEL.HEAD.N_ORI_BINS_PER_DIM
        )
        model.eval()
        model.cpu()

        # verify that there is no negative weights in batchnorm layers
        check_model(model)

        # Eval model
        # score, error = evaluation(model, data, spe_utils, split['eval'], torch.device('cpu'))

        # Save model and config
        save_model(os.path.join(save_folder, 'model'), model, bit_width)
        save_config(config, os.path.join(save_folder, 'config.yaml'))

        # Export Brevitas model to ONNX
        # For now only export the feature extractor (backbone) as the split into two branches in the head of the network seem not to be supported by FINN.
        img_shape = tuple(next(iter(data['valid']))[0]['torch'].size())
        model_finn = ModelWrapper(
            bo.export_finn_onnx(model.features, img_shape,
                                os.path.join(save_folder, 'export', 'export.onnx'))
        )

        # Add pre-processing (to give normalized [0-1] images to the accelerator)
        model_finn = add_pre_processing(model_finn)
        model_file = os.path.join(save_folder, 'export', 'with_preprocessing.onnx')
        model_finn.save(model_file)

        # Build the FINN accelerator
        assert config.FINN.FIFO.SIZING_METHOD in ('largefifo_rtlsim', 'characterize')
        auto_fifo = build_cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM if config.FINN.FIFO.SIZING_METHOD == 'largefifo_rtlsim' \
            else build_cfg.AutoFIFOSizingMethod.CHARACTERIZE

        # For more details about FINN Build configuration (inside Docker container):
        # /tools/finn/src/finn/builder/build_dataflow_config.py

        cfg = build_cfg.DataflowBuildConfig(
            steps=select_build_steps(config.MODEL.BACKBONE.RESIDUAL),
            output_dir=os.path.join(save_folder, 'finn_output'),
            folding_config_file=None,
            synth_clk_period_ns=config.FINN.ACCEL.CLK_PERIOD_NS,
            target_fps=int((10**9/config.FINN.ACCEL.CLK_PERIOD_NS) / config.FINN.ACCEL.TARGET_CYCLES_PER_FRAME),
            board=config.FINN.BOARD.NAME,
            shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
            # FIFO
            # Allow very large FIFOs: FIFO nodes with depth larger than 32768 will be split
            split_large_fifos=config.FINN.FIFO.SPLIT_LARGE,
            auto_fifo_depths=config.FINN.FIFO.AUTO_DEPTH,
            #: When `auto_fifo_depths = True`, select which method will be used (default: RTL SIM)
            # auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM,
            auto_fifo_strategy = auto_fifo,
            # If True, use Python instead of c++ RTL SIM (default: False)
            # In my experience it produces smaller FIFOs with less FPGA resource usage
            force_python_rtlsim = config.FINN.FIFO.RTL_SIM,
            #: Memory resource type for large FIFOs
            large_fifo_mem_style = build_cfg.LargeFIFOMemStyle.AUTO,
            generate_outputs=[
                build_cfg.DataflowOutputType.PYNQ_DRIVER,
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                build_cfg.DataflowOutputType.BITFILE,
                build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.OOC_SYNTH,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            ],
        )

        # Create a logger specific to the experiment
        logger = logging.getLogger(f'Experiment_{exp_id}')
        file_handler = logging.FileHandler(os.path.join(save_folder, 'error.log'))
        file_handler.setLevel(logging.ERROR)
        logger.addHandler(file_handler)

        try:
            build.build_dataflow_cfg(model_file, cfg)
        except Exception as e:
            # Log the exception
            logger.exception(f'An exception occurred: {e}')
            print(f"An exception occurred, see {os.path.join(save_folder, 'error.log')}")

        else:
            # Remove the file handler to avoid creating an empty log file
            logger.removeHandler(file_handler)
            os.remove(os.path.join(save_folder, 'error.log'))


if __name__ == "__main__":
    main()
