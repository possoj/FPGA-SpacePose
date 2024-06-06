"""
Copyright (c) 2024 Julien Posso
"""

import os

import torch
from qonnx.core.modelwrapper import ModelWrapper
from src.tools.utils import RunningAverage
from src.config.build.config import load_config, save_config
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model, save_model
from src.boards.boards_cfg import import_board
from src.finn.spe_finn import SPEFinn
from src.finn.finn_deploy import RemotePynq


def main():

    experiment_name = input('Select the experiment name (in the experiments/build folder) you want to deploy on a board: ')
    experiment_path = os.path.join('experiments', 'build', experiment_name)
    print(f'Loading experiment {experiment_path}')
    assert os.path.exists(experiment_path), f'path {experiment_path} does not exists'
    local_deploy_path = os.path.join(experiment_path, 'finn_output', 'deploy')
    assert os.path.exists(local_deploy_path), f'path {local_deploy_path} does not exists'

    config = load_config(os.path.join(experiment_path, 'config.yaml'))

    camera = load_camera(config.DATA.PATH)

    spe_utils = SPEUtils(camera, config.MODEL.HEAD.ORI, config.MODEL.HEAD.N_ORI_BINS_PER_DIM,
                         config.DATA.ORI_SMOOTH_FACTOR, config.MODEL.HEAD.POS)

    rot_augment = False
    other_augment = False
    data_shuffle = True
    seed = 1001

    data, split = load_dataset(spe_utils, config.DATA.PATH, config.DATA.BATCH_SIZE, config.DATA.IMG_SIZE,
                               rot_augment, other_augment, data_shuffle, seed)

    params_path = os.path.join(experiment_path, 'model', 'parameters.pt')
    bit_width_path = os.path.join(experiment_path, 'model', 'bit_width.json')
    manual_copy = False

    model_brevitas, bit_width = import_model(
        data, config.MODEL.BACKBONE.NAME, config.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy, residual=config.MODEL.BACKBONE.RESIDUAL, quantization=config.MODEL.QUANTIZATION,
        ori_mode=config.MODEL.HEAD.ORI, n_ori_bins_per_dim=config.MODEL.HEAD.N_ORI_BINS_PER_DIM
    )
    model_brevitas.eval()  # Important!!

    board = import_board(config.FINN.BOARD.NAME)

    remote_path = os.path.join(config.FINN.BOARD.DEPLOYMENT_FOLDER, experiment_name)
    freq_mhz = float(input('Enter the clock frequency in MHz (info on the post_route_timing.rpt report), '
                           'e.g.: 187.512: '))

    pynq = RemotePynq(local_deploy_path, remote_path, board)
    pynq.modify_deploy_folder(freq_mhz)
    pynq.deploy_to_pynq()

    dataflow_model = ModelWrapper(
        os.path.join(experiment_path, 'finn_output', 'intermediate_models', 'dataflow_parent.onnx')
    )
    spe_finn = SPEFinn(pynq, model_brevitas, data[split['eval'][0]], spe_utils, dataflow_model)

    # img, pose = next(iter(data['valid']))
    # score_finn, score_torch = spe_finn.predict_and_compare(img, pose, print_results=True)

    running_avg_finn = RunningAverage(keys=('esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'))
    running_avg_torch = RunningAverage(keys=('esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'))

    for i, (img, true_pose) in enumerate(data[split['eval'][0]]):
        print(f'\n\n\nImage {i}:')
        score_finn, score_torch = spe_finn.predict_and_compare(img['torch'], true_pose, print_results=True)
        running_avg_finn.update(score_finn)
        running_avg_torch.update(score_torch)
        print(f"AVG FINN: {running_avg_finn.get_multiple(keys=('esa_score', 'ori_error', 'pos_error'))}")
        print(f"AVG TORCH: {running_avg_torch.get_multiple(keys=('esa_score', 'ori_error', 'pos_error'))}")
        # Stop experiment before the end
        if i == 0:
            break

    spe_finn.throughput_test()


if __name__ == "__main__":
    main()