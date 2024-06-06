import os
import glob
import logging

from src.config.train.config import load_config, save_config
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model, save_model, save_bit_width
from src.solver.loss import SPELoss
from src.solver.optimizer import import_optimizer
from src.tools.training import train
from src.tools.utils import select_device, set_seed, prepare_directories
from src.tools.utils import save_score_error
from src.tools.evaluation import evaluation


def main():
    """Main function to run the experiments."""

    # Configure the logging settings at the beginning of the main function
    logging.basicConfig(level=logging.ERROR)

    seed = 1001
    cfg_root = os.path.join('src', 'config', 'train')
    cfg_template = 'exp_'

    set_seed(seed)
    device = select_device()

    # List experiments to run (both files and directories that start with config_ in the config folder)
    experiment_list = [x for x in os.listdir(cfg_root) if x.startswith(cfg_template)]

    # Run multiple experiments
    for exp in experiment_list:
        exp_path = os.path.join(cfg_root, exp)

        # Pytorch YAML config are in the config folder. The YAML file starts with "cfg_template"
        if os.path.isfile(exp_path):
            cfg = load_config(exp_path)
            exp_id, _ = os.path.splitext(exp.replace(cfg_template, ''))
            bit_width_path = None
        # Brevitas YAML config should come with a JSON file that parametrize the bit-width
        else:
            yaml_file = glob.glob(exp_path + '/*.yaml')
            assert len(yaml_file) == 1, f"Should have only one YAML file in the directory {exp_path}"
            cfg = load_config(yaml_file[0])
            exp_id = exp.replace(cfg_template, '')
            json_file = glob.glob(exp_path + '/*.json')
            assert len(json_file) == 1, f"Should have only one JSON file in the directory {exp_path}"
            bit_width_path = json_file[0]

        if os.path.exists(os.path.join('experiments', 'train', f'{cfg_template}{exp_id}')):
            print(f"\nExperiment {os.path.join('experiments', 'train', f'{cfg_template}{exp_id}')} already exist\n\n")
            continue

        save_folder = prepare_directories('experiments', 'train', f'{cfg_template}{exp_id}',
                                          ('model', 'results', 'tensorboard'))
        print(f"\nLoad config file {os.path.join(cfg_root, exp)}")
        print(f"\nResults will be saved to {save_folder}\n")

        camera = load_camera(cfg.DATA.PATH)
        spe_utils = SPEUtils(camera, cfg.MODEL.HEAD.ORI, cfg.MODEL.HEAD.N_ORI_BINS_PER_DIM,
                             cfg.DATA.ORI_SMOOTH_FACTOR, cfg.MODEL.HEAD.POS)

        data, split = load_dataset(spe_utils, cfg.DATA.PATH, cfg.DATA.BATCH_SIZE, cfg.DATA.IMG_SIZE,
                                   cfg.DATA.ROT_AUGMENT, cfg.DATA.OTHER_AUGMENT, cfg.DATA.SHUFFLE, seed)

        model, model_bit_width = import_model(
            data, cfg.MODEL.BACKBONE.NAME, cfg.MODEL.HEAD.NAME, cfg.MODEL.PRETRAINED_PATH, bit_width_path,
            cfg.MODEL.MANUAL_COPY, residual=cfg.MODEL.BACKBONE.RESIDUAL, quantization=cfg.MODEL.QUANTIZATION,
            ori_mode=cfg.MODEL.HEAD.ORI, n_ori_bins_per_dim=cfg.MODEL.HEAD.N_ORI_BINS_PER_DIM
        )

        print(f"Number of trainable parameters in the model:"
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

        spe_loss = SPELoss(cfg.MODEL.HEAD.ORI, cfg.MODEL.HEAD.POS, beta=1, norm_distance=True)
        optimizer, scheduler = import_optimizer(
            model, cfg.TRAIN.LR, cfg.TRAIN.OPTIM, cfg.TRAIN.MOMENTUM, cfg.TRAIN.DECAY,
            cfg.TRAIN.SCHEDULER, cfg.TRAIN.MILESTONES, cfg.TRAIN.GAMMA, verbose=True
        )

        tensorboard_cfg = {
            'log_folder': os.path.join(save_folder, 'tensorboard'),
            'save_model': False, # will raise warnings with brevitas models if True
            'save_parameters': False,
        }

        # Save config
        save_config(cfg, os.path.join(save_folder, 'config.yaml'))
        if 'brevitas' in cfg.MODEL.BACKBONE.NAME or 'brevitas' in cfg.MODEL.HEAD.NAME:
            save_bit_width(os.path.join(save_folder, 'model'), model_bit_width, 'bit_width.json')

        # Create a logger specific to the experiment
        logger = logging.getLogger(f'Experiment_{exp_id}')
        file_handler = logging.FileHandler(os.path.join(save_folder, 'error.log'))
        file_handler.setLevel(logging.ERROR)
        logger.addHandler(file_handler)

        try:
            # Training
            model, loss, score, error = train(
                model, data, cfg.TRAIN.N_EPOCH, spe_utils, spe_loss, scheduler,
                optimizer, tensorboard_cfg, split['train'], device, clip_batchnorm=True, amp=False
            )
            save_score_error(score, error, path=os.path.join(save_folder, 'results'), name='train.xlsx')

            # Evaluation
            score, error = evaluation(model, data, spe_utils, split['eval'], device)
            save_score_error(score, error, path=os.path.join(save_folder, 'results'), name='eval.xlsx')

            # Save model (without bit-width because it is saved before)
            save_model(os.path.join(save_folder, 'model'), model)

        except Exception as e:
            # Log the exception
            logger.exception(f'An exception occurred: {e}')
            print(f"An exception occurred, see {os.path.join(save_folder, 'error.log')}")

        else:
            # Remove the file handler to avoid creating an empty log file
            logger.removeHandler(file_handler)
            os.remove(os.path.join(save_folder, 'error.log'))


if __name__ == '__main__':
    main()
