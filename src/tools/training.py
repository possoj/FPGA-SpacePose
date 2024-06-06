"""
Copyright (c) 2024 Julien Posso
"""

import torch
import copy
import sys
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src.solver.loss import SPELoss
from src.spe.spe_utils import SPEUtils
from src.tools.utils import RunningAverage


def train(model: torch.nn.Module,
          dataloader: Dict[str, torch.utils.data.DataLoader],
          n_epochs: int,
          spe_utils: SPEUtils,
          spe_loss: SPELoss,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          optimizer: torch.optim.Optimizer,
          tensorboard: Dict[str, Union[str, bool]],
          split: Tuple[str, ...] = ('train', 'valid'),
          device: torch.device = torch.device('cpu'),
          clip_batchnorm: bool = False,
          amp: bool = False,
          ) -> Tuple[torch.nn.Module, Dict[str, List[float]], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """
    Train the model for a given number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (Dict[str, torch.utils.data.DataLoader]): A dictionary containing the data loaders for different splits.
        n_epochs (int): The number of epochs to train the model.
        spe_utils (SPEUtils): Placeholder for spe_utils type.
        spe_loss (SPELoss): Placeholder for spe_loss type.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        optimizer (torch.optim.Optimizer): The optimizer.
        tensorboard (Dict[str, Union[str, bool]]): the tensorboard configuration.
        split (Tuple[str, ...], optional): A tuple containing the split names (train, valid, ...). Defaults to ('train', 'valid').
        device (torch.device, optional): The device to run the training on. Defaults to 'cpu'.
        clip_batchnorm (bool, optional): Whether to clip batch normalization weights. Defaults to False.
        amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.

    Returns:
        Tuple[torch.nn.Module, Dict[str, List[float]], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]: A tuple containing the trained model,
        the recorded losses, the recorded scores, and the recorded errors.
    """

    assert 'train' in split and 'valid' in split, "The split must contain at least 'train' and 'valid'"

    best_loss = 1e6
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    model.to(device)

    # Record loss/score/error during training
    rec_loss = {x: [] for x in split}
    rec_score = {x: {'ori': [], 'pos': [], 'esa': []} for x in split}
    rec_error = {x: {'ori': [], 'pos': []} for x in split}

    # Create SummaryWriter for TensorBoard
    writer = SummaryWriter(tensorboard['log_folder'])
    # save model with TensorBoard for visualization
    img, _ = next(iter(dataloader['train']))

    if tensorboard['save_model']:
        # will raise warnings with brevitas models if True
        writer.add_graph(model, img['torch'].to(device))
        writer.flush()

    # Epoch loop
    for epoch in range(1, n_epochs + 1):
        for phase in split:
            running_avg = RunningAverage(keys=('loss', 'esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'))

            # Set model to train or eval mode
            model.train() if phase == 'train' else model.eval()
            if phase == 'train':
                print('')

            # batch loop
            loop = tqdm(dataloader[phase], desc=f"Train epoch {epoch} - {phase}",
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=140, file=sys.stdout)

            for images, targets in loop:
                # Send images and targets to GPU memory if device is CUDA
                images = images['torch'].to(device)
                targets['ori'], targets['pos'] = targets['ori'].to(device), targets['pos'].to(device)

                # Runs the forward pass under autocast
                with torch.cuda.amp.autocast(enabled=amp):
                    with torch.set_grad_enabled(phase == 'train'):
                        ori, pos = model(images)
                        pred_pose, ori_loss = spe_utils.process_output_nn(ori, pos)

                    # Compute solver
                    loss = spe_loss.compute_loss(ori_loss, pos, targets)

                # Runs the backward pass
                if phase == 'train':
                    # Init gradient
                    optimizer.zero_grad()
                    # Backpropagation
                    scaler.scale(loss).backward()
                    # Update parameters
                    scaler.step(optimizer)

                    if clip_batchnorm:
                        with torch.no_grad():
                            for layer in model.modules():
                                if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                                    # Clip BN weights in the range [0, 1] for FINN build
                                    # FINN cannot absorb negative Multiplications into MultiThreshold nodes
                                    layer.weight.clamp_(0, 1)

                    # Updates the scale for next iteration.
                    scaler.update()

                # Compute evaluation metrics
                eval_metrics = spe_utils.get_score(targets, pred_pose)
                eval_metrics.update({'loss': loss.item()})
                running_avg.update(eval_metrics, images.size(0))

                # Update progress bar
                loop.set_postfix(running_avg.get_multiple(keys=('loss', 'ori_error', 'pos_error', 'esa_score')))

            # Store solver and score for printing
            rec_loss[phase].append(running_avg.get('loss'))
            rec_score[phase]['ori'].append(running_avg.get('ori_score'))
            rec_score[phase]['pos'].append(running_avg.get('pos_score'))
            rec_score[phase]['esa'].append(running_avg.get('esa_score'))
            rec_error[phase]['ori'].append(running_avg.get('ori_error'))
            rec_error[phase]['pos'].append(running_avg.get('pos_error'))

            running_loss = running_avg.get('loss')

            if phase == 'train':
                # Update learning rate at the end of epoch loop (train)
                scheduler.step(running_loss if isinstance(scheduler, ReduceLROnPlateau) else None)
            elif phase == 'valid':
                # Selecting the best model on validation set (lower solver is better)
                # model selection done on validation
                if running_loss < best_loss:
                    best_model = copy.deepcopy(model.state_dict())
                    best_loss = running_loss
                    best_epoch = epoch

            # Write values to TensorBoard
            for key in ('loss', 'esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'):
                writer.add_scalar(f'{key}/{phase}', running_avg.get(key), epoch)

            if tensorboard['save_parameters']:
                for name, parameter in model.named_parameters():
                    writer.add_histogram(name, parameter, epoch)

    # Load best model
    model.load_state_dict(best_model)
    print('Best epoch:', best_epoch)

    # Close the SummaryWriter
    writer.flush()
    writer.close()

    return model, rec_loss, rec_score, rec_error
