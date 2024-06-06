"""
Copyright (c) 2024 Julien Posso
"""

import sys
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch

from src.spe.spe_utils import SPEUtils
from src.tools.utils import RunningAverage


def evaluation(
    model: torch.nn.Module,
    dataloader: Dict[str, torch.utils.data.DataLoader],
    spe_utils: SPEUtils,
    split: Tuple[str, ...] = ('test', 'valid'),
    device: torch.device = torch.device('cpu')
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """
    Evaluate the model on specified splits and record scores, and errors.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (Dict[str, torch.utils.data.DataLoader]): A dictionary containing the data loaders for different splits.
        spe_utils (SPEUtils): An instance of the SPEUtils class.
        split (Tuple[str, ...], optional): A tuple containing the split names (train, valid, ...). Defaults to ('test', 'valid').
        device (torch.device, optional): The device to run the evaluation on. Defaults to 'cpu'.

    Returns:
        Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]: A tuple containing the recorded scores and errors.
    """

    model.to(device)
    model.eval()

    # Record loss/score/error during evaluation
    rec_score = {x: {'ori': [], 'pos': [], 'esa': []} for x in split}
    rec_error = {x: {'ori': [], 'pos': []} for x in split}

    for phase in split:
        running_avg = RunningAverage(keys=('esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'))

        # Batch loop
        loop = tqdm(dataloader[phase], desc=f"Eval - {phase}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    ncols=130, file=sys.stdout)

        for images, targets in loop:
            # Send images and targets to GPU memory if device is CUDA
            images = images['torch'].to(device)
            targets['ori'], targets['pos'] = targets['ori'].to(device), targets['pos'].to(device)

            with torch.no_grad():
                ori, pos = model(images)
                pred_pose, _ = spe_utils.process_output_nn(ori, pos)

            # Compute evaluation metrics
            eval_metrics = spe_utils.get_score(targets, pred_pose)
            running_avg.update(eval_metrics, images.size(0))

            # Update progress bar
            loop.set_postfix(running_avg.get_multiple(keys=('ori_error', 'pos_error', 'esa_score')))

        # Store scores and errors for printing
        rec_score[phase]['ori'].append(running_avg.get('ori_score'))
        rec_score[phase]['pos'].append(running_avg.get('pos_score'))
        rec_score[phase]['esa'].append(running_avg.get('esa_score'))
        rec_error[phase]['ori'].append(running_avg.get('ori_error'))
        rec_error[phase]['pos'].append(running_avg.get('pos_error'))

    return rec_score, rec_error
