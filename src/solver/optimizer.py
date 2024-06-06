"""
Copyright (c) 2024 Julien Posso
"""

import torch
from typing import Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

def import_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 0.01,
    optimizer: str = 'SGD',
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    scheduler: str = 'MultiStepLR',
    milestones: Tuple[int, ...] = (5, 15),
    gamma: float = 0.1,
    verbose: bool = True
) -> Tuple[Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Import and configure an optimizer and scheduler for training a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to optimize.
        learning_rate (float): The learning rate for the optimizer (default: 0.01).
        optimizer (str): The optimizer algorithm to use, either 'SGD' or 'Adam' (default: 'SGD').
        momentum (float): The momentum factor for SGD optimizer (default: 0.9).
        weight_decay (float): The weight decay for the optimizer (default: 0.0).
        scheduler (str): The scheduler algorithm to use, either 'OnPlateau' or 'MultiStepLR' (default: 'MultiStepLR').
        milestones (Tuple[int, ...]): The milestones for the MultiStepLR scheduler (default: (5, 15)).
        gamma (float): The gamma factor for the scheduler (default: 0.1).
        verbose (bool): Whether to print scheduler updates (default: True).

    Returns:
        Tuple[Optimizer, torch.optim.lr_scheduler]: The configured optimizer and scheduler.

    Raises:
        AssertionError: If the optimizer or scheduler argument is invalid.
    """

    assert optimizer in ('SGD', 'Adam')
    assert scheduler in ('OnPlateau', 'MultiStepLR')

    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if scheduler == 'OnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=gamma,
            patience=milestones[0],
            verbose=verbose
        )
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones,
            gamma=gamma,
            verbose=verbose
        )

    return optimizer, scheduler

