"""
Copyright (c) 2024 Julien Posso
"""

import torch
from torch.nn.modules import Module
from typing import Union


class POSREGLoss(Module):
    """
    Loss function used for the Position branch in regression configuration.

    Args:
        reduction (str): Type of reduction to apply to the loss. Must be 'mean' or 'sum'.
        norm_distance (bool): Whether to normalize the distance. Default is True.
    """
    def __init__(self, reduction: str = 'mean', norm_distance: bool = True):
        super().__init__()
        assert reduction in ('mean', 'sum'), "reduction must be 'mean' or 'sum'"
        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.norm_distance = norm_distance

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            pred (torch.Tensor): Estimated position.
            target (torch.Tensor): Ground truth position.

        Returns:
            torch.Tensor: Position regression loss.
        """
        loss = torch.linalg.norm(pred - target)
        if self.norm_distance:
            loss = loss / torch.linalg.norm(target)
        return self.reduction(loss)


class ORIREGLoss(Module):
    """
    Loss function used for the Orientation branch in regression configuration.

    Args:
        reduction (str): Type of reduction to apply to the loss. Must be 'mean' or 'sum'.
        norm_distance (bool): Whether to normalize the distance. Default is True.
    """
    def __init__(self, reduction: str = 'mean', norm_distance: bool = True):
        super(ORIREGLoss, self).__init__()

        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.norm_distance = norm_distance

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, target_pos: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            pred (torch.Tensor): Estimated orientation.
            target (torch.Tensor): Ground truth orientation.
            target_pos (torch.Tensor, optional): Ground truth position. Default is None.

        Returns:
            torch.Tensor: Orientation regression loss.
        """
        inter_sum = torch.abs(torch.sum(pred * target, dim=1, keepdim=True))
        # Scaling down intermediate sum to avoid nan of arccos(x) when x > 1. See scoring for more details
        if True in inter_sum[inter_sum > 1.01]:
            raise ValueError("Error while computing orientation Loss")

        inter_sum[inter_sum > 1] = 0
        loss = torch.arccos(inter_sum)
        if self.norm_distance:
            loss = loss / torch.linalg.norm(target_pos, dim=1, keepdim=True)
        return self.reduction(loss)


class ORICLASSLoss(Module):
    """
    Loss function used for the Orientation branch in classification configuration.

    Args:
        reduction (str): Type of reduction to apply to the loss. Must be 'mean' or 'sum'.
    """
    def __init__(self, reduction: str = 'mean'):
        super(ORICLASSLoss, self).__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum

    def forward(self, pred_ori: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            pred_ori (torch.Tensor): Predicted orientation probabilities.
            target (torch.Tensor): Ground truth orientation.

        Returns:
            torch.Tensor: Orientation classification loss.
        """
        loss = self.reduction(torch.sum(-(target * torch.log(pred_ori)), dim=1))
        if True in torch.isnan(loss):
            raise ValueError("Error while computing orientation Loss")
        return loss


def import_loss(ori_mode: str = 'classification', pos_mode: str = 'regression', norm_distance: bool = False):
    """
    Imports the appropriate loss functions based on the orientation and position modes.

    Args:
        ori_mode (str): Orientation mode. Must be 'regression' or 'classification'.
        pos_mode (str): Position mode. Must be 'regression'.
        norm_distance (bool): Whether to normalize the distance. Default is False.

    Returns:
        tuple: Tuple containing the orientation loss and position loss functions.
    """
    if ori_mode == 'regression':
        ori_criterion = ORIREGLoss(norm_distance=norm_distance)
    elif ori_mode == 'classification':
        ori_criterion = ORICLASSLoss()
    else:
        raise ValueError('Orientation estimation type has to be either \'Regression\' or \'Classification\'')

    if pos_mode == 'regression':
        pos_criterion = POSREGLoss(norm_distance=norm_distance)
    else:
        raise ValueError('Position estimation type must be \'Regression\' (Classification not implemented)')

    return ori_criterion, pos_criterion


class SPELoss:
    """
    Satellite Pose Estimation Loss class.

    Args:
        ori_mode (str): Orientation mode. Must be 'regression' or 'classification'.
        pos_mode (str): Position mode. Must be 'regression'.
        beta (float): Weight parameter for the orientation loss. Default is 1.
        norm_distance (bool): Whether to normalize the distance for orientation loss. Default is False.
    """
    def __init__(
        self, ori_mode: str, pos_mode: str, beta: float = 1, norm_distance: bool = False
    ):
        self.ori_mode = ori_mode
        self.pos_mode = pos_mode
        self.beta = beta
        self.ori_criterion, self.pos_criterion = import_loss(ori_mode, pos_mode, norm_distance)

    def compute_loss(self, ori: torch.Tensor, pos: torch.Tensor, target: dict) -> torch.Tensor:
        """
        Computes the overall loss.

        Args:
            ori (torch.Tensor): Estimated orientation.
            pos (torch.Tensor): Estimated position.
            target (dict): Dictionary containing the ground truth orientation and position.

        Returns:
            torch.Tensor: Overall loss.
        """
        if self.ori_mode == 'regression':
            ori_loss = self.ori_criterion(ori, target['ori'], target['pos'])
        else:
            ori_loss = self.ori_criterion(ori, target['ori'])

        pos_loss = self.pos_criterion(pos, target['pos'])
        loss = self.beta * ori_loss + pos_loss
        return loss
