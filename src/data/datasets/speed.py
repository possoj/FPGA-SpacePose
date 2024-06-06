"""
Copyright (c) 2024 Julien Posso
"""

import os
import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

from src.data.utils import seed_worker, SPEDataset, CustomRotation
from src.spe.spe_utils import SPEUtils


class Camera:
    """Utility class for accessing camera parameters"""

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)


def import_speed(
    spe_utils: SPEUtils,
    path: str,
    batch_size: int = 1,
    img_size: Tuple[int, int] = (240, 240),
    rot_augment: bool = False,
    other_augment: bool = False,
    shuffle: bool = False,
    seed: int = 1001,
) -> Tuple[dict, dict]:
    """
    Import SPEED dataset: https://zenodo.org/record/6327547

    Args:
        spe_utils: SPEUtils object.
        path: Path to the dataset.
        batch_size: Batch size for the dataloaders.
        img_size: Desired size of the images after resizing.
        rot_augment: Flag indicating whether to perform rotation augmentation.
        other_augment: Flag indicating whether to perform data augmentation.
        shuffle: Flag indicating whether to shuffle data.
        seed: Seed for reproducibility.

    Returns:
        A tuple containing two dictionaries:
        1. A dictionary containing the dataloaders for different splits of the dataset.
        2. A dictionary defining the split structure for the dataset, with keys 'train' and 'eval'.
    """
    # Reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    g = torch.Generator().manual_seed(seed)

    default_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # Probability of applying rotation augmentation if rot_augment is True
    rot_probability = 0.5
    # Maximum rotation magnitude in degrees for augmentation
    rot_max_magnitude = 50.0

    if other_augment:
        train_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])
    else:
        train_transforms = default_transforms

    rot_transform = CustomRotation(spe_utils, rot_probability, rot_max_magnitude) if rot_augment else None

    # No need to import test and real test as we cannot submit the NN predictions on ESA website anymore
    datasets = {
        'train': SPEDataset(spe_utils, train_transforms, rot_transform,
                            os.path.join(path, 'images', 'train'), os.path.join(path, 'train_no_valid.json')),
        'valid': SPEDataset(spe_utils, train_transforms, rot_transform,
                            os.path.join(path, 'images', 'train'), os.path.join(path, 'valid.json')),
        'real': SPEDataset(spe_utils, train_transforms, rot_transform,
                            os.path.join(path, 'images', 'real'), os.path.join(path, 'real.json')),
    }

    # Set number of workers to zero if debug mode
    n_workers = 0 if sys.gettrace() else 16

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            worker_init_fn=seed_worker,
            generator=g
        ) for x in datasets.keys()
    }

    split = {
        'train': ('train', 'valid', 'real'),
        'eval': ('valid', 'real'),
    }

    return dataloaders, split


def import_speed_camera():
    """Import SPEED camera settings"""
    return Camera()