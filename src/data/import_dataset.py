"""
Copyright (c) 2024 Julien Posso
"""

import os.path
from src.data.datasets.speed import import_speed, import_speed_camera
from src.data.datasets.speed_plus import import_speed_plus, import_speed_plus_camera
from src.spe.spe_utils import SPEUtils
from typing import Tuple


def load_dataset(
        spe_utils: SPEUtils,
        path: str = '../dataset/path/',
        batch_size: int = 1,
        img_size: Tuple[int, int] = (240, 240),
        rot_augment: bool = False,
        other_augment: bool = False,
        shuffle: bool = False,
        seed: int = 1001,
) -> Tuple[dict, dict]:
    """
    Import Spacecraft POSE estimation datasets.

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
        Tuple[dict, Dict[str, Tuple[str, ...]]]: A tuple containing:
        - data (dict): Dictionary containing the imported dataset (dataloaders).
        - split (Dict[str, Tuple[str, ...]]): Dictionary mapping split names to tuples of subset names.
            Example: {'training': ('train', 'valid', 'real'), 'evaluation': ('valid', 'real')}
    """
    assert os.path.exists(path), f"Dataset path {path} does not exist"

    dataset_name = os.path.split(path)[-1]

    if dataset_name == 'speed':
        data, split = import_speed(spe_utils, path, batch_size, img_size, rot_augment, other_augment, shuffle, seed)
    elif dataset_name == 'speed_plus':
        data, split = import_speed_plus(spe_utils, path, batch_size, img_size, rot_augment, other_augment, shuffle, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return data, split


def load_camera(path: str = '../dataset/path/'):
    """
    Import Spacecraft POSE estimation camera.

    Args:
        path: Path to the dataset.

    Returns:
        camera: Utility class for accessing camera parameters.
    """

    assert os.path.exists(path), f"Dataset path {path} does not exist"

    dataset_name = os.path.split(path)[-1]

    if dataset_name == 'speed':
        camera = import_speed_camera()
    elif dataset_name == 'speed_plus':
        camera = import_speed_plus_camera()
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return camera
