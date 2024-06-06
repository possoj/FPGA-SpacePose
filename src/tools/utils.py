"""
Copyright (c) 2024 Julien Posso
"""

import os
import random
import string
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class RunningAverage:
    """Class to compute and store running averages of values"""

    def __init__(self, keys: Tuple[str, ...] = ('loss', 'accuracy')):
        """
        Initialize the RunningAverage object.

        Args:
            keys (tuple, optional): Tuple of keys for values to be tracked. Defaults to ('loss', 'accuracy').
        """
        self.running = {x: AverageMeter() for x in keys}

    def update(self, values: Dict[str, float], batch_size: int = 1) -> None:
        """
        Update the running average values with new values.

        Args:
            values (dict): Dictionary of key-value pairs to update the running averages.
            batch_size (int, optional): Batch size for weight adjustment. Defaults to 1.
        """
        for key, value in values.items():
            self.running[key].update(value, batch_size)

    def get_multiple(self, keys: Tuple[str, ...] = ('loss', 'accuracy')) -> Dict[str, float]:
        """
        Get the running average values for multiple keys.

        Args:
            keys (tuple, optional): Tuple of keys to retrieve the running averages. Defaults to ('loss', 'accuracy').

        Returns:
            dict: Dictionary of key-value pairs containing the running averages.
        """
        avg = {x: self.running[x].get_avg() for x in keys}
        return avg

    def get(self, key: str) -> float:
        """
        Get the running average value for a specific key.

        Args:
            key (str): Key to retrieve the running average value.

        Returns:
            float: Running average value for the specified key.

        Raises:
            AssertionError: If the specified key does not exist in the running averages dictionary.
        """
        assert key in self.running.keys(), f"Error: '{key}' not found in {self.running.keys()}"
        return self.running[key].get_avg()


class AverageMeter:
    """Class to compute and store the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        """Reset the average meter values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the average meter with a new value.

        Args:
            val (float): New value to update the average meter.
            n (int, optional): Weight for the value. Defaults to 1.
        """
        self.val = float(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self) -> float:
        """
        Get the current average value.

        Returns:
            float: Current average value.
        """
        return self.avg


def prepare_directories(
        root: str = 'experiments',
        phase: str = 'train',
        name: str = 'exp_0',
        sub_folders: Tuple[str, ...] = ('model', 'results', 'tensorboard')
    ) -> str:
    """Prepare directories for FINN build.

    Args:
        root (str): Root directory for experiments.
        phase (str): Phase of the experiment, either 'train' or 'build'.
        name (str): Name of the experiment folder.
        sub_folders (Tuple[str, ...]): Tuple of sub-folder names to be created.

    Returns:
        str: Experiment folder path.
    """

    assert phase in ('train', 'build')

    exp_folder = os.path.join(root, phase, name)

    if os.path.exists(exp_folder):
        print(f'Path {exp_folder} already exists.')
        answer = None
        while answer not in ('y', 'n'):
            answer = input('Would you like to delete it? (y/n): ')
        if answer == 'y':
            shutil.rmtree(exp_folder)
        else:
            exp_folder = exp_folder + random.choice(string.ascii_letters)
            assert not os.path.exists(exp_folder), f'Error, folder {exp_folder} already exists'

    # Create experiment folder and sub-folders
    os.makedirs(exp_folder)
    for folder in sub_folders:
        os.makedirs(os.path.join(exp_folder, folder))

    return exp_folder


def select_device() -> torch.device:
    """
    Select and return the appropriate device (GPU or CPU) for computation.

    Returns:
        torch.device: Selected device for computation.
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            gpu_nb = input("Select GPU number:")
            device = torch.device(f"cuda:{gpu_nb}")
        else:
            device = torch.device("cuda:0")
            print(f"Device used: {device}\n")
    else:
        device = torch.device("cpu")
        print(f"Device used: {device}\n")
    return device


def set_seed(seed: int = 1) -> None:
    """
    Set manual seeds for reproducibility.

    Args:
        seed (int): The seed value to set.

    See Also:
        - PyTorch documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    torch.manual_seed(seed)
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.use_deterministic_algorithms(False)  # TODO: true ??? Issues when true with forward pass on a quantized model

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"


def save_score_error(
    score: Optional[Dict[str, Dict[str, list]]] = None,
    error: Optional[Dict[str, Dict[str, list]]] = None,
    path: str = "results/",
    name: str = "score.xlsx"
    ) -> None:
    """
    Save score and/or error data to an Excel (xlsx) file.

    Args:
        score (Optional[Dict[str, Dict[str, list]]]): Dictionary containing score data with splits as keys. Defaults to None.
        error (Optional[Dict[str, Dict[str, list]]]): Dictionary containing error data with splits as keys. Defaults to None.
        path (str): Path where the file will be saved. Defaults to "results/".
        name (str): Name of the Excel file. Defaults to "score.xlsx".
    """
    writer = pd.ExcelWriter(os.path.join(path, name), engine='xlsxwriter')

    if score is not None:
        for split in list(score.keys()):
            pd.DataFrame(data=score[split]).to_excel(writer, sheet_name=f'score_{split}')

    if error is not None:
        for split in list(error.keys()):
            pd.DataFrame(data=error[split]).to_excel(writer, sheet_name=f'error_{split}')

    writer.save()
    writer.close()


def load_score_error(path: str = "results/", name: str = "score.xlsx") -> tuple:
    """
    Load score and error data from an Excel (xlsx) file.

    Args:
        path (str): Path where the file is located. Defaults to "results/".
        name (str): Name of the Excel file. Defaults to "score.xlsx".

    Returns:
        tuple: A tuple containing the loaded score and error data dictionaries.
    """
    score = {}
    error = {}

    file_path = os.path.join(path, name)
    excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

    for sheet_name, data in excel_data.items():
        if sheet_name.startswith('score_'):
            split = sheet_name.split('score_')[1]
            score[split] =  data.iloc[0:, 1:].to_dict(orient='list')
        elif sheet_name.startswith('error_'):
            split = sheet_name.split('error_')[1]
            error[split] =  data.iloc[0:, 1:].to_dict(orient='list')

    return score, error
