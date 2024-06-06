"""
Copyright (c) 2024 Julien Posso
"""

import random
import copy
import json
import os
import re

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BrightnessContrast(object):
    """ Adjust brightness and contrast of the image in a fashion of
        OpenCV's convertScaleAbs, where

        newImage = alpha * image + beta

        image: torch.Tensor image (0 ~ 1)
        alpha: multiplicative factor
        beta:  additive factor (0 ~ 255)
    """
    def __init__(self, alpha=(0.5, 2.0), beta=(-25, 25)):
        self.alpha = torch.tensor(alpha).log()
        self.beta  = torch.tensor(beta)/255

    def __call__(self, image):
        # Contrast - multiplicative factor
        loga = torch.rand(1) * (self.alpha[1] - self.alpha[0]) + self.alpha[0]
        a = loga.exp()

        # Brightness - additive factor
        b = torch.rand(1) * (self.beta[1]  - self.beta[0])  + self.beta[0]

        # Apply
        image = torch.clamp(a*image + b, 0, 1)

        return image


class GaussianNoise(object):
    """ Add random Gaussian white noise

        image: torch.Tensor image (0 ~ 1)
        std:   noise standard deviation (0 ~ 255)
    """
    def __init__(self, std=25):
        self.std = std/255

    def __call__(self, image):
        noise = torch.randn(image.shape, dtype=torch.float32) * self.std
        image = torch.clamp(image + noise, 0, 1)
        return image


class AddGaussianNoise(object):
    """
    Add Gaussian noise to a tensor.

    Args:
        mean (float): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """Apply Gaussian noise to a tensor"""
        return torch.abs(tensor + torch.randn(tensor.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomRotation(object):
    """
    A custom transformation class for rotating images and corresponding pose data in spacecraft pose estimation tasks.

    This transformation randomly applies rotation to the input image and updates the orientation and position data accordingly.

    Args:
        spe_utils (SPEUtils): The SPEUtils instance used for performing image and pose transformations.
        rot_probability (float, optional): The probability of applying a rotation to the image and pose. Default is 0.5.
        rot_max_magnitude (float, optional): The maximum rotation magnitude in degrees. Default is 50.0.
    """

    def __init__(self, spe_utils, rot_probability: float = 0.5, rot_max_magnitude: float = 50.0):
        self.spe_utils = spe_utils
        self.rot_probability = rot_probability
        self.rot_max_magnitude = rot_max_magnitude

    def __call__(self, image, pose):
        """
        Apply the custom transformation to an image and its corresponding pose data.

        This method randomly decides whether to apply a rotation based on `rot_probability`. If rotation is applied,
        it uses the `rotate_image` method from the `spe_utils` instance to rotate the image and update the pose data.

        Args:
            image (PIL.Image or numpy.ndarray): The input image to be transformed.
            pose (dict): The pose data associated with the image. Should contain 'ori' for orientation and 'pos' for position.

        Returns:
            tuple: A tuple containing the transformed image and updated pose data as a dictionary.
        """
        ori = pose['ori']
        pos = pose['pos']

        # Randomly decide whether to apply rotation
        dice = np.random.rand()
        if dice >= self.rot_probability:
            # Apply rotation transformation
            image, ori, pos = self.spe_utils.rotate_image(image, ori, pos, self.rot_max_magnitude)

        # Update pose data
        pose_updated = {'ori': ori, 'pos': pos}
        return image, pose_updated


def get_image_number(path_and_pose):
    image_name = os.path.basename(path_and_pose[0])
    numbers_only = re.sub(r'[^0-9]', '', image_name)
    return int(numbers_only)


class SPEDataset(Dataset):
    """
    A dataset class for spacecraft pose estimation tasks, capable of applying transformations to images and their corresponding pose labels.

    The class handles loading and transforming images and labels for training and evaluation in spacecraft pose estimation models.

    Args:
        spe_utils (SPEUtils): The SPEUtils instance used for encoding orientation and other utility operations.
        transform (callable, optional): Standard PyTorch transform to be applied to each image.
        rot_transform (CustomTransform, optional): Custom rotation transform that acts on both images and their corresponding pose data.
        images_path (str, optional): The directory path where the images are stored. Default is "../datasets/images".
        labels_path (str, optional): The file path of the JSON file containing the labels. Default is "../datasets/labels.json".
    """

    def __init__(self, spe_utils, transform=None, rot_transform=None, images_path="../datasets/images",
                 labels_path="../datasets/labels.json"):
        self.spe_utils = spe_utils
        self.transform = transform
        self.rot_transform = rot_transform

        # Load labels from JSON file
        with open(labels_path, 'r') as f:
            target_list = json.load(f)

        # Determine quaternion name based on dataset
        q_name = 'q_vbs2tango_true' if 'speed_plus' in images_path else 'q_vbs2tango'

        # Organize data into a dictionary with image paths, orientations, and positions
        if 'bbox' in list(target_list[0].keys()):
            # If bounding box available in the dataset
            self.data = {
                os.path.join(images_path, target['filename']): {
                    'ori': torch.tensor(target[q_name]), 'pos': torch.tensor(target['r_Vo2To_vbs_true']),
                    'bbox': torch.tensor(target['bbox'])
                } for target in target_list
            }
        else:
            self.data = {
                os.path.join(images_path, target['filename']): {
                    'ori': torch.tensor(target[q_name]), 'pos': torch.tensor(target['r_Vo2To_vbs_true'])
                } for target in target_list
            }

        # Sort images by filename for consistency, especially in video sequences
        self.data = dict(sorted(self.data.items(), key=get_image_number))

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image and its corresponding pose data
        image_path, target = copy.deepcopy(list(self.data.items())[idx])
        image = Image.open(image_path).convert("RGB")
        img = {
            'original': torch.tensor(np.array(image)),
            'torch': None,
            'path': image_path
        }

        # Apply custom rotation transform if defined
        if self.rot_transform:
            image, target = self.rot_transform(image, target)

        # Apply standard image transformation if defined
        if self.transform:
            image = self.transform(image)

        img['torch'] = copy.deepcopy(image)

        # Encode orientation for classification tasks
        if self.spe_utils.ori_mode == 'classification':
            target['ori_original'] = target['ori']
            target['ori'] = self.spe_utils.encode_ori(target['ori'])

        return img, target


def seed_worker(worker_id):
    """This function is used as the `worker_init_fn` parameter in `torch.utils.data.DataLoader` to set the random seed
    for each worker process in a way that ensures reproducibility"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
