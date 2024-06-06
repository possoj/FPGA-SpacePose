import os
import json
import ast
import warnings

import torch
import torchvision.models

from src.modeling.backbone.small import QSmallBackbone
from src.modeling.backbone.mobilenet_v2 import QMobileNetV2, MobileNetV2, QSmallMobile
from src.modeling.head.ursonet import QURSONetHead, URSONetHead
from src.modeling.common.pytorch_layers import ModelWrapper


def load_bit_width(path: str):
    """Load bit-width configuration from a JSON file.

    Args:
        path (str): The path to the JSON file containing the bit width configuration.

    Returns:
        dict or None: A dictionary representing the loaded bit width configuration, or None if the file is not found.

    Raises:
        FileNotFoundError: If the specified file is not found.

    Warning:
        If the file is not found, a warning is issued, and the default bit width defined in the model code is used.
    """
    try:
        with open(path, 'r') as f:
            content = json.load(f)

        for key, value in content.items():
            if key == 'inverted_residual':
                content[key] = [ast.literal_eval(val) for val in value]
            else:
                content[key] = ast.literal_eval(value)

        return content
    except FileNotFoundError:
        warnings.warn(f'Bit width path {path} not found.\n '
                      f'The default bit_width defined in the code of the model is used')
        return None


def save_bit_width(save_folder: str, bit_width: dict, bit_width_name: str = 'bit_width.json') -> None:
    """
    Save the bit-width dictionary to a JSON file.

    Args:
        save_folder (str): The path to the folder where the file will be saved.
        bit_width (dict): The bit-width dictionary to be saved.
        bit_width_name (str, optional): The name of the bit-width JSON file. Defaults to 'bit_width.json'.
    """
    assert bit_width is not None

    # Convert bit-width values to strings
    str_bit_width = {
        key: str(value) if key != 'inverted_residual' else [str(line) for line in value]
        for key, value in bit_width.items()
    }

    # Save the bit-width dictionary to JSON file
    with open(os.path.join(save_folder, bit_width_name), 'w') as f:
        json.dump(str_bit_width, f, indent=4)


def save_model(save_folder: str, model: torch.nn.Module, bit_width: dict = None,
               model_name: str = 'parameters.pt', bit_width_name: str = 'bit_width.json'):
    """
    Save the model and optional bit-width information.

    Args:
        save_folder (str): The folder where the model and bit-width information will be saved.
        model (torch.nn.Module): The model to be saved.
        bit_width (dict, optional): Dictionary containing bit-width information (default: None).
        model_name (str, optional): Name of the model file (default: 'parameters.pt').
        bit_width_name (str, optional): Name of the bit-width file (default: 'bit_width.json').
    """
    # Save model
    model.cpu()
    model.eval()
    torch.save(model.state_dict(), os.path.join(save_folder, model_name))

    # Save bit-width
    if bit_width is not None:
        save_bit_width(save_folder, bit_width, bit_width_name)


def copy_state_dict(state_dict_1, state_dict_2, act_quant=False):
    """
    Manual copy of state_dict_1 in state_dict_2.
    Why ? Because when copying a state dict to another with load_state_dict, the values of weight are copied only
    when keys are the same in both state_dict, even if strict=False.
    Set act_quant to True only if both state_dict comes from quantized networks and have done one inference before
    """

    state1_keys = list(state_dict_1.keys())
    state2_keys = list(state_dict_2.keys())

    # In these keys, fp32_state_dict = quantized_state_dict
    # Convolutions and FC layers: "weight" and "bias"
    # Batchnorm layers: "running_mean", "running_var", "num_batches_tracked"
    keys = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
    # In a FP32 state_dict, there is no "act_quant" key

    if act_quant:
        keys.append("act_quant")

    for key in keys:
        state1_key = [i for i in state1_keys if key in i]
        state2_key = [i for i in state2_keys if key in i]

        for x in range(len(state1_key)):
            state_dict_2[state2_key[x]] = state_dict_1[state1_key[x]]

    return state_dict_2


def import_model(
        data: dict,
        backbone_name: str,
        head_name: str,
        params_path: str = None,
        bit_width_path: str = None,
        manual_copy: bool = False,
        in_channels: int = 3,
        batchnorm: bool = True,
        residual: bool = True,
        quantization: bool = True,
        ori_mode: str = 'classification',
        n_ori_bins_per_dim: int = 12
    ):
    """Import and check the Brevitas model"""

    # Load bit-width
    if 'brevitas' in backbone_name or 'brevitas'in head_name:
        if params_path is not None:
            assert bit_width_path is not None, 'the bit_width_path of a brevitas model must be specified ' \
                                               'if parameters are loaded'
        bit_width = load_bit_width(bit_width_path)
    else:
        # Pytorch model
        bit_width = None

    # Configure backbone
    backbone_map = {
        'mobilenet_v2_brevitas': QMobileNetV2,
        'mobilenet_v2_pytorch': MobileNetV2,
        'small_brevitas': QSmallBackbone,
        'small_mobile_brevitas': QSmallMobile
    }
    assert backbone_name in backbone_map.keys(), f'backbone {backbone_name} does not exist'

    common_backbone_config = {
        'in_channels': in_channels,
        'batchnorm': batchnorm,
        'residual_connections': residual,
    }

    # Backbone parameters
    last_return_quant = True if 'brevitas' in backbone_name and 'brevitas' in head_name and quantization else False
    backbone_config = {
        'mobilenet_v2_brevitas': {'out_channels': 1280, 'quantization': quantization, 'bit_width': bit_width,
                                  'last_return_quant': last_return_quant},
        'small_mobile_brevitas': {'out_channels': 64, 'quantization': quantization, 'bit_width': bit_width,
                                  'last_return_quant': last_return_quant},
        'small_brevitas': {'out_channels': 32, 'quantization': quantization, 'bit_width': bit_width,
                           'last_return_quant': last_return_quant},
        'mobilenet_v2_pytorch': {'out_channels': 1280},
    }
    backbone_config[backbone_name].update(common_backbone_config)
    backbone = backbone_map[backbone_name](**backbone_config[backbone_name])

    head_map = {
        'ursonet_brevitas': QURSONetHead,
        'ursonet_pytorch': URSONetHead,
    }

    n_ori_outputs = 4 if ori_mode == 'regression' else n_ori_bins_per_dim ** 3
    head_config = {
        'n_feature_maps': backbone_config[backbone_name]['out_channels'],
        'n_ori_outputs': n_ori_outputs,
        'n_pos_outputs': 3,
        'bias': True,
        'dropout_rate': 0.2
    }

    if 'brevitas' in head_name:
        img, _ = next(iter(data[list(data.keys())[0]]))
        img = img['torch']
        img_size = (img.size(2), img.size(3))
        # correspondence table between image size and kernel size.
        avg_pooling_size = {
            (240, 240): (8, 8),
            (240, 384): (8, 12),
            (480, 768): (15, 24)
        }
        assert img_size in avg_pooling_size.keys(), f'You should update the avg_pooling_size with the new image size'
        head_config.update({'pool_kernel': avg_pooling_size[img_size],
                            'quantization': quantization,
                            'bit_width': bit_width})

    head = head_map[head_name](**head_config)

    # Merge Backbone and Head
    model = ModelWrapper(backbone, head)
    # need to run one inference for the quantization parameters to appear in the state dict
    img, _ = next(iter(data[list(data.keys())[0]]))
    _ = model(img['torch'])

    if params_path is not None:
        assert os.path.isfile(params_path), f'parameters not found {params_path}'
        if manual_copy:
            model.load_state_dict(copy_state_dict(torch.load(params_path), model.state_dict()))
        else:
            model.load_state_dict(torch.load(params_path))
    else:
        # Use pretrained parameters of Pytorch backbone on ImageNet
        if 'mobilenet_v2' in backbone_name:
            print(f'Load weights...')
            try:
                model.features.load_state_dict(copy_state_dict(torchvision.models.mobilenet_v2(pretrained=True).
                                                           features.state_dict(), model.features.state_dict()))
            except:
                print('Failed to load pretrained weights')

    return model, bit_width