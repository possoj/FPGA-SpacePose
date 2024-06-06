import os
import shutil
from src.modeling.model import save_bit_width
from src.config.train.config import load_config, save_config

network_bit_width = [

    # Parameters bit-width

    # Inverted residual bloc
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(6, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(4, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(3, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(2, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(1, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (6, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (4, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (3, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (2, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (1, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (6,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (4,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (3,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (2,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (1,)] for _ in range(17)]
    },

    # First conv
    {
        'image': 8,
        'first_conv': (6, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (4, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (3, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (2, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (1, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # Last conv
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (6, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (4, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (3, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (2, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (1, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # Fully connected
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (6, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (4, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (3, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (2, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (1, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # Activations

    # Image
    {
        'image': 6,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 4,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 3,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # Inverted residual activations
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 6), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 4), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 3), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 2), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 1), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 6), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 4), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 3), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 2), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 1), (8,)] for _ in range(17)]
    },

    # First conv activations

    {
        'image': 8,
        'first_conv': (8, 6),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 4),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 3),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 2),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 1),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # last conv activation
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 6),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 4),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 3),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 2),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 1),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # shared activation
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 6,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 4,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 3,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # Pooling bit-width
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 6,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 4,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 8),
        'shared_act': 8,
        'pooling': 3,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    # FC bias
    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 6),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 4),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

    {
        'image': 8,
        'first_conv': (8, 8),
        'last_conv': (8, 8),
        'fully_connected': (8, 3),
        'shared_act': 8,
        'pooling': 8,
        'inverted_residual': [[(8, 8), (8, 8), (8,)] for _ in range(17)]
    },

]

inverted_residual_bw = []
# 1st 1x1 conv weights (expansion)
for x in range(16):
    inverted_residual_bw.append(
        [[(None, None), (8, 8), (8,)]] + \
        [[(1, 8), (8, 8), (8,)] if x == y else [(8, 8), (8, 8), (8,)] for y in range(16)]
    )

# Depthwise conv, weights
for x in range(17):
    inverted_residual_bw.append([[(8, 8), (1, 8), (8,)] if x == y else [(8, 8), (8, 8), (8,)] for y in range(17)])

# 2nd 1x1 conv weights (projection)
for x in range(17):
    inverted_residual_bw.append([[(8, 8), (8, 8), (1,)] if x == y else [(8, 8), (8, 8), (8,)] for y in range(17)])

# 1st 1x1 conv activations (expansion)
for x in range(16):
    inverted_residual_bw.append(
        [[(None, None), (8, 8), (8,)]] + \
        [[(8, 3), (8, 8), (8,)] if x == y else [(8, 8), (8, 8), (8,)] for y in range(16)]
    )

# Depthwise conv, activations
for x in range(17):
    inverted_residual_bw.append([[(8, 8), (8, 3), (8,)] if x == y else [(8, 8), (8, 8), (8,)] for y in range(17)])

for bw in inverted_residual_bw:
    network_bit_width.append(
        {
            'image': 8,
            'first_conv': (8, 8),
            'last_conv': (8, 8),
            'fully_connected': (8, 8),
            'shared_act': 8,
            'pooling': 8,
            'inverted_residual': bw
        }
    )


if __name__ == '__main__':

    exp_root = 'src/config/train/bit_width_experiment'
    # Remove previous experiment configs
    if os.path.exists(os.path.join(exp_root, 'configs')):
        shutil.rmtree(os.path.join(exp_root, 'configs'))

    cfg = load_config(os.path.join(exp_root, 'config.yaml'))
    for idx, bit_width in enumerate(network_bit_width):
        exp_folder = os.path.join(exp_root, 'configs', f'exp_{idx}')
        os.makedirs(exp_folder, exist_ok=True)
        save_bit_width(exp_folder, bit_width, 'bit_width.json')
        save_config(cfg, os.path.join(exp_folder, 'config.yaml'))
