import os.path
from yacs.config import CfgNode as ConfigurationNode

_C = ConfigurationNode()

_C.FINN = ConfigurationNode()
_C.FINN.BOARD = ConfigurationNode()
_C.FINN.BOARD.NAME = 'ZCU104'
_C.FINN.BOARD.DEPLOYMENT_FOLDER = '/home/xilinx/spacecraft_pose_estimation'

_C.FINN.ACCEL = ConfigurationNode()
_C.FINN.ACCEL.TARGET_CYCLES_PER_FRAME = 800_000
_C.FINN.ACCEL.CLK_PERIOD_NS = 5

_C.FINN.FIFO = ConfigurationNode()
_C.FINN.FIFO.SPLIT_LARGE = True
_C.FINN.FIFO.AUTO_DEPTH = True
_C.FINN.FIFO.SIZING_METHOD = 'largefifo_rtlsim'
_C.FINN.FIFO.RTL_SIM = True

_C.MODEL = ConfigurationNode()
_C.MODEL.PATH = 'models/fp32_12bins_model.pt'
_C.MODEL.MANUAL_COPY = False
_C.MODEL.QUANTIZATION = True

_C.MODEL.BACKBONE = ConfigurationNode()
_C.MODEL.BACKBONE.NAME = 'mobilenet_v2_pytorch'
_C.MODEL.BACKBONE.RESIDUAL = True

_C.MODEL.HEAD = ConfigurationNode()
_C.MODEL.HEAD.NAME = 'ursonet_pytorch'
_C.MODEL.HEAD.ORI = 'classification'
_C.MODEL.HEAD.POS = 'regression'
_C.MODEL.HEAD.N_ORI_BINS_PER_DIM = 12

_C.DATA = ConfigurationNode()
_C.DATA.BATCH_SIZE = 1
_C.DATA.PATH = '../datasets/speed'
_C.DATA.IMG_SIZE = (240, 240)
_C.DATA.ORI_SMOOTH_FACTOR = 3


def load_config(path=None):
    """Load configuration for FINN build. Optionally load parameters from a YAML file"""
    if path is not None:
        assert os.path.isfile(path), f'File {path} does not exist'
        _C.merge_from_file(path)
    # assert _C.MODEL.BACKBONE.NAME in ('mobilenet_v2_brevitas', 'mobilenet_v2_pytorch', 'small_brevitas')
    # assert _C.MODEL.HEAD.NAME in ('ursonet_brevitas', 'ursonet_pytorch')
    assert _C.MODEL.HEAD.ORI in ('classification', 'regression')
    assert _C.MODEL.HEAD.POS == 'regression', 'classification not implemented yet'
    return _C.clone()


def save_config(config, path=None):
    """Save the configuration to a YAML file"""
    assert os.path.exists(os.path.dirname(path)), f'Path {path} does not exists'
    config.dump(stream=open(path, 'w'))
