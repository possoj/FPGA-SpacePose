"""
Copyright (c) 2024 Julien Posso
"""


class Ultra96:
    """Config class that contains specific parameters of Ultra96 board"""
    name = 'Ultra96'
    ip = '192.168.3.1'  # IP of the board""
    port = 22
    username = 'xilinx'
    password = 'xilinx'


class ZCU104:
    """Config class that contains specific parameters of ZCU104 board"""
    name = 'ZCU104'
    ip = '192.168.2.99'   # IP of the board
    port = 22
    username = 'xilinx'
    password = 'xilinx'


def import_board(board_name):
    """Import the config class that contains the board parameters"""
    board_list = ('Ultra96', 'ZCU104')
    assert board_name in board_list, f'Only support these boards: {board_list}'
    class_obj = globals()[board_name]
    board = class_obj()
    return board
