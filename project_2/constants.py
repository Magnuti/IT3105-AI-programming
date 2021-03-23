from enum import Enum


class GameType(Enum):
    NIM = "nim"
    HEX = "hex"


class BoardCell(Enum):
    EMPTY_CELL = 0
    PLAYER_0_CELL = 1
    PLAYER_0_CELL_PART_OF_WINNING_PATH = 2
    PLAYER_1_CELL = 3
    PLAYER_1_CELL_PART_OF_WINNING_PATH = 4


class ActivationFunction(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"


class Optimizer(Enum):
    ADAGRAD = "Adagrad"
    SGD = "SGD"
    RMSProp = "RMSProp"
    ADAM = "Adam"
