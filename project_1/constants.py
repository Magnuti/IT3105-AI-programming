from enum import Enum


class BoardType(Enum):
    TRIANGLE = "triangle"
    DIAMOND = "diamond"


class CriticType(Enum):
    TABLE = "table"
    NEURAL_NETWORK = "nn"


class BoardCell(Enum):
    EMPTY_CELL = 0
    FULL_CELL = 1
    PRUNED_CELL = 2
    JUMPED_FROM_CELL = 3
    JUMPED_TO_CELL = 4


class StateStatus(Enum):
    IN_PROGRESS = 0
    SUCCESS_FINISH = 1
    INCOMPLETE_FINISH = 2


class EpsilonDecayFunction(Enum):
    EXPONENTIAL = "exponential"
    REVERSED_SIGMOID = "reversed_sigmoid"
    LINEAR = "linear"
