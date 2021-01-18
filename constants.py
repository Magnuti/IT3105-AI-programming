from enum import Enum

# TODO set all enum names to UPPER_CASE as specified by Python's naming convention


class BoardType(Enum):
    Triangle = "triangle"
    Diamond = "diamond"


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
