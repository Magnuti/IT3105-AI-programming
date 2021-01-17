from enum import Enum


class BoardType(Enum):
    Triangle = "triangle"
    Diamond = "diamond"


class BoardCell(Enum):
    EMPTY_CELL = 0
    FULL_CELL = 1
    PRUNED_CELL = 2
    JUMPED_FROM_CELL = 3
    JUMPED_TO_CELL = 4
