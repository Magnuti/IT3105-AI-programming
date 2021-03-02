from enum import Enum


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
