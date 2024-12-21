from enum import Enum

class ActivationFunction(Enum) :
    """Enum for activation function options, for convenience"""
    RELU = 1
    SIGMOID = 2
    SOFTMAX = 3 # IT IS NOT WORKING SADLY; MAY TRY TO IMPLEMENT IN FUTURE
    TANH = 4