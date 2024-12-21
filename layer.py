import numpy as np
from utils import ActivationFunction

class Layer :
    """Defines a layer, consisting of weights, biases, and an activation function"""
    def __init__(self, input_size : int, output_size : int, type : ActivationFunction) :
        """Initializes an layer of the given size and type.

        Arguments:
            input_size: the size of the previous layer, or input for the first layer
            output_size: the size of this layer's output
            type: the activation function for this layer"""
        self.weights = 2*np.random.sample((input_size, output_size))-1
        self.biases = np.random.sample(output_size) - .5
        self.type = type
        self.input = None
        self.output = None

    def activate(self, arr : np.ndarray) :
        """Computes the ReLu, Sigmoid, and Softmax activation functions.
            Note that Softmax has not been fully implemented for this network"""
        if self.type == ActivationFunction.RELU :
            return np.maximum(arr, 0)
        elif self.type == ActivationFunction.SIGMOID :
            # return 1/(1 + np.exp(-arr))
            return np.where(
                arr >= 0, # condition
                1 / (1 + np.exp(-arr)), # For positive values
                np.exp(arr) / (1 + np.exp(arr)) # For negative values
            )
        elif self.type == ActivationFunction.SOFTMAX :
            exps = np.exp(arr - np.max(arr))
            return exps / np.sum(exps)
        elif self.type == ActivationFunction.TANH :
            return np.tanh(arr)
    
    def partial(self) :
        """Computes partial derivatives of the enumerated functions"""
        if self.type == ActivationFunction.RELU :
            return (self.output >= 0)
        elif self.type == ActivationFunction.SIGMOID :
            return self.output * (1-self.output)
        elif self.type == ActivationFunction.SOFTMAX :
            return
        elif self.type == ActivationFunction.TANH :
            return 1-np.square(self.output)
    
    def f_propagate(self, input : np.ndarray) :
        """Computes the output of the layer given an input, including activation"""
        self.input = input
        raw_output = self.input @ self.weights + self.biases
        output = self.activate(raw_output)
        self.output = output
        return output
    
    def b_propagate(self, output_error : np.ndarray, learning_rate) :
        """Updates weights and biases at the given learning rate after computing their
        gradients with respect to the output"""
        inactivated_gradients = self.partial() * output_error
        # input_error will be output_error for the next layer
        input_error = inactivated_gradients @ self.weights.T
        weights_error = self.input[None].T @ inactivated_gradients[None]
        
        self.weights -= learning_rate * weights_error # negative gradient
        # bias error is the same as output error (after inactivating)
        self.biases -= learning_rate * inactivated_gradients
        return input_error