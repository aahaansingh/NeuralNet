import numpy as np
import random
from layer import Layer

class Network :
    """Defines a network as a list of layers"""
    def __init__(self, output_size : int, batch_size : int, layer_specs : list) :
        """Initializes an network with the given layers.

        Arguments:
            output_size: the number of features of the output
            layer_specs: A list of TUPLES containing, in order, input size and activation function"""
        self.batch_size = batch_size
        self.layers : list[Layer] = []
        for i in range(len(layer_specs)) :
            if i == len(layer_specs)-1 :
                output_layer = Layer(layer_specs[i][0], output_size, layer_specs[i][1])
                self.layers.append(output_layer)
            else :
                new_layer = Layer(layer_specs[i][0], layer_specs[i+1][0], layer_specs[i][1])
                self.layers.append(new_layer)
    
    def loss(self, output : np.ndarray, actual : np.ndarray) :
        """Calculates loss between output and the actual results"""
        # May implement cross-entropy loss w/softmax in the future
        return np.linalg.norm(output - actual) # RMSE
        
    def partial_loss(self, output : np.ndarray, actual : np.ndarray) :
        """The partial derivative of the loss, used in backprop"""
        return 2*(output-actual)/actual.size
    
    def predict(self, input_data : np.ndarray) :
        """Runs predictions (forward prop) on the input data, a 2D array whose rows are
        individual samples"""
        result = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                # run the sample copy thru the neural network
                output = layer.f_propagate(output)
            result.append(output)
        return np.array(result)
        
    def train(self, x : np.ndarray, y : np.ndarray, epochs : int, learning_rate) :
        """Trains the neural network on the given training data for the specified number
        of epochs and at the specified learning rate"""
        for epoch in range(epochs) :
            avg_err = 0
            selected_samples = range(len(y))
            # batch selection
            if self.batch_size < len(y) :
                selected_samples = random.sample(selected_samples, self.batch_size)
            # conduct training step
            for i in selected_samples :
                output = x[i]
                for layer in self.layers:
                    output = layer.f_propagate(output)
                loss = self.loss(output, y[i])
                avg_err += loss
                pd_err = self.partial_loss(output, y[i])
                for layer in reversed(self.layers):
                    pd_err = layer.b_propagate(pd_err, learning_rate)
            avg_err /= len(selected_samples)
            print(f"Epoch {epoch}: Average error is {avg_err}")