import numpy as np
from network import Network
from utils import ActivationFunction
from keras.api.datasets import mnist

def xor() :
    """Runs the network on the XOR logical operator to make sure that it is actually functional"""
    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[1,0], [0,1], [0,1], [1,0]]) # We are looking at probability of categorization

    net = Network(2, 16, [(2, ActivationFunction.SIGMOID), 
                      (600, ActivationFunction.SIGMOID),
                      (60, ActivationFunction.SIGMOID),
                      (4, ActivationFunction.RELU)])

    net.train(x_train, y_train, epochs=500, learning_rate=0.4)
    out = net.predict(x_train)
    print(out)

def mnist_run() :
    """Runs the specified network on the MNIST dataset for digit recognition"""
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # Preparing X: flatten and normalize
    train_X_flattened = train_X.reshape(train_X.shape[0],(train_X.shape[1]*train_X.shape[2]))
    test_X_flattened = test_X.reshape(test_X.shape[0],(test_X.shape[1]*test_X.shape[2]))
    # Accuracy improves massively as a result of normalization! dDon't use the raw data!
    train_X_flattened = train_X_flattened/255
    test_X_flattened = test_X_flattened/255

    # Preparing y: transform values to one-hot encoded vectors for classification
    encoded_train_y = np.zeros((train_y.size, train_y.max() + 1))
    encoded_train_y[np.arange(train_y.size), train_y] = 1

    encoded_test_y = np.zeros((test_y.size, test_y.max() + 1))
    encoded_test_y[np.arange(test_y.size), test_y] = 1

    # Initialize the network and train
    # Best performance was achieved with sigmoid activation function for all layers (ReLU didn't work)
    net = Network(encoded_train_y.shape[1], 128, [
        (train_X_flattened.shape[1], ActivationFunction.SIGMOID), 
                      (256, ActivationFunction.SIGMOID),
                      (80, ActivationFunction.SIGMOID),
                      (40, ActivationFunction.SIGMOID)]) 
    net.train(train_X_flattened, encoded_train_y, epochs=1500, learning_rate=0.1)

    # get output, take the max as the predicted value and compare to the actual
    out = net.predict(test_X_flattened)
    out = (out == out.max(axis=1)[:,None]).astype(int)
    error = np.sum(np.all(out == encoded_test_y, axis=1))/encoded_test_y.shape[0]
    print(f"Accuracy: {error}")


def main() :
    mnist_run()

if __name__ == "__main__" :
    main()
