This is an implementation of a standard neural network using NumPy. The implementation allows one to add an arbitrary number of layers with an arbitrary number of neurons and using the 
ReLU, sigmoid, and tanh activation functions, although only sigmoid appears to be working properly. 

Also included in `main.py` are examples running the network on XOR and the MNIST 
dataset. The highest accuracy on MNIST, using the network and learning parameters shown, is 93%; however, it takes about 5 minutes of training and 1500 epochs to achieve this. I suspect 
that there is a bug in the backpropagation/activation functions. I also hope to add softmax and cross-entropy loss in the future.
