import numpy as np
import nnfs
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data
nnfs.init()

# Define a simple dense layer class
# This class will handle the weights, biases, and forward pass of the layer 
class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation function
class Activation_ReLU:
     # Forward pass
     def forward(self, inputs):
          # Calculate output values from input
          self.output = np.maximum(0, inputs)

#Softmax activation
class Activation_Softmax:
     # Forward pass
     def forward(self, inputs):
          # Get unnormalized probabilities
          exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
          # Normalize them for each sample
          probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
          self.output = probabilities
          

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Visualize dataset
plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Let's see initial weights and biases
print(">>> Initial weights and biases of the first layer:")
print(dense1.weights)
print(dense1.biases)
print("_______________________________________")

# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print(">>> Output of the first few samples:")
print(dense1.output[:5])
print("_______________________________________")

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print(activation1.output[:5])
print("_______________________________________")

# Visualize output of the first layer
plt.scatter(dense1.output[:, 0], dense1.output[:, 1], c=y, cmap='brg')
plt.title("Output before activation")
plt.show()
# Visualize output of the first layer with ReLU activation
plt.scatter(activation1.output[:, 0], activation1.output[:, 1], c=y, cmap='brg')
plt.title("Output after ReLU activation")
plt.show()

#  Creat second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# Creat Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])
# Visualize output of the second layer
plt.scatter(dense2.output[:, 0], dense2.output[:, 1], c=y, cmap='brg')
plt.title("Output of second layer before activation")
plt.show()
# Visualize output of the second layer with Softmax activation
plt.scatter(activation2.output[:, 0], activation2.output[:, 1], c=y, cmap='brg')
plt.title("Output of second layer after Softmax activation")
plt.show()
# Print final output shape
print("Final output shape:", activation2.output.shape)
