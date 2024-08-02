import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import Vector, dot
from scratch.neural_networks.perceptrons import step_function


# The topology of the brain is enormously complicated, so it’s common to 
# approximate it with an idealized feed-forward neural network that consists of 
# discrete layers of neurons, each connected to the next. This typically entails 
# an input layer (which receives inputs and feeds them forward unchanged), one 
# or more “hidden layers” (each of which consists of neurons that take the 
# outputs of the previous layer, performs some calculation, and passes the 
# result to the next layer), and an output layer (which produces the final
# outputs).

# Just like in the perceptron, each (noninput) neuron has a weight corresponding 
# to each of its inputs and a bias. To make our representation simpler, we’ll 
# add the bias to the end of our weights vector and give each neuron a bias 
# input that always equals 1.

# As with the perceptron, for each neuron we’ll sum up the products of its 
# inputs and its weights. But here, rather than outputting the step_function 
# applied to that product, we’ll output a smooth approximation of it. Here we’ll 
# use the sigmoid function.

def sigmoid(t: float) -> float:
    """
    Same as logistic.
    """

    return 1 / (1 + math.exp(-t))

# Why use sigmoid instead of the simpler step_function? In order to train a 
# neural network, we need to use calculus, and in order to use calculus, we need 
# smooth functions. step_function isn’t even continuous, and sigmoid is a good 
# smooth approximation of it.

xs = np.linspace(-10, 10, 100)
step_ys = [step_function(x) for x in xs]
sigmoid_ys = [sigmoid(x) for x in xs]
plt.plot(xs, sigmoid_ys)
plt.plot(xs, step_ys, linestyle="--")
plt.title("The sigmoid function")
plt.show()

# We then calculate the output as:

def neuron_output(weight: Vector, inputs: Vector) -> float:
    """
    Weights includes the bias term, input includes a 1.
    """

    return sigmoid(dot(weight, inputs))

# Given this function, we can represent a neuron simply as a vector of weights 
# whose length is one more than the number of inputs to that neuron (because of 
# the bias weight). Then we can represent a neural network as a list of (noninput) 
# layers, where each layer is just a list of the neurons in that layer.

# That is, we’ll represent a neural network as a list (layers) of lists (neurons) 
# of vectors (weights).

# Given such a representation, using the neural network is quite simple:

def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """

    outputs: List[Vector] = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
            for neuron in layer]
        outputs.append(output)

        # Then the input to the next layer is the output of this one
        input_vector = output

    return outputs

# Now it’s easy to build the XOR gate that we couldn’t build with a single perceptron.
# We just need to scale the weights up so that the neuron_outputs are either 
# really close to 0 or really close to 1:

xor_network = [
    # hidden layer
    [
        [20, 20, -30], # 'and neuron
        [20, 20, -10], # 'or' neuron
    ],
    # output layer
    [
        [-60, 60, -30], # '2nd input but not 1st input' neuron
    ]
]

# feed_forward returns the outputs of all layers, so the [-1] gets the
# final output, and the [0] gets the value out of the resulting vector
assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001

# For a given input (which is a two-dimensional vector), the hidden layer 
# produces a two-dimensional vector consisting of the “and” of the two input 
# values and the “or” of the two input values.

# And the output layer takes a two-dimensional vector and computes “second 
# element but not first element.” The result is a network that performs “or, 
# but not and,” which is precisely XOR.

# One suggestive way of thinking about this is that the hidden layer is 
# computing features of the input data (in this case “and” and “or”) and the 
# output layer is combining those features in a way that generates the desired 
# output.
