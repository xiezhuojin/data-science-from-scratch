from typing import List
import random
from pprint import pprint

import tqdm

from scratch.linear_algebra.vectors import Vector, dot
from scratch.neural_networks.feed_forward_neural_networks import feed_forward
from scratch.gradient_descent.using_the_gradient import gradient_step


# Usually we don’t build neural networks by hand. This is in part because we 
# use them to solve much bigger problems—an image recognition problem might 
# involve hundreds or thousands of neurons.  And it’s in part because we usually 
# won’t be able to “reason out” what the neurons should be.

# Instead (as usual) we use data to train neural networks. The typical approach 
# is an algorithm called backpropagation, which uses gradient descent or one of 
# its variants.

# Imagine we have a training set that consists of input vectors and corresponding 
# target output vectors.  For example, in our previous xor_network example, the 
# input vector [1, 0] corresponded to the target output [1]. Imagine that our 
# network has some set of weights.  We then adjust the weights using the 
# following algorithm:

#   1. Run feed_forward on an input vector to produce the outputs of all the 
#      neurons in the network.
#   2. We know the target output, so we can compute a loss that’s the sum of the 
#      squared errors.
#   3. Compute the gradient of this loss as a function of the output neuron’s weights.
#   4. "Propagate” the gradients and errors backward to compute the gradients
#      with respect to the hidden neurons’ weights.
#   1. Take a gradient descent step.

# Typically we run this algorithm many times for our entire training set until 
# the network converges.

# To start with, let’s write the function to compute the gradients:

def sqerror_gradients(network: List[List[Vector]], input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector, and a target vector, make a prediction 
    and compute the gradient of the sqquared error loss with respect to the 
    neuron weights.
    """

    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # gradients with respoect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]
    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                      for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]
    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                        dot(output_deltas, [n[i] for n in network[-1]])
                    for i, hidden_output in enumerate(hidden_outputs)]
    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]
    
    return [hidden_grads, output_grads]

# The math behind the preceding calculations is not terribly difficult, but it 
# involves some tedious calculus and careful attention to detail, so I’ll leave 
# it as an exercise for you.

# Armed with the ability to compute gradients, we can now train neural networks. 
# Let’s try to learn the XOR network we previously designed by hand.

# We’ll start by generating the training data and initializing our neural 
# network with random weights:

random.seed(0)

# training data
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

# start with random weights
network = [ # hidden layer: 2 inputs -> 2 outputs
            [[random.random() for _ in range(2 + 1)],   # 1st hidden neuron
             [random.random() for _ in range(2 + 1)]],  # 2nd hidden neuron
            # output layer: 2 inputs -> 1 output
            [[random.random() for _ in range(2 + 1)]]   # 1st output neuron
          ]

# As usual, we can train it using gradient descent. One difference from our 
# previous examples is that here we have several parameter vectors, each with 
# its own gradient, which means we’ll have to call gradient_step for each of them.

learning_rate = 1.0

for epoch in tqdm.trange(20000, desc="neural net for xor"):
    for x, y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)

        # Take a gradient step for each neuron in each layer
        network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                   for layer, layer_grad in zip(network, gradients)]

# check that it learned XOR
assert feed_forward(network, [0, 0])[-1][0] < 0.01
assert feed_forward(network, [0, 1])[-1][0] > 0.99
assert feed_forward(network, [1, 0])[-1][0] > 0.99
assert feed_forward(network, [1, 1])[-1][0] < 0.01

pprint(network)