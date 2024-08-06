from typing import Iterable
import random

from scratch.working_with_data.exploring_your_data import inverse_normal_cdf
from scratch.linear_algebra.vectors import dot
from scratch.deep_learning.the_layer_abstraction import Layer
from scratch.deep_learning.the_tensor import Tensor, shape


# The other piece we’ll need to duplicate the neural networks from Chapter 18 is 
# a “linear” layer that represents the dot(weights, inputs) part of the neurons.

# This layer will have parameters, which we’d like to initialize with random values.

# It turns out that the initial parameter values can make a huge difference in 
# how quickly (and sometimes whether) the network trains. If weights are too 
# big, they may produce large outputs in a range where the activation function 
# has near-zero gradients. And parts of the network that have zero gradients 
# necessarily can’t learn anything via gradient descent.

# Accordingly, we’ll implement three different schemes for randomly generating 
# our weight tensors. The first is to choose each value from the random uniform 
# distribution on [0, 1]—that is, as a random.random(). The second (and default) 
# is to choose each value randomly from a standard normal distribution. And the 
# third is to use Xavier initialization, where each weight is initialized with 
# a random draw from a normal distribution with mean 0 and 
# variance 2 / (num_inputs + num_outputs). It turns out this often works nicely 
# for neural network weights. We’ll implement these with a random_uniform 
# function and a random_normal function:

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int, mean: float=0.0, variance: float=1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]

# And then wrap them all in a random_tensor function:

def random_tensor(*dims: int, init: str="normal") -> Tensor:
    if init == "normal":
        return random_normal(*dims)
    elif init == "uniform":
        return random_uniform(*dims)
    elif init == "xavier":
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")

# Now we can define our linear layer. We need to initialize it with the dimension 
# of the inputs (which tells us how many weights each neuron needs), the 
# dimension of the outputs (which tells us how many neurons we should have), and 
# the initialization scheme we want:

class Linear(Layer):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 init: str="xavier") -> None:
        """
        A layer of output_dim neurons, each with input_dim weights (and a bias).
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.w[o] is the weights for the oth neuron
        self.w = random_tensor(output_dim, input_dim, init=init)

        # self.b[o] is the bias for the oth neuron.
        self.b = random_tensor(output_dim, init=init)

# The forward method is easy to implement. We’ll get one output per neuron, 
# which we stick in a vector. And each neuron’s output is just the dot of its 
# weights with the input, plus its bias:

    def forward(self, input: Tensor) -> Tensor:
        # Save the input to use in the backward pass.
        self.input = input

        # Return the vector of neuron outputs.
        return [dot(input, self.w[o]) + self.b[o]
                for o in range(self.output_dim)]

# The backward method is more involved, but if you know calculus it’s not 
# difficult:

    def backward(self, gradient: Tensor) -> Tensor:
        # Each b[o] gets added to output[o], which means
        # the gradient of b is the same as the output gradient.
        self.b_grad = gradient

        # Each w[o][i] multiplies input[i] and gets added to output[o].
        # So its gradient is input[i] * gradient[o].
        self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]

        # Each input[i] multiplies every w[o][i] and gets added to every
        # output[o]. So its gradient is the sum of w[o][i] * gradient[o]
        # across all the outputs.
        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
                for i in range(self.input_dim)]

# Finally, here we do need to implement params and grads. We have two parameters 
# and two corresponding gradients:

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]
