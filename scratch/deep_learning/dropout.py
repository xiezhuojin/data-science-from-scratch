import random
from operator import mul

from scratch.deep_learning.the_layer_abstraction import Layer
from scratch.deep_learning.the_tensor import Tensor, tensor_apply, tensor_combine


# Like most machine learning models, neural networks are prone to overfitting to
# their training data. We’ve previously seen ways to ameliorate this; for example, 
# in “Regularization” we penalized large weights and that helped prevent overfitting.

# A common way of regularizing neural networks is using dropout. At training time, 
# we randomly turn off each neuron (that is, replace its output with 0) with some 
# fixed probability. This means that the network can’t learn to depend on any 
# individual neuron, which seems to help with overfitting.

# At evaluation time, we don’t want to dropout any neurons, so a Dropout layer 
# will need to know whether it’s training or not. In addition, at training time 
# a Dropout layer only passes on some random fraction of its input. To make its 
# output comparable during evaluation, we’ll scale down the outputs (uniformly) 
# using that same fraction:

class Dropout(Layer):

    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True

    def forward(self, input: Tensor) -> Tensor:
        if self.train:
            # Create a mask of 0s and 1s shaped like the input using the specified 
            # probability
            self.mask = tensor_apply(
                lambda _: 0 if random.random() < self.p else 1,
                input
            )
            # Multiply by the mask to dropout inputs
            return tensor_combine(mul, input, self.mask)
        else:
            # During evaluation just scale down the outputs uniformly
            return tensor_apply(lambda x: x * (1 - self.p), input)

    def backward(self, gradient: Tensor) -> Tensor:
        if self.train:
            # Only propagate the gradients when mask == 1.
            return tensor_combine(mul, gradient, self.mask)
        else:
            raise RuntimeError("don't call backward when not in train mode")

# We’ll use this to help prevent our deep learning models from overfitting.
