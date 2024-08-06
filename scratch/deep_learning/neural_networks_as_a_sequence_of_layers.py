from typing import List, Iterable

from scratch.deep_learning.the_tensor import Tensor
from scratch.deep_learning.the_layer_abstraction import Layer
from scratch.deep_learning.the_linear_layer import Linear
from scratch.deep_learning.the_layer_abstraction import Sigmoid


# We’d like to think of neural networks as sequences of layers, so let’s come 
# up with a way to combine multiple layers into one. The resulting neural
# network is itself a layer, and it implements the Layer methods in the
# obvious ways:

class Sequential(Layer):
    """
    A layer consisting of a sequence fo other layers, It's up to you to make sure 
    that the output of each layer makes sense as the input to the next layer.
    """

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        """
        Just forward the input through the layers in order.
        """

        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, gradient):
        """
        Just backpropagate the gradient through the layers in reverse.
        """

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def params(self) -> Iterable[Tensor]:
        """
        Just return the params from each layer.
        """

        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        """
        Just return grads from each layer.
        """

        return (grad for layer in self.layers for grad in layer.grads())

# “o we could represent the neural network we used for XOR as:

xor_net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1),
    Sigmoid()
])