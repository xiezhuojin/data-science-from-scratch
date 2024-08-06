import math

import matplotlib.pyplot as plt
import numpy as np

from scratch.deep_learning.the_tensor import Tensor, tensor_apply, tensor_combine
from scratch.deep_learning.the_layer_abstraction import Layer


# The sigmoid function has fallen out of favor for a couple of reasons. One 
# reason is that sigmoid(0) equals 1/2, which means that a neuron whose inputs 
# sum to 0 has a positive output. Another is that its gradient is very close to 
# 0 for very large and very small inputs, which means that its gradients can get 
# “saturated” and its weights can get stuck.

# One popular replacement is tanh (“hyperbolic tangent”), which is a different 
# sigmoid-shaped function that ranges from –1 to 1 and outputs 0 if its input 
# is 0. The derivative of tanh(x) is just 1 - tanh(x) ** 2, which makes the 
# layer easy to write:

def tanh(x: float) -> float:
    # If x is very large or very small, tanh is (essentially) 1 or -1.
    # We check for this because, e.g., math.exp(1000) raises an error
    if x < -100: return -1
    elif x > 100: return 1

    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)

xs = np.linspace(-200, 200, 1000)
ys = [tanh(x) for x in xs]

plt.plot(xs, ys)
plt.title("tanh")
plt.show()


class Tanh(Layer):

    def forward(self, input: Tensor) -> Tensor:
        # Save tanh output to use in backward pass.
        self.tanh = tensor_apply(tanh, input)
        return self.tanh

    def backward(self, gradient):
        return tensor_combine(
            lambda tanh, grad: (1 - tanh ** 2) * grad,
            self.tanh,
            gradient
        )

# In larger networks another popular replacement is Relu, which is 0 for 
# negative inputs and the identity for positive inputs:

def relu(x: float) -> float:
    return max(x, 0)

ys = [relu(x) for x in xs]

plt.plot(xs, ys)
plt.title("relu")
plt.show()


class Relu(Layer):

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return tensor_apply(relu, input)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
            lambda x, grad: grad if x > 0 else 0,
            self.input,
            gradient
        )
