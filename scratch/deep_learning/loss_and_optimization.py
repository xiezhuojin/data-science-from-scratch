from typing import List

from scratch.deep_learning.the_tensor import Tensor, tensor_combine, tensor_sum, \
    zeros_like
from scratch.deep_learning.the_layer_abstraction import Layer

# Previously we wrote out individual loss functions and gradient functions for 
# our models. Here we’ll want to experiment with different loss functions, so 
# (as usual) we’ll introduce a new Loss abstraction that encapsulates both the 
# loss computation and the gradient computation:

class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """
        How good are our predictions? (Larger numbers are worse.)
        """

        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """
        How does the loss change as the predictions change?
        """

        raise NotImplementedError

# We’ve already worked many times with the loss that’s the sum of the squared 
# errors, so we should have an easy time implementing that. The only trick is 
# that we’ll need to use tensor_combine:

class SSE(Loss):
    """
    Loss function that computes the sum of the squared errors.
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Compute the tensor of squared differences
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2,
            predicted,
            actual
        )
        # And just add them up
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted, actual: 2 * (predicted - actual),
            predicted,
            actual
        )

# The last piece to figure out is gradient descent. Throughout the book we’ve 
# done all of our gradient descent manually by having a training loop that 
# involves something like:
# theta = gradient_step(theta, grad, -learning_rate)”

# Here that won’t quite work for us, for a couple reasons. The first is that our 
# neural nets will have many parameters, and we’ll need to update all of them.
# The second is that we’d like to be able to use more clever variants of gradient
# descent, and we don’t want to have to rewrite them each time.

# Accordingly, we’ll introduce a (you guessed it) Optimizer abstraction, of which 
# gradient descent will be a specific instance:

class Optimizer:
    """
    An optimizer updates the weights of a layer (in place) using information 
    known by either the layer or the optimizer (or by both)
    """

    def step(self, layer: Layer):
        raise NotImplementedError

# After that it’s easy to implement gradient descent, again using tensor_combine:

class GradientDescent(Optimizer):

    def __init__(self, learning_rates: float=0.1) -> None:
        self.lr = learning_rates

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # Update param using a gradient step
            param[:] = tensor_combine(
                lambda param, grad: param - self.lr * grad,
                param,
                grad
            )

# The only thing that’s maybe surprising is the “slice assignment,” which is a 
# reflection of the fact that reassigning a list doesn’t change its original 
# value. That is, if you just did param = tensor_combine(. . .), you would be 
# redefining the local variable param, but you would not be affecting the 
# original parameter tensor stored in the layer. If you assign to the slice [:], 
# however, it actually changes the values inside the list.”

# To demonstrate the value of this abstraction, let’s implement another optimizer 
# that uses momentum. The idea is that we don’t want to overreact to each new 
# gradient, and so we maintain a running average of the gradients we’ve seen, 
# updating it with each new gradient and taking a step in the direction of the 
# average:

class Momentum(Optimizer):

    def __init__(self,
                 learning_rate: float,
                 momentum: float=0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []

    def step(self, layer: Layer) -> None:
        # If we have no previous updates, start with all zeros:
        if not len(self.updates):
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
            # Apply momentum
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad
            )

            # Then take a gradient step
            param[:] = tensor_combine(
                lambda p, u: p - self.lr * u,
                param,
                update
            )

# Because we used an Optimizer abstraction, we can easily switch between our 
# different optimizers.
