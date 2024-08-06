from typing import Iterable, Tuple

from scratch.neural_networks.feed_forward_neural_networks import sigmoid
from scratch.deep_learning.the_tensor import Tensor, tensor_apply, tensor_combine


# In the previous chapter we built a simple neural net that allowed us to stack 
# two layers of neurons, each of which computed sigmoid(dot(weights, inputs)).

# Although that’s perhaps an idealized representation of what an actual neuron 
# does, in practice we’d like to allow a wider variety of things. Perhaps we’d 
# like the neurons to remember something about their previous inputs. Perhaps 
# we’d like to use a different activation function than sigmoid. And frequently 
# we’d like to use more than two layers. (Our feed_forward function actually 
# handled any number of layers, but our gradient computations did not.)

# In this chapter we’ll build machinery for implementing such a variety of 
# neural networks. Our fundamental abstraction will be the Layer, something that 
# knows how to apply some function to its inputs and that knows how to 
# backpropagate gradients.

class Layer:
    """
    Our neural networks will be computed of layers, each of which knows how to 
    do some computation on its inputs inthe "forward" direction and propagate 
    gradients in the backward" direction.
    """

    def forward(self, input):
        """
        Note the lack of types, We're not going to be prescriptive about what 
        kinds of inputs layers can take and what kinds of outputs they can 
        return.
        """

        raise NotImplementedError

    def backward(self, gradient):
        """
        Similarly, we're not going to be prescriptive about what the gradient 
        looks like, It's up to you the user to make sure that you're doing things 
        sensibly.
        """

        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """
        Returns the parameters fo this layer, The default implementation returns 
        nothing, so that if you have a layer with no parameters you don't have 
        to implement this.
        """

        return ()

    def grads(self) -> Iterable[Tensor]:
        """
        Returns the gradients, in the same order as params().
        """

        return ()

# The forward and backward methods will have to be implemented in our concrete 
# subclasses. Once we build a neural net, we’ll want to train it using gradient 
# descent, which “means we’ll want to update each parameter in the network 
# using its gradient. Accordingly, we insist that each layer be able to tell us 
# its parameters and gradients.

# Some layers (for example, a layer that applies sigmoid to each of its inputs)
# have no parameters to update, so we provide a default implementation that
# handles that case.

# Let’s look at that layer:

class Sigmoid(Layer):

    def forward(self, input: Tensor):
        """
        Apply sigmoid to each elment of the input tensor, and save the results 
        to use in backpropagation.
        """

        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids

    def backward(self, gradient: Tensor):
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                              self.sigmoids,
                              gradient)

# There are a couple of things to notice here. One is that during the forward 
# pass we saved the computed sigmoids so that we could use them later in the 
# backward pass. Our layers will typically need to do this sort of thing.

# Second, you may be wondering where the sig * (1 - sig) * grad comes from. 
# This is just the chain rule from calculus and corresponds to the 
# output * (1 - output) * (output - target) term in our previous neural networks.

# Finally, you can see how we were able to make use of the tensor_apply and the 
# tensor_combine functions. Most of our layers will use these functions similarly.