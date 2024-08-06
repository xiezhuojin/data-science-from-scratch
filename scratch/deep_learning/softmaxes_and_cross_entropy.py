import math
import random

import tqdm

from scratch.deep_learning.the_tensor import Tensor, is_1d, \
    tensor_combine, tensor_sum
from scratch.deep_learning.neural_networks_as_a_sequence_of_layers import Sequential
from scratch.deep_learning.the_linear_layer import Linear
from scratch.deep_learning.other_activation_functions import Tanh
from scratch.deep_learning.loss_and_optimization import Momentum
from scratch.deep_learning.loss_and_optimization import Loss
from scratch.deep_learning.example_fizz_buzz_revisited import NUMBER_HIDDEN, xs, ys, \
    fizzbuzz_accuracy


# The neural net we used in the previous section ended in a Sigmoid layer, which 
# means that its output was a vector of numbers between 0 and 1. In particular, 
# it could output a vector that was entirely 0s, or it could output a vector 
# that was entirely 1s. Yet when we’re doing classification problems, we’d like 
# to output a 1 for the correct class and a 0 for all the incorrect classes. 
# Generally our predictions will not be so perfect, but we’d at least like to 
# predict an actual probability distribution over the classes.

# For example, if we have two classes, and our model outputs [0, 0], it’s hard
# to make much sense of that. It doesn’t think the output belongs in either class?

# But if our model outputs [0.4, 0.6], we can interpret it as a prediction that 
# there’s a probability of 0.4 that our input belongs to the first class and 0.6 
# that our input belongs to the second class.

# In order to accomplish this, we typically forgo the final Sigmoid layer and 
# instead use the softmax function, which converts a vector of real numbers to a 
# vector of probabilities. We compute exp(x) for each number in the vector, which 
# results in a vector of positive numbers. After that, we just divide each of 
# those positive numbers by the sum, which gives us a bunch of positive numbers 
# that add up to 1—that is, a vector of probabilities.

# If we ever end up trying to compute, say, exp(1000) we will get a Python error, 
# so before taking the exp we subtract off the largest value. This turns out to 
# result in the same probabilities; it’s just safer to compute in Python:

def softmax(tensor: Tensor) -> Tensor:
    """
    Softmax along the last dimension.
    """

    if is_1d(tensor):
        # Subtract largest value for numerial stability
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]

        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps for exp_i in exps]
    else:
        return [softmax(t) for t in tensor]

# Once our network produces probabilities, we often use a different loss 
# function called cross-entropy (or sometimes “negative log likelihood”).

# You may recall that in “Maximum Likelihood Estimation”, we justified the use
# of least squares in linear regression by appealing to the fact that (under 
# certain assumptions) the least squares coefficients maximized the likelihood
# of the observed data.

# Here we can do something similar: if our network outputs are probabilities,
# the cross-entropy loss represents the negative log likelihood of the observed
# data, which means that minimizing that loss is the same as maximizing the log 
# likelihood (and hence the likelihood) of the training data.

# Typically we won’t include the softmax function as part of the neural network 
# itself. This is because it turns out that if softmax is part of your loss 
# function but not part of the network itself, the gradients of the loss with 
# respect to the network outputs are very easy to compute.

class SoftmaxCrossEntropy(Loss):
    """
    This is the negative-log-likelihood of the observed values, given the neural 
    net model. So if we choose weights to minimize it, our model will be 
    maximizing the likelihood of the observed data.
    """

    def loss(self,  predicted: Tensor, actual: Tensor) -> float:
        # Apply softmax to ge probabilities
        probabilities = softmax(predicted)

        # This will be log p_i for the actual class i and 0 for the other classes.
        # We add a tiny amount to p to avoid taking log(0)
        likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
                                     probabilities,
                                     actual)

        # And then we just sum up the negatives.
        return -tensor_sum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)

        return tensor_combine(lambda p, actual: p - actual,
                              probabilities,
                              actual)

# If I now train the same Fizz Buzz network using SoftmaxCrossEntropy loss, I 
# find that it typically trains much faster (that is, in many fewer epochs).
# Presumably this is because it is much easier to find weights that softmax to 
# a given distribution than it is to find weights that sigmoid to a given 
# distribution.

random.seed(0)

net = Sequential([
    Linear(10, NUMBER_HIDDEN, init="uniform"),
    Tanh(),
    Linear(NUMBER_HIDDEN, 4, init="uniform")
    # No final sigmoid layer now
])

optimizer = Momentum(0.1, 0.9)
loss = SoftmaxCrossEntropy()

# with tqdm.trange(100) as t:
#     for epoch in t:
#         epoch_loss = 0.0

#         for x, y in zip(xs, ys):
#             predicted = net.forward(x)
#             epoch_loss += loss.loss(predicted, y)
#             gradient = loss.gradient(predicted, y)
#             net.backward(gradient)

#             optimizer.step(net)

#         accuracy = fizzbuzz_accuracy(101, 1024, net)
#         t.set_description(f"fb loss: {epoch_loss:.3f} acc: {accuracy:.2f}")

# # Again check results on the test net
# print("test results", fizzbuzz_accuracy(1, 101, net))
