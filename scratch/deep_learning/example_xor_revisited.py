import random

import tqdm

from scratch.deep_learning.neural_networks_as_a_sequence_of_layers import Sequential
from scratch.deep_learning.the_linear_layer import Linear
from scratch.deep_learning.the_layer_abstraction import Sigmoid
from scratch.deep_learning.loss_and_optimization import GradientDescent, SSE


# Let’s see how easy it is to use our new framework to train a network that can 
# compute XOR. We start by re-creating the training data:

xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

random.seed(0)

net = Sequential([
    Linear(2, 2),
    Sigmoid(),
    Linear(2, 1)
])

# We can now write a simple training loop, except that now we can use the 
# abstractions of Optimizer and Loss. This allows us to easily try different ones:

optimizer = GradientDescent(learning_rates=0.1)
loss = SSE()

with tqdm.trange(3000) as t:
    for epoch in t:
        epoch_loss = 0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        t.set_description(f"xor loss {epoch_loss:0.3f}")

# This should train quickly, and you should see the loss go down. And now we can 
# inspect the weights:

for param in net.params():
    print(param)
    