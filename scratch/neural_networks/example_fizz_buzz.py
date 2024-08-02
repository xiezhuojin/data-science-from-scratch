from typing import List
import random

import tqdm

from scratch.linear_algebra.vectors import Vector, squared_distance
from scratch.neural_networks.backpropagation import sqerror_gradients, \
    feed_forward, gradient_step


def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

assert fizz_buzz_encode(2) == [1, 0, 0, 0]
assert fizz_buzz_encode(6) == [0, 1, 0, 0]
assert fizz_buzz_encode(10) == [0, 0, 1, 0]
assert fizz_buzz_encode(30) == [0, 0, 0, 1]

def binary_encode(x: int) -> Vector:
    binary: List[int] = []

    for _ in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary

#                             1  2  4  8 16 32 64 128 256 512
assert binary_encode(0)   == [0, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(1)   == [1, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(10)  == [0, 1, 0, 1, 0, 0, 0, 0,  0,  0]
assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0,  0,  0]
assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1,  1,  1]

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

# Next, let’s create a neural network with random initial weights. It will have 
# 10 input neurons (since we’re representing our inputs as 10-dimensional 
# vectors) and 4 output neurons (since we’re representing our targets as 
# 4-dimensional vectors). We’ll give it 25 hidden units, but we’ll use a variable 
# for that so it’s easy to change:

NUM_HIDDEN = 25

network = [
    # hidden layer: 10 inputs -> NUM_HIDDEN outputs
    [[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],

    # output layer: NUMBER_HIDDEN inputs -> 4 outputs
    [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
]

# That’s it. Now we’re ready to train. Because this is a more involved problem 
# (and there are a lot more things to mess up), we’d like to closely monitor the 
# training process. In particular, for each epoch we’ll track the sum of squared 
# errors and print them out. We want to make sure they decrease:

learning_rate = 1.0

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = feed_forward(network, x)[-1]
            epoch_loss += squared_distance(predicted, y)
            gradients = sqerror_gradients(network, x, y)

            # Take a gradient step for each neuron in each layer
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                    for layer, layer_grad in zip(network, gradients)]

        t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")

# At last we’re ready to solve our original problem. We have one remaining issue.
# Our network will produce a four-dimensional vector of numbers, but we want a 
# single prediction. We’ll do that by taking the argmax, which is the index of 
# the largest value:

def argmax(xs: list) -> int:
    """
    Returns the index of the largest value.
    """

    return xs.index(max(xs))

assert argmax([0, -1]) == 0               # items[0] is largest
assert argmax([-1, 0]) == 1               # items[1] is largest
assert argmax([-1, 10, 5, 20, -3]) == 3   # items[3] is largest

# Now we can finally solve “FizzBuzz”:

num_correct = 0

for i in range(1, 101):
    x = binary_encode(i)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizz_buzz_encode(i))
    labels = [str(i), "fizz", "buzz", "fizzbuzz"]

    print(i, labels[predicted], labels[actual])
    if actual == predicted:
        num_correct += 1

print(num_correct, "/", 100)