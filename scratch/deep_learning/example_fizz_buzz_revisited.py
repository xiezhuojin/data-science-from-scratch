import random

import tqdm

from scratch.neural_networks.example_fizz_buzz import binary_encode, \
    fizz_buzz_encode, argmax
from scratch.deep_learning.neural_networks_as_a_sequence_of_layers import Sequential
from scratch.deep_learning.the_linear_layer import Linear
from scratch.deep_learning.the_layer_abstraction import Sigmoid
from scratch.deep_learning.other_activation_functions import Tanh
from scratch.deep_learning.loss_and_optimization import Momentum, SSE

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUMBER_HIDDEN = 25

random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUMBER_HIDDEN, init="uniform"),
    Tanh(),
    Linear(input_dim=NUMBER_HIDDEN, output_dim=4, init="uniform"),
    Sigmoid()
])

def fizzbuzz_accuracy(low: int, hi: int, net: Linear) -> float:
    number_correct = 0
    for n in range(low, hi):
        x = binary_encode(n)
        predicted = argmax(net.forward(x))
        actual = argmax(fizz_buzz_encode(n))
        if predicted == actual:
            number_correct += 1

    return number_correct / (hi - low)

optimizer = Momentum(0.1, 0.9)
loss = SSE()

# with tqdm.trange(1000) as t:
#     for epoch in t:
#         epoch_loss = 0

#         for x, y in zip(xs, ys):
#             predicted = net.forward(x)
#             epoch_loss += loss.loss(predicted, y)
#             gradient = loss.gradient(predicted, y)
#             net.backward(gradient)
#             optimizer.step(net)

#         accuracy = fizzbuzz_accuracy(101, 1024, net)
#         t.set_description(f"fb loss: {epoch_loss:.2f}")
    
# print("test_results", fizzbuzz_accuracy(1, 101, net))

# After 1,000 training iterations, the model gets 90% accuracy on the test set; 
# if you keep training it longer, it should do even better. (I don’t think it’s 
# possible to train to 100% accuracy with only 25 hidden units, but it’s 
# definitely possible if you go up to 50 hidden units.)
