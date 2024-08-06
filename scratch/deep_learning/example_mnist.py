from typing import List
import random

import tqdm
import mnist
import matplotlib.pyplot as plt

from scratch.deep_learning.the_tensor import shape, tensor_sum, Tensor
from scratch.deep_learning.the_layer_abstraction import Layer
from scratch.deep_learning.neural_networks_as_a_sequence_of_layers import Sequential
from scratch.deep_learning.loss_and_optimization import Loss, Optimizer, Momentum
from scratch.deep_learning.the_linear_layer import Linear
from scratch.deep_learning.softmaxes_and_cross_entropy import SoftmaxCrossEntropy
from scratch.deep_learning.other_activation_functions import Tanh
from scratch.deep_learning.dropout import Dropout
from scratch.neural_networks.example_fizz_buzz import argmax

# MNIST is a dataset of handwritten digits that everyone uses to learn deep learning.
# It is available in a somewhat tricky binary format, so we’ll install the mnist 
# library to work with it.

# And then we can load the data:

mnist.temporary_dir = lambda: "/workspaces/data-science-from-scratch/data-science-from-scratch/mnist"

train_images = mnist.train_images().tolist()
train_labels = mnist.train_labels().tolist()

assert shape(train_images) == [60000, 28, 28]
assert shape(train_labels) == [60000]

# Let’s plot the first 100 training images to see what they look like
fig, ax = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        ax[i][j].imshow(train_images[10 * i + j], cmap="Greys")
        ax[i][j].xaxis.set_visible(False)
        ax[i][j].yaxis.set_visible(False)
plt.show()

# Each image is 28 × 28 pixels, but our linear layers can only deal with
# one-dimensional inputs, so we’ll just flatten them (and also divide by 256 to 
# get them between 0 and 1). In addition, our neural net will train better if our 
# inputs are 0 on average, so we’ll subtract out the average value:

# Compute the average pixel value
avg = tensor_sum(train_images) / 60000 / 28 / 28

test_images = mnist.test_images().tolist()
test_labels = mnist.test_labels().tolist()

# Recenter, rescale, and flatten
train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                for image in train_images]
test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                for image in test_images]

assert shape(train_images) == [60000, 784], "images should be flattened"
assert shape(test_images) == [10000, 784], "images should be flattened"

# After centering, average pixel should be very close to 0
assert -0.0001 < tensor_sum(train_images) < 0.0001

# We also want to one-hot-encode the targets, since we have 10 outputs. First 
# let’s write a one_hot_encode function:

def one_hot_encode(i: int, num_labels: int=10) -> List[float]:
    return [1 if j == i else 0 for j in range(num_labels)]

assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert one_hot_encode(2, num_labels=5) == [0, 0, 1, 0, 0]

# and then apply it to our data:
train_labels = [one_hot_encode(label) for label in train_labels]
test_labels = [one_hot_encode(label) for label in test_labels]

assert shape(train_labels) == [60000, 10]
assert shape(test_labels) == [10000, 10]

# One of the strengths of our abstractions is that we can use the same 
# training/evaluation loop with a variety of models. So let’s write that first. 
# We’ll pass it our model, the data, a loss function, and (if we’re training) 
# an optimizer. It will make a pass through our data, track performance, and
# (if we passed in an optimizer) update our parameters:

def loop(model: Layer,
         images: List[Tensor],
         labels: List[Tensor],
         loss: Loss,
         optimizer: Optimizer=None) -> None:
    correct = 0
    total_loss = 0.0

    with tqdm.trange(len(images)) as t:
        for i in t:
            predicted = model.forward(images[i])             # Predict.
            if argmax(predicted) == argmax(labels[i]):       # Check for
                correct += 1                                 # correctness.
            total_loss += loss.loss(predicted, labels[i])    # Compute loss.

            # If we're training, backpropagate gradient and update weights.
            if optimizer is not None:
                gradient = loss.gradient(predicted, labels[i])
                model.backward(gradient)
                optimizer.step(model)

            # And update our metrics in the progress bar.
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            t.set_description(f"mnist loss: {avg_loss:.3f} acc: {acc:.3f}")

random.seed(0)

model = Linear(784, 10)
loss = SoftmaxCrossEntropy()
optimizer = Momentum(0.01, 0.99)

# # Train on the traning data
# loop(model, test_images, test_labels, loss, optimizer)

# # Test on the test data (no optimizer means just evaluate)
# loop(model, test_images, test_labels, loss)

# This gets about 89% accuracy. Let’s see if we can do better with a deep neural 
# network. We’ll use two hidden layers, the first with 30 neurons, and the second 
# with 10 neurons. And we’ll use our Tanh activation:

dropout1 = Dropout(0.1)
dropout2 = Dropout(0.1)

model = Sequential([
    Linear(784, 30),
    dropout1,
    Tanh(),
    Linear(30, 10),
    dropout2,
    Tanh(),
    Linear(10, 10)
])

# And we can just use the same training loop!

optimizer = Momentum(learning_rate=0.01, momentum=0.99)
loss = SoftmaxCrossEntropy()

# Enable dropout and train (takes > 20 minutes on my laptop!)
dropout1.train = dropout2.train = True
# loop(model, train_images, train_labels, loss, optimizer)

# Disable dropout and evaluate
dropout1.train = dropout2.train = False
# loop(model, test_images, test_labels, loss)

# Our deep model gets better than 92% accuracy on the test set, which is a nice 
# improvement from the simple logistic model.