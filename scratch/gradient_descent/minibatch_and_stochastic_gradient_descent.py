import random
from typing import TypeVar, List, Iterator

from scratch.linear_algebra.vectors import vector_mean
from scratch.gradient_descent.using_the_gradient import gradient_step
from scratch.gradient_descent.using_gradient_descent_to_fix_models import linear_gradient, inputs

# minibatch gradient descent, in which we compute the gradient based on a "minibatch"
# sampled from the larger dataset:

T = TypeVar("T") # this allow us to type "generic" functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool=True) -> Iterator:
    """
    Generates `batch-size`-sized minibatches from the dataset.
    """

    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle:
        random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
learning_rate = 0.001
for epoch in range(5_000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"

# Another variation is stochastic gradient descent, in which you take gradient
# steps based on one traning example at a time:
for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, step_size=-learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"

# The terminology for the various flavors of gradient descent is not uniform. The
# "compute the gradient for the whole dataset" approach is often called batch
# gradient descent, and some people say stochastic gradient descent when referring
# to the minibatch version (of which one-point-at-at-time version is a special
# case).