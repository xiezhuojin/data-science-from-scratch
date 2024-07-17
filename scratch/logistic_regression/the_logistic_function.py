import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import Vector, dot, vector_sum


def logistic(x: float) -> float:
    return 1.0 / (1 + math.exp(-x))

# also known as sigmod
sigmod = logistic

xs = np.linspace(-10, 10, 100)
ys = [logistic(x) for x in xs]

plt.plot(xs, ys)
plt.title("logistic function")
plt.show()

# As its input gets large and positive, it gets closer and closer to 1. As its 
# input gets large and negative, it gets closer and closer to 0.  Additionally, 
# it has the convenient property that its derivative is given by:

def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)

def _negative_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    """
    The negative log likelihood for one data point.
    """

    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))

def negative_log_likelihood(xs: List[Vector], ys: List[float], beta: Vector) -> float:
    return sum(_negative_log_likelihood(x, y, beta) for x, y in zip(xs, ys))

def _negative_log_partical_j(x: Vector, y: float, beta: Vector, j: int) -> float:
    """
    The jth partial derivative for one data point. Here i is the index of the data
    point.
    """

    return -(y - logistic(dot(x, beta))) * x[j]

def _negative_log_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    """
    The gradient for one data point.
    """

    return [_negative_log_partical_j(x, y, beta, j) for j in range(len(beta))]

def negative_log_gradient(xs: List[Vector], ys: List[float], beta: Vector) -> Vector:
    return vector_sum([_negative_log_gradient(x, y, beta) for x, y in zip(xs, ys)])