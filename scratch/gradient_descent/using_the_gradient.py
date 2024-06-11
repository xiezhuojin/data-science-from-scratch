import random

from scratch.linear_algebra.vectors import Vector
from scratch.linear_algebra.vectors import distance, add, scaler_multiply


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Move 'step_size' in the 'gradient' direction from `v`.
    """

    assert len(v) == len(gradient)
    step = scaler_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# pick a random starting point
v = [random.uniform(-10, 10) for _ in range(3)]

for epoch in range(1_000):
    grad = sum_of_squares_gradient(v) # compute the gradient at v
    v = gradient_step(v, grad, -0.01) # take a negative gradient step
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001
