import random

import tqdm

from scratch.gradient_descent.using_the_gradient import gradient_step
from scratch.simple_linear_regression.the_model import error, sum_of_sqerrors, \
    num_friends_good, daily_minutes_good

# If we write theta = [alpha, beta], we can also solve this using gradient descent:

num_epochs = 10_000
random.seed(0)

guess = [random.random(), random.random()]
learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess
        # Partial derivative of loss with respect to alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        # Partial derivative of loss with respect to alpha
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        # Compute loss to stick in tqdm description
        loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:0.3f}")

        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)


alpha, beta = guess
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905