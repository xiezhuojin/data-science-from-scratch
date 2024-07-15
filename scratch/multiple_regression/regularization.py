import random
from typing import List

import tqdm

from scratch.linear_algebra.vectors import Vector, dot, add, vector_mean
from scratch.gradient_descent.using_the_gradient import gradient_step
from scratch.multiple_regression.fitting_the_model import error, sqerror_gradient, \
    inputs, daily_minutes_good, learning_rate, beta
from scratch.multiple_regression.goodness_of_fit import multiple_r_squared

# In practice, you’d often like to apply linear regression to datasets with 
# large numbers of variables.  This creates a couple of extra wrinkles. First, 
# the more variables you use, the more likely you are to overfit your model to 
# the training set. And second, the more nonzero coefficients you have, the 
# harder it is to make sense of them. If the goal is to explain some phenomenon,
#  a sparse model with three factors might be more useful than a slightly better 
# model with hundreds.

# Regularization is an approach in which we add to the error term a penalty that 
# gets larger as beta gets larger.  We then minimize the combined error and 
# penalty.  The more importance we place on the penalty term, the more we 
# discourage large coefficients.

# For example, in ridge regression, e add a penalty proportional to the sum of 
# the squares of the beta_i (except that typically we don’t penalize beta_0, 
# the constant term):

# alpha is a *hyperparameter* controlling how harsh the penalty is.
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector, y: float, beta: Vector, alpha: float) -> float:
    """
    Estimate error plus ridge penalty on beta.
    """

    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

# We can then plug this into gradient descent in the usual way:

def ridge_penalty_gradient(beta: Vector, alpha: Vector) -> Vector:
    """
    Gradient of just the ridge penalty.
    """

    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: Vector, y: float, beta: Vector, alpha: float) -> Vector:
    """
    The gradient corresponding to the ith squared error term including the ridge 
    penalty.
    """

    return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))

# And then we just need to modify the least_squares_fit function to use the 
# sqerror_ridge_gradient instead of sqerror_gradient.

def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate: float=0.001,
                            num_steps: int=1000,
                            batch_size: int=1) -> Vector:
    """
    Find the beta that minimizes the sum of squared errors assuming the model 
    y = dot(x, beta)
    """

    # start with a random guess
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start: start + batch_size]
            batch_ys = ys[start: start + batch_size]
            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

# With alpha set to 0, there’s no penalty at all and we get the same results as 
# before:

random.seed(0)
beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1, # alpha
                                   learning_rate, 5000, 25)

# [30.8, 0.95, -1.83, 0.54]
assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69

beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1,  # alpha
                                 learning_rate, 5000, 25)
# [30.6, 0.90, -1.68, 0.10]
assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good,10,  # alpha
                                  learning_rate, 5000, 25)
# [28.3, 0.67, -0.90, -0.01]
assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6

print(beta)
print(beta_0_1)
print(beta_1)
print(beta_10)

# In particular, the coefficient on "PhD" vanishes as we increase the penalty, 
# which accords with our previous result that it wasn't significantly different 
# from 0.

# Another approach is lasso regression, which use the penalty:
def lasso_penalty(beta: Vector, alpha: float) -> float:
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])

# Whereas the ridge penalty shrank the coefficients overall, the lasso penalty 
# tends to force coefficients to be 0, which makes it good for learning sparse 
# models. Unfortunately, it’s not amenable to gradient descent, which means 
# that we won’t be able to solve it from scratch.