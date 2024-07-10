from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scratch.linear_algebra.vectors import Vector
from scratch.statistics.describing_a_single_set_of_data import mean, de_mean
from scratch.statistics.correlation import correlation, standard_deviation, \
    num_friends_good, daily_minutes_good


def predict(alpha: float, beta: float, x_i: float) -> float:

    return  beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha when the actual value is y_i.
    """

    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum([error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y)])

# The lease squares solution is to choose the alpha and beta that make 
# sum_of_sqerrors as small as possible.

# Using calculus (or tedious algebra), the error-minimizing alpha and beta are 
# give by:

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y, find the least-squares of alpha and beta.
    """

    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)

    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]
# Should find that y = 3x - 5
assert least_squares_fit(x, y) == (-5, 3)

# Now it's easy to apply this to the outlierless data from Chapter 5:
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905

plt.scatter(num_friends_good, daily_minutes_good)
plt.title("Simple Linear Regression Model")
xs = [x for x in range(50)]
plt.plot(xs, [beta * x + alpha for x in xs])
plt.show()

data = pd.DataFrame(
    [[x, y] for x, y in zip(num_friends_good, daily_minutes_good)],
    columns=["x", "y"]
)
sns.regplot(x="x", y="y", data=data, scatter=True, ci=None)
plt.show()

# Of course, we need a better way to figure out how well we've fit the data than 
# staring at the graph. A common measure is the coefficient of determination 
# (or R-squared), which measures the fraction of the total variation in the 
# dependent variable that is captured by the model:

def total_sum_of_squares(y: Vector) -> float:
    """
    The total squared variation of y_i's from their mean.
    """

    y = de_mean(y)

    return sum(y_i ** 2 for y_i in y)

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    The fration of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model.
    """

    return 1 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330

# Recall that we chose the alpha and beta that minimized the sum of the squared 
# prediction errors. A linear model we could have chosen is "alway predict mean(y)" 
# (corresponding to alpha = mean(y) and beta = 0), whose sum of squared errors 
# exactly equals its total sum of squares. The means an R-squared of 0, which 
# indicates a model that (obviously, in this case) performs no better than 
# jusing predicting the mean.

# Clearly, the least squares model must be at least as good as that one, which 
# means that the sum of the squared errors is at most the total sum of squares, 
# which means that the R-squared must be at least 0. And the sum of squared errors 
# must be at least 0, which means that the R-squared can be at most 1.

# The highter the number, the better our model fits the data. Here we calculate 
# an R-squared of 0.329, which tells us that our model is only sort of okay at 
# fitting the data, and that clearly there are other factors at play.
