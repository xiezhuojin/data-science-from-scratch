import random

import tqdm

from scratch.machine_learning.overfitting_and_underfitting import train_test_split
from scratch.gradient_descent.using_the_gradient import gradient_step
from scratch.working_with_data.rescaling import scale
from scratch.logistic_regression.the_problem import rescaled_xs, ys, xs
from scratch.logistic_regression.the_logistic_function import negative_log_gradient, \
    negative_log_likelihood


random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(rescaled_xs, ys, 0.33)

learning_rate = 0.01

beta_unscaled = [random.random() for _ in range(3)]
with tqdm.trange(5000) as t:
    for epoch in t:
        gradient = negative_log_gradient(x_train, y_train, beta_unscaled)
        beta_unscaled = gradient_step(beta_unscaled, gradient, -learning_rate)
        loss = negative_log_likelihood(x_train, y_train, beta_unscaled)
        t.set_description(f"loss: {loss:0.3f} beta: {beta_unscaled}")
print(beta_unscaled)

# These are coeffients for the rescaled data, but we can transform them back to 
# the original data as well:

means, stdevs = scale(xs)
beta_unscaled = [
    beta_unscaled[0] - beta_unscaled[1] * means[1] / stdevs[1] - beta_unscaled[2] * means[2] / stdevs[2],
    beta_unscaled[1] / stdevs[1],
    beta_unscaled[2] / stdevs[2],
]
print(beta_unscaled)

# Unfortunately, these are not as easy to interpret as linear regression 
# coefficients.  All else being equal, an extra year of experience adds 1.6 to 
# the input of logistic.  All else being equal, an extra $10,000 of salary 
# subtracts 2.88 from the input of logistic.

# The impact on the output, however, depends on the other inputs as well. If 
# dot(beta, x_i) is already large (corresponding to a probability close to 1), 
# increasing it even by a lot cannot affect the probability very much. If it’s 
# close to 0, increasing it just a little might increase the probability quite 
# a bit.

# What we can say is that—all else being equal—people with more experience are 
# more likely to pay for accounts.  And that—all else being equal—people with 
# higher salaries are less likely to pay for accounts.  (This was also somewhat 
# apparent when we plotted the data.)