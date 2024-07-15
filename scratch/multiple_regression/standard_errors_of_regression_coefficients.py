from typing import Tuple, List
import random

from scratch.linear_algebra.vectors import Vector
from scratch.statistics.describing_a_single_set_of_data import standard_deviation
from scratch.probability.the_normal_distribution import normal_cdf
from scratch.multiple_regression.fitting_the_model import least_squares_fit, \
    learning_rate, inputs, daily_minutes_good
from scratch.multiple_regression.digression_the_bootstrap import bootstrap_statistic

# We can take the same approach to estimating the standard errors of our 
# regression coefficients.  We repeatedly take a bootstrap_sample of our data 
# and estimate beta based on that sample.  If the coefficient corresponding to 
# one of the independent variables (say, num_friends) doesn’t vary much across 
# samples, then we can be confident that our estimate is relatively tight. If 
# the coefficient varies greatly across samples, then we can’t be at all 
# confident in our estimate.

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]

    beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    print("bootstrap sample", beta)
    return beta

random.seed(0)
bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                      estimate_sample_beta,
                                      100)

# After which we can estimate the standard deviation of each coefficient:
bootstrap_standard_errors = [
    standard_deviation(beta[i] for beta in bootstrap_betas)
    for i in range(4)
]

print(bootstrap_standard_errors)

def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
        # if the coefficient is positive, we need to compute twice the probability 
        # of seeing an even *larger* value
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        # otherwise twich the probability of seeing a *smaller* value
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)

assert p_value(30.58, 1.27)   < 0.001  # constant term
assert p_value(0.972, 0.103)  < 0.001  # num_friends
assert p_value(-1.865, 0.155) < 0.001  # work_hours
assert p_value(0.923, 1.249)  > 0.4    # phd

# While most of the coefficients have very small p-values (suggesting that they 
# are indeed nonzero), the coefficient for “PhD” is not “significantly” different 
# from 0, which makes it likely that the coefficient for “PhD” is random rather 
# than meaningful.